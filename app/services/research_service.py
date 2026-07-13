"""
知行·认知加速器 — 研究管道服务

串联完整的后端流程：搜索 → 提取事件 → 分析关系 → 组织章节 → 生成报告。
这是"搜一个词 → 看到因果脉络"的核心管道。

流程：
    1. Firecrawl 搜索关键词，获取搜索结果
    2. DeepSeek 从搜索结果提取结构化事件
    3. DeepSeek 分析事件间关联关系
    4. DeepSeek 将事件组织成章节
    5. 组装 Report JSON
    6. 存入数据库

用法：
    service = ResearchService()
    report = await service.research("AI行业")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import settings
from app.services.firecrawl_service import (
    FirecrawlSearchService,
    TavilySearchService,
    SearchResult,
)
from app.services.deepseek_service import (
    DeepSeekService,
    ExtractedEvent,
    AnalyzedRelation,
    ChapterOutline,
    GeneratedInsight,
)
from app.services.query_classifier import classify_query, QueryProfile

logger = logging.getLogger(__name__)


class ResearchError(Exception):
    """研究流程异常"""

    pass


@dataclass
class ResearchResult:
    """研究结果

    包含完整的报告数据：事件列表、关系列表、章节结构、引导摘要、AI 洞察。
    """

    query: str
    summary: str
    events: list[ExtractedEvent]
    relations: list[AnalyzedRelation]
    chapters: list[ChapterOutline]
    insight: GeneratedInsight | None = None
    query_profile: dict | None = None
    source_status: str = "real"  # "real" | "fallback"
    warning: str | None = None

    def to_dict(self) -> dict:
        """转为字典格式（用于组装 Report JSON）"""
        return {
            "query": self.query,
            "summary": self.summary,
            "events": [e.to_dict() for e in self.events],
            "relations": [r.to_dict() for r in self.relations],
            "chapters": [c.to_dict() for c in self.chapters],
            "insight": self.insight.to_dict() if self.insight else None,
            "query_profile": self.query_profile,
            "source_status": self.source_status,
            "warning": self.warning,
        }


class ResearchService:
    """研究管道服务

    串联 Firecrawl 搜索 + DeepSeek AI 处理，生成完整的研究报告。

    依赖：
        - FirecrawlSearchService（搜索）
        - DeepSeekService（事件提取 + 关系分析 + 章节组织）
    """

    def __init__(
        self,
        search_service: FirecrawlSearchService | TavilySearchService | None = None,
        ai_service: DeepSeekService | None = None,
    ) -> None:
        """初始化研究服务

        Args:
            search_service: 搜索服务，默认自动选择
                - 如果配置了 Tavily API Key，优先用 Tavily
                - 否则用 Firecrawl（需自部署）
            ai_service: DeepSeek AI 服务，默认自动创建
        """
        if search_service is None:
            if settings.TAVILY_API_KEY:
                search_service = TavilySearchService()
                logger.info("使用 Tavily 搜索服务")
            else:
                search_service = FirecrawlSearchService()
                logger.info("使用 Firecrawl 搜索服务")
        self.search_service = search_service
        self.ai_service = ai_service or DeepSeekService()

    async def research(
        self,
        query: str,
        search_limit: int = 10,
        max_events: int = 8,
    ) -> ResearchResult:
        """执行完整研究流程

        Args:
            query: 搜索关键词
            search_limit: Firecrawl 搜索结果数量上限
            max_events: 最多提取的事件数量

        Returns:
            完整的研究结果（事件 + 关系 + 章节 + 摘要）

        Raises:
            ResearchError: 流程中任一步骤失败
        """
        logger.info("开始研究流程: query='%s'", query)

        # Step 0: 主题分类
        query_profile = await classify_query(query)
        logger.info("主题分类完成: type='%s', rewritten='%s'", query_profile.topic_type, query_profile.rewritten_query[:50])

        # 搜索时使用改写后的查询词
        search_query = query_profile.rewritten_query

        # Step 1: 搜索
        try:
            search_results = await self._search(search_query, limit=search_limit, query_profile=query_profile.to_dict())
        except Exception as e:
            logger.error("搜索失败: %s", e)
            raise ResearchError(f"搜索失败: {e}") from e

        if not search_results:
            logger.warning("搜索结果为空")
            return ResearchResult(
                query=query,
                summary="未找到与该关键词相关的信息。",
                events=[],
                relations=[],
                chapters=[],
                insight=None,
                query_profile=query_profile.to_dict(),
                source_status=getattr(self, '_source_status', 'real'),
                warning=getattr(self, '_warning', None),
            )

        # Step 2: 提取事件
        try:
            events = await self._extract_events(query, search_results, max_events, query_profile.to_dict())
        except Exception as e:
            logger.error("事件提取失败: %s", e)
            raise ResearchError(f"事件提取失败: {e}") from e

        if not events:
            return ResearchResult(
                query=query,
                summary="找到了相关信息，但无法提取结构化事件。",
                events=[],
                relations=[],
                chapters=[],
                insight=None,
                query_profile=query_profile.to_dict(),
                source_status=getattr(self, '_source_status', 'real'),
                warning=getattr(self, '_warning', None),
            )

        # Step 3: 分析关系
        try:
            relations = await self._analyze_relations(query, events, query_profile.to_dict())
        except Exception as e:
            logger.warning("关系分析失败（非致命）: %s", e)
            relations = []

        # Step 4: 组织章节
        try:
            chapters = await self._organize_chapters(query, events, relations, query_profile.to_dict())
        except Exception as e:
            logger.warning("章节组织失败（非致命）: %s", e)
            chapters = []

        # Step 5: 生成洞察（非致命）
        try:
            insight = await self._generate_insight(query, events, relations, query_profile.to_dict())
        except Exception as e:
            logger.warning("洞察生成失败（非致命）: %s", e)
            insight = None

        # Step 6: 生成引导摘要
        summary = self._generate_summary(query, events, chapters)

        logger.info(
            "研究流程完成: query='%s', 事件=%d, 关系=%d, 章节=%d, 洞察=%s",
            query,
            len(events),
            len(relations),
            len(chapters),
            "有" if insight else "无",
        )

        return ResearchResult(
            query=query,
            summary=summary,
            events=events,
            relations=relations,
            chapters=chapters,
            insight=insight,
            query_profile=query_profile.to_dict(),
            source_status=getattr(self, '_source_status', 'real'),
            warning=getattr(self, '_warning', None),
        )

    async def _search(self, query: str, limit: int, query_profile: dict | None = None) -> list[SearchResult]:
        """Step 1: 搜索互联网信息

        搜索优先级：
        1. Tavily（免费，无需自部署）
        2. Firecrawl（需自部署）
        3. 兜底预设数据（保证流程能跑通）
        """
        logger.info("Step 1: 搜索 '%s'", query)

        # 尝试主搜索服务
        try:
            results = await self.search_service.search(query, limit=limit)
            if results:
                logger.info("搜索成功: 返回 %d 条结果", len(results))
                return results
            logger.warning("搜索服务返回空结果")
        except Exception as e:
            logger.warning("主搜索服务失败: %s", e)

        # 如果主服务是 Tavily，尝试降级到 Firecrawl
        if isinstance(self.search_service, TavilySearchService):
            try:
                logger.info("Tavily 失败，尝试降级到 Firecrawl")
                fallback = FirecrawlSearchService()
                results = await fallback.search(query, limit=limit)
                if results:
                    return results
            except Exception as e2:
                logger.warning("Firecrawl 也失败: %s", e2)

        # 最终兜底：预设数据
        logger.warning("所有搜索服务均不可用，使用兜底数据")
        self._source_status = "fallback"
        self._warning = "当前报告使用演示/兜底数据生成，非真实搜索结果，请勿用于正式判断。"
        return _fallback_search_results(query, query_profile)

    async def _extract_events(
        self,
        query: str,
        search_results: list[SearchResult],
        max_events: int,
        query_profile: dict | None = None,
    ) -> list[ExtractedEvent]:
        """Step 2: 用 DeepSeek 提取结构化事件"""
        logger.info("Step 2: 提取事件")
        return await self.ai_service.extract_events(query, search_results, max_events, query_profile)

    async def _analyze_relations(
        self,
        query: str,
        events: list[ExtractedEvent],
        query_profile: dict | None = None,
    ) -> list[AnalyzedRelation]:
        """Step 3: 用 DeepSeek 分析事件间关系"""
        logger.info("Step 3: 分析关系")
        return await self.ai_service.analyze_relations(query, events, query_profile)

    async def _organize_chapters(
        self,
        query: str,
        events: list[ExtractedEvent],
        relations: list[AnalyzedRelation],
        query_profile: dict | None = None,
    ) -> list[ChapterOutline]:
        """Step 4: 用 DeepSeek 组织章节"""
        logger.info("Step 4: 组织章节")
        return await self.ai_service.organize_chapters(query, events, relations, query_profile)

    async def _generate_insight(
        self,
        query: str,
        events: list[ExtractedEvent],
        relations: list[AnalyzedRelation],
        query_profile: dict | None = None,
    ) -> GeneratedInsight:
        """Step 5: 用 DeepSeek 生成洞察"""
        logger.info("Step 5: 生成洞察")
        return await self.ai_service.generate_insight(query, events, relations, query_profile)

    def _generate_summary(
        self,
        query: str,
        events: list[ExtractedEvent],
        chapters: list[ChapterOutline],
    ) -> str:
        """Step 5: 生成引导摘要（2-3 句）

        基于事件和章节信息，生成简洁的引导语。
        Phase 1 用规则生成，Task 12 可改用 AI 生成。
        """
        if not events:
            return f"关于「{query}」，暂未找到足够的信息。"

        parts = [f"关于「{query}」，"]

        if chapters:
            chapter_titles = [ch.title for ch in chapters]
            parts.append(f"以下分为 {len(chapters)} 章：{'、'.join(chapter_titles)}。")
        else:
            parts.append(f"共整理了 {len(events)} 个关键事件。")

        # 取第一个事件作为切入点
        first_event = events[0]
        parts.append(f"从「{first_event.title}」开始，")

        if len(events) > 1:
            last_event = events[-1]
            parts.append(f"到「{last_event.title}」结束。")

        return "".join(parts)


# ============================================================
# 兜底搜索数据
# ============================================================

def _fallback_search_results(query: str, query_profile: dict | None = None) -> list[SearchResult]:
    """生成兜底搜索结果

    当 Firecrawl 不可用时，根据查询关键词生成预设搜索结果，
    让 DeepSeek AI 能继续处理（提取事件、分析关系、生成洞察）。

    根据 query_profile 的 topic_type 生成不同主题方向的兜底内容，
    避免所有主题都被套成行业分析报告。

    这不是真实的搜索结果，只是让流程能跑通的占位数据。
    生产环境应部署 Firecrawl 服务获取真实搜索结果。
    """
    topic_type = (query_profile or {}).get("topic_type", "general")

    if topic_type == "daily_word":
        return [
            SearchResult(
                title=f"{query} 词义与来源",
                url="https://example.com/fallback/1",
                markdown=(
                    f"# {query} 的词义与来源\n\n"
                    f"## 基本含义\n"
                    f"「{query}」是一个常见的日常用语，其基本含义随着语境变化而有所不同。\n"
                    f"从语言学角度看，它承载着特定的情感色彩和社交功能。\n\n"
                    f"## 文化含义\n"
                    f"在不同文化背景下，「{query}」的使用方式反映了人际关系的微妙变化，\n"
                    f"体现了语言作为社会纽带的作用。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 的社会功能",
                url="https://example.com/fallback/2",
                markdown=(
                    f"# {query} 的社会功能\n\n"
                    f"## 社交语境\n"
                    f"「{query}」在日常生活中承担着润滑人际关系的功能，\n"
                    f"是建立、维持和调节社会关系的重要语言工具。\n\n"
                    f"## 群体差异\n"
                    f"不同年龄层和社交圈对「{query}」的使用频率和方式存在显著差异，\n"
                    f"反映了代际文化和社交习惯的变迁。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 的当代变化",
                url="https://example.com/fallback/3",
                markdown=(
                    f"# {query} 在当代的变化\n\n"
                    f"## 数字时代的演变\n"
                    f"随着即时通讯和社交媒体的普及，「{query}」的使用场景和形式发生了变化，\n"
                    f"表情包、缩写等新形式不断涌现。\n\n"
                    f"## 人机交互\n"
                    f"在 AI 助手和智能设备普及的背景下，「{query}」也成为人与机器交互的入口，\n"
                    f"其语义和功能正在被重新定义。\n"
                ),
            ),
        ]

    elif topic_type == "history":
        return [
            SearchResult(
                title=f"{query} 历史背景",
                url="https://example.com/fallback/1",
                markdown=(
                    f"# {query} 的历史背景\n\n"
                    f"## 时代背景\n"
                    f"「{query}」所处的时代背景复杂多元，政治、经济、文化因素交织，\n"
                    f"构成了理解其历史意义的基本框架。\n\n"
                    f"## 起源与发展\n"
                    f"从起源到演变，「{query}」经历了多个阶段，每个阶段都有其独特的特征和推动力。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 关键阶段",
                url="https://example.com/fallback/2",
                markdown=(
                    f"# {query} 的关键历史阶段\n\n"
                    f"## 重要转折点\n"
                    f"在「{query}」的发展历程中，存在若干关键转折点，\n"
                    f"这些节点深刻影响了后续的历史走向。\n\n"
                    f"## 代表人物与事件\n"
                    f"多个重要人物和标志性事件塑造了「{query}」的历史面貌，\n"
                    f"他们的决策和行动产生了深远的影响。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 的长期影响",
                url="https://example.com/fallback/3",
                markdown=(
                    f"# {query} 的长期影响\n\n"
                    f"## 制度与文化遗产\n"
                    f"「{query}」留下的制度遗产和文化印记至今仍在发挥作用，\n"
                    f"深刻影响着当代社会的结构和价值观。\n\n"
                    f"## 今天的影子\n"
                    f"从现代社会中仍能找到「{query}」的历史投影，\n"
                    f"理解这段历史有助于我们更好地认识当下。\n"
                ),
            ),
        ]

    elif topic_type == "geography":
        return [
            SearchResult(
                title=f"{query} 地理位置与自然环境",
                url="https://example.com/fallback/1",
                markdown=(
                    f"# {query} 的地理位置\n\n"
                    f"## 自然地理\n"
                    f"「{query}」地处独特的地理位置，其地形、气候和自然资源\n"
                    f"共同塑造了该区域的自然环境和战略价值。\n\n"
                    f"## 区域特征\n"
                    f"该区域的地理特征决定了其在交通、贸易和军事上的重要角色。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 的历史角色",
                url="https://example.com/fallback/2",
                markdown=(
                    f"# {query} 的历史角色\n\n"
                    f"## 文明交汇\n"
                    f"「{query}」作为文明交流的枢纽，见证了多种文化的碰撞与融合，\n"
                    f"是贸易路线和文化传播的关键节点。\n\n"
                    f"## 历史变迁\n"
                    f"不同时期，「{query}」的战略地位发生变化，\n"
                    f"反映了地缘政治格局的演变。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 的地缘意义",
                url="https://example.com/fallback/3",
                markdown=(
                    f"# {query} 的地缘政治意义\n\n"
                    f"## 地缘价值\n"
                    f"「{query}」在地缘政治中具有独特价值，其地理位置影响着区域力量平衡。\n\n"
                    f"## 当代变化\n"
                    f"随着全球化进程和区域合作深化，「{query}」的战略角色正在被重新定义，\n"
                    f"文明的交流与碰撞仍在继续。\n"
                ),
            ),
        ]

    elif topic_type == "social_phenomenon":
        return [
            SearchResult(
                title=f"{query} 现象描述",
                url="https://example.com/fallback/1",
                markdown=(
                    f"# {query}：现象概述\n\n"
                    f"## 现象特征\n"
                    f"「{query}」作为一种社会现象，近年来引发广泛关注。\n"
                    f"其表现形式多样，在不同群体中呈现出不同的特点。\n\n"
                    f"## 传播范围\n"
                    f"该现象从特定圈层扩散至更广泛的社会群体，\n"
                    f"通过社交媒体加速传播，成为公共讨论的热点话题。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 成因分析",
                url="https://example.com/fallback/2",
                markdown=(
                    f"# {query} 的成因分析\n\n"
                    f"## 深层原因\n"
                    f"「{query}」的产生有其深层社会原因，\n"
                    f"经济压力、社会结构变化和文化转型共同推动了这一现象的出现。\n\n"
                    f"## 群体心理\n"
                    f"从社会心理学角度看，该现象反映了特定群体的心理诉求和应对策略，\n"
                    f"是集体情绪的一种表达方式。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 的未来走向",
                url="https://example.com/fallback/3",
                markdown=(
                    f"# {query} 的未来走向\n\n"
                    f"## 社会影响\n"
                    f"「{query}」对社会文化、消费行为和代际关系产生了多方面影响，\n"
                    f"其长期效应仍在显现。\n\n"
                    f"## 趋势预判\n"
                    f"该现象可能演变为新的社会常态，也可能随环境变化而消退，\n"
                    f"其走向取决于经济、政策和文化等多重因素的互动。\n"
                ),
            ),
        ]

    elif topic_type == "tech_business":
        return [
            SearchResult(
                title=f"{query} 最新动态",
                url="https://example.com/fallback/1",
                markdown=(
                    f"# {query} 最新动态\n\n"
                    f"## 行业概览\n"
                    f"{query} 领域近期发生多项重要变化，"
                    f"技术进步和市场政策共同推动行业发展。\n\n"
                    f"## 关键事件\n"
                    f"1. 政策层面：相关部门发布新政策，支持{query}发展\n"
                    f"2. 技术层面：核心技术取得突破，效率提升 30%\n"
                    f"3. 市场层面：头部企业加大投入，行业竞争加剧\n"
                ),
            ),
            SearchResult(
                title=f"{query} 行业分析",
                url="https://example.com/fallback/2",
                markdown=(
                    f"# {query} 行业深度分析\n\n"
                    f"## 市场现状\n"
                    f"{query} 市场规模持续增长，预计未来三年保持 15% 年增速。\n\n"
                    f"## 竞争格局\n"
                    f"行业呈现寡头竞争态势，头部三家企业占据 60% 市场份额。\n"
                    f"新兴企业通过差异化策略切入细分市场。\n\n"
                    f"## 投资机会\n"
                    f"产业链上下游均有投资机会，技术层和应用层值得关注。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 未来趋势",
                url="https://example.com/fallback/3",
                markdown=(
                    f"# {query} 未来趋势展望\n\n"
                    f"## 技术趋势\n"
                    f"AI 驱动的自动化将成为{query}领域的核心驱动力。\n\n"
                    f"## 政策趋势\n"
                    f"监管趋严，合规要求提高，行业进入规范化发展阶段。\n\n"
                    f"## 市场趋势\n"
                    f"全球化布局加速，中国企业出海面临新机遇和挑战。\n"
                ),
            ),
        ]

    else:
        # 通用：基本概念、相关背景、发展趋势
        return [
            SearchResult(
                title=f"{query} 基本概念",
                url="https://example.com/fallback/1",
                markdown=(
                    f"# {query} 基本概念\n\n"
                    f"## 定义与内涵\n"
                    f"「{query}」是一个值得深入探索的主题，\n"
                    f"其核心含义涉及多个维度，需要从不同角度加以理解。\n\n"
                    f"## 关键特征\n"
                    f"理解「{query}」需要把握其基本特征和内在逻辑，\n"
                    f"这些特征构成了进一步分析的基础。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 相关背景",
                url="https://example.com/fallback/2",
                markdown=(
                    f"# {query} 的相关背景\n\n"
                    f"## 背景信息\n"
                    f"「{query}」所处的背景涉及历史、社会、文化等多个层面，\n"
                    f"这些背景因素共同塑造了它的面貌和意义。\n\n"
                    f"## 关联主题\n"
                    f"「{query}」与多个相关主题存在联系，\n"
                    f"理解这些关联有助于构建更完整的认知图景。\n"
                ),
            ),
            SearchResult(
                title=f"{query} 发展趋势",
                url="https://example.com/fallback/3",
                markdown=(
                    f"# {query} 的发展趋势\n\n"
                    f"## 当前状态\n"
                    f"「{query}」在当前阶段呈现出新的特征和变化，\n"
                    f"值得关注的动态正在不断涌现。\n\n"
                    f"## 未来展望\n"
                    f"从发展趋势看，「{query}」可能在未来发生重要变化，\n"
                    f"这些变化对相关领域都有潜在影响。\n"
                ),
            ),
        ]
