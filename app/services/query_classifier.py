"""
知行 · AI 深度调研助手 — 主题分类器

规则优先 + DeepSeek 兜底，判断用户输入属于哪种主题类型，
并生成对应的查询改写、分析焦点和语调。

核心设计：
    1. 先用规则快速判断明显类型（你好→daily_word, 唐朝→history 等）
    2. 规则判断不了时调用 DeepSeek 分类
    3. 返回 QueryProfile，供后续所有步骤使用
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# 主题类型定义
# ============================================================

# 9 种主题类型 → 中文显示名
TOPIC_TYPE_DISPLAY: dict[str, str] = {
    "daily_word": "日常词语",
    "abstract_concept": "抽象概念",
    "history": "历史主题",
    "geography": "地理空间",
    "person_or_org": "人物/组织",
    "social_phenomenon": "社会现象",
    "tech_business": "科技/商业",
    "question": "问题研究",
    "general": "综合探索",
}

# 每种类型对应的模板名
TOPIC_TYPE_TEMPLATE: dict[str, str] = {
    "daily_word": "concept_exploration",
    "abstract_concept": "concept_exploration",
    "history": "historical_context",
    "geography": "geo_civilization",
    "person_or_org": "biography_profile",
    "social_phenomenon": "social_analysis",
    "tech_business": "tech_business_analysis",
    "question": "question_decomposition",
    "general": "general_exploration",
}

# 每种类型的默认语调
TOPIC_TYPE_TONE: dict[str, str] = {
    "daily_word": "轻知识、启发式",
    "abstract_concept": "思辨、多角度",
    "history": "历史叙事",
    "geography": "空间视角、文明交融",
    "person_or_org": "传记式、评价性",
    "social_phenomenon": "现实观察、社会分析",
    "tech_business": "专业、分析性",
    "question": "拆解式、解答性",
    "general": "开放探索",
}


@dataclass
class QueryProfile:
    """查询画像 — 描述用户输入的主题类型和分析方向"""

    original_query: str
    topic_type: str
    template: str
    rewritten_query: str
    analysis_focus: str
    tone: str
    display_type: str
    confidence: float = 0.8
    classified_by: str = "rule"  # "rule" 或 "llm"

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "topic_type": self.topic_type,
            "template": self.template,
            "rewritten_query": self.rewritten_query,
            "analysis_focus": self.analysis_focus,
            "tone": self.tone,
            "display_type": self.display_type,
            "confidence": self.confidence,
            "classified_by": self.classified_by,
        }


# ============================================================
# 规则分类
# ============================================================

# 问题型输入关键词
_QUESTION_KEYWORDS = ["为什么", "如何", "怎么", "有没有", "吗", "？", "?", "为何", "怎样", "为何"]
# 科技/商业关键词
_TECH_BUSINESS_KEYWORDS = ["竞品", "对比", "VS", "vs", "商业化", "市场", "工具", "平台", "产品", "公司", "行业", "赛道", "融资", "估值", "增长", "规模"]
# 历史关键词
_HISTORY_KEYWORDS = ["朝", "代", "战争", "革命", "帝国", "王朝", "冷战", "丝绸之路", "工业革命", "二战", "一战", "古代", "历史"]
# 社会现象关键词
_SOCIAL_PHENOMON_KEYWORDS = ["内卷", "躺平", "搭子", "松弛感", "丧", "焦虑", "社恐", "消费降级", "县城", "婆罗门"]
# 地理关键词
_GEOGRAPHY_KEYWORDS = ["海峡", "山脉", "高原", "盆地", "运河", "岛屿", "半岛", "流域", "沙漠", "洋", "海", "河", "山"]
# 日常问候
_DAILY_WORDS = ["你好", "hello", "hi", "嗨", "早上好", "晚上好", "再见", "谢谢", "早安", "晚安"]
# 抽象概念关键词
_ABSTRACT_CONCEPT_KEYWORDS = ["自由", "孤独", "边界", "效率", "信任", "正义", "公平", "美", "真理", "意义", "时间", "空间", "存在", "意识"]


def classify_by_rules(query: str) -> QueryProfile | None:
    """规则分类 — 快速判断明显类型

    返回 None 表示规则无法判断，需要调用 LLM。
    """
    q = query.strip().lower()

    # 1. 日常问候
    for word in _DAILY_WORDS:
        if q == word or q == word.lower():
            return _make_profile(
                query, "daily_word",
                rewritten=f"{query} 问候语 人际沟通 社会关系 人机交互 AI助手 对话式交互",
                focus="从语言、社会关系、人机交互和 AI 产品入口角度探索「你好」的意义",
                confidence=0.95,
            )

    # 2. 问题型输入
    for kw in _QUESTION_KEYWORDS:
        if kw in query:
            return _make_profile(
                query, "question",
                rewritten=query,
                focus=f"拆解问题「{query}」，从多个角度给出解释、深层原因和判断建议",
                confidence=0.9,
            )

    # 3. 社会现象
    for kw in _SOCIAL_PHENOMON_KEYWORDS:
        if kw in query:
            return _make_profile(
                query, "social_phenomenon",
                rewritten=f"{query} 社会现象 成因 传播 群体心理 趋势 青年文化",
                focus=f"分析「{query}」的来源、传播路径、社会心理和未来变化",
                confidence=0.88,
            )

    # 4. 历史主题
    for kw in _HISTORY_KEYWORDS:
        if kw in query:
            return _make_profile(
                query, "history",
                rewritten=f"{query} 历史 政治 经济 文化 制度 影响 时间线",
                focus=f"从历史脉络、制度、文化和长期影响角度理解「{query}」",
                confidence=0.85,
            )

    # 5. 科技/商业
    for kw in _TECH_BUSINESS_KEYWORDS:
        if kw in query:
            return _make_profile(
                query, "tech_business",
                rewritten=f"{query} 趋势 产品 竞品 商业化 市场 技术",
                focus=f"分析「{query}」的技术趋势、产品形态、竞争格局和商业机会",
                confidence=0.85,
            )

    # 6. 地理空间
    for kw in _GEOGRAPHY_KEYWORDS:
        if kw in query and len(query) <= 10:
            return _make_profile(
                query, "geography",
                rewritten=f"{query} 地理 文明 贸易 历史 地缘政治",
                focus=f"从地理位置、文明交流、贸易网络和地缘政治角度理解「{query}」",
                confidence=0.8,
            )

    # 7. 抽象概念
    for kw in _ABSTRACT_CONCEPT_KEYWORDS:
        if kw in query and len(query) <= 6:
            return _make_profile(
                query, "abstract_concept",
                rewritten=f"{query} 概念 哲学 心理 社会 定义 历史 演变",
                focus=f"从定义、历史演变、多学科理解和现实表现角度探索「{query}」",
                confidence=0.8,
            )

    # 规则无法判断
    return None


def _make_profile(
    query: str,
    topic_type: str,
    rewritten: str,
    focus: str,
    confidence: float = 0.85,
) -> QueryProfile:
    """快速构造 QueryProfile"""
    return QueryProfile(
        original_query=query,
        topic_type=topic_type,
        template=TOPIC_TYPE_TEMPLATE.get(topic_type, "general_exploration"),
        rewritten_query=rewritten,
        analysis_focus=focus,
        tone=TOPIC_TYPE_TONE.get(topic_type, "开放探索"),
        display_type=TOPIC_TYPE_DISPLAY.get(topic_type, "综合探索"),
        confidence=confidence,
        classified_by="rule",
    )


# ============================================================
# LLM 分类（DeepSeek 兜底）
# ============================================================

_CLASSIFY_SYSTEM_PROMPT = """你是一个主题分类器。判断用户输入属于以下哪种类型：

1. daily_word — 日常词语（你好、吃饭、回家、等待）
2. abstract_concept — 抽象概念（自由、孤独、边界、效率）
3. history — 历史主题（唐朝、冷战、丝绸之路、工业革命）
4. geography — 地理空间（地中海、长安、深圳、东北）
5. person_or_org — 人物/组织（孔子、拿破仑、OpenAI、飞书）
6. social_phenomenon — 社会现象（内卷、躺平、搭子文化、松弛感）
7. tech_business — 科技/商业（AI Agent、Dify、跨境电商、SaaS）
8. question — 问题型输入（为什么年轻人焦虑？如何选择职业？）
9. general — 兜底，无法归入以上类型

请返回 JSON：
{
  "topic_type": "类型名",
  "rewritten_query": "改写后的搜索词（补充上下文关键词）",
  "analysis_focus": "从哪些角度探索这个主题（一句话）",
  "confidence": 0.0-1.0
}

注意：
- rewritten_query 应补充 5-8 个相关关键词，帮助搜索更精准
- analysis_focus 要具体，不要泛泛说"分析这个主题"
- confidence 是你对分类的确信度"""


async def classify_by_llm(query: str) -> QueryProfile:
    """调用 DeepSeek 进行主题分类

    规则无法判断时使用。
    """
    try:
        client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=60.0,
        )

        response = await client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": f"请分类：{query}"},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        topic_type = result.get("topic_type", "general")
        # 校验类型合法性
        if topic_type not in TOPIC_TYPE_DISPLAY:
            topic_type = "general"

        return QueryProfile(
            original_query=query,
            topic_type=topic_type,
            template=TOPIC_TYPE_TEMPLATE.get(topic_type, "general_exploration"),
            rewritten_query=result.get("rewritten_query", query),
            analysis_focus=result.get("analysis_focus", f"从多个角度探索「{query}」"),
            tone=TOPIC_TYPE_TONE.get(topic_type, "开放探索"),
            display_type=TOPIC_TYPE_DISPLAY.get(topic_type, "综合探索"),
            confidence=float(result.get("confidence", 0.7)),
            classified_by="llm",
        )

    except Exception as e:
        logger.warning("LLM 分类失败，使用 general 兜底: %s", e)
        return _make_profile(
            query, "general",
            rewritten=query,
            focus=f"从多个角度探索「{query}」",
            confidence=0.5,
        )


# ============================================================
# 主入口
# ============================================================

async def classify_query(query: str) -> QueryProfile:
    """主题分类入口 — 规则优先，DeepSeek 兜底

    Args:
        query: 用户原始输入

    Returns:
        QueryProfile，包含主题类型、改写查询、分析焦点等
    """
    # Step 1: 规则分类
    rule_result = classify_by_rules(query)
    if rule_result is not None and rule_result.confidence >= 0.8:
        logger.info("规则分类成功: query='%s', type='%s'", query, rule_result.topic_type)
        return rule_result

    # Step 2: LLM 分类
    logger.info("规则无法判断，调用 LLM 分类: query='%s'", query)
    return await classify_by_llm(query)
