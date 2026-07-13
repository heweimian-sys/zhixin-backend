"""
知行·认知加速器 — AI 洞察生成测试（Task 12）

测试覆盖：
1. GeneratedInsight 数据类（创建、默认值、to_dict）
2. generate_insight() 方法（正常生成、空事件、API 异常、格式异常）
3. ResearchService 集成洞察生成（非致命错误处理）
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.deepseek_service import (
    DeepSeekService,
    DeepSeekError,
    ExtractedEvent,
    AnalyzedRelation,
    GeneratedInsight,
)
from app.services.research_service import ResearchService, ResearchResult
from app.services.firecrawl_service import SearchResult


# ============================================================
# 模拟数据
# ============================================================

def _make_events() -> list[ExtractedEvent]:
    """创建模拟事件列表"""
    return [
        ExtractedEvent(
            title="Claude 4 发布",
            summary="Anthropic 发布 Claude 4，多模态能力突破",
            date="2024-03-15",
            confidence=0.9,
        ),
        ExtractedEvent(
            title="OpenAI 加速 GPT-5 开发",
            summary="受竞争压力影响，OpenAI 加速开发",
            date="2024-04-01",
            confidence=0.75,
        ),
        ExtractedEvent(
            title="AI 芯片需求激增",
            summary="大模型竞争推动 GPU 需求激增",
            date="2024-05-01",
            confidence=0.8,
        ),
    ]


def _make_relations() -> list[AnalyzedRelation]:
    """创建模拟关系列表"""
    return [
        AnalyzedRelation(
            from_event_index=0,
            to_event_index=1,
            type="causal",
            description="Claude 4 发布给 OpenAI 带来竞争压力",
            confidence=0.85,
        ),
        AnalyzedRelation(
            from_event_index=0,
            to_event_index=2,
            type="causal",
            description="大模型竞争推动 GPU 需求激增",
            confidence=0.8,
        ),
    ]


MOCK_INSIGHT_RESPONSE = {
    "title": "模型层战争结束，应用层刚刚开始",
    "body": "过去一年大模型竞争焦点从参数规模转向多模态能力和成本效率。Claude 4 的发布标志着模型层趋同化加速，未来差异化将主要体现在应用层。GPU 供需矛盾短期无解，端侧 AI 芯片是突破口。AI Agent 作为连接模型和用户的新层，是下一个增长点。",
    "judgments": [
        "GPU供需矛盾短期无解，端侧AI芯片是突破口",
        "模型层趋同化加速，差异化在应用层",
        "AI Agent是下一个增长点",
    ],
    "suggestions": {
        "投资者": ["关注端侧AI芯片赛道", "AI Agent工具链是投资蓝海"],
        "创业者": ["AI Agent工具链有差异化机会", "多模态垂直应用是蓝海"],
        "求职者": ["多模态应用开发技能需求激增", "AI infra人才仍然稀缺"],
    },
}


# ============================================================
# GeneratedInsight 数据类测试
# ============================================================


class TestGeneratedInsight:
    """测试 GeneratedInsight 数据类"""

    def test_create_with_full_data(self):
        """测试完整数据创建"""
        insight = GeneratedInsight(
            title="测试洞察标题",
            body="测试详细分析",
            judgments=["判断1", "判断2"],
            suggestions={"投资者": ["建议1"], "创业者": ["建议2"]},
        )
        assert insight.title == "测试洞察标题"
        assert insight.body == "测试详细分析"
        assert len(insight.judgments) == 2
        assert "投资者" in insight.suggestions

    def test_default_values(self):
        """测试默认值"""
        insight = GeneratedInsight(title="只有标题")
        assert insight.title == "只有标题"
        assert insight.body == ""
        assert insight.judgments == []
        assert insight.suggestions == {}

    def test_to_dict(self):
        """测试转字典"""
        insight = GeneratedInsight(
            title="标题",
            body="正文",
            judgments=["判断"],
            suggestions={"角色": ["建议"]},
        )
        d = insight.to_dict()
        assert d["title"] == "标题"
        assert d["body"] == "正文"
        assert d["judgments"] == ["判断"]
        assert d["suggestions"] == {"角色": ["建议"]}

    def test_to_dict_with_defaults(self):
        """测试默认值转字典"""
        insight = GeneratedInsight(title="标题")
        d = insight.to_dict()
        assert d["judgments"] == []
        assert d["suggestions"] == {}


# ============================================================
# DeepSeekService.generate_insight() 测试
# ============================================================


class TestGenerateInsight:
    """测试 DeepSeekService.generate_insight()"""

    @pytest.mark.asyncio
    async def test_generate_insight_normal(self):
        """测试正常生成洞察"""
        service = DeepSeekService(api_key="test-key")

        with patch.object(
            service, "chat_json", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = MOCK_INSIGHT_RESPONSE

            insight = await service.generate_insight(
                "AI行业", _make_events(), _make_relations()
            )

            assert insight.title == "模型层战争结束，应用层刚刚开始"
            assert len(insight.judgments) == 3
            assert "投资者" in insight.suggestions
            assert len(insight.suggestions["投资者"]) == 2

            # 验证 chat_json 被调用
            mock_chat.assert_called_once()
            args = mock_chat.call_args
            # chat_json(system_prompt, user_prompt) — user_prompt 包含查询关键词
            assert "AI行业" in args.args[1]

    @pytest.mark.asyncio
    async def test_generate_insight_empty_events(self):
        """测试空事件列表"""
        service = DeepSeekService(api_key="test-key")

        insight = await service.generate_insight("测试", [], [])

        assert insight.title == ""
        assert "暂无" in insight.body

    @pytest.mark.asyncio
    async def test_generate_insight_api_error(self):
        """测试 API 异常"""
        service = DeepSeekService(api_key="test-key")

        with patch.object(
            service, "chat_json", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.side_effect = DeepSeekError("API 不可用")

            with pytest.raises(DeepSeekError):
                await service.generate_insight(
                    "AI行业", _make_events(), _make_relations()
                )

    @pytest.mark.asyncio
    async def test_generate_insight_missing_fields(self):
        """测试返回缺少字段"""
        service = DeepSeekService(api_key="test-key")

        with patch.object(
            service, "chat_json", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = {"title": "只有标题"}

            insight = await service.generate_insight(
                "AI行业", _make_events(), _make_relations()
            )

            assert insight.title == "只有标题"
            assert insight.body == ""
            assert insight.judgments == []
            assert insight.suggestions == {}

    @pytest.mark.asyncio
    async def test_generate_insight_no_relations(self):
        """测试没有关系时也能生成洞察"""
        service = DeepSeekService(api_key="test-key")

        with patch.object(
            service, "chat_json", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = MOCK_INSIGHT_RESPONSE

            insight = await service.generate_insight(
                "美食文化", _make_events(), []
            )

            assert insight.title == "模型层战争结束，应用层刚刚开始"
            mock_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_insight_temperature(self):
        """测试温度参数（洞察生成应该用更高温度）"""
        service = DeepSeekService(api_key="test-key")

        with patch.object(
            service, "chat_json", new_callable=AsyncMock
        ) as mock_chat:
            mock_chat.return_value = MOCK_INSIGHT_RESPONSE

            await service.generate_insight(
                "AI行业", _make_events(), _make_relations()
            )

            # 验证温度参数
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs.get("temperature") == 0.4

    @pytest.mark.asyncio
    async def test_generate_insight_format_error(self):
        """测试返回格式异常"""
        service = DeepSeekService(api_key="test-key")

        with patch.object(
            service, "chat_json", new_callable=AsyncMock
        ) as mock_chat:
            # 返回非字典数据导致 TypeError
            mock_chat.return_value = "not a dict"

            insight = await service.generate_insight(
                "AI行业", _make_events(), _make_relations()
            )

            # 应该返回空洞察而不是抛异常
            assert insight.title == ""


# ============================================================
# ResearchService 集成洞察测试
# ============================================================


class TestResearchServiceInsight:
    """测试 ResearchService 集成洞察生成"""

    @pytest.mark.asyncio
    async def test_research_with_insight(self):
        """测试完整流程包含洞察"""
        search_results = [
            SearchResult(title="测试", url="https://example.com", markdown="内容"),
        ]
        events = _make_events()
        relations = _make_relations()
        insight = GeneratedInsight(
            title="测试洞察",
            body="测试分析",
            judgments=["判断1"],
            suggestions={"投资者": ["建议1"]},
        )

        service = ResearchService()

        with patch.object(
            service.search_service, "search", new_callable=AsyncMock
        ) as mock_search, patch.object(
            service.ai_service, "extract_events", new_callable=AsyncMock
        ) as mock_extract, patch.object(
            service.ai_service, "analyze_relations", new_callable=AsyncMock
        ) as mock_relations, patch.object(
            service.ai_service, "organize_chapters", new_callable=AsyncMock
        ) as mock_chapters, patch.object(
            service.ai_service, "generate_insight", new_callable=AsyncMock
        ) as mock_insight:
            mock_search.return_value = search_results
            mock_extract.return_value = events
            mock_relations.return_value = relations
            mock_chapters.return_value = []
            mock_insight.return_value = insight

            result = await service.research("AI行业")

            assert result.insight is not None
            assert result.insight.title == "测试洞察"
            assert result.insight.judgments == ["判断1"]

            # 验证 generate_insight 被调用（新架构含 query_profile 参数）
            mock_insight.assert_called_once()
            call_args = mock_insight.call_args
            assert call_args.args[0] == "AI行业"
            assert call_args.args[1] == events
            assert call_args.args[2] == relations

    @pytest.mark.asyncio
    async def test_research_insight_failure_non_fatal(self):
        """测试洞察生成失败不中断流程"""
        search_results = [
            SearchResult(title="测试", url="https://example.com", markdown="内容"),
        ]
        events = _make_events()

        service = ResearchService()

        with patch.object(
            service.search_service, "search", new_callable=AsyncMock
        ) as mock_search, patch.object(
            service.ai_service, "extract_events", new_callable=AsyncMock
        ) as mock_extract, patch.object(
            service.ai_service, "analyze_relations", new_callable=AsyncMock
        ) as mock_relations, patch.object(
            service.ai_service, "organize_chapters", new_callable=AsyncMock
        ) as mock_chapters, patch.object(
            service.ai_service, "generate_insight", new_callable=AsyncMock
        ) as mock_insight:
            mock_search.return_value = search_results
            mock_extract.return_value = events
            mock_relations.return_value = []
            mock_chapters.return_value = []
            mock_insight.side_effect = Exception("洞察生成失败")

            result = await service.research("AI行业")

            # 洞察失败不影响其他结果
            assert result.insight is None
            assert len(result.events) == 3
            assert result.summary  # 摘要仍然生成了

    @pytest.mark.asyncio
    async def test_research_result_to_dict_with_insight(self):
        """测试 to_dict 包含洞察"""
        insight = GeneratedInsight(
            title="测试",
            body="分析",
            judgments=["j1"],
            suggestions={"投资者": ["s1"]},
        )
        result = ResearchResult(
            query="测试",
            summary="摘要",
            events=[],
            relations=[],
            chapters=[],
            insight=insight,
        )

        d = result.to_dict()
        assert d["insight"] is not None
        assert d["insight"]["title"] == "测试"
        assert d["insight"]["judgments"] == ["j1"]

    @pytest.mark.asyncio
    async def test_research_result_to_dict_without_insight(self):
        """测试 to_dict 无洞察时返回 None"""
        result = ResearchResult(
            query="测试",
            summary="摘要",
            events=[],
            relations=[],
            chapters=[],
            insight=None,
        )

        d = result.to_dict()
        assert d["insight"] is None

    @pytest.mark.asyncio
    async def test_research_empty_search_insight_none(self):
        """测试搜索为空且兜底也为空时洞察为 None"""
        service = ResearchService()

        with patch.object(
            service.search_service, "search", new_callable=AsyncMock
        ) as mock_search, patch(
            "app.services.research_service._fallback_search_results", return_value=[]
        ), patch(
            "app.services.research_service.FirecrawlSearchService"
        ) as MockFC:
            MockFC.return_value.search = AsyncMock(return_value=[])
            mock_search.return_value = []

            result = await service.research("不存在的关键词")

            assert result.insight is None
            assert result.events == []

    @pytest.mark.asyncio
    async def test_research_empty_events_insight_none(self):
        """测试事件为空时洞察为 None"""
        search_results = [
            SearchResult(title="测试", url="https://example.com", markdown="内容"),
        ]

        service = ResearchService()

        with patch.object(
            service.search_service, "search", new_callable=AsyncMock
        ) as mock_search, patch.object(
            service.ai_service, "extract_events", new_callable=AsyncMock
        ) as mock_extract:
            mock_search.return_value = search_results
            mock_extract.return_value = []

            result = await service.research("测试")

            assert result.insight is None
            assert result.events == []
