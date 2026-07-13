"""
知行·认知加速器 — 端到端联调测试（Task 14）

测试完整流程：
1. 后端 API 全流程（搜索→提取→分析→组织→洞察→返回 JSON）
2. API 响应结构完整性验证（前端能正确消费）
3. 静态 Mock 数据验证（不依赖外部 API 也能展示）
4. 边界场景处理
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.services.firecrawl_service import SearchResult
from app.services.deepseek_service import (
    ExtractedEvent,
    AnalyzedRelation,
    ChapterOutline,
    GeneratedInsight,
)
from app.services.research_service import ResearchService, ResearchResult
from app.main import ResearchRequest


# ============================================================
# 模拟数据 — 完整的研究结果
# ============================================================

def _mock_search_results() -> list[SearchResult]:
    """模拟 Firecrawl 搜索结果"""
    return [
        SearchResult(
            title="Claude 4 发布",
            url="https://example.com/1",
            markdown="# Claude 4\n\nAnthropic 发布 Claude 4，多模态能力突破。",
        ),
        SearchResult(
            title="GPT-5 加速开发",
            url="https://example.com/2",
            markdown="# GPT-5\n\nOpenAI 加速 GPT-5 开发。",
        ),
        SearchResult(
            title="GPU 需求激增",
            url="https://example.com/3",
            markdown="# GPU 短缺\n\n大模型推动 GPU 需求激增。",
        ),
    ]


def _mock_events() -> list[ExtractedEvent]:
    """模拟 AI 提取的事件"""
    return [
        ExtractedEvent(
            title="Claude 4 发布",
            summary="Anthropic 发布 Claude 4，多模态能力突破",
            date="2024-03-15",
            sources=[{"name": "Anthropic", "url": "https://example.com/1"}],
            key_quote="Claude 4 在多模态基准测试中超越了所有竞争对手",
            confidence=0.9,
        ),
        ExtractedEvent(
            title="OpenAI 加速 GPT-5 开发",
            summary="受竞争压力影响，OpenAI 加速开发 GPT-5",
            date="2024-04-01",
            sources=[{"name": "TechCrunch", "url": "https://example.com/2"}],
            key_quote="GPT-5 预计将在今年晚些时候发布",
            confidence=0.75,
        ),
        ExtractedEvent(
            title="AI 芯片需求激增",
            summary="大模型竞争推动 GPU 需求激增",
            date="2024-05-01",
            sources=[{"name": "Reuters", "url": "https://example.com/3"}],
            key_quote="GPU 供需矛盾预计将持续到 2025 年",
            confidence=0.8,
        ),
    ]


def _mock_relations() -> list[AnalyzedRelation]:
    """模拟 AI 分析的关系"""
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


def _mock_chapters() -> list[ChapterOutline]:
    """模拟 AI 组织的章节"""
    return [
        ChapterOutline(title="第一章·模型之争", event_indices=[0, 1]),
        ChapterOutline(title="第二章·产业影响", event_indices=[2]),
    ]


def _mock_insight() -> GeneratedInsight:
    """模拟 AI 生成的洞察"""
    return GeneratedInsight(
        title="模型层战争结束，应用层刚刚开始",
        body="过去一年大模型竞争焦点从参数规模转向多模态能力和成本效率。",
        judgments=[
            "GPU供需矛盾短期无解",
            "模型层趋同化加速，差异化在应用层",
            "AI Agent是下一个增长点",
        ],
        suggestions={
            "投资者": ["关注端侧AI芯片赛道", "AI Agent工具链是投资蓝海"],
            "创业者": ["AI Agent工具链有差异化机会"],
            "求职者": ["多模态应用开发技能需求激增"],
        },
    )


# ============================================================
# 后端全流程测试
# ============================================================


class TestFullPipeline:
    """测试后端完整流程：搜索→提取→分析→组织→洞察→返回"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_insight(self):
        """测试完整流程含洞察"""
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
            mock_search.return_value = _mock_search_results()
            mock_extract.return_value = _mock_events()
            mock_relations.return_value = _mock_relations()
            mock_chapters.return_value = _mock_chapters()
            mock_insight.return_value = _mock_insight()

            result = await service.research("AI行业")

            # 验证完整结果
            assert result.query == "AI行业"
            assert len(result.events) == 3
            assert len(result.relations) == 2
            assert len(result.chapters) == 2
            assert result.insight is not None
            assert result.insight.title == "模型层战争结束，应用层刚刚开始"
            assert len(result.insight.judgments) == 3
            assert "投资者" in result.insight.suggestions

            # 验证调用链顺序
            mock_search.assert_called_once()
            mock_extract.assert_called_once()
            mock_relations.assert_called_once()
            mock_chapters.assert_called_once()
            mock_insight.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_pipeline_to_dict_structure(self):
        """测试 to_dict 返回前端可消费的完整 JSON 结构"""
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
            mock_search.return_value = _mock_search_results()
            mock_extract.return_value = _mock_events()
            mock_relations.return_value = _mock_relations()
            mock_chapters.return_value = _mock_chapters()
            mock_insight.return_value = _mock_insight()

            result = await service.research("AI行业")
            data = result.to_dict()

            # 验证 JSON 结构完整性
            assert "query" in data
            assert "summary" in data
            assert "events" in data
            assert "relations" in data
            assert "chapters" in data
            assert "insight" in data

            # 验证事件结构
            evt = data["events"][0]
            assert "title" in evt
            assert "summary" in evt
            assert "date" in evt
            assert "sources" in evt
            assert "key_quote" in evt
            assert "confidence" in evt

            # 验证关系结构
            rel = data["relations"][0]
            assert "from_event_index" in rel
            assert "to_event_index" in rel
            assert "type" in rel
            assert "description" in rel
            assert "confidence" in rel

            # 验证章节结构
            ch = data["chapters"][0]
            assert "title" in ch
            assert "event_indices" in ch

            # 验证洞察结构
            ins = data["insight"]
            assert "title" in ins
            assert "body" in ins
            assert "judgments" in ins
            assert "suggestions" in ins

    @pytest.mark.asyncio
    async def test_pipeline_all_non_fatal_failures(self):
        """测试所有非致命步骤都失败时不中断流程"""
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
            mock_search.return_value = _mock_search_results()
            mock_extract.return_value = _mock_events()
            mock_relations.side_effect = Exception("关系分析失败")
            mock_chapters.side_effect = Exception("章节组织失败")
            mock_insight.side_effect = Exception("洞察生成失败")

            result = await service.research("AI行业")

            # 核心数据仍然返回
            assert len(result.events) == 3
            assert result.relations == []
            assert result.chapters == []
            assert result.insight is None
            assert result.summary  # 摘要仍然生成


# ============================================================
# API 端点测试
# ============================================================


class TestResearchAPI:
    """测试 /api/research API 端点"""

    @pytest.mark.asyncio
    async def test_api_research_success(self, client):
        """测试 API 正常返回"""
        with patch(
            "app.main.ResearchService"
        ) as MockService:
            mock_instance = MockService.return_value
            mock_instance.research = AsyncMock(
                return_value=ResearchResult(
                    query="AI行业",
                    summary="关于AI行业的调研摘要",
                    events=_mock_events(),
                    relations=_mock_relations(),
                    chapters=_mock_chapters(),
                    insight=_mock_insight(),
                )
            )

            response = await client.post(
                "/api/research",
                json={"query": "AI行业"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "AI行业"
            assert len(data["events"]) == 3
            assert data["insight"] is not None
            assert data["insight"]["title"] == "模型层战争结束，应用层刚刚开始"

    @pytest.mark.asyncio
    async def test_api_research_empty_query(self, client):
        """测试空查询返回 400"""
        response = await client.post(
            "/api/research",
            json={"query": ""},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_api_research_whitespace_query(self, client):
        """测试纯空格查询返回 400"""
        response = await client.post(
            "/api/research",
            json={"query": "   "},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_api_research_internal_error(self, client):
        """测试后端异常返回 500"""
        from app.services.research_service import ResearchError

        with patch(
            "app.main.ResearchService"
        ) as MockService:
            mock_instance = MockService.return_value
            mock_instance.research = AsyncMock(
                side_effect=ResearchError("搜索服务不可用")
            )

            response = await client.post(
                "/api/research",
                json={"query": "测试"},
            )
            assert response.status_code == 500
            assert "搜索服务不可用" in response.json()["detail"]


# ============================================================
# 静态 Mock 数据验证
# ============================================================


class TestMockDataValidation:
    """验证 Mock 数据结构完整，前端可正确消费"""

    def test_mock_events_have_required_fields(self):
        """验证模拟事件包含所有必须字段"""
        events = _mock_events()
        for evt in events:
            assert evt.title
            assert evt.summary
            assert evt.date is not None
            assert len(evt.sources) > 0
            assert evt.confidence > 0

    def test_mock_relations_have_valid_indices(self):
        """验证模拟关系索引在事件列表范围内"""
        events = _mock_events()
        relations = _mock_relations()
        for rel in relations:
            assert 0 <= rel.from_event_index < len(events)
            assert 0 <= rel.to_event_index < len(events)
            assert rel.type in AnalyzedRelation.VALID_TYPES

    def test_mock_chapters_cover_all_events(self):
        """验证模拟章节覆盖所有事件"""
        events = _mock_events()
        chapters = _mock_chapters()
        all_indices = set()
        for ch in chapters:
            all_indices.update(ch.event_indices)
        assert all_indices == set(range(len(events)))

    def test_mock_insight_has_all_roles(self):
        """验证模拟洞察包含所有角色建议"""
        insight = _mock_insight()
        assert "投资者" in insight.suggestions
        assert "创业者" in insight.suggestions
        assert "求职者" in insight.suggestions
        for role, suggestions in insight.suggestions.items():
            assert len(suggestions) > 0

    def test_full_result_json_serializable(self):
        """验证完整结果可 JSON 序列化"""
        result = ResearchResult(
            query="AI行业",
            summary="测试摘要",
            events=_mock_events(),
            relations=_mock_relations(),
            chapters=_mock_chapters(),
            insight=_mock_insight(),
        )
        # 不抛异常即通过
        json_str = json.dumps(result.to_dict(), ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["query"] == "AI行业"

    def test_result_without_insight_json_serializable(self):
        """验证无洞察的结果也可 JSON 序列化"""
        result = ResearchResult(
            query="测试",
            summary="摘要",
            events=_mock_events(),
            relations=[],
            chapters=[],
            insight=None,
        )
        data = result.to_dict()
        assert data["insight"] is None
        json_str = json.dumps(data, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["insight"] is None
