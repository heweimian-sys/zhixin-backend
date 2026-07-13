"""
知行·认知加速器 — 研究管道测试（Task 7）

测试 ResearchService 串联完整流程：
搜索 → 提取事件 → 分析关系 → 组织章节 → 生成报告
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.firecrawl_service import SearchResult
from app.services.deepseek_service import (
    ExtractedEvent,
    AnalyzedRelation,
    ChapterOutline,
)
from app.services.research_service import ResearchService, ResearchResult


# ============================================================
# 模拟数据
# ============================================================

def _mock_search_results() -> list[SearchResult]:
    return [
        SearchResult(
            title="Claude 4 发布",
            url="https://example.com/1",
            markdown="# Claude 4\n\n内容...",
        ),
        SearchResult(
            title="GPT-5 加速",
            url="https://example.com/2",
            markdown="# GPT-5\n\n内容...",
        ),
    ]


def _mock_events() -> list[ExtractedEvent]:
    return [
        ExtractedEvent(
            title="Claude 4 发布",
            summary="Anthropic 发布 Claude 4",
            date="2024-03-15",
        ),
        ExtractedEvent(
            title="GPT-5 加速",
            summary="OpenAI 加速开发",
            date="2024-04-01",
        ),
    ]


def _mock_relations() -> list[AnalyzedRelation]:
    return [
        AnalyzedRelation(
            from_event_index=0,
            to_event_index=1,
            type="causal",
            description="Claude 4 发布导致 GPT-5 加速",
        )
    ]


def _mock_chapters() -> list[ChapterOutline]:
    return [
        ChapterOutline(title="第一章·模型之争", event_indices=[0, 1]),
    ]


# ============================================================
# 辅助：创建 mock 的 ResearchService
# ============================================================

def _create_mocked_research_service(
    search_results=None,
    events=None,
    relations=None,
    chapters=None,
    search_error=None,
    extract_error=None,
):
    """创建所有子服务都被 mock 的 ResearchService"""
    mock_search = AsyncMock()
    if search_error:
        mock_search.search = AsyncMock(side_effect=search_error)
    else:
        mock_search.search = AsyncMock(return_value=search_results if search_results is not None else _mock_search_results())

    mock_ai = AsyncMock()
    if extract_error:
        mock_ai.extract_events = AsyncMock(side_effect=extract_error)
    else:
        mock_ai.extract_events = AsyncMock(return_value=events if events is not None else _mock_events())
    mock_ai.analyze_relations = AsyncMock(return_value=relations if relations is not None else _mock_relations())
    mock_ai.organize_chapters = AsyncMock(return_value=chapters if chapters is not None else _mock_chapters())

    return ResearchService(search_service=mock_search, ai_service=mock_ai)


# ============================================================
# ResearchResult 数据结构测试
# ============================================================

def test_research_result_to_dict():
    """测试 ResearchResult.to_dict"""
    result = ResearchResult(
        query="AI行业",
        summary="测试摘要",
        events=_mock_events(),
        relations=_mock_relations(),
        chapters=_mock_chapters(),
    )
    d = result.to_dict()

    assert d["query"] == "AI行业"
    assert d["summary"] == "测试摘要"
    assert len(d["events"]) == 2
    assert len(d["relations"]) == 1
    assert len(d["chapters"]) == 1
    assert d["chapters"][0]["title"] == "第一章·模型之争"


# ============================================================
# 完整流程测试
# ============================================================

@pytest.mark.asyncio
async def test_research_full_flow():
    """测试完整研究流程：搜索→提取→分析→组织"""
    service = _create_mocked_research_service()

    result = await service.research("AI行业")

    assert result.query == "AI行业"
    assert len(result.events) == 2
    assert len(result.relations) == 1
    assert len(result.chapters) == 1
    assert "AI行业" in result.summary


@pytest.mark.asyncio
async def test_research_search_called_with_correct_params():
    """测试搜索服务被正确调用"""
    service = _create_mocked_research_service()

    await service.research("AI行业", search_limit=5, max_events=3)

    # 新架构：搜索使用改写后的查询词（包含原始关键词）
    service.search_service.search.assert_called_once()
    call_args = service.search_service.search.call_args
    assert call_args.kwargs.get("limit") == 5
    assert "AI行业" in call_args.args[0]
    service.ai_service.extract_events.assert_called_once()
    # 验证 max_events 传递（新架构: extract_events(query, results, max_events, query_profile)）
    call_args = service.ai_service.extract_events.call_args
    assert call_args.args[2] == 3  # max_events 是第3个位置参数


@pytest.mark.asyncio
async def test_research_empty_search_results():
    """测试搜索结果为空 — 主搜索返回空时降级到兜底数据

    新契约：_search 有三级降级链（Tavily → Firecrawl → 兜底），
    主搜索返回空不再导致整个流程返回空，而是降级到兜底预设数据。
    """
    service = _create_mocked_research_service(search_results=[])

    # mock Firecrawl 也不可用，确保走兜底预设数据
    with patch("app.services.research_service.FirecrawlSearchService") as MockFC:
        MockFC.return_value.search = AsyncMock(side_effect=Exception("Firecrawl 不可用"))
        result = await service.research("不存在的关键词")

    # 兜底数据返回 3 条，AI 会从中提取事件（mock extract_events 返回 2 个）
    assert result.query == "不存在的关键词"
    assert len(result.events) == 2  # mock extract_events 的返回值
    service.ai_service.extract_events.assert_called_once()


@pytest.mark.asyncio
async def test_research_search_error():
    """测试搜索失败 — 主搜索报错时降级到兜底数据

    新契约：搜索失败不再抛 ResearchError，而是降级到兜底预设数据，
    保证 AI 处理流程不中断。
    """
    from app.services.firecrawl_service import FirecrawlConnectionError

    service = _create_mocked_research_service(search_error=FirecrawlConnectionError("连接失败"))

    # mock Firecrawl 也不可用，确保走兜底预设数据
    with patch("app.services.research_service.FirecrawlSearchService") as MockFC:
        MockFC.return_value.search = AsyncMock(side_effect=Exception("Firecrawl 不可用"))
        result = await service.research("AI行业")

    # 降级到兜底数据，流程正常继续
    assert result.query == "AI行业"
    assert len(result.events) == 2  # mock extract_events 的返回值
    service.ai_service.extract_events.assert_called_once()


@pytest.mark.asyncio
async def test_default_search_service_selection():
    """测试默认搜索服务选择逻辑

    - TAVILY_API_KEY 有值时 → 默认 TavilySearchService
    - TAVILY_API_KEY 为空时 → 默认 FirecrawlSearchService
    """
    from app.services.firecrawl_service import TavilySearchService, FirecrawlSearchService

    # 场景 1：配置了 Tavily Key
    with patch("app.services.research_service.settings") as mock_settings:
        mock_settings.TAVILY_API_KEY = "tvly-test-key"
        service = ResearchService()
        assert isinstance(service.search_service, TavilySearchService)

    # 场景 2：未配置 Tavily Key
    with patch("app.services.research_service.settings") as mock_settings:
        mock_settings.TAVILY_API_KEY = ""
        service = ResearchService()
        assert isinstance(service.search_service, FirecrawlSearchService)


@pytest.mark.asyncio
async def test_fallback_chain_tavily_then_firecrawl():
    """测试降级链：Tavily 失败 → Firecrawl 命中"""
    from app.services.firecrawl_service import FirecrawlConnectionError, TavilySearchService

    # 主搜索（Tavily）失败
    mock_tavily = AsyncMock()
    mock_tavily.search = AsyncMock(side_effect=FirecrawlConnectionError("Tavily 不可用"))

    service = ResearchService(search_service=mock_tavily, ai_service=AsyncMock())

    # mock Firecrawl 返回结果
    mock_fc_results = _mock_search_results()
    with patch("app.services.research_service.FirecrawlSearchService") as MockFC:
        MockFC.return_value.search = AsyncMock(return_value=mock_fc_results)
        # mock AI 服务
        service.ai_service.extract_events = AsyncMock(return_value=_mock_events())
        service.ai_service.analyze_relations = AsyncMock(return_value=_mock_relations())
        service.ai_service.organize_chapters = AsyncMock(return_value=_mock_chapters())

        result = await service.research("AI行业")

    assert result.query == "AI行业"
    assert len(result.events) == 2


@pytest.mark.asyncio
async def test_research_extract_error():
    """测试事件提取失败"""
    from app.services.deepseek_service import DeepSeekError

    service = _create_mocked_research_service(extract_error=DeepSeekError("API失败"))

    with pytest.raises(Exception) as exc_info:
        await service.research("AI行业")

    assert "事件提取失败" in str(exc_info.value)


@pytest.mark.asyncio
async def test_research_relations_error_non_fatal():
    """测试关系分析失败不中断流程"""
    from app.services.deepseek_service import DeepSeekError

    mock_search = AsyncMock()
    mock_search.search = AsyncMock(return_value=_mock_search_results())

    mock_ai = AsyncMock()
    mock_ai.extract_events = AsyncMock(return_value=_mock_events())
    mock_ai.analyze_relations = AsyncMock(side_effect=DeepSeekError("关系分析失败"))
    mock_ai.organize_chapters = AsyncMock(return_value=_mock_chapters())

    service = ResearchService(search_service=mock_search, ai_service=mock_ai)
    result = await service.research("AI行业")

    # 关系分析失败，但流程继续
    assert len(result.events) == 2
    assert len(result.relations) == 0  # 失败后为空
    assert len(result.chapters) == 1


@pytest.mark.asyncio
async def test_research_chapters_error_non_fatal():
    """测试章节组织失败不中断流程"""
    from app.services.deepseek_service import DeepSeekError

    mock_search = AsyncMock()
    mock_search.search = AsyncMock(return_value=_mock_search_results())

    mock_ai = AsyncMock()
    mock_ai.extract_events = AsyncMock(return_value=_mock_events())
    mock_ai.analyze_relations = AsyncMock(return_value=_mock_relations())
    mock_ai.organize_chapters = AsyncMock(side_effect=DeepSeekError("章节失败"))

    service = ResearchService(search_service=mock_search, ai_service=mock_ai)
    result = await service.research("AI行业")

    assert len(result.events) == 2
    assert len(result.relations) == 1
    assert len(result.chapters) == 0  # 失败后为空


@pytest.mark.asyncio
async def test_research_summary_generation():
    """测试引导摘要生成"""
    service = _create_mocked_research_service()

    result = await service.research("新能源")

    assert "新能源" in result.summary
    assert "第一章" in result.summary  # 包含章节标题


@pytest.mark.asyncio
async def test_research_summary_no_chapters():
    """测试无章节时的摘要"""
    service = _create_mocked_research_service(chapters=[])

    result = await service.research("AI行业")

    # 没有章节时，摘要中包含事件数量
    assert "2" in result.summary  # 2 个事件


@pytest.mark.asyncio
async def test_research_to_dict():
    """测试完整结果转字典"""
    service = _create_mocked_research_service()

    result = await service.research("AI行业")
    d = result.to_dict()

    assert d["query"] == "AI行业"
    assert isinstance(d["events"], list)
    assert isinstance(d["relations"], list)
    assert isinstance(d["chapters"], list)
    assert isinstance(d["summary"], str)
