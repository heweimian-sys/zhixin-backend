"""
知行·认知加速器 — Tavily 搜索服务测试

测试 TavilySearchService 的完整逻辑：
- 正常搜索返回结果
- 空结果处理
- 连接错误 / 超时处理
- 未配置 API Key 报错
- Bearer 认证头断言
- max_results 钳制（Tavily 限制 0-20）
- search_depth / topic 进入 payload
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.firecrawl_service import (
    FirecrawlAPIError,
    FirecrawlConnectionError,
    SearchResult,
    TavilySearchService,
)


# ============================================================
# 模拟数据
# ============================================================

MOCK_TAVILY_RESPONSE = {
    "query": "AI行业",
    "answer": "AI 行业正在快速发展...",
    "results": [
        {
            "title": "Claude 4 发布，多模态能力突破",
            "url": "https://example.com/news/claude-4",
            "content": "Anthropic 发布了最新 AI 模型 Claude 4，具备强大的多模态能力。",
            "score": 0.95,
            "raw_content": "# Claude 4 发布\n\nAnthropic 发布了最新 AI 模型 Claude 4...",
        },
        {
            "title": "OpenAI 加速 GPT-5 开发",
            "url": "https://example.com/news/gpt5",
            "content": "受竞争压力影响，OpenAI 正在加速 GPT-5 的开发进程。",
            "score": 0.88,
            "raw_content": "# GPT-5 开发加速\n\nOpenAI 正在加速 GPT-5...",
        },
    ],
    "response_time": 1.23,
}

MOCK_TAVILY_EMPTY_RESPONSE = {
    "query": "不存在的关键词",
    "answer": "",
    "results": [],
    "response_time": 0.5,
}


# ============================================================
# 辅助函数：创建 mock httpx.AsyncClient
# ============================================================

def _mock_httpx_client(
    response_json: dict | None = None,
    status_code: int = 200,
    exc: Exception | None = None,
) -> MagicMock:
    """创建 mock 的 httpx.AsyncClient 实例"""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    if exc:
        mock_client.post = AsyncMock(side_effect=exc)
    else:
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = response_json
        mock_response.text = json.dumps(response_json or {})
        mock_client.post = AsyncMock(return_value=mock_response)

    return mock_client


# ============================================================
# 测试用例
# ============================================================

@pytest.mark.asyncio
async def test_tavily_normal_search():
    """测试正常搜索返回结果"""
    mock_client = _mock_httpx_client(MOCK_TAVILY_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = TavilySearchService(api_key="tvly-test-key")
        results = await service.search("AI行业", limit=10)

    assert len(results) == 2
    assert results[0].title == "Claude 4 发布，多模态能力突破"
    assert results[0].url == "https://example.com/news/claude-4"
    assert results[0].description.startswith("Anthropic 发布了")
    assert results[0].markdown.startswith("# Claude 4 发布")
    assert results[1].title == "OpenAI 加速 GPT-5 开发"


@pytest.mark.asyncio
async def test_tavily_empty_results():
    """测试搜索返回空结果"""
    mock_client = _mock_httpx_client(MOCK_TAVILY_EMPTY_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = TavilySearchService(api_key="tvly-test-key")
        results = await service.search("不存在的关键词")

    assert len(results) == 0


@pytest.mark.asyncio
async def test_tavily_connection_error():
    """测试连接错误处理"""
    mock_client = _mock_httpx_client(exc=httpx.ConnectError("Connection refused"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = TavilySearchService(api_key="tvly-test-key")
        with pytest.raises(FirecrawlConnectionError) as exc_info:
            await service.search("AI行业")

    assert "Tavily" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tavily_timeout_error():
    """测试请求超时处理"""
    mock_client = _mock_httpx_client(exc=httpx.TimeoutException("Timeout"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = TavilySearchService(api_key="tvly-test-key", timeout=5)
        with pytest.raises(FirecrawlConnectionError) as exc_info:
            await service.search("AI行业")

    assert "Tavily" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tavily_missing_api_key():
    """测试未配置 API Key 时报错"""
    service = TavilySearchService(api_key="")

    with pytest.raises(FirecrawlConnectionError) as exc_info:
        await service.search("AI行业")

    assert "Tavily API Key 未配置" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tavily_bearer_auth_header():
    """测试 Bearer 认证头 — Authorization: Bearer tvly-xxx"""
    mock_client = _mock_httpx_client(MOCK_TAVILY_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = TavilySearchService(api_key="tvly-test-key")
        await service.search("AI行业")

    call_args = mock_client.post.call_args
    headers = call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer tvly-test-key"
    assert headers["Content-Type"] == "application/json"
    # 确认 api_key 不在 body 中
    payload = call_args.kwargs["json"]
    assert "api_key" not in payload


@pytest.mark.asyncio
async def test_tavily_max_results_clamping():
    """测试 max_results 钳制 — 传 30，断言 payload 中为 20（Tavily 上限）"""
    mock_client = _mock_httpx_client(MOCK_TAVILY_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = TavilySearchService(api_key="tvly-test-key")
        await service.search("AI行业", limit=30)

    call_args = mock_client.post.call_args
    payload = call_args.kwargs["json"]
    assert payload["max_results"] == 20  # 钳制到上限


@pytest.mark.asyncio
async def test_tavily_search_depth_topic_in_payload():
    """测试 search_depth 和 topic 进入请求 payload"""
    mock_client = _mock_httpx_client(MOCK_TAVILY_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = TavilySearchService(
            api_key="tvly-test-key",
            search_depth="advanced",
            topic="news",
        )
        await service.search("AI行业", limit=5)

    call_args = mock_client.post.call_args
    payload = call_args.kwargs["json"]
    assert payload["search_depth"] == "advanced"
    assert payload["topic"] == "news"
    assert payload["query"] == "AI行业"
    assert payload["include_raw_content"] == "markdown"
    assert payload["include_answer"] is True


@pytest.mark.asyncio
async def test_tavily_api_error_status():
    """测试 API 返回非 200 状态码"""
    mock_client = _mock_httpx_client(
        response_json={"error": "Invalid API key"},
        status_code=401,
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = TavilySearchService(api_key="tvly-invalid-key")
        with pytest.raises(FirecrawlAPIError) as exc_info:
            await service.search("AI行业")

    assert exc_info.value.status_code == 401
