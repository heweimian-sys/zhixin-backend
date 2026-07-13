"""
知行·认知加速器 — Firecrawl 搜索服务测试

使用 unittest.mock 直接 mock httpx.AsyncClient，
确保测试的是真实的 FirecrawlSearchService.search() 代码逻辑。

测试覆盖：
- 正常搜索返回结果（含 Markdown）
- 空结果处理
- 连接错误处理（服务未启动）
- 请求超时处理
- API 返回非 200 状态码
- API 返回 success=false
- search_simple 方法不请求 Markdown
- search 方法请求 Markdown
- 自定义 limit 参数
- SearchResult 数据解析（正常 + 缺少字段）
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.firecrawl_service import (
    FirecrawlAPIError,
    FirecrawlConnectionError,
    FirecrawlSearchService,
    SearchResult,
)


# ============================================================
# 模拟数据
# ============================================================

MOCK_FIRECRAWL_RESPONSE = {
    "success": True,
    "data": [
        {
            "markdown": "# Claude 4 发布\n\nAnthropic 发布了 Claude 4...",
            "metadata": {
                "title": "Claude 4 发布，多模态能力突破",
                "description": "Anthropic 发布最新 AI 模型 Claude 4",
                "sourceURL": "https://example.com/news/claude-4",
                "url": "https://example.com/news/claude-4",
            },
        },
        {
            "markdown": "# GPT-5 开发加速\n\nOpenAI 正在加速 GPT-5...",
            "metadata": {
                "title": "OpenAI 加速 GPT-5 开发",
                "description": "受竞争压力影响，OpenAI 加速开发",
                "sourceURL": "https://example.com/news/gpt5",
                "url": "https://example.com/news/gpt5",
            },
        },
    ],
}

MOCK_EMPTY_RESPONSE = {
    "success": True,
    "data": [],
}

MOCK_ERROR_RESPONSE = {
    "success": False,
    "error": "Invalid API key",
}


# ============================================================
# 辅助函数：创建 mock httpx.AsyncClient
# ============================================================

def _mock_httpx_client(
    response_json: dict | None = None,
    status_code: int = 200,
    exc: Exception | None = None,
) -> MagicMock:
    """创建 mock 的 httpx.AsyncClient 实例

    Args:
        response_json: 返回的 JSON 数据
        status_code: HTTP 状态码
        exc: 要抛出的异常（优先于 response）
    """
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
# SearchResult 数据解析测试（纯数据，无需 mock）
# ============================================================

def test_search_result_from_firecrawl():
    """测试从 Firecrawl 响应数据构建 SearchResult"""
    item = MOCK_FIRECRAWL_RESPONSE["data"][0]
    result = SearchResult.from_firecrawl(item)

    assert result.title == "Claude 4 发布，多模态能力突破"
    assert result.url == "https://example.com/news/claude-4"
    assert result.description == "Anthropic 发布最新 AI 模型 Claude 4"
    assert result.markdown.startswith("# Claude 4 发布")
    assert result.source_url == "https://example.com/news/claude-4"


def test_search_result_missing_metadata():
    """测试缺少 metadata 字段时的容错处理"""
    item = {"markdown": "仅内容，无 metadata"}
    result = SearchResult.from_firecrawl(item)

    assert result.title == ""
    assert result.url == ""
    assert result.markdown == "仅内容，无 metadata"


def test_search_result_to_dict():
    """测试 to_dict 方法"""
    result = SearchResult(title="测试", url="https://test.com", markdown="内容")
    d = result.to_dict()

    assert d["title"] == "测试"
    assert d["url"] == "https://test.com"
    assert d["markdown_length"] == 2


# ============================================================
# 搜索服务测试
# ============================================================

@pytest.mark.asyncio
async def test_search_success():
    """测试正常搜索返回结果"""
    mock_client = _mock_httpx_client(MOCK_FIRECRAWL_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(base_url="http://test:3002/v1")
        results = await service.search("AI行业", limit=10)

    assert len(results) == 2
    assert results[0].title == "Claude 4 发布，多模态能力突破"
    assert results[1].title == "OpenAI 加速 GPT-5 开发"
    assert results[0].markdown.startswith("# Claude 4 发布")
    assert results[0].url == "https://example.com/news/claude-4"


@pytest.mark.asyncio
async def test_search_empty_results():
    """测试搜索返回空结果"""
    mock_client = _mock_httpx_client(MOCK_EMPTY_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(base_url="http://test:3002/v1")
        results = await service.search("不存在的关键词")

    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_connection_error():
    """测试连接错误处理（服务未启动）"""
    mock_client = _mock_httpx_client(exc=httpx.ConnectError("Connection refused"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(base_url="http://test:3002/v1")
        with pytest.raises(FirecrawlConnectionError) as exc_info:
            await service.search("AI行业")

    assert "无法连接" in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_timeout_error():
    """测试请求超时处理"""
    mock_client = _mock_httpx_client(exc=httpx.TimeoutException("Timeout"))

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(base_url="http://test:3002/v1", timeout=5)
        with pytest.raises(FirecrawlConnectionError) as exc_info:
            await service.search("AI行业")

    assert "超时" in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_api_error_status():
    """测试 API 返回非 200 状态码"""
    mock_client = _mock_httpx_client(
        response_json={"error": "Server error"},
        status_code=500,
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(base_url="http://test:3002/v1")
        with pytest.raises(FirecrawlAPIError) as exc_info:
            await service.search("AI行业")

    assert exc_info.value.status_code == 500
    assert "500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_success_false():
    """测试 API 返回 success=false"""
    mock_client = _mock_httpx_client(MOCK_ERROR_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(base_url="http://test:3002/v1")
        with pytest.raises(FirecrawlAPIError) as exc_info:
            await service.search("AI行业")

    assert "Invalid API key" in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_requests_markdown():
    """测试 search 方法请求 Markdown 内容"""
    mock_client = _mock_httpx_client(MOCK_FIRECRAWL_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(base_url="http://test:3002/v1")
        await service.search("AI行业", limit=10)

    # 验证请求参数
    call_args = mock_client.post.call_args
    sent_payload = call_args.kwargs["json"]
    assert sent_payload["scrapeOptions"] == {"formats": ["markdown"]}
    assert sent_payload["query"] == "AI行业"
    assert sent_payload["limit"] == 10


@pytest.mark.asyncio
async def test_search_simple_no_markdown():
    """测试 search_simple 方法不请求 Markdown"""
    mock_client = _mock_httpx_client(MOCK_FIRECRAWL_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(base_url="http://test:3002/v1")
        await service.search_simple("AI行业", limit=5)

    call_args = mock_client.post.call_args
    sent_payload = call_args.kwargs["json"]
    assert "scrapeOptions" not in sent_payload
    assert sent_payload["query"] == "AI行业"
    assert sent_payload["limit"] == 5


@pytest.mark.asyncio
async def test_search_custom_limit():
    """测试自定义返回数量"""
    mock_client = _mock_httpx_client(MOCK_FIRECRAWL_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(base_url="http://test:3002/v1")
        await service.search("AI行业", limit=3)

    call_args = mock_client.post.call_args
    assert call_args.kwargs["json"]["limit"] == 3


@pytest.mark.asyncio
async def test_search_api_key_header():
    """测试 API Key 传递到请求头"""
    mock_client = _mock_httpx_client(MOCK_FIRECRAWL_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(
            base_url="http://test:3002/v1",
            api_key="test-api-key",
        )
        await service.search("AI行业")

    call_args = mock_client.post.call_args
    headers = call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-api-key"


@pytest.mark.asyncio
async def test_search_no_api_key_header():
    """测试无 API Key 时不发送 Authorization 头"""
    mock_client = _mock_httpx_client(MOCK_FIRECRAWL_RESPONSE)

    with patch("httpx.AsyncClient", return_value=mock_client):
        service = FirecrawlSearchService(
            base_url="http://test:3002/v1",
            api_key="",
        )
        await service.search("AI行业")

    call_args = mock_client.post.call_args
    headers = call_args.kwargs["headers"]
    assert "Authorization" not in headers
