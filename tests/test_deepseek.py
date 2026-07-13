"""
知行·认知加速器 — DeepSeek AI 服务测试

使用 unittest.mock 模拟 OpenAI SDK 的 AsyncOpenAI 客户端，
无需真实 API Key 即可测试所有逻辑。

测试覆盖：
- ExtractedEvent 数据结构
- chat_json 方法（正常 + JSON 解析失败 + API 异常）
- extract_events 方法（正常 + 空结果 + 异常处理）
- 提示词构建逻辑
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.deepseek_service import (
    DeepSeekError,
    DeepSeekResponseError,
    DeepSeekService,
    ExtractedEvent,
)
from app.services.firecrawl_service import SearchResult


# ============================================================
# 模拟数据
# ============================================================

MOCK_DEEPSEEK_RESPONSE = {
    "events": [
        {
            "title": "Claude 4 发布，多模态能力突破",
            "summary": "Anthropic 发布 Claude 4，在推理和多模态方面显著提升，被称为AI行业的Sputnik时刻。",
            "date": "2024-03-15",
            "sources": [
                {"name": "The Verge", "url": "https://example.com/news/claude-4"},
            ],
            "key_quote": "这是AI行业的Sputnik时刻",
            "confidence": 0.9,
        },
        {
            "title": "OpenAI 加速 GPT-5 开发",
            "summary": "受 Claude 4 竞争压力影响，OpenAI 加速 GPT-5 开发进程，预计提前发布。",
            "date": "2024-04-01",
            "sources": [
                {"name": "TechCrunch", "url": "https://example.com/news/gpt5"},
            ],
            "key_quote": None,
            "confidence": 0.75,
        },
    ]
}

MOCK_EMPTY_DEEPSEEK_RESPONSE = {
    "events": []
}

MOCK_INVALID_JSON_CONTENT = "这不是有效的JSON"


# ============================================================
# 搜索结果模拟数据
# ============================================================

def _make_search_results() -> list[SearchResult]:
    """创建模拟搜索结果"""
    return [
        SearchResult(
            title="Claude 4 发布",
            url="https://example.com/news/claude-4",
            description="Anthropic 发布最新模型",
            markdown="# Claude 4 发布\n\nAnthropic 发布了 Claude 4...",
            source_url="https://example.com/news/claude-4",
        ),
        SearchResult(
            title="GPT-5 加速",
            url="https://example.com/news/gpt5",
            description="OpenAI 加速开发",
            markdown="# GPT-5 加速\n\nOpenAI 正在加速 GPT-5...",
            source_url="https://example.com/news/gpt5",
        ),
    ]


# ============================================================
# ExtractedEvent 数据结构测试
# ============================================================

def test_extracted_event_creation():
    """测试 ExtractedEvent 创建"""
    event = ExtractedEvent(
        title="测试事件",
        summary="测试摘要",
        date="2024-01-01",
        sources=[{"name": "来源", "url": "https://test.com"}],
        key_quote="关键引述",
        confidence=0.8,
    )

    assert event.title == "测试事件"
    assert event.summary == "测试摘要"
    assert event.date == "2024-01-01"
    assert len(event.sources) == 1
    assert event.key_quote == "关键引述"
    assert event.confidence == 0.8


def test_extracted_event_defaults():
    """测试 ExtractedEvent 默认值"""
    event = ExtractedEvent(title="测试", summary="摘要")

    assert event.date is None
    assert event.sources == []
    assert event.key_quote is None
    assert event.confidence == 0.7


def test_extracted_event_to_dict():
    """测试 to_dict 方法"""
    event = ExtractedEvent(title="测试", summary="摘要", confidence=0.9)
    d = event.to_dict()

    assert d["title"] == "测试"
    assert d["summary"] == "摘要"
    assert d["confidence"] == 0.9
    assert d["sources"] == []


# ============================================================
# chat_json 方法测试
# ============================================================

def _mock_openai_response(content: str) -> MagicMock:
    """创建模拟的 OpenAI 响应对象"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


@pytest.mark.asyncio
async def test_chat_json_success():
    """测试 chat_json 正常返回 JSON"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps({"key": "value"}))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        result = await service.chat_json("system prompt", "user prompt")

    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_chat_json_parse_error():
    """测试 chat_json JSON 解析失败"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(MOCK_INVALID_JSON_CONTENT)
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        with pytest.raises(DeepSeekResponseError) as exc_info:
            await service.chat_json("system", "user")

    assert "JSON 解析失败" in str(exc_info.value)


@pytest.mark.asyncio
async def test_chat_json_api_error():
    """测试 chat_json API 调用异常"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("API connection failed")
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        with pytest.raises(DeepSeekError) as exc_info:
            await service.chat_json("system", "user")

    assert "API 调用失败" in str(exc_info.value)


@pytest.mark.asyncio
async def test_chat_json_passes_response_format():
    """测试 chat_json 传递 JSON Output 参数"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps({"ok": True}))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        await service.chat_json("system", "user")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert call_kwargs["model"] == "deepseek-chat"


# ============================================================
# extract_events 方法测试
# ============================================================

@pytest.mark.asyncio
async def test_extract_events_success():
    """测试正常提取事件"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(MOCK_DEEPSEEK_RESPONSE))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        events = await service.extract_events("AI行业", _make_search_results())

    assert len(events) == 2
    assert events[0].title == "Claude 4 发布，多模态能力突破"
    assert events[0].date == "2024-03-15"
    assert events[0].confidence == 0.9
    assert events[0].key_quote == "这是AI行业的Sputnik时刻"
    assert len(events[0].sources) == 1

    assert events[1].title == "OpenAI 加速 GPT-5 开发"
    assert events[1].key_quote is None
    assert events[1].confidence == 0.75


@pytest.mark.asyncio
async def test_extract_events_empty_search_results():
    """测试搜索结果为空时返回空列表"""
    mock_client = AsyncMock()

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        events = await service.extract_events("AI行业", [])

    assert len(events) == 0
    # 不应该调用 API
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_extract_events_empty_response():
    """测试 AI 返回空事件列表"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(MOCK_EMPTY_DEEPSEEK_RESPONSE))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        events = await service.extract_events("AI行业", _make_search_results())

    assert len(events) == 0


@pytest.mark.asyncio
async def test_extract_events_api_error():
    """测试提取事件时 API 异常"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("Connection refused")
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        with pytest.raises(DeepSeekError):
            await service.extract_events("AI行业", _make_search_results())


@pytest.mark.asyncio
async def test_extract_events_malformed_event_skipped():
    """测试格式异常的事件被跳过"""
    malformed_response = {
        "events": [
            {
                "title": "正常事件",
                "summary": "正常摘要",
                "date": "2024-01-01",
                "sources": [],
                "confidence": 0.8,
            },
            {
                "title": "置信度异常的事件",
                "summary": "摘要",
                "confidence": "不是一个数字",  # 会触发 ValueError
            },
        ]
    }

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(malformed_response))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        events = await service.extract_events("AI行业", _make_search_results())

    # 正常事件保留，异常事件跳过
    assert len(events) == 1
    assert events[0].title == "正常事件"


@pytest.mark.asyncio
async def test_extract_events_custom_max_events():
    """测试自定义最大事件数"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(MOCK_DEEPSEEK_RESPONSE))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        await service.extract_events("AI行业", _make_search_results(), max_events=5)

    # 验证提示词中包含 max_events
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert "5" in user_message


@pytest.mark.asyncio
async def test_extract_events_prompt_contains_query():
    """测试提示词包含查询关键词"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(MOCK_DEEPSEEK_RESPONSE))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        await service.extract_events("新能源政策", _make_search_results())

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert "新能源政策" in user_message


@pytest.mark.asyncio
async def test_extract_events_prompt_contains_search_content():
    """测试提示词包含搜索结果内容"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(MOCK_DEEPSEEK_RESPONSE))
    )

    search_results = [
        SearchResult(
            title="测试标题",
            url="https://test.com",
            markdown="这是测试内容，应该出现在提示词中",
        ),
    ]

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        await service.extract_events("AI", search_results)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert "测试标题" in user_message
    assert "这是测试内容" in user_message


# ============================================================
# 客户端配置测试
# ============================================================

def test_service_uses_config_defaults():
    """测试服务使用配置文件中的默认值"""
    with patch("app.services.deepseek_service.AsyncOpenAI"):
        service = DeepSeekService()

    assert service.model is not None
    assert service.base_url is not None
