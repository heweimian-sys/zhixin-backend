"""
知行·认知加速器 — 关系分析和章节组织测试（Task 5/6）
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.deepseek_service import (
    AnalyzedRelation,
    ChapterOutline,
    DeepSeekService,
    ExtractedEvent,
)


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


MOCK_RELATIONS_RESPONSE = {
    "relations": [
        {
            "from_event_index": 0,
            "to_event_index": 1,
            "type": "causal",
            "description": "Claude 4 发布给 OpenAI 带来竞争压力，迫使其加速 GPT-5 开发",
            "confidence": 0.85,
        },
        {
            "from_event_index": 0,
            "to_event_index": 2,
            "type": "causal",
            "description": "Claude 4 的发布推动整个行业对算力的需求",
            "confidence": 0.7,
        },
        {
            "from_event_index": 1,
            "to_event_index": 2,
            "type": "dependency",
            "description": "GPT-5 开发依赖更多 GPU 资源",
            "confidence": 0.8,
        },
    ]
}

MOCK_CHAPTERS_RESPONSE = {
    "chapters": [
        {
            "title": "第一章·模型之争",
            "event_indices": [0, 1],
        },
        {
            "title": "第二章·算力瓶颈",
            "event_indices": [2],
        },
    ]
}


def _mock_openai_response(content: str) -> MagicMock:
    """创建模拟的 OpenAI 响应对象"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


# ============================================================
# AnalyzedRelation 数据结构测试
# ============================================================

def test_analyzed_relation_creation():
    """测试 AnalyzedRelation 创建"""
    rel = AnalyzedRelation(
        from_event_index=0,
        to_event_index=1,
        type="causal",
        description="A 导致 B",
        confidence=0.85,
    )
    assert rel.from_event_index == 0
    assert rel.to_event_index == 1
    assert rel.type == "causal"
    assert rel.confidence == 0.85


def test_analyzed_relation_defaults():
    """测试 AnalyzedRelation 默认值"""
    rel = AnalyzedRelation(from_event_index=0, to_event_index=1, type="competitive")
    assert rel.description == ""
    assert rel.confidence == 0.7


def test_analyzed_relation_to_dict():
    """测试 to_dict"""
    rel = AnalyzedRelation(from_event_index=0, to_event_index=1, type="chain")
    d = rel.to_dict()
    assert d["from_event_index"] == 0
    assert d["to_event_index"] == 1
    assert d["type"] == "chain"


# ============================================================
# ChapterOutline 数据结构测试
# ============================================================

def test_chapter_outline_creation():
    """测试 ChapterOutline 创建"""
    ch = ChapterOutline(title="第一章·模型之争", event_indices=[0, 1])
    assert ch.title == "第一章·模型之争"
    assert ch.event_indices == [0, 1]


def test_chapter_outline_to_dict():
    """测试 to_dict"""
    ch = ChapterOutline(title="第二章", event_indices=[2, 3])
    d = ch.to_dict()
    assert d["title"] == "第二章"
    assert d["event_indices"] == [2, 3]


# ============================================================
# analyze_relations 方法测试
# ============================================================

@pytest.mark.asyncio
async def test_analyze_relations_success():
    """测试正常分析关系"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(MOCK_RELATIONS_RESPONSE))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        relations = await service.analyze_relations("AI行业", _make_events())

    assert len(relations) == 3
    assert relations[0].type == "causal"
    assert relations[0].from_event_index == 0
    assert relations[0].to_event_index == 1
    assert relations[0].confidence == 0.85


@pytest.mark.asyncio
async def test_analyze_relations_insufficient_events():
    """测试事件不足 2 个时跳过分析"""
    mock_client = AsyncMock()

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        relations = await service.analyze_relations("AI", [_make_events()[0]])

    assert len(relations) == 0
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_relations_empty_response():
    """测试 AI 返回空关系列表"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps({"relations": []}))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        relations = await service.analyze_relations("AI", _make_events())

    assert len(relations) == 0


@pytest.mark.asyncio
async def test_analyze_relations_out_of_bounds_index():
    """测试索引越界的关系被跳过"""
    bad_response = {
        "relations": [
            {
                "from_event_index": 0,
                "to_event_index": 1,
                "type": "causal",
                "description": "正常关系",
                "confidence": 0.8,
            },
            {
                "from_event_index": 0,
                "to_event_index": 99,  # 越界
                "type": "causal",
                "description": "越界关系",
                "confidence": 0.5,
            },
        ]
    }
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(bad_response))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        relations = await service.analyze_relations("AI", _make_events())

    # 正常的保留，越界的跳过
    assert len(relations) == 1
    assert relations[0].to_event_index == 1


@pytest.mark.asyncio
async def test_analyze_relations_prompt_contains_events():
    """测试提示词包含事件信息"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(MOCK_RELATIONS_RESPONSE))
    )

    events = _make_events()
    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        await service.analyze_relations("AI行业", events)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert "AI行业" in user_message
    assert "Claude 4 发布" in user_message


# ============================================================
# organize_chapters 方法测试
# ============================================================

@pytest.mark.asyncio
async def test_organize_chapters_success():
    """测试正常组织章节"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(MOCK_CHAPTERS_RESPONSE))
    )

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        events = _make_events()
        chapters = await service.organize_chapters("AI行业", events, [])

    assert len(chapters) == 2
    assert chapters[0].title == "第一章·模型之争"
    assert chapters[0].event_indices == [0, 1]
    assert chapters[1].title == "第二章·算力瓶颈"
    assert chapters[1].event_indices == [2]


@pytest.mark.asyncio
async def test_organize_chapters_empty_events():
    """测试事件为空时跳过"""
    mock_client = AsyncMock()

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        chapters = await service.organize_chapters("AI", [], [])

    assert len(chapters) == 0
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_organize_chapters_prompt_contains_relations():
    """测试提示词包含关系信息"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(json.dumps(MOCK_CHAPTERS_RESPONSE))
    )

    events = _make_events()
    relations = [
        AnalyzedRelation(
            from_event_index=0,
            to_event_index=1,
            type="causal",
            description="Claude 4 导致 GPT-5 加速",
        )
    ]

    with patch("app.services.deepseek_service.AsyncOpenAI", return_value=mock_client):
        service = DeepSeekService(api_key="test-key")
        await service.organize_chapters("AI", events, relations)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert "关联关系" in user_message
    assert "causal" in user_message
