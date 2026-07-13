"""
知行·认知加速器 — 数据模型测试

测试所有模型的 CRUD 操作和关联关系，
包括级联删除验证。
"""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from app.models.report import Report
from app.models.event import Event
from app.models.relation import Relation
from app.models.insight import Insight
from app.models.topic import Topic


# ============================================================
# Report 模型测试
# ============================================================

@pytest.mark.asyncio
async def test_create_report(db_session):
    """测试创建报告"""
    report = Report(
        query="AI行业",
        summary="2024年是AI行业变化最大的一年...",
    )
    db_session.add(report)
    await db_session.commit()
    await db_session.refresh(report)

    assert report.id is not None
    assert report.query == "AI行业"
    assert report.summary.startswith("2024年")
    assert report.generated_at is not None


@pytest.mark.asyncio
async def test_report_with_chapters(db_session):
    """测试报告的章节 JSON 字段"""
    report = Report(
        query="AI行业",
        summary="...",
        chapters=[
            {"title": "第一章·模型之争", "event_ids": [1, 2]},
            {"title": "第二章·应用爆发", "event_ids": [3, 4]},
        ],
    )
    db_session.add(report)
    await db_session.commit()
    await db_session.refresh(report)

    assert len(report.chapters) == 2
    assert report.chapters[0]["title"] == "第一章·模型之争"
    assert report.chapters[1]["event_ids"] == [3, 4]


# ============================================================
# Event 模型测试
# ============================================================

@pytest.mark.asyncio
async def test_create_event(db_session):
    """测试创建事件并关联到报告"""
    report = Report(query="AI行业", summary="...")
    db_session.add(report)
    await db_session.flush()

    event = Event(
        report_id=report.id,
        title="Claude 4 发布，多模态能力突破",
        summary="Anthropic 发布 Claude 4，在推理和多模态方面显著提升",
        date=date(2024, 3, 15),
        sources=[{"name": "The Verge", "url": "https://example.com"}],
        confidence=0.9,
        key_quote="这是AI行业的Sputnik时刻",
    )
    db_session.add(event)
    await db_session.commit()
    await db_session.refresh(event)

    assert event.id is not None
    assert event.report_id == report.id
    assert len(event.sources) == 1
    assert event.sources[0]["name"] == "The Verge"
    assert event.confidence == 0.9


@pytest.mark.asyncio
async def test_report_events_relationship(db_session):
    """测试报告与事件的一对多关系"""
    from sqlalchemy.orm import selectinload

    report = Report(query="AI行业", summary="...")
    db_session.add(report)
    await db_session.flush()

    event1 = Event(report_id=report.id, title="事件A", summary="摘要A")
    event2 = Event(report_id=report.id, title="事件B", summary="摘要B")
    db_session.add_all([event1, event2])
    await db_session.commit()

    # 重新查询并预加载关系（异步上下文不能懒加载）
    result = await db_session.execute(
        select(Report).options(selectinload(Report.events)).where(Report.id == report.id)
    )
    report = result.scalar_one()

    assert len(report.events) == 2
    titles = [e.title for e in report.events]
    assert "事件A" in titles
    assert "事件B" in titles


# ============================================================
# Relation 模型测试
# ============================================================

@pytest.mark.asyncio
async def test_create_relation(db_session):
    """测试创建事件间的关联关系"""
    report = Report(query="AI行业", summary="...")
    db_session.add(report)
    await db_session.flush()

    event1 = Event(report_id=report.id, title="Claude 4 发布", summary="...")
    event2 = Event(report_id=report.id, title="OpenAI 加速 GPT-5 开发", summary="...")
    db_session.add_all([event1, event2])
    await db_session.flush()

    relation = Relation(
        report_id=report.id,
        from_event_id=event1.id,
        to_event_id=event2.id,
        type="causal",
        description="Claude 4 的竞争压力迫使 OpenAI 加速 GPT-5 开发",
        confidence=0.85,
    )
    db_session.add(relation)
    await db_session.commit()
    await db_session.refresh(relation)

    assert relation.id is not None
    assert relation.type == "causal"
    assert relation.from_event_id == event1.id
    assert relation.to_event_id == event2.id
    assert relation.confidence == 0.85


@pytest.mark.asyncio
async def test_relation_types(db_session):
    """测试五种关系类型"""
    report = Report(query="半导体", summary="...")
    db_session.add(report)
    await db_session.flush()

    events = []
    for i in range(6):
        e = Event(report_id=report.id, title=f"事件{i}", summary="...")
        events.append(e)
    db_session.add_all(events)
    await db_session.flush()

    relation_types = ["causal", "competitive", "contains", "dependency", "chain"]
    for i, rtype in enumerate(relation_types):
        r = Relation(
            report_id=report.id,
            from_event_id=events[i].id,
            to_event_id=events[i + 1].id,
            type=rtype,
            description=f"测试 {rtype}",
        )
        db_session.add(r)
    await db_session.commit()

    result = await db_session.execute(
        select(Relation).where(Relation.report_id == report.id)
    )
    relations = result.scalars().all()
    assert len(relations) == 5
    types_in_db = [r.type for r in relations]
    for rtype in relation_types:
        assert rtype in types_in_db


# ============================================================
# Insight 模型测试
# ============================================================

@pytest.mark.asyncio
async def test_create_insight(db_session):
    """测试创建 AI 洞察"""
    report = Report(query="AI行业", summary="...")
    db_session.add(report)
    await db_session.flush()

    insight = Insight(
        report_id=report.id,
        title="模型层战争结束，应用层刚刚开始",
        body="随着基础模型能力趋同，竞争焦点正在从模型层转向应用层...",
        judgments=[
            "GPU 供需矛盾短期无解",
            "端侧 AI 是下一个增长点",
        ],
        suggestions={
            "投资": ["关注端侧芯片", "关注 AI Agent 工具链"],
            "创业": ["多模态应用开发"],
            "求职": ["AI 应用工程师需求激增"],
        },
        related_event_ids=[1, 2],
        related_relation_ids=[1],
    )
    db_session.add(insight)
    await db_session.commit()
    await db_session.refresh(insight)

    assert insight.id is not None
    assert len(insight.judgments) == 2
    assert "投资" in insight.suggestions
    assert len(insight.suggestions["投资"]) == 2
    assert insight.related_event_ids == [1, 2]


# ============================================================
# Topic 模型测试
# ============================================================

@pytest.mark.asyncio
async def test_create_topic(db_session):
    """测试创建主题"""
    topic = Topic(
        name="大模型军备竞赛",
        type="主题",
        description="各大科技公司在大模型领域的激烈竞争",
    )
    db_session.add(topic)
    await db_session.commit()
    await db_session.refresh(topic)

    assert topic.id is not None
    assert topic.name == "大模型军备竞赛"
    assert topic.type == "主题"


@pytest.mark.asyncio
async def test_topic_types(db_session):
    """测试不同类型的主题"""
    topics_data = [
        ("OpenAI", "公司", "AI 研究公司"),
        ("Transformer", "技术", "神经网络架构"),
        ("Sam Altman", "人物", "OpenAI CEO"),
        ("AI监管法案", "政策", "欧盟 AI Act"),
    ]
    for name, ttype, desc in topics_data:
        t = Topic(name=name, type=ttype, description=desc)
        db_session.add(t)
    await db_session.commit()

    result = await db_session.execute(select(Topic))
    topics = result.scalars().all()
    assert len(topics) == 4
    types = [t.type for t in topics]
    assert "公司" in types
    assert "技术" in types
    assert "人物" in types
    assert "政策" in types


# ============================================================
# 级联删除测试
# ============================================================

@pytest.mark.asyncio
async def test_cascade_delete_report(db_session):
    """测试删除报告时级联删除事件、关系、洞察"""
    report = Report(query="AI行业", summary="...")
    db_session.add(report)
    await db_session.flush()

    event1 = Event(report_id=report.id, title="事件A", summary="...")
    event2 = Event(report_id=report.id, title="事件B", summary="...")
    db_session.add_all([event1, event2])
    await db_session.flush()

    relation = Relation(
        report_id=report.id,
        from_event_id=event1.id,
        to_event_id=event2.id,
        type="causal",
        description="A 导致 B",
    )
    db_session.add(relation)

    insight = Insight(
        report_id=report.id,
        title="测试洞察",
        body="...",
    )
    db_session.add(insight)
    await db_session.commit()

    report_id = report.id
    event1_id = event1.id
    relation_id = relation.id
    insight_id = insight.id

    # 删除报告
    await db_session.delete(report)
    await db_session.commit()

    # 验证事件已级联删除
    event_result = await db_session.execute(
        select(Event).where(Event.id == event1_id)
    )
    assert event_result.scalar_one_or_none() is None

    # 验证关系已级联删除
    relation_result = await db_session.execute(
        select(Relation).where(Relation.id == relation_id)
    )
    assert relation_result.scalar_one_or_none() is None

    # 验证洞察已级联删除
    insight_result = await db_session.execute(
        select(Insight).where(Insight.id == insight_id)
    )
    assert insight_result.scalar_one_or_none() is None

    # 验证报告已删除
    report_result = await db_session.execute(
        select(Report).where(Report.id == report_id)
    )
    assert report_result.scalar_one_or_none() is None
