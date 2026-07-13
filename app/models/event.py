"""
知行·认知加速器 — 信息事件模型

Event 是信息流的核心节点，代表一个可追溯的信息事件。
每个 Event 属于一个 Report，包含标题、摘要、日期、来源、确信度和关键引述。
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Event(Base, TimestampMixin):
    """信息事件节点

    示例：
        title: "Claude 4 发布，多模态能力突破"
        summary: "Anthropic 发布 Claude 4，在推理和多模态方面显著提升..."
        date: 2024-03-15
        sources: [{"name": "The Verge", "url": "https://..."}]
        confidence: 0.9
        key_quote: "这是AI行业的Sputnik时刻"
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"),
        comment="所属报告 ID",
    )
    title: Mapped[str] = mapped_column(String(500), comment="事件标题")
    summary: Mapped[str] = mapped_column(Text, comment="AI 生成摘要")
    date: Mapped[date | None] = mapped_column(Date, nullable=True, comment="事件发生日期")
    sources: Mapped[list] = mapped_column(
        JSON,
        default=list,
        comment='来源列表，格式：[{"name": "来源名", "url": "链接"}]',
    )
    confidence: Mapped[float] = mapped_column(
        Float, default=0.0, comment="AI 确信度 0-1"
    )
    key_quote: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="关键引述"
    )

    # 关联关系
    report: Mapped["Report"] = relationship(back_populates="events")
    # 该事件指向其他事件的关系（作为起点）
    outgoing_relations: Mapped[list["Relation"]] = relationship(
        foreign_keys="Relation.from_event_id",
        back_populates="from_event",
    )
    # 其他事件指向该事件的关系（作为终点）
    incoming_relations: Mapped[list["Relation"]] = relationship(
        foreign_keys="Relation.to_event_id",
        back_populates="to_event",
    )

    def __repr__(self) -> str:
        return f"<Event(id={self.id}, title='{self.title[:30]}...')>"
