"""
知行·认知加速器 — 关联关系模型

Relation 是事件间的关联边，描述事件之间的因果、竞争、包含、依赖或连锁关系。
这是"因果网络"的核心，让信息不只是时间线，而是有脉络的网络。

关系类型：
    causal      因果关系   A → B   "Claude 4 发布" → "OpenAI 加速 GPT-5"
    competitive 竞争关系   A ↔ B   "GPT-5" ↔ "Claude 4"
    contains    包含关系   A ⊂ B   "端侧AI芯片" ⊂ "半导体产业链"
    dependency  技术依赖   A ⇢ B   "AI应用" ⇢ "GPU供应"
    chain       连锁反应   A → B → C  "成本下降 → 创业潮 → 人才短缺"
"""
from __future__ import annotations

from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Relation(Base, TimestampMixin):
    """关联边：连接两个 Event

    from_event → to_event，附带关系类型、描述和确信度。
    """

    __tablename__ = "relations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"),
        comment="所属报告 ID",
    )
    from_event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"),
        comment="起点事件 ID",
    )
    to_event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"),
        comment="终点事件 ID",
    )
    type: Mapped[str] = mapped_column(
        String(50),
        comment="关系类型：causal/competitive/contains/dependency/chain",
    )
    description: Mapped[str] = mapped_column(
        Text, default="", comment="关系描述，例如'Claude 4 的竞争压力迫使 OpenAI 提前发布'"
    )
    confidence: Mapped[float] = mapped_column(
        Float, default=0.0, comment="AI 确信度 0-1"
    )

    # 关联关系
    report: Mapped["Report"] = relationship(back_populates="relations")
    from_event: Mapped["Event"] = relationship(
        foreign_keys=[from_event_id],
        back_populates="outgoing_relations",
    )
    to_event: Mapped["Event"] = relationship(
        foreign_keys=[to_event_id],
        back_populates="incoming_relations",
    )

    def __repr__(self) -> str:
        return f"<Relation(id={self.id}, type='{self.type}', {self.from_event_id}→{self.to_event_id})>"
