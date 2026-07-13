"""
知行·认知加速器 — AI 洞察模型

Insight 是 AI 生成的趋势判断和行动建议，
基于事件列表和关系列表分析得出，属于"第三层认知：洞察行动"。
"""
from __future__ import annotations

from sqlalchemy import ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Insight(Base, TimestampMixin):
    """AI 洞察

    示例：
        title: "模型层战争结束，应用层刚刚开始"
        body: "详细分析..."
        judgments: ["GPU供需矛盾短期无解", "端侧AI是下一个增长点"]
        suggestions: {"投资": ["关注端侧芯片"], "创业": ["AI Agent工具链"], "求职": ["多模态应用开发"]}
        related_event_ids: [1, 3, 5]
        related_relation_ids: [2, 4]
    """

    __tablename__ = "insights"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"),
        comment="所属报告 ID",
    )
    title: Mapped[str] = mapped_column(String(500), comment="洞察标题")
    body: Mapped[str] = mapped_column(Text, default="", comment="详细分析")
    judgments: Mapped[list] = mapped_column(
        JSON,
        default=list,
        comment='关键判断列表，如 ["GPU供需矛盾无解", "端侧AI是增长点"]',
    )
    suggestions: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        comment='按角色分组的建议，如 {"投资": [...], "创业": [...]}',
    )
    related_event_ids: Mapped[list] = mapped_column(
        JSON,
        default=list,
        comment="关联事件 ID 列表",
    )
    related_relation_ids: Mapped[list] = mapped_column(
        JSON,
        default=list,
        comment="关联关系 ID 列表",
    )

    # 关联关系
    report: Mapped["Report"] = relationship(back_populates="insights")

    def __repr__(self) -> str:
        return f"<Insight(id={self.id}, title='{self.title[:30]}...')>"
