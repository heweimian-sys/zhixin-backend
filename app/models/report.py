"""
知行·认知加速器 — 报告模型

Report 是一次搜索生成的完整报告，是数据层级中的顶层实体。
包含查询关键词、引导摘要、章节结构，以及关联的事件、关系和洞察。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Report(Base, TimestampMixin):
    """完整报告

    一次搜索 → 生成一份报告，包含：
    - query: 用户输入的关键词
    - summary: 开头引导摘要（2-3 句）
    - chapters: 章节结构，JSON 格式
    - events: 关联的事件列表
    - relations: 关联的关系列表
    - insights: 关联的洞察列表

    chapters 格式：
        [
            {"title": "第一章·模型之争", "event_ids": [1, 2, 3]},
            {"title": "第二章·应用爆发", "event_ids": [4, 5]}
        ]
    """

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String(255), comment="用户输入的搜索关键词")
    summary: Mapped[str] = mapped_column(Text, default="", comment="开头引导摘要")
    chapters: Mapped[list] = mapped_column(
        JSON,
        default=list,
        comment='章节结构，格式：[{"title": "章节标题", "event_ids": [1, 2]}]',
    )
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="报告生成时间",
    )

    # 关联关系（删除报告时级联删除子记录）
    events: Mapped[list["Event"]] = relationship(
        back_populates="report",
        cascade="all, delete-orphan",
    )
    relations: Mapped[list["Relation"]] = relationship(
        back_populates="report",
        cascade="all, delete-orphan",
    )
    insights: Mapped[list["Insight"]] = relationship(
        back_populates="report",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Report(id={self.id}, query='{self.query}')>"
