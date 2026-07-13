"""
知行·认知加速器 — 主题/实体模型

Topic 代表从信息中提取的主题或实体，
例如"大模型军备竞赛"（主题）、"OpenAI"（公司）、"Transformer"（技术）。

Phase 1 中 Topic 是独立实体，Phase 2+ 可用于跨报告关联。
"""
from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Topic(Base, TimestampMixin):
    """主题/实体

    类型枚举：主题 / 公司 / 人物 / 技术 / 政策
    """

    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), comment="主题名称")
    type: Mapped[str] = mapped_column(String(100), default="主题", comment="类型：主题/公司/人物/技术/政策")
    description: Mapped[str] = mapped_column(Text, default="", comment="主题描述")

    def __repr__(self) -> str:
        return f"<Topic(id={self.id}, name='{self.name}', type='{self.type}')>"
