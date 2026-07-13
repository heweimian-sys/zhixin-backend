"""
知行·认知加速器 — 模型基类

Base：所有 SQLAlchemy 模型的基类（DeclarativeBase）
TimestampMixin：通用时间戳字段（created_at, updated_at）
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有模型的基类，SQLAlchemy 2.0 声明式基类"""
    pass


class TimestampMixin:
    """时间戳混入：为模型自动添加创建时间和更新时间"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
