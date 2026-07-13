"""
知行·认知加速器 — 事件 Schema
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict


class Source(BaseModel):
    """信息来源"""

    name: str
    url: str = ""


class EventBase(BaseModel):
    """事件基础字段"""

    title: str
    summary: str
    date: date | None = None
    sources: list[Source] = []
    confidence: float = 0.0
    key_quote: str | None = None


class EventCreate(EventBase):
    """创建事件时的请求体"""

    report_id: int


class EventResponse(EventBase):
    """返回给前端的事件数据"""

    id: int
    report_id: int

    model_config = ConfigDict(from_attributes=True)
