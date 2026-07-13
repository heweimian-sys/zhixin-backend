"""
知行·认知加速器 — 报告 Schema
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.event import EventResponse
from app.schemas.relation import RelationResponse
from app.schemas.insight import InsightResponse


class Chapter(BaseModel):
    """章节结构"""

    title: str
    event_ids: list[int] = []


class ReportBase(BaseModel):
    """报告基础字段"""

    query: str
    summary: str = ""
    chapters: list[Chapter] = []


class ReportCreate(ReportBase):
    """创建报告时的请求体"""

    pass


class ReportResponse(ReportBase):
    """返回给前端的完整报告数据"""

    id: int
    generated_at: datetime
    events: list[EventResponse] = []
    relations: list[RelationResponse] = []
    insights: list[InsightResponse] = []

    model_config = ConfigDict(from_attributes=True)
