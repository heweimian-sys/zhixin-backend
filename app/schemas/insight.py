"""
知行·认知加速器 — AI 洞察 Schema
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class InsightBase(BaseModel):
    """洞察基础字段"""

    title: str
    body: str = ""
    judgments: list[str] = []
    suggestions: dict[str, list[str]] = {}
    related_event_ids: list[int] = []
    related_relation_ids: list[int] = []


class InsightCreate(InsightBase):
    """创建洞察时的请求体"""

    report_id: int


class InsightResponse(InsightBase):
    """返回给前端的洞察数据"""

    id: int
    report_id: int

    model_config = ConfigDict(from_attributes=True)
