"""
知行·认知加速器 — 关联关系 Schema
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RelationBase(BaseModel):
    """关联关系基础字段"""

    from_event_id: int
    to_event_id: int
    type: str  # causal / competitive / contains / dependency / chain
    description: str = ""
    confidence: float = 0.0


class RelationCreate(RelationBase):
    """创建关联关系时的请求体"""

    report_id: int


class RelationResponse(RelationBase):
    """返回给前端的关联关系数据"""

    id: int
    report_id: int

    model_config = ConfigDict(from_attributes=True)
