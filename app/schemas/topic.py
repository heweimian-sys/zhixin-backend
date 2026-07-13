"""
知行·认知加速器 — 主题 Schema
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TopicBase(BaseModel):
    """主题基础字段"""

    name: str
    type: str = "主题"
    description: str = ""


class TopicCreate(TopicBase):
    """创建主题时的请求体"""

    pass


class TopicResponse(TopicBase):
    """返回给前端的主题数据"""

    id: int

    model_config = ConfigDict(from_attributes=True)
