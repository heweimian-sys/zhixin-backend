"""
知行·认知加速器 — 模型注册

导入所有模型，确保它们注册到 Base.metadata，
这样 init_db() 调用 Base.metadata.create_all() 时能创建所有表。
"""
from app.models.base import Base, TimestampMixin
from app.models.topic import Topic
from app.models.event import Event
from app.models.relation import Relation
from app.models.insight import Insight
from app.models.report import Report

__all__ = [
    "Base",
    "TimestampMixin",
    "Topic",
    "Event",
    "Relation",
    "Insight",
    "Report",
]
