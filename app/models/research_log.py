"""
知行·认知加速器 — 用户研究记录模型

记录每次 /api/research 请求的输入、输出、状态和耗时，供管理员后台查看。
"""
from __future__ import annotations

from sqlalchemy import Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ResearchLog(Base, TimestampMixin):
    """一次用户研究请求的完整记录"""

    __tablename__ = "research_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String(255), index=True, comment="用户原始输入")
    status: Mapped[str] = mapped_column(String(32), default="success", index=True, comment="success/failed")
    request_json: Mapped[dict] = mapped_column(JSON, default=dict, comment="请求参数")
    response_json: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="后端输出结果")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="失败原因")
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="处理耗时毫秒")
    client_ip: Mapped[str | None] = mapped_column(String(120), nullable=True, comment="请求 IP")
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True, comment="浏览器 User-Agent")

    def to_admin_dict(self) -> dict:
        """返回管理员后台使用的数据结构"""
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status,
            "request": self.request_json,
            "response": self.response_json,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
