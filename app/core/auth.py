"""
知行·认知加速器 — 简单认证

使用 Bearer Token 保护 API 端点，防止公网部署后被滥用。
本地开发时可不设密码，自动跳过认证。
"""
from __future__ import annotations

from fastapi import HTTPException, Header

from app.core.config import settings


async def verify_token(authorization: str = Header(None)):
    """验证 Bearer Token

    如果 ACCESS_PASSWORD 为空（本地开发），跳过认证。
    如果设置了密码，必须提供正确的 Bearer Token。

    Args:
        authorization: Authorization 请求头

    Raises:
        HTTPException 401: 认证失败
    """
    # 本地开发模式：不设密码，跳过认证
    if not settings.ACCESS_PASSWORD:
        return True

    # 检查是否有 Authorization 头
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="缺少认证信息，请在请求头添加 Authorization: Bearer <密码>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 验证 Token
    expected = f"Bearer {settings.ACCESS_PASSWORD}"
    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="认证失败，密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True
