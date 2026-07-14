"""
知行·认知加速器 — FastAPI 应用入口

启动方式：
    cd 后端
    uvicorn app.main:app --reload

访问：
    API 文档：http://localhost:8000/docs
    健康检查：http://localhost:8000/health
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import verify_admin_token, verify_token
from app.core.config import settings
from app.core.db import get_db, init_db
from app.models.research_log import ResearchLog
from app.services.research_service import ResearchService, ResearchError


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化数据库，关闭时清理资源"""
    # 启动
    await init_db()
    yield
    # 关闭（目前无需额外清理）


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    description="知行·认知加速器 — 让信息像好文章一样自然流入大脑",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.APP_NAME,
        "description": "认知加速器 — 输入关键词，看到信息的因果脉络",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================
# 研究接口
# ============================================================

class ResearchRequest(BaseModel):
    """研究请求体"""

    query: str
    search_limit: int = 10
    max_events: int = 8


async def save_research_log(
    db: AsyncSession,
    *,
    req: ResearchRequest,
    request: Request,
    status: str,
    duration_ms: int,
    response_json: dict | None = None,
    error_message: str | None = None,
) -> None:
    """保存一次研究请求记录；失败不影响主流程返回"""
    try:
        client_ip = None
        if request.client:
            client_ip = request.client.host
        forwarded_for = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        log = ResearchLog(
            query=req.query.strip(),
            status=status,
            request_json={
                "query": req.query,
                "search_limit": req.search_limit,
                "max_events": req.max_events,
            },
            response_json=response_json,
            error_message=error_message,
            duration_ms=duration_ms,
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent"),
        )
        db.add(log)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.warning("保存研究记录失败: %s", e)


@app.post("/api/research")
async def research(
    req: ResearchRequest,
    request: Request,
    _=Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """研究接口：输入关键词，返回带因果脉络的完整报告

    流程：搜索 → 提取事件 → 分析关系 → 组织章节 → 生成摘要
    """
    started_at = time.perf_counter()

    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="查询关键词不能为空")

    service = ResearchService()

    try:
        result = await service.research(
            query=req.query.strip(),
            search_limit=req.search_limit,
            max_events=req.max_events,
        )
    except ResearchError as e:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        await save_research_log(
            db,
            req=req,
            request=request,
            status="failed",
            duration_ms=duration_ms,
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))

    response_json = result.to_dict()
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    await save_research_log(
        db,
        req=req,
        request=request,
        status="success",
        duration_ms=duration_ms,
        response_json=response_json,
    )

    return response_json


@app.get("/api/admin/research-logs")
async def list_research_logs(
    _=Depends(verify_admin_token),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """管理员后台：查看用户输入和输出记录"""
    result = await db.execute(
        select(ResearchLog)
        .order_by(ResearchLog.created_at.desc(), ResearchLog.id.desc())
        .offset(offset)
        .limit(limit)
    )
    logs = result.scalars().all()
    return {
        "items": [log.to_admin_dict() for log in logs],
        "limit": limit,
        "offset": offset,
    }
