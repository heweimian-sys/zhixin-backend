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

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.auth import verify_token
from app.core.config import settings
from app.core.db import init_db
from app.services.research_service import ResearchService, ResearchError


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


@app.post("/api/research")
async def research(req: ResearchRequest, _=Depends(verify_token)):
    """研究接口：输入关键词，返回带因果脉络的完整报告

    流程：搜索 → 提取事件 → 分析关系 → 组织章节 → 生成摘要
    """
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
        raise HTTPException(status_code=500, detail=str(e))

    return result.to_dict()
