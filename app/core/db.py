"""
知行·认知加速器 — 数据库连接

使用 SQLAlchemy 2.0 异步模式 + SQLite（aiosqlite 驱动）。
SQLite 无需额外安装数据库服务，数据存储在文件中。
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
# Base 从 models/base.py 导入，确保所有模型共用同一个 Base
from app.models.base import Base
# 导入所有模型，确保它们注册到 Base.metadata
# 这样 init_db() 调用 create_all 时能创建所有表
import app.models  # noqa: F401


def _ensure_data_dir() -> None:
    """确保 SQLite 数据库文件所在目录存在"""
    # 从 DATABASE_URL 提取文件路径
    # 格式：sqlite+aiosqlite:///./data/zhixin.db
    db_path = settings.DATABASE_URL.split("///")[-1]
    db_dir = Path(db_path).parent
    if str(db_dir) != ".":
        db_dir.mkdir(parents=True, exist_ok=True)


# 确保数据目录存在
_ensure_data_dir()

# 创建异步引擎
# echo=True 时打印 SQL 日志（DEBUG 模式下开启）
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    # SQLite 需要关闭线程检查
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)

# 创建会话工厂
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """FastAPI 依赖：获取数据库会话

    用法：
        @app.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session() as session:
        yield session


async def init_db() -> None:
    """初始化数据库：创建所有表

    在应用启动时调用（main.py 的 lifespan 中）。
    生产环境建议用 Alembic 管理迁移，Phase 1 先用自动建表。
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
