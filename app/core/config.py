"""
知行·认知加速器 — 配置管理

使用 pydantic-settings 从环境变量读取配置。
.env 文件存放敏感信息，不提交到 GitHub（已在 .gitignore 中）。
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，从 .env 文件读取"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用配置
    APP_NAME: str = "知行"
    DEBUG: bool = True

    # 数据库（SQLite，文件存储在 data/ 目录）
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/zhixin.db"

    # Firecrawl 搜索服务（自部署）
    FIRECRAWL_API_URL: str = "http://localhost:3002/v1"
    FIRECRAWL_API_KEY: str = ""

    # Tavily 搜索服务（免费，无需自部署）
    # 注册：https://tavily.com
    TAVILY_API_KEY: str = ""

    # DeepSeek AI 服务
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # 访问密码（保护 API 不被滥用）
    # 留空 = 本地开发模式，跳过认证
    ACCESS_PASSWORD: str = ""

    # CORS（前端地址，逗号分隔多个）
    CORS_ORIGINS: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        """将逗号分隔的 CORS 字符串转为列表"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


# 全局配置单例
settings = Settings()
