"""
知行·认知加速器 — Firecrawl 搜索服务

封装自部署 Firecrawl 的搜索 API，提供统一的搜索接口。
Firecrawl 搜索 + 爬取：一次调用返回搜索结果及其 Markdown 内容。

API 文档：https://docs.firecrawl.dev/api-reference/search
自部署文档：https://github.com/firecrawl/firecrawl

请求格式（POST /v1/search）：
    {
        "query": "AI行业",
        "limit": 10,
        "scrapeOptions": {"formats": ["markdown"]}
    }

响应格式（v1）：
    {
        "success": true,
        "data": [
            {
                "markdown": "页面内容...",
                "metadata": {
                    "title": "页面标题",
                    "description": "页面描述",
                    "sourceURL": "https://...",
                    "url": "https://..."
                }
            }
        ]
    }
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class FirecrawlError(Exception):
    """Firecrawl 服务异常基类"""

    pass


class FirecrawlConnectionError(FirecrawlError):
    """连接 Firecrawl 服务失败（服务未启动、网络不通）"""

    pass


class FirecrawlAPIError(FirecrawlError):
    """Firecrawl API 返回错误（非 success 响应）"""

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class SearchResult:
    """搜索结果数据结构

    封装 Firecrawl 返回的单条搜索结果，
    提取标题、URL、描述和 Markdown 内容。
    """

    title: str = ""
    url: str = ""
    description: str = ""
    markdown: str = ""
    source_url: str = ""
    # 原始数据，保留以备后续扩展
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_firecrawl(cls, item: dict) -> "SearchResult":
        """从 Firecrawl API 返回的单条数据构建 SearchResult

        Args:
            item: Firecrawl 返回的 data 数组中的单条记录
        """
        metadata = item.get("metadata", {}) or {}
        return cls(
            title=metadata.get("title", ""),
            url=metadata.get("url", metadata.get("sourceURL", "")),
            description=metadata.get("description", ""),
            markdown=item.get("markdown", ""),
            source_url=metadata.get("sourceURL", metadata.get("url", "")),
            raw=item,
        )

    def to_dict(self) -> dict:
        """转为字典格式（方便日志和调试）"""
        return {
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "markdown_length": len(self.markdown),
        }

    def __repr__(self) -> str:
        return f"<SearchResult(title='{self.title[:30]}...', url='{self.url[:50]}...')>"


class FirecrawlSearchService:
    """Firecrawl 搜索服务

    封装自部署 Firecrawl 的搜索接口，提供异步搜索方法。

    用法：
        service = FirecrawlSearchService()
        results = await service.search("AI行业", limit=10)
        for r in results:
            print(r.title, r.url, r.markdown[:100])

    依赖配置（.env）：
        FIRECRAWL_API_URL=http://localhost:3002/v1
        FIRECRAWL_API_KEY=your-key（自部署可为空）
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
    ) -> None:
        """初始化搜索服务

        Args:
            base_url: Firecrawl API 地址，默认从配置读取
            api_key: API 密钥，默认从配置读取
            timeout: 请求超时秒数
        """
        self.base_url = (base_url or settings.FIRECRAWL_API_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.FIRECRAWL_API_KEY
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        """构建请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def search(
        self,
        query: str,
        limit: int = 10,
        scrape_markdown: bool = True,
    ) -> list[SearchResult]:
        """执行搜索

        Args:
            query: 搜索关键词
            limit: 返回结果数量上限
            scrape_markdown: 是否抓取页面 Markdown 内容（关闭则只返回标题+URL）

        Returns:
            搜索结果列表

        Raises:
            FirecrawlConnectionError: 连接失败
            FirecrawlAPIError: API 返回错误
        """
        url = f"{self.base_url}/search"
        payload: dict = {
            "query": query,
            "limit": limit,
        }
        if scrape_markdown:
            payload["scrapeOptions"] = {"formats": ["markdown"]}

        logger.info("Firecrawl 搜索: query='%s', limit=%d", query, limit)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                )
        except httpx.ConnectError as e:
            logger.error("Firecrawl 连接失败: %s", e)
            raise FirecrawlConnectionError(
                f"无法连接 Firecrawl 服务 ({self.base_url})，请确认服务已启动"
            ) from e
        except httpx.TimeoutException as e:
            logger.error("Firecrawl 请求超时: %s", e)
            raise FirecrawlConnectionError(
                f"Firecrawl 服务响应超时 ({self.timeout}s)"
            ) from e

        if response.status_code != 200:
            error_msg = f"Firecrawl API 返回错误状态码: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f", 详情: {error_detail}"
            except Exception:
                error_msg += f", 响应: {response.text[:200]}"
            logger.error(error_msg)
            raise FirecrawlAPIError(error_msg, status_code=response.status_code)

        data = response.json()

        if not data.get("success", False):
            error_msg = f"Firecrawl API 返回失败: {data.get('error', '未知错误')}"
            logger.error(error_msg)
            raise FirecrawlAPIError(error_msg)

        # 解析结果
        raw_results = data.get("data", [])
        results = [SearchResult.from_firecrawl(item) for item in raw_results]

        logger.info("Firecrawl 搜索完成: 返回 %d 条结果", len(results))
        return results

    async def search_simple(self, query: str, limit: int = 10) -> list[SearchResult]:
        """简化搜索：不抓取 Markdown 内容，只返回标题+URL+描述

        速度更快，适合只需要链接列表的场景。
        """
        return await self.search(query, limit=limit, scrape_markdown=False)


class TavilySearchService:
    """Tavily 搜索服务 — 免费、AI 优化、无需自部署

    Tavily 是专为 AI Agent 设计的搜索 API，
    返回结构化的搜索结果（含标题、内容、URL、markdown 原文）。

    注册地址：https://tavily.com
    免费额度：1000 credits/月（basic=1 credit，advanced=2 credits）
    认证方式：HTTP 请求头 Authorization: Bearer <tvly-xxx>

    用法：
        service = TavilySearchService()
        results = await service.search("AI行业", limit=10)

    依赖配置（.env）：
        TAVILY_API_KEY=tvly-your-key
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 30,
        search_depth: str = "basic",
        topic: str = "general",
    ) -> None:
        """初始化 Tavily 搜索服务

        Args:
            api_key: Tavily API 密钥，默认从配置读取
            timeout: 请求超时秒数
            search_depth: 搜索深度（basic/fast/ultra-fast/advanced）
            topic: 搜索类别（general/news/finance）
        """
        self.api_key = api_key if api_key is not None else settings.TAVILY_API_KEY
        self.base_url = "https://api.tavily.com"
        self.timeout = timeout
        self.search_depth = search_depth
        self.topic = topic

    def _get_headers(self) -> dict[str, str]:
        """构建请求头 — Bearer 认证"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """执行搜索

        Args:
            query: 搜索关键词
            limit: 返回结果数量上限（自动钳制到 1-20）

        Returns:
            搜索结果列表

        Raises:
            FirecrawlConnectionError: 配置缺失或网络错误
            FirecrawlAPIError: API 返回错误
        """
        if not self.api_key:
            raise FirecrawlConnectionError(
                "Tavily API Key 未配置，请在 .env 中设置 TAVILY_API_KEY"
            )

        # Tavily 官方限制 max_results 范围 0-20
        max_results = max(1, min(int(limit), 20))

        logger.info("Tavily 搜索: query='%s', max_results=%d", query, max_results)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "query": query,
                        "max_results": max_results,
                        "search_depth": self.search_depth,
                        "topic": self.topic,
                        "include_answer": True,
                        "include_raw_content": "markdown",
                    },
                    headers=self._get_headers(),
                )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.error("Tavily 搜索失败: %s", e)
            raise FirecrawlConnectionError(f"Tavily 服务不可用: {e}") from e

        if response.status_code != 200:
            error_msg = f"Tavily API 返回错误: HTTP {response.status_code}"
            try:
                detail = response.json()
                error_msg += f", {detail.get('error', detail)})"
            except Exception:
                pass
            logger.error(error_msg)
            raise FirecrawlAPIError(error_msg, status_code=response.status_code)

        data = response.json()
        results = data.get("results", [])

        parsed = []
        for r in results:
            content = r.get("content", "")
            raw = r.get("raw_content", "") or content
            parsed.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=content,
                markdown=raw,
            ))

        logger.info("Tavily 搜索完成: 返回 %d 条结果", len(parsed))
        return parsed
