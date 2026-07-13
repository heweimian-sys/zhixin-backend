"""
知行·认知加速器 — 健康检查测试

验证应用启动、数据库初始化、基础接口可用。
"""
import pytest


@pytest.mark.asyncio
async def test_health_check(client):
    """测试健康检查接口返回正常状态"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "知行"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """测试根路径返回应用信息"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "知行"
    assert "docs" in data
    assert "health" in data


@pytest.mark.asyncio
async def test_db_session(db_session):
    """测试数据库会话可用"""
    from sqlalchemy import text
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1
