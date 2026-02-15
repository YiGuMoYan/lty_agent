import time
import asyncio
from typing import Dict, Any
from qdrant_client import QdrantClient


async def check_llm_health() -> Dict[str, Any]:
    """检查 LLM 服务健康状态"""
    from rag_core.llm.llm_client import LLMClient
    from rag_core.utils.logger import logger

    try:
        client = LLMClient.get_instance()
        # 简单测试调用 - 发送一个简单的聊天请求
        test_messages = [{"role": "user", "content": "ping"}]
        response = await client.chat(test_messages)

        if response and "服务暂时不可用" not in response:
            return {"status": "ok", "model": client.model_name}
        else:
            return {"status": "error", "error": "LLM 返回无效响应"}
    except Exception as e:
        logger.error(f"[HealthCheck] LLM 健康检查失败: {e}")
        return {"status": "error", "error": str(e)}


def check_qdrant_health() -> Dict[str, Any]:
    """检查 Qdrant 向量数据库健康状态"""
    from rag_core.utils.logger import logger
    import config

    try:
        # 使用与 FactIndexer 相同的路径初始化
        persist_directory = config.VECTOR_STORE_PATH

        # 尝试连接并获取集合信息
        client = QdrantClient(path=persist_directory)

        # 检查集合是否存在
        collection_name = "lty_facts"
        if client.collection_exists(collection_name):
            info = client.get_collection(collection_name)
            return {
                "status": "ok",
                "collection": collection_name,
                "points_count": info.points_count,
                "persist_directory": persist_directory
            }
        else:
            return {
                "status": "warning",
                "error": f"集合 '{collection_name}' 不存在",
                "persist_directory": persist_directory
            }
    except Exception as e:
        logger.error(f"[HealthCheck] Qdrant 健康检查失败: {e}")
        return {"status": "error", "error": str(e)}


async def check_health() -> Dict[str, Any]:
    """
    检查所有服务健康状态

    返回格式:
    {
        "status": "healthy" | "unhealthy" | "degraded",
        "timestamp": 1234567890.123,
        "llm": {...},
        "vector_db": {...}
    }
    """
    from rag_core.utils.logger import logger

    start_time = time.time()

    # 并行检查各服务
    llm_result, qdrant_result = await asyncio.gather(
        check_llm_health(),
        asyncio.to_thread(check_qdrant_health)
    )

    # 判断整体状态
    if llm_result["status"] == "ok" and qdrant_result["status"] == "ok":
        overall_status = "healthy"
    elif llm_result["status"] == "error" or qdrant_result["status"] == "error":
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    health_info = {
        "status": overall_status,
        "timestamp": start_time,
        "llm": llm_result,
        "vector_db": qdrant_result,
        "check_duration_ms": int((time.time() - start_time) * 1000)
    }

    logger.info(f"[HealthCheck] 健康检查完成: {overall_status}")
    return health_info


if __name__ == "__main__":
    async def main():
        health = await check_health()
        print(health)

    asyncio.run(main())
