import sys
import os
import json
from datetime import datetime
from loguru import logger
from config import BASE_DIR

# 确保日志目录存在
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 移除默认的 handler
logger.remove()


def serialize_log(record: dict) -> str:
    """将日志记录序列化为 JSON 格式"""
    log_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": record["level"].name,
        "logger": record["name"],
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"]
    }

    # 添加异常信息
    if record["exception"]:
        log_data["exception"] = str(record["exception"])

    return json.dumps(log_data, ensure_ascii=False)


# 添加控制台输出 (JSON 格式)
logger.add(
    sys.stdout,
    format="{message}",
    serialize=serialize_log,
    level="INFO"
)

# 添加文件输出 (JSON 格式，每天轮转，保留 10 天，最大 10MB)
logger.add(
    os.path.join(LOG_DIR, "lty_agent_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="10 days",
    compression="zip",
    format="{message}",
    serialize=serialize_log,
    level="DEBUG",
    encoding="utf-8"
)

# 导出配置好的 logger
__all__ = ["logger"]
