import sys
import os
from loguru import logger
from config import BASE_DIR

# 确保日志目录存在
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 移除默认的 handler
logger.remove()

# 添加控制台输出 (带颜色)
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# 添加文件输出 (每天轮转，保留 10 天，最大 10MB)
logger.add(
    os.path.join(LOG_DIR, "lty_agent_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="10 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    encoding="utf-8"
)

# 导出配置好的 logger
__all__ = ["logger"]
