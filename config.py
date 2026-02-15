import os
from typing import Optional
from dotenv import load_dotenv
from rag_core.utils.logger import logger

load_dotenv()

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'dataset', 'vector_store', 'qdrant_lty')
SONG_DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'song', 'lyrics.jsonl')
TOPICS_MASTER_PATH = os.path.join(BASE_DIR, 'dataset', 'data_gen', 'topics_master.json')

# Prompt Paths
PROMPT_PATH = os.path.join(BASE_DIR, 'prompt', 'SYSTEM_PROMPT_FRIEND')

# Non-sensitive config with defaults
DEFAULT_RESPONSE_STYLE = os.getenv("DEFAULT_RESPONSE_STYLE", "casual").lower()
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
WS_PORT = int(os.getenv("WS_PORT", "8765"))

# LLM Config - Chat Stage (Conversation)
CHAT_API_BASE = os.getenv("CHAT_API_BASE", "http://localhost:11434/v1")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "lty_v6:7b")
# API密钥 - 无默认值，必须通过环境变量设置
CHAT_API_KEY: Optional[str] = os.getenv("CHAT_API_KEY")

# LLM 可配置参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# LLM Config - Info Gathering Stage (Data Generation/Search)
GEN_API_BASE = os.getenv("GEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "qwen-plus")
# API密钥 - 无默认值，必须通过环境变量设置
GEN_API_KEY: Optional[str] = os.getenv("GEN_API_KEY")

# Embedding Config
# EMBEDDING_BACKEND: "local" = local BGE-M3, "cloud" = Dashscope cloud API
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "cloud").lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_LOCAL_PATH = os.getenv("EMBEDDING_LOCAL_PATH", os.path.join(BASE_DIR, "models", "Xorbits", "bge-m3"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# TTS (Text-to-Speech) Config
TTS_SERVER = os.getenv("TTS_SERVER", "http://cosyvoice-lty-dzxzbwpnzo.cn-hangzhou.fcapp.run")
TTS_ENABLED = os.getenv("TTS_ENABLED", "True").lower() == "true"


def validate_config():
    """验证必要的配置项"""
    required = []
    optional = []

    # 根据后端类型决定需要的API密钥
    if EMBEDDING_BACKEND == "cloud":
        # 如果使用云端embedding，需要验证
        if os.getenv("EMBEDDING_API_KEY"):
            optional.append("EMBEDDING_API_KEY")

    # 检查必要的API密钥
    if CHAT_API_KEY is None:
        required.append("CHAT_API_KEY")
    if GEN_API_KEY is None and os.getenv("USE_GEN_LLM", "False").lower() == "true":
        required.append("GEN_API_KEY")

    if required:
        raise ValueError(f"Missing required environment variables: {', '.join(required)}")

    if optional:
        logger.warning(f"[Config] Warning: Optional environment variables not set: {', '.join(optional)}")

    logger.info("[Config] Configuration validated successfully")


# 模块加载时验证
validate_config()

