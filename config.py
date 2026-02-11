import os
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'dataset', 'vector_store', 'qdrant_lty')
SONG_DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'song', 'lyrics.jsonl')
TOPICS_MASTER_PATH = os.path.join(BASE_DIR, 'dataset', 'data_gen', 'topics_master.json')

# Prompt Paths
PROMPT_PATH = os.path.join(BASE_DIR, 'prompt', 'SYSTEM_PROMPT_FRIEND')

DEFAULT_RESPONSE_STYLE = os.getenv("DEFAULT_RESPONSE_STYLE", "casual").lower()

# LLM Config - Chat Stage (Conversation)
CHAT_API_BASE = os.getenv("CHAT_API_BASE", "http://localhost:11434/v1")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "lty_v6:7b")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "ollama")

# LLM Config - Info Gathering Stage (Data Generation/Search)
GEN_API_BASE = os.getenv("GEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "qwen-plus")
GEN_API_KEY = os.getenv("GEN_API_KEY", "")

# Embedding Config
# EMBEDDING_BACKEND: "local" = local BGE-M3, "cloud" = Dashscope cloud API
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "cloud").lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_LOCAL_PATH = os.getenv("EMBEDDING_LOCAL_PATH", os.path.join(BASE_DIR, "models", "Xorbits", "bge-m3"))

DEBUG = os.getenv("DEBUG", "True").lower() == "true"

