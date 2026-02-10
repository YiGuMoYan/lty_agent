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

DEBUG = os.getenv("DEBUG", "True").lower() == "true"

