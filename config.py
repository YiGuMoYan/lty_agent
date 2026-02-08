import os
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Adjust if config is in root, but plan says config.py is in root. 
# If config.py is in root `c:\Users\YiGuMoYan\OneDrive\Desktop\rag_lty\config.py`
# Then BASE_DIR is `c:\Users\YiGuMoYan\OneDrive\Desktop\rag_lty`
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_DIR = os.path.join(BASE_DIR, 'lty_universal')
GRAPH_PATH = os.path.join(BASE_DIR, 'lty_graph.json')
PICKLE_PATH = os.path.join(BASE_DIR, 'lty_graph.pkl')
SONG_DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'song', 'lyrics.jsonl')
PROMPT_PATH = os.path.join(BASE_DIR, 'prompt', 'SYSTEM_PROMPT_SMART')

# LLM Config - Chat Stage (Conversation)
CHAT_API_BASE = os.getenv("CHAT_API_BASE", "http://localhost:11434/v1")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "lty_v6:7b")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "ollama")

# LLM Config - Info Gathering Stage (Data Generation/Search)
# Defaulting to Dashscope/Qwen as it supports search
GEN_API_BASE = os.getenv("GEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "qwen-plus")
GEN_API_KEY = os.getenv("GEN_API_KEY", "")

DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Graph Config
MAX_HOP_DEPTH = 2
