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

# LLM Config
OLLAMA_URL = os.getenv("LLM_API_BASE", "http://192.168.1.221:11434/api/chat")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "lty_v6:7b")
DEBUG = True

# Graph Config
MAX_HOP_DEPTH = 2
