import os
from typing import List, Union
from openai import OpenAI
import numpy as np

import config

# Type definitions to replace chromadb.api.types
Documents = List[str]
Embeddings = List[List[float]]

class EmbeddingFunction:
    """
    Abstract base class for embedding functions.
    """
    def __call__(self, input: Documents) -> Embeddings:
        raise NotImplementedError

class DashScopeEmbeddingFunction(EmbeddingFunction):
    """
    Custom Embedding Function using Dashscope (Aliyun).
    Optimized for Chinese content.
    """
    def __init__(self, api_key=None, model_name="text-embedding-v2"):
        self.api_key = api_key or config.GEN_API_KEY
        self.model_name = model_name
        if not self.api_key:
            self.api_key = os.getenv("DASHSCOPE_API_KEY")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        try:
            clean_inputs = [text.replace("\n", " ") for text in input]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=clean_inputs
            )
            data = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in data]
            return embeddings
        except Exception as e:
            print(f"[Embedding] Error: {e}")
            raise e

class LocalBGEEmbeddingFunction(EmbeddingFunction):
    """
    Local BGE-M3 Embedding Function using SentenceTransformer.
    High performance Chinese/Multilingual model.
    """
    def __init__(self, model_path):
        print(f"[Embedding] Loading local BGE-M3 model from: {model_path}")
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedding] Using device: {device.upper()}")
        self.model = SentenceTransformer(model_path, device=device)

    def __call__(self, input: Documents) -> Embeddings:
        # SentenceTransformer encode returns ndarray
        # Ensure normalization for cosine similarity
        self.model.max_seq_length = 1024
        output = self.model.encode(input, normalize_embeddings=True)
        return output.tolist()

def get_embedding_function():
    """
    Factory to get the best configured embedding function.
    """
    # 1. Check for Local BGE-M3 Model
    # Path: ./models/Xorbits/bge-m3
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_model_path = os.path.join(base_dir, "models", "Xorbits", "bge-m3")

    if os.path.exists(local_model_path):
        print("[Embedding] Found local BGE-M3 model. Using Local Backend.")
        return LocalBGEEmbeddingFunction(local_model_path)

    # 2. Prefer Dashscope for Chinese Quality (Cloud)
    if config.GEN_API_KEY:
        print("[Embedding] Using Dashscope (text-embedding-v2)")
        return DashScopeEmbeddingFunction(api_key=config.GEN_API_KEY)

    # 3. Fallback
    print("[Embedding] Error: No GEN_API_KEY and no local model found.")
    raise ValueError("No embedding model available. Please configure GEN_API_KEY or download local model.")


