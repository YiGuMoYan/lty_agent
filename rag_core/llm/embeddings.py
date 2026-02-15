import os
from typing import List
from openai import OpenAI

import config

Documents = List[str]
Embeddings = List[List[float]]


class EmbeddingFunction:
    def __call__(self, input: Documents) -> Embeddings:
        raise NotImplementedError


class DashScopeEmbeddingFunction(EmbeddingFunction):
    """
    Dashscope cloud embedding (text-embedding-v3).
    """
    def __init__(self, api_key=None, model_name=None, dimensions=None):
        self.api_key = api_key or config.GEN_API_KEY or os.getenv("DASHSCOPE_API_KEY", "")
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.dimensions = dimensions or config.EMBEDDING_DIM
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=config.GEN_API_BASE
        )

    def __call__(self, input: Documents) -> Embeddings:
        clean_inputs = [text.replace("\n", " ") for text in input]
        response = self.client.embeddings.create(
            model=self.model_name,
            input=clean_inputs,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in data]


class LocalBGEEmbeddingFunction(EmbeddingFunction):
    """
    Local BGE-M3 embedding via SentenceTransformer.
    """
    def __init__(self, model_path=None):
        model_path = model_path or config.EMBEDDING_LOCAL_PATH
        print(f"[Embedding] Loading local BGE-M3 model from: {model_path}")
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedding] Using device: {device.upper()}")
        self.model = SentenceTransformer(model_path, device=device)

    def __call__(self, input: Documents) -> Embeddings:
        self.model.max_seq_length = 1024
        output = self.model.encode(input, normalize_embeddings=True)
        return output.tolist()


def get_embedding_function():
    backend = config.EMBEDDING_BACKEND

    if backend == "local":
        if not os.path.exists(config.EMBEDDING_LOCAL_PATH):
            raise ValueError(f"[Embedding] Local model not found: {config.EMBEDDING_LOCAL_PATH}")
        print(f"[Embedding] Using local BGE-M3 ({config.EMBEDDING_LOCAL_PATH})")
        return LocalBGEEmbeddingFunction()

    if backend == "cloud":
        if not (config.GEN_API_KEY or os.getenv("DASHSCOPE_API_KEY")):
            # raise ValueError("[Embedding] Cloud mode requires GEN_API_KEY or DASHSCOPE_API_KEY") # Warning instead
            print("[Embedding] Warning: Cloud mode requires GEN_API_KEY. Using mock/empty if fails.")
        print(f"[Embedding] Using Dashscope cloud ({config.EMBEDDING_MODEL_NAME}, dim={config.EMBEDDING_DIM})")
        return DashScopeEmbeddingFunction()

    raise ValueError(f"[Embedding] Unknown EMBEDDING_BACKEND: '{backend}'. Use 'local' or 'cloud'.")
