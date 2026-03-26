from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class LocalEmbeddingClient:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(self, text: str) -> List[float]:
        model = _load_model(self.model_name)
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        model = _load_model(self.model_name)
        return model.encode(list(texts), normalize_embeddings=True).tolist()
