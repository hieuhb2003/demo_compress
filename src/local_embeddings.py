from __future__ import annotations

from typing import Iterable, List

from openai import OpenAI


class OpenAIEmbeddingClient:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str = ""):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=text_list,
        )
        return [item.embedding for item in response.data]
