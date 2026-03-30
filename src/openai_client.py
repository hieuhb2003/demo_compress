from __future__ import annotations

import time
from typing import List, Tuple

from openai import OpenAI

from src.config import Settings


class OpenAIClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**kwargs)

    def chat_completion(self, messages: List[dict]) -> Tuple[str, dict, float]:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            seed=self.settings.chat_seed,
        )
        latency = time.perf_counter() - started
        message = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
        }
        return message, usage, latency
