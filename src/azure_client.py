from __future__ import annotations

import time
from typing import List, Tuple

from openai import AzureOpenAI

from src.config import Settings


class AzureAIClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AzureOpenAI(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_endpoint,
        )

    def chat_completion(self, messages: List[dict]) -> Tuple[str, dict, float]:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.settings.azure_deployment,
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
