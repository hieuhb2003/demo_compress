from __future__ import annotations

from typing import Iterable

import tiktoken


def get_encoding(model_hint: str = "gpt-4o-mini") -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_hint)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model_hint: str = "gpt-4o-mini") -> int:
    return len(get_encoding(model_hint).encode(text))


def count_message_tokens(messages: Iterable[dict], model_hint: str = "gpt-4o-mini") -> int:
    total = 0
    for message in messages:
        total += 4
        total += count_tokens(message.get("content", ""), model_hint)
    return total + 2
