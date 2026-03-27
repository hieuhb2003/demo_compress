from __future__ import annotations

from functools import lru_cache
from threading import Lock


LLMLINGUA_LOCK = Lock()


@lru_cache(maxsize=1)
def _build_llmlingua():
    from llmlingua import PromptCompressor

    return PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
    )


def compress_history_context(text: str, rate: float) -> tuple[str, bool, str | None]:
    if not text.strip():
        return text, False, None
    try:
        with LLMLINGUA_LOCK:
            compressor = _build_llmlingua()
            result = compressor.compress_prompt(
                text,
                rate=rate,
                force_tokens=["\n", "[", "]"],
            )
        if isinstance(result, dict):
            compressed = result.get("compressed_prompt", text)
        else:
            compressed = result
        applied = not (compressed == text)
        return compressed, applied, None
    except Exception as exc:
        return text, False, f"{type(exc).__name__}: {exc}"
