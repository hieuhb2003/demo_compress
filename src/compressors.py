from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _build_llmlingua():
    from llmlingua import PromptCompressor

    return PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
    )


def compress_history_context(text: str, rate: float) -> tuple[str, bool]:
    if not text.strip():
        return text, False
    try:
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
        return compressed, True
    except Exception:
        return text, False
