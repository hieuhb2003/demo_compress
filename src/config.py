from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_embedding_model: str
    embedding_base_url: str
    summary_retrieval_mode: str
    rag_top_k: int
    summary_top_k: int
    llmlingua_rate: float
    chat_seed: int
    llm_judge_model: str
    llm_judge_base_url: str


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "").strip(),
        summary_retrieval_mode=os.getenv("SUMMARY_RETRIEVAL_MODE", "embedding").strip().lower(),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        summary_top_k=int(os.getenv("SUMMARY_TOP_K", "1")),
        llmlingua_rate=float(os.getenv("LLMLINGUA_RATE", "0.5")),
        chat_seed=int(os.getenv("CHAT_SEED", "42")),
        llm_judge_model=os.getenv("LLM_JUDGE_MODEL", "gpt-4o-mini").strip(),
        llm_judge_base_url=os.getenv("LLM_JUDGE_BASE_URL", "").strip(),
    )


def missing_required_settings(settings: Settings) -> list[str]:
    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.openai_model:
        missing.append("OPENAI_MODEL")
    return missing
