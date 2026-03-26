from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    azure_endpoint: str
    azure_api_key: str
    azure_deployment: str
    azure_api_version: str
    local_embedding_model: str
    summary_retrieval_mode: str
    rag_top_k: int
    summary_top_k: int
    llmlingua_rate: float


def load_settings() -> Settings:
    return Settings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip(),
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip(),
        local_embedding_model=os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ).strip(),
        summary_retrieval_mode=os.getenv("SUMMARY_RETRIEVAL_MODE", "embedding").strip().lower(),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        summary_top_k=int(os.getenv("SUMMARY_TOP_K", "1")),
        llmlingua_rate=float(os.getenv("LLMLINGUA_RATE", "0.5")),
    )


def missing_required_settings(settings: Settings) -> list[str]:
    missing = []
    if not settings.azure_endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not settings.azure_api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not settings.azure_deployment:
        missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not settings.azure_api_version:
        missing.append("AZURE_OPENAI_API_VERSION")
    if not settings.local_embedding_model:
        missing.append("LOCAL_EMBEDDING_MODEL")
    return missing
