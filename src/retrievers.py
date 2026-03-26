from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from rank_bm25 import BM25Okapi

from src.models import DocumentChunk, SummaryRecord


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def bm25_retrieve(query: str, items: Iterable[str], top_k: int) -> list[int]:
    corpus = [item.split() for item in items]
    if not corpus:
        return []
    engine = BM25Okapi(corpus)
    scores = engine.get_scores(query.split())
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return ranked[:top_k]


def retrieve_summary_records(
    query: str,
    summaries: list[SummaryRecord],
    top_k: int,
    mode: str,
    query_embedding: Optional[list[float]] = None,
) -> list[SummaryRecord]:
    if not summaries or top_k <= 0:
        return []
    if mode == "embedding" and query_embedding is not None:
        ranked = sorted(
            [summary for summary in summaries if summary.embedding is not None],
            key=lambda summary: cosine_similarity(query_embedding, summary.embedding or []),
            reverse=True,
        )
        return ranked[:top_k]
    indices = bm25_retrieve(query, [summary.text for summary in summaries], top_k)
    return [summaries[index] for index in indices]


def retrieve_document_chunks(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int,
    mode: str,
    query_embedding: Optional[list[float]] = None,
) -> list[DocumentChunk]:
    if not chunks or top_k <= 0:
        return []
    if mode == "embedding" and query_embedding is not None:
        ranked = sorted(
            [chunk for chunk in chunks if chunk.embedding is not None],
            key=lambda chunk: cosine_similarity(query_embedding, chunk.embedding or []),
            reverse=True,
        )
        if ranked:
            return ranked[:top_k]
    indices = bm25_retrieve(query, [chunk.text for chunk in chunks], top_k)
    return [chunks[index] for index in indices]
