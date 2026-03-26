from __future__ import annotations

from typing import List

from src.azure_client import AzureAIClient
from src.compressors import compress_history_context
from src.config import Settings
from src.local_embeddings import LocalEmbeddingClient
from src.models import ConversationState, DocumentChunk, PromptArtifacts, SummaryRecord
from src.retrievers import retrieve_document_chunks, retrieve_summary_records
from src.summarizers import (
    ensure_summary_jobs,
    harvest_completed_summaries,
    maybe_embed_summary,
    require_summary_for_turn_count,
    wait_for_required_summary,
)
from src.tokenizer import count_message_tokens, count_tokens


SYSTEM_PROMPT = (
    "You are a helpful assistant in a prompt-compression demo. "
    "Answer precisely, keep continuity with prior turns, and use retrieved document context when relevant."
)


def _render_turns(turns) -> str:
    lines = []
    for turn in turns:
        lines.append(f"[Turn {turn.turn_index}] User: {turn.user_message}")
        lines.append(f"[Turn {turn.turn_index}] Assistant: {turn.assistant_message}")
    return "\n".join(lines)


def _render_summaries(summaries: List[SummaryRecord]) -> str:
    lines = []
    for summary in summaries:
        lines.append(
            f"[Summary block {summary.block_index + 1} turns {summary.start_turn}-{summary.end_turn}]\n{summary.text}"
        )
    return "\n\n".join(lines)


def _render_rag_context(chunks: List[DocumentChunk]) -> str:
    if not chunks:
        return ""
    lines = ["[Retrieved document context]"]
    for chunk in chunks:
        lines.append(f"Source: {chunk.source_name}")
        lines.append(chunk.text)
    return "\n".join(lines)


def _build_messages(system_prompt: str, history_context: str, user_message: str) -> list[dict]:
    content = history_context.strip() if history_context.strip() else "No prior history."
    user_payload = f"{content}\n\n[Current user message]\n{user_message}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]


def _attach_rag_context(base_context: str, rag_chunks: List[DocumentChunk]) -> str:
    rag_text = _render_rag_context(rag_chunks)
    if not rag_text:
        return base_context
    if not base_context.strip():
        return rag_text
    return f"{base_context}\n\n{rag_text}"


def prepare_prompt(
    method_key: str,
    state: ConversationState,
    user_message: str,
    app_rag_chunks: list[DocumentChunk],
    settings: Settings,
    azure_client: AzureAIClient,
    embedding_client: LocalEmbeddingClient,
) -> PromptArtifacts:
    needs_summaries = method_key != "full_history"
    if needs_summaries:
        harvest_completed_summaries(state)
        ensure_summary_jobs(state, azure_client)

    query_embedding = None
    if settings.summary_retrieval_mode == "embedding":
        try:
            query_embedding = embedding_client.embed(user_message)
        except Exception:
            query_embedding = None

    if needs_summaries:
        required_block = require_summary_for_turn_count(state, len(state.turns))
        if required_block is not None:
            wait_for_required_summary(state, required_block)
            harvest_completed_summaries(state)

        for summary in state.summaries:
            if summary.embedding is None and settings.summary_retrieval_mode == "embedding":
                maybe_embed_summary(summary, embedding_client)

    rag_chunks = retrieve_document_chunks(
        user_message,
        app_rag_chunks,
        top_k=settings.rag_top_k,
        mode=settings.summary_retrieval_mode,
        query_embedding=query_embedding,
    )

    if method_key == "full_history":
        context_text = _attach_rag_context(_render_turns(state.turns), rag_chunks)
        retrieved_summaries: list[SummaryRecord] = []
    elif method_key in {"summary_window", "summary_window_llmlingua"}:
        last_summary_turn = state.summaries[-1].end_turn if state.summaries else 0
        context_parts = []
        if state.summaries:
            context_parts.append(_render_summaries(state.summaries))
        remaining_turns = [turn for turn in state.turns if turn.turn_index > last_summary_turn]
        if remaining_turns:
            context_parts.append(_render_turns(remaining_turns))
        context_text = _attach_rag_context("\n\n".join(part for part in context_parts if part), rag_chunks)
        retrieved_summaries = []
    elif method_key in {"summary_retrieval", "summary_retrieval_llmlingua"}:
        latest_summary = state.summaries[-1:] if state.summaries else []
        older_summaries = state.summaries[:-1] if len(state.summaries) > 1 else []
        retrieved_summaries = retrieve_summary_records(
            user_message,
            older_summaries,
            top_k=settings.summary_top_k,
            mode=settings.summary_retrieval_mode,
            query_embedding=query_embedding,
        )
        parts = []
        if latest_summary:
            parts.append(_render_summaries(latest_summary))
        if retrieved_summaries:
            parts.append("[Retrieved older summaries]\n" + _render_summaries(retrieved_summaries))
        last_summary_turn = state.summaries[-1].end_turn if state.summaries else 0
        remaining_turns = [turn for turn in state.turns if turn.turn_index > last_summary_turn]
        if remaining_turns:
            parts.append(_render_turns(remaining_turns))
        context_text = _attach_rag_context("\n\n".join(part for part in parts if part), rag_chunks)
    else:
        raise ValueError(f"Unsupported method: {method_key}")

    compressed_context = None
    compressible = method_key in {"summary_window_llmlingua", "summary_retrieval_llmlingua"}
    effective_context = context_text
    if compressible and context_text.strip():
        compressed_context, _ = compress_history_context(context_text, settings.llmlingua_rate)
        effective_context = compressed_context or context_text

    messages = _build_messages(SYSTEM_PROMPT, effective_context, user_message)
    raw_messages = _build_messages(SYSTEM_PROMPT, context_text, user_message)

    return PromptArtifacts(
        system_prompt=SYSTEM_PROMPT,
        context_text=context_text,
        user_message=user_message,
        rag_chunks=rag_chunks,
        retrieved_summaries=retrieved_summaries,
        raw_prompt_preview=raw_messages[1]["content"],
        compressed_context=compressed_context,
        estimated_input_tokens=count_message_tokens(messages),
        compressed_input_tokens=count_tokens(effective_context) if compressed_context is not None else None,
    )


def build_messages_from_artifacts(artifacts: PromptArtifacts) -> list[dict]:
    context = artifacts.compressed_context if artifacts.compressed_context is not None else artifacts.context_text
    return _build_messages(artifacts.system_prompt, context, artifacts.user_message)
