from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from src.azure_client import AzureAIClient
from src.config import Settings
from src.local_embeddings import LocalEmbeddingClient
from src.models import AppState, ChatTurn, ConversationState, DocumentChunk, MethodMetrics, MethodResult, PromptArtifacts
from src.prompt_builders import build_messages_from_artifacts, prepare_prompt
from src.retrievers import retrieve_document_chunks
from src.summarizers import ensure_summary_jobs


METHOD_LABELS = {
    "summary_window": "Method 1: Summary Window",
    "summary_retrieval": "Method 2: Summary Retrieval",
    "summary_window_llmlingua": "Method 3: Summary Window + LLMLingua",
    "summary_retrieval_llmlingua": "Method 4: Summary Retrieval + LLMLingua",
    "full_history": "Method 5: Full History",
}


def build_initial_state() -> AppState:
    method_states = {
        key: ConversationState(method_key=key, label=label)
        for key, label in METHOD_LABELS.items()
    }
    return AppState(method_states=method_states)


def _compression_ratio(artifacts: PromptArtifacts) -> float:
    if artifacts.compressed_context is None:
        return 1.0
    original = max(1, len(artifacts.context_text))
    return len(artifacts.compressed_context) / original


def _compute_query_embedding(
    user_message: str,
    settings: Settings,
    embedding_client: LocalEmbeddingClient,
) -> list[float] | None:
    if settings.summary_retrieval_mode != "embedding":
        return None
    try:
        return embedding_client.embed(user_message)
    except Exception:
        return None


def _compute_shared_rag_chunks(
    user_message: str,
    app_state: AppState,
    settings: Settings,
    query_embedding: list[float] | None,
) -> list[DocumentChunk]:
    return retrieve_document_chunks(
        user_message,
        app_state.rag_chunks,
        top_k=settings.rag_top_k,
        mode=settings.summary_retrieval_mode,
        query_embedding=query_embedding,
    )


def _run_single_method(
    method_key: str,
    state: ConversationState,
    user_message: str,
    app_state: AppState,
    settings: Settings,
    azure_client: AzureAIClient,
    embedding_client: LocalEmbeddingClient,
    query_embedding: list[float] | None,
    shared_rag_chunks: list[DocumentChunk],
) -> MethodResult:
    turn_index = len(state.turns) + 1
    artifacts = prepare_prompt(
        method_key,
        state,
        user_message,
        app_state.rag_chunks,
        settings,
        azure_client,
        embedding_client,
        query_embedding=query_embedding,
        precomputed_rag_chunks=shared_rag_chunks,
    )
    messages = build_messages_from_artifacts(artifacts)
    assistant_message, usage, latency = azure_client.chat_completion(messages)
    turn = ChatTurn(
        turn_index=turn_index,
        user_message=user_message,
        assistant_message=assistant_message,
        created_at=time.time(),
    )
    state.turns.append(turn)
    if method_key != "full_history":
        ensure_summary_jobs(state, azure_client)
    metrics = MethodMetrics(
        turn_index=turn_index,
        method_key=method_key,
        estimated_input_tokens=artifacts.estimated_input_tokens,
        actual_input_tokens=usage["prompt_tokens"],
        actual_output_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        latency_seconds=latency,
        compression_ratio=_compression_ratio(artifacts),
    )
    state.metrics_history.append(metrics)
    return MethodResult(
        method_key=method_key,
        label=state.label,
        assistant_message=assistant_message,
        prompt_artifacts=artifacts,
        metrics=metrics,
    )


def run_all_methods(
    app_state: AppState,
    user_message: str,
    settings: Settings,
    azure_client: AzureAIClient,
    embedding_client: LocalEmbeddingClient,
) -> list[MethodResult]:
    query_embedding = _compute_query_embedding(user_message, settings, embedding_client)
    shared_rag_chunks = _compute_shared_rag_chunks(user_message, app_state, settings, query_embedding)

    with ThreadPoolExecutor(max_workers=len(app_state.method_states)) as executor:
        futures = [
            executor.submit(
                _run_single_method,
                method_key,
                state,
                user_message,
                app_state,
                settings,
                azure_client,
                embedding_client,
                query_embedding,
                shared_rag_chunks,
            )
            for method_key, state in app_state.method_states.items()
        ]
    return [future.result() for future in futures]
