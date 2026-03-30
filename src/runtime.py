from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import current_thread

from src.config import Settings
from src.local_embeddings import OpenAIEmbeddingClient
from src.models import AppState, ChatTurn, ConversationState, DocumentChunk, MethodMetrics, MethodResult, PromptArtifacts
from src.openai_client import OpenAIClient
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
    embedding_client: OpenAIEmbeddingClient,
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


def _api_call_in_thread(
    openai_client: OpenAIClient,
    messages: list[dict],
) -> tuple:
    """Execute OpenAI API call in a thread. Return (message, usage, latency, thread_name)."""
    thread_name = current_thread().name
    message, usage, latency = openai_client.chat_completion(messages)
    return message, usage, latency, thread_name


def run_all_methods(
    app_state: AppState,
    user_message: str,
    settings: Settings,
    openai_client: OpenAIClient,
    embedding_client: OpenAIEmbeddingClient,
) -> list[MethodResult]:
    query_embedding = _compute_query_embedding(user_message, settings, embedding_client)
    shared_rag_chunks = _compute_shared_rag_chunks(user_message, app_state, settings, query_embedding)

    # Step 1: Prepare all prompts SEQUENTIALLY in main thread
    #         (LLMLingua compression runs here safely, no fork issues)
    preparations: dict[str, tuple[PromptArtifacts, list[dict], float]] = {}
    for method_key, state in app_state.method_states.items():
        prep_start = time.perf_counter()
        artifacts = prepare_prompt(
            method_key,
            state,
            user_message,
            app_state.rag_chunks,
            settings,
            openai_client,
            embedding_client,
            query_embedding=query_embedding,
            precomputed_rag_chunks=shared_rag_chunks,
        )
        messages = build_messages_from_artifacts(artifacts)
        prep_time = time.perf_counter() - prep_start
        preparations[method_key] = (artifacts, messages, prep_time)

    # Step 2: Send ALL API calls in parallel via threads (HTTP I/O, thread-safe)
    with ThreadPoolExecutor(max_workers=len(preparations), thread_name_prefix="method") as executor:
        api_futures = {
            method_key: executor.submit(
                _api_call_in_thread,
                openai_client,
                messages,
            )
            for method_key, (artifacts, messages, prep_time) in preparations.items()
        }
        api_results = {key: future.result() for key, future in api_futures.items()}

    # Step 3: Collect results and update state in main thread
    results = []
    for method_key, state in app_state.method_states.items():
        artifacts, _, prep_time = preparations[method_key]
        assistant_message, usage, api_latency, thread_name = api_results[method_key]

        turn_index = len(state.turns) + 1
        turn = ChatTurn(
            turn_index=turn_index,
            user_message=user_message,
            assistant_message=assistant_message,
            created_at=time.time(),
        )
        state.turns.append(turn)

        if method_key != "full_history":
            ensure_summary_jobs(state, openai_client)

        metrics = MethodMetrics(
            turn_index=turn_index,
            method_key=method_key,
            estimated_input_tokens=artifacts.estimated_input_tokens,
            actual_input_tokens=usage["prompt_tokens"],
            actual_output_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            latency_seconds=api_latency,
            compression_ratio=_compression_ratio(artifacts),
            prep_time=prep_time,
            thread_name=thread_name,
        )
        state.metrics_history.append(metrics)

        results.append(MethodResult(
            method_key=method_key,
            label=state.label,
            assistant_message=assistant_message,
            prompt_artifacts=artifacts,
            metrics=metrics,
        ))

    return results
