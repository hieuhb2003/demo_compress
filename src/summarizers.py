from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.openai_client import OpenAIClient
from src.local_embeddings import OpenAIEmbeddingClient
from src.models import ChatTurn, ConversationState, SummaryRecord


SUMMARY_BLOCK_SIZE = 6
SUMMARY_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _turns_to_text(turns: list[ChatTurn]) -> str:
    parts = []
    for turn in turns:
        parts.append(f"[Turn {turn.turn_index}] User: {turn.user_message}")
        parts.append(f"[Turn {turn.turn_index}] Assistant: {turn.assistant_message}")
    return "\n".join(parts)


def _generate_summary(
    openai_client: OpenAIClient,
    block_index: int,
    turns: list[ChatTurn],
) -> SummaryRecord:
    prompt = (
        "Summarize these conversation turns as briefly as possible while preserving all key information: "
        "commitments, entities, constraints, preferences, unresolved questions, and facts needed for future turns. "
        "Use bullet points. Be extremely concise.\n\n"
        f"{_turns_to_text(turns)}"
    )
    messages = [
        {"role": "system", "content": "You summarize chat history for future retrieval. Be concise."},
        {"role": "user", "content": prompt},
    ]
    text, _, _ = openai_client.chat_completion(messages)
    return SummaryRecord(
        block_index=block_index,
        start_turn=turns[0].turn_index,
        end_turn=turns[-1].turn_index,
        text=text.strip(),
    )


def ensure_summary_jobs(state: ConversationState, openai_client: OpenAIClient) -> None:
    completed_blocks = len(state.turns) // SUMMARY_BLOCK_SIZE
    for block_index in range(completed_blocks):
        already_done = any(summary.block_index == block_index for summary in state.summaries)
        already_pending = block_index in state.pending_summary_blocks
        if already_done or already_pending:
            continue
        block_turns = state.turns[block_index * SUMMARY_BLOCK_SIZE : (block_index + 1) * SUMMARY_BLOCK_SIZE]
        state.pending_summary_blocks[block_index] = SUMMARY_EXECUTOR.submit(
            _generate_summary,
            openai_client,
            block_index,
            block_turns,
        )


def harvest_completed_summaries(state: ConversationState) -> None:
    completed_blocks = []
    for block_index, future in state.pending_summary_blocks.items():
        if future.done():
            summary = future.result()
            completed_blocks.append(block_index)
            if all(existing.block_index != summary.block_index for existing in state.summaries):
                state.summaries.append(summary)
    for block_index in completed_blocks:
        state.pending_summary_blocks.pop(block_index, None)
    state.summaries.sort(key=lambda item: item.block_index)


def require_summary_for_turn_count(state: ConversationState, turn_count: int) -> Optional[int]:
    completed_blocks = turn_count // SUMMARY_BLOCK_SIZE
    for block_index in range(completed_blocks):
        boundary_turn = (block_index + 1) * SUMMARY_BLOCK_SIZE
        if turn_count == boundary_turn:
            continue
        has_summary = any(summary.block_index == block_index for summary in state.summaries)
        if not has_summary:
            return block_index
    return None


def wait_for_required_summary(state: ConversationState, block_index: int) -> None:
    future = state.pending_summary_blocks.get(block_index)
    if not future:
        return
    summary = future.result()
    state.pending_summary_blocks.pop(block_index, None)
    if all(existing.block_index != summary.block_index for existing in state.summaries):
        state.summaries.append(summary)
        state.summaries.sort(key=lambda item: item.block_index)


def maybe_embed_summary(summary: SummaryRecord, embedding_client: OpenAIEmbeddingClient) -> None:
    try:
        summary.embedding = embedding_client.embed(summary.text)
    except Exception:
        summary.embedding = None
