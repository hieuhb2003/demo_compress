from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import Settings
from src.models import (
    AppState,
    ChatTurn,
    ConversationState,
    DocumentChunk,
    JudgeReference,
    JudgeScore,
    MethodMetrics,
    MethodResult,
    PromptArtifacts,
    SummaryRecord,
)


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "conversation_snapshots.db"


@dataclass
class SnapshotBundle:
    snapshot_id: int
    saved_at: float
    settings_payload: dict
    app_state: AppState
    latest_results: list[MethodResult]
    name: str = ""


@dataclass
class SnapshotInfo:
    snapshot_id: int
    saved_at: float
    name: str


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_at REAL NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                settings_json TEXT NOT NULL,
                app_state_json TEXT NOT NULL,
                latest_results_json TEXT NOT NULL
            )
            """
        )
        # Migration: add name column if missing (existing DBs)
        cursor = conn.execute("PRAGMA table_info(conversation_snapshots)")
        columns = {row[1] for row in cursor.fetchall()}
        if "name" not in columns:
            conn.execute("ALTER TABLE conversation_snapshots ADD COLUMN name TEXT NOT NULL DEFAULT ''")
        conn.commit()


def settings_to_payload(settings: Settings) -> dict:
    payload = asdict(settings)
    payload.pop("openai_api_key", None)
    return payload


def merge_settings(base_settings: Settings, payload: dict | None) -> Settings:
    if not payload:
        return base_settings
    return Settings(
        openai_api_key=base_settings.openai_api_key,
        openai_base_url=payload.get("openai_base_url", base_settings.openai_base_url),
        openai_model=payload.get("openai_model", base_settings.openai_model),
        openai_embedding_model=payload.get("openai_embedding_model", base_settings.openai_embedding_model),
        embedding_base_url=payload.get("embedding_base_url", base_settings.embedding_base_url),
        summary_retrieval_mode=payload.get("summary_retrieval_mode", base_settings.summary_retrieval_mode),
        rag_top_k=int(payload.get("rag_top_k", base_settings.rag_top_k)),
        summary_top_k=int(payload.get("summary_top_k", base_settings.summary_top_k)),
        llmlingua_rate=float(payload.get("llmlingua_rate", base_settings.llmlingua_rate)),
        chat_seed=int(payload.get("chat_seed", base_settings.chat_seed)),
        llm_judge_model=payload.get("llm_judge_model", base_settings.llm_judge_model),
        llm_judge_base_url=payload.get("llm_judge_base_url", base_settings.llm_judge_base_url),
    )


def save_snapshot(
    app_state: AppState,
    latest_results: list[MethodResult],
    settings: Settings,
    name: str = "",
    db_path: Path = DB_PATH,
) -> int:
    init_db(db_path)
    saved_at = time.time()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO conversation_snapshots (
                saved_at,
                name,
                settings_json,
                app_state_json,
                latest_results_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                saved_at,
                name.strip(),
                json.dumps(settings_to_payload(settings)),
                json.dumps(_serialize_app_state(app_state)),
                json.dumps(_serialize_method_results(latest_results)),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def load_snapshot(snapshot_id: int, db_path: Path = DB_PATH) -> SnapshotBundle:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, saved_at, name, settings_json, app_state_json, latest_results_json
            FROM conversation_snapshots
            WHERE id = ?
            """,
            (snapshot_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Snapshot id={snapshot_id} was not found.")

    _, saved_at, name, settings_json, app_state_json, latest_results_json = row
    settings_payload = json.loads(settings_json)
    return SnapshotBundle(
        snapshot_id=int(snapshot_id),
        saved_at=float(saved_at),
        settings_payload=settings_payload,
        app_state=_deserialize_app_state(json.loads(app_state_json)),
        latest_results=_deserialize_method_results(json.loads(latest_results_json)),
        name=name or "",
    )


def list_snapshots(db_path: Path = DB_PATH) -> list[SnapshotInfo]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, saved_at, name FROM conversation_snapshots ORDER BY id DESC"
        ).fetchall()
    return [SnapshotInfo(snapshot_id=r[0], saved_at=r[1], name=r[2] or "") for r in rows]


def update_snapshot(
    snapshot_id: int,
    app_state: AppState,
    latest_results: list[MethodResult],
    settings: Settings,
    name: str | None = None,
    db_path: Path = DB_PATH,
) -> None:
    init_db(db_path)
    saved_at = time.time()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT id, name FROM conversation_snapshots WHERE id = ?", (snapshot_id,)).fetchone()
        if row is None:
            raise ValueError(f"Snapshot id={snapshot_id} was not found.")
        final_name = name.strip() if name is not None else row[1]
        conn.execute(
            """
            UPDATE conversation_snapshots
            SET saved_at = ?, name = ?, settings_json = ?, app_state_json = ?, latest_results_json = ?
            WHERE id = ?
            """,
            (
                saved_at,
                final_name,
                json.dumps(settings_to_payload(settings)),
                json.dumps(_serialize_app_state(app_state)),
                json.dumps(_serialize_method_results(latest_results)),
                snapshot_id,
            ),
        )
        conn.commit()


def delete_snapshot(snapshot_id: int, db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM conversation_snapshots WHERE id = ?", (snapshot_id,))
        conn.commit()


# ── Serialization ──────────────────────────────────────────────

def _serialize_app_state(app_state: AppState) -> dict:
    return {
        "method_states": {
            method_key: _serialize_conversation_state(state)
            for method_key, state in app_state.method_states.items()
        },
        "rag_chunks": [_serialize_document_chunk(chunk) for chunk in app_state.rag_chunks],
        "judge_references": [_serialize_judge_reference(ref) for ref in app_state.judge_references],
        "judge_scores": [_serialize_judge_score(score) for score in app_state.judge_scores],
    }


def _serialize_conversation_state(state: ConversationState) -> dict:
    return {
        "method_key": state.method_key,
        "label": state.label,
        "turns": [_serialize_chat_turn(turn) for turn in state.turns],
        "summaries": [_serialize_summary_record(summary) for summary in state.summaries],
        "metrics_history": [_serialize_method_metrics(metric) for metric in state.metrics_history],
    }


def _serialize_chat_turn(turn: ChatTurn) -> dict:
    return {
        "turn_index": turn.turn_index,
        "user_message": turn.user_message,
        "assistant_message": turn.assistant_message,
        "created_at": turn.created_at,
    }


def _serialize_summary_record(summary: SummaryRecord) -> dict:
    return {
        "block_index": summary.block_index,
        "start_turn": summary.start_turn,
        "end_turn": summary.end_turn,
        "text": summary.text,
        "embedding": summary.embedding,
    }


def _serialize_document_chunk(chunk: DocumentChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "source_name": chunk.source_name,
        "text": chunk.text,
        "embedding": chunk.embedding,
    }


def _serialize_method_metrics(metric: MethodMetrics) -> dict:
    return {
        "turn_index": metric.turn_index,
        "method_key": metric.method_key,
        "estimated_input_tokens": metric.estimated_input_tokens,
        "actual_input_tokens": metric.actual_input_tokens,
        "actual_output_tokens": metric.actual_output_tokens,
        "total_tokens": metric.total_tokens,
        "latency_seconds": metric.latency_seconds,
        "compression_ratio": metric.compression_ratio,
        "prep_time": metric.prep_time,
        "thread_name": metric.thread_name,
    }


def _serialize_judge_reference(ref: JudgeReference) -> dict:
    return {
        "turn_index": ref.turn_index,
        "question": ref.question,
        "reference_answer": ref.reference_answer,
    }


def _serialize_judge_score(score: JudgeScore) -> dict:
    return {
        "turn_index": score.turn_index,
        "method_key": score.method_key,
        "score": score.score,
        "reasoning": score.reasoning,
    }


def _serialize_method_results(results: list[MethodResult]) -> list[dict]:
    return [
        {
            "method_key": result.method_key,
            "label": result.label,
            "assistant_message": result.assistant_message,
            "prompt_artifacts": _serialize_prompt_artifacts(result.prompt_artifacts),
            "metrics": _serialize_method_metrics(result.metrics),
        }
        for result in results
    ]


def _serialize_prompt_artifacts(artifacts: PromptArtifacts) -> dict:
    return {
        "system_prompt": artifacts.system_prompt,
        "context_text": artifacts.context_text,
        "user_message": artifacts.user_message,
        "rag_chunks": [_serialize_document_chunk(chunk) for chunk in artifacts.rag_chunks],
        "retrieved_summaries": [_serialize_summary_record(summary) for summary in artifacts.retrieved_summaries],
        "raw_prompt_preview": artifacts.raw_prompt_preview,
        "compressed_context": artifacts.compressed_context,
        "estimated_input_tokens": artifacts.estimated_input_tokens,
        "compressed_input_tokens": artifacts.compressed_input_tokens,
        "compression_attempted": artifacts.compression_attempted,
        "compression_applied": artifacts.compression_applied,
        "compression_error": artifacts.compression_error,
    }


# ── Deserialization ────────────────────────────────────────────

def _deserialize_app_state(payload: dict) -> AppState:
    return AppState(
        method_states={
            method_key: _deserialize_conversation_state(state_payload)
            for method_key, state_payload in payload.get("method_states", {}).items()
        },
        rag_chunks=[_deserialize_document_chunk(chunk) for chunk in payload.get("rag_chunks", [])],
        judge_references=[_deserialize_judge_reference(ref) for ref in payload.get("judge_references", [])],
        judge_scores=[_deserialize_judge_score(score) for score in payload.get("judge_scores", [])],
    )


def _deserialize_conversation_state(payload: dict) -> ConversationState:
    return ConversationState(
        method_key=payload["method_key"],
        label=payload["label"],
        turns=[_deserialize_chat_turn(turn) for turn in payload.get("turns", [])],
        summaries=[_deserialize_summary_record(summary) for summary in payload.get("summaries", [])],
        pending_summary_blocks={},
        metrics_history=[_deserialize_method_metrics(metric) for metric in payload.get("metrics_history", [])],
    )


def _deserialize_chat_turn(payload: dict) -> ChatTurn:
    return ChatTurn(
        turn_index=int(payload["turn_index"]),
        user_message=str(payload["user_message"]),
        assistant_message=str(payload["assistant_message"]),
        created_at=float(payload["created_at"]),
    )


def _deserialize_summary_record(payload: dict) -> SummaryRecord:
    return SummaryRecord(
        block_index=int(payload["block_index"]),
        start_turn=int(payload["start_turn"]),
        end_turn=int(payload["end_turn"]),
        text=str(payload["text"]),
        embedding=payload.get("embedding"),
    )


def _deserialize_document_chunk(payload: dict) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=str(payload["chunk_id"]),
        source_name=str(payload["source_name"]),
        text=str(payload["text"]),
        embedding=payload.get("embedding"),
    )


def _deserialize_method_metrics(payload: dict) -> MethodMetrics:
    return MethodMetrics(
        turn_index=int(payload["turn_index"]),
        method_key=str(payload["method_key"]),
        estimated_input_tokens=int(payload["estimated_input_tokens"]),
        actual_input_tokens=int(payload["actual_input_tokens"]),
        actual_output_tokens=int(payload["actual_output_tokens"]),
        total_tokens=int(payload["total_tokens"]),
        latency_seconds=float(payload["latency_seconds"]),
        compression_ratio=float(payload["compression_ratio"]),
        prep_time=float(payload.get("prep_time", 0.0)),
        thread_name=str(payload.get("thread_name", "")),
    )


def _deserialize_judge_reference(payload: dict) -> JudgeReference:
    return JudgeReference(
        turn_index=int(payload["turn_index"]),
        question=str(payload["question"]),
        reference_answer=str(payload["reference_answer"]),
    )


def _deserialize_judge_score(payload: dict) -> JudgeScore:
    return JudgeScore(
        turn_index=int(payload["turn_index"]),
        method_key=str(payload["method_key"]),
        score=float(payload["score"]),
        reasoning=str(payload["reasoning"]),
    )


def _deserialize_method_results(payload: list[dict]) -> list[MethodResult]:
    return [
        MethodResult(
            method_key=str(item["method_key"]),
            label=str(item["label"]),
            assistant_message=str(item["assistant_message"]),
            prompt_artifacts=_deserialize_prompt_artifacts(item["prompt_artifacts"]),
            metrics=_deserialize_method_metrics(item["metrics"]),
        )
        for item in payload
    ]


def _deserialize_prompt_artifacts(payload: dict) -> PromptArtifacts:
    compressed_input_tokens = payload.get("compressed_input_tokens")
    return PromptArtifacts(
        system_prompt=str(payload["system_prompt"]),
        context_text=str(payload["context_text"]),
        user_message=str(payload["user_message"]),
        rag_chunks=[_deserialize_document_chunk(chunk) for chunk in payload.get("rag_chunks", [])],
        retrieved_summaries=[
            _deserialize_summary_record(summary) for summary in payload.get("retrieved_summaries", [])
        ],
        raw_prompt_preview=str(payload.get("raw_prompt_preview", "")),
        compressed_context=payload.get("compressed_context"),
        estimated_input_tokens=int(payload.get("estimated_input_tokens", 0)),
        compressed_input_tokens=(None if compressed_input_tokens is None else int(compressed_input_tokens)),
        compression_attempted=bool(payload.get("compression_attempted", False)),
        compression_applied=bool(payload.get("compression_applied", False)),
        compression_error=payload.get("compression_error"),
    )
