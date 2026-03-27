from __future__ import annotations

import html
import json
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from src.azure_client import AzureAIClient
from src.charts import line_chart, metrics_dataframe
from src.config import load_settings, missing_required_settings
from src.local_embeddings import LocalEmbeddingClient
from src.models import ChatTurn, MethodMetrics
from src.persistence import init_db, load_snapshot, merge_settings, save_snapshot
from src.prompt_builders import prepare_prompt
from src.rag import chunk_text, extract_text_from_upload
from src.runtime import METHOD_LABELS, build_initial_state, run_all_methods
from src.tokenizer import count_tokens


st.set_page_config(page_title="Prompt Compression Demo", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embedding_client(model_name: str) -> LocalEmbeddingClient:
    return LocalEmbeddingClient(model_name)


def get_app_state():
    if "app_state" not in st.session_state:
        st.session_state.app_state = build_initial_state()
    return st.session_state.app_state


def reset_state():
    st.session_state.app_state = build_initial_state()
    st.session_state.pop("latest_results", None)
    st.session_state.pop("active_settings_payload", None)
    st.session_state.pop("current_snapshot_id", None)
    st.session_state.pop("current_snapshot_saved_at", None)
    st.session_state.pop("last_saved_snapshot_id", None)


def _escape(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def _normalize_turns(payload) -> list[dict[str, str]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("turns"), list):
            items = payload["turns"]
        elif isinstance(payload.get("conversation"), list):
            items = payload["conversation"]
        elif isinstance(payload.get("messages"), list):
            return _turns_from_messages(payload["messages"])
        else:
            raise ValueError("JSON must contain `turns`, `conversation`, or `messages`.")
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Conversation JSON must be an object or a list.")

    turns = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Turn #{index} must be an object.")
        user_message = item.get("user_message") or item.get("user") or item.get("question")
        assistant_message = item.get("assistant_message") or item.get("assistant") or item.get("answer")
        if user_message is None or assistant_message is None:
            raise ValueError(
                f"Turn #{index} must contain `user_message`/`user` and `assistant_message`/`assistant`."
            )
        turns.append(
            {
                "user_message": str(user_message),
                "assistant_message": str(assistant_message),
            }
        )
    return turns


def _turns_from_messages(messages) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("`messages` must be a list.")

    turns = []
    pending_user = None
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"Message #{index} must be an object.")
        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")
        if content is None:
            raise ValueError(f"Message #{index} is missing `content`.")
        content = str(content)
        if role == "user":
            if pending_user is not None:
                raise ValueError("Found two consecutive user messages in uploaded JSON.")
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                raise ValueError("Assistant message appears before a user message in uploaded JSON.")
            turns.append({"user_message": pending_user, "assistant_message": content})
            pending_user = None
    if pending_user is not None:
        raise ValueError("Uploaded `messages` ends with a user message without an assistant reply.")
    return turns


def backfill_imported_metrics(app_state, imported_turns, settings, azure_client, embedding_client) -> int:
    replay_turns = imported_turns[:10]
    for state in app_state.method_states.values():
        state.turns.clear()
        state.summaries.clear()
        state.pending_summary_blocks.clear()
        state.metrics_history.clear()

    for turn in replay_turns:
        for method_key, state in app_state.method_states.items():
            artifacts = prepare_prompt(
                method_key,
                state,
                turn.user_message,
                app_state.rag_chunks,
                settings,
                azure_client,
                embedding_client,
            )
            output_tokens = count_tokens(turn.assistant_message)
            state.metrics_history.append(
                MethodMetrics(
                    turn_index=turn.turn_index,
                    method_key=method_key,
                    estimated_input_tokens=artifacts.estimated_input_tokens,
                    actual_input_tokens=artifacts.estimated_input_tokens,
                    actual_output_tokens=output_tokens,
                    total_tokens=artifacts.estimated_input_tokens + output_tokens,
                    latency_seconds=0.0,
                    compression_ratio=(
                        len(artifacts.compressed_context) / max(1, len(artifacts.context_text))
                        if artifacts.compressed_context is not None
                        else 1.0
                    ),
                )
            )
            state.turns.append(
                ChatTurn(
                    turn_index=turn.turn_index,
                    user_message=turn.user_message,
                    assistant_message=turn.assistant_message,
                    created_at=turn.created_at,
                )
            )

    if len(imported_turns) > len(replay_turns):
        for state in app_state.method_states.values():
            state.turns.extend(
                ChatTurn(
                    turn_index=turn.turn_index,
                    user_message=turn.user_message,
                    assistant_message=turn.assistant_message,
                    created_at=turn.created_at,
                )
                for turn in imported_turns[len(replay_turns):]
            )

    return len(replay_turns)


def import_conversation_json(raw_bytes: bytes, app_state, settings, azure_client, embedding_client) -> tuple[int, int]:
    payload = json.loads(raw_bytes.decode("utf-8"))
    normalized_turns = _normalize_turns(payload)
    imported_turns = [
        ChatTurn(
            turn_index=index,
            user_message=turn["user_message"],
            assistant_message=turn["assistant_message"],
            created_at=time.time(),
        )
        for index, turn in enumerate(normalized_turns, start=1)
    ]

    estimated_turns = backfill_imported_metrics(
        app_state,
        imported_turns,
        settings,
        azure_client,
        embedding_client,
    )

    st.session_state.pop("latest_results", None)
    return len(imported_turns), estimated_turns


def render_overview_cards(latest_results):
    if not latest_results:
        st.info("Submit a message to see outputs.")
        return

    cards = []
    for result in latest_results:
        cards.append(
            {
                "Method": result.label,
                "Turn": result.metrics.turn_index,
                "Prompt": result.metrics.actual_input_tokens,
                "Output": result.metrics.actual_output_tokens,
                "Total": result.metrics.total_tokens,
                "Latency (s)": round(result.metrics.latency_seconds, 2),
                "Compress": round(result.metrics.compression_ratio, 2),
                "Preview": result.assistant_message[:140].replace("\n", " "),
            }
        )
    st.dataframe(pd.DataFrame(cards), width="stretch", hide_index=True)


def render_method_windows(app_state, latest_results):
    st.markdown(
        """
        <style>
        .method-pane {
            border: 1px solid #d9d9d9;
            border-radius: 12px;
            padding: 12px;
            background: #fafafa;
            height: 78vh;
            overflow-y: auto;
            font-size: 14px;
        }
        .method-head {
            position: sticky;
            top: -12px;
            background: #fafafa;
            padding: 0 0 10px 0;
            border-bottom: 1px solid #e6e6e6;
            margin-bottom: 10px;
        }
        .method-title {
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 8px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 6px;
            margin-bottom: 8px;
        }
        .metric-cell {
            background: white;
            border: 1px solid #ececec;
            border-radius: 8px;
            padding: 6px 8px;
        }
        .metric-label {
            font-size: 11px;
            color: #666;
        }
        .metric-value {
            font-size: 13px;
            font-weight: 700;
        }
        .chat-block {
            background: white;
            border: 1px solid #ececec;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .turn-title {
            font-weight: 700;
            margin-bottom: 6px;
        }
        .role-user {
            color: #7a4b00;
            margin-bottom: 6px;
        }
        .role-assistant {
            color: #0d4d8b;
        }
        .aux-block {
            background: #fff;
            border: 1px dashed #d6d6d6;
            border-radius: 10px;
            padding: 8px;
            margin-top: 8px;
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    latest_by_method = {result.method_key: result for result in latest_results}
    panes = st.columns(5)
    for pane, (method_key, state) in zip(panes, app_state.method_states.items()):
        latest_result = latest_by_method.get(method_key)
        prompt_tokens = latest_result.metrics.actual_input_tokens if latest_result else 0
        total_tokens = latest_result.metrics.total_tokens if latest_result else 0
        latency = f"{latest_result.metrics.latency_seconds:.2f}s" if latest_result else "-"
        compression = f"{latest_result.metrics.compression_ratio:.2f}" if latest_result else "-"

        body = [
            '<div class="method-pane">',
            '<div class="method-head">',
            f'<div class="method-title">{html.escape(state.label)}</div>',
            '<div class="metric-grid">',
            f'<div class="metric-cell"><div class="metric-label">Turns</div><div class="metric-value">{len(state.turns)}</div></div>',
            f'<div class="metric-cell"><div class="metric-label">Summaries</div><div class="metric-value">{len(state.summaries)}</div></div>',
            f'<div class="metric-cell"><div class="metric-label">Prompt</div><div class="metric-value">{prompt_tokens}</div></div>',
            f'<div class="metric-cell"><div class="metric-label">Total</div><div class="metric-value">{total_tokens}</div></div>',
            f'<div class="metric-cell"><div class="metric-label">Latency</div><div class="metric-value">{latency}</div></div>',
            f'<div class="metric-cell"><div class="metric-label">Compress</div><div class="metric-value">{compression}</div></div>',
            '</div>',
            '</div>',
        ]

        for turn in state.turns:
            body.extend(
                [
                    '<div class="chat-block">',
                    f'<div class="turn-title">Turn {turn.turn_index}</div>',
                    f'<div class="role-user"><strong>User</strong><br>{_escape(turn.user_message)}</div>',
                    f'<div class="role-assistant"><strong>Assistant</strong><br>{_escape(turn.assistant_message)}</div>',
                    '</div>',
                ]
            )

        if state.summaries:
            body.append('<div class="aux-block"><strong>Available Summaries</strong><br>')
            for summary in state.summaries:
                body.append(
                    f'Block {summary.block_index + 1} ({summary.start_turn}-{summary.end_turn})<br>{_escape(summary.text)}<br><br>'
                )
            body.append('</div>')

        if latest_result is not None:
            if latest_result.prompt_artifacts.retrieved_summaries:
                body.append('<div class="aux-block"><strong>Retrieved Summaries This Turn</strong><br>')
                for summary in latest_result.prompt_artifacts.retrieved_summaries:
                    body.append(f'{_escape(summary.text)}<br><br>')
                body.append('</div>')
            if latest_result.prompt_artifacts.rag_chunks:
                body.append('<div class="aux-block"><strong>RAG Chunks This Turn</strong><br>')
                for chunk in latest_result.prompt_artifacts.rag_chunks:
                    body.append(f'{html.escape(chunk.source_name)}<br>{_escape(chunk.text[:500])}<br><br>')
                body.append('</div>')

        body.append('</div>')
        with pane:
            st.markdown("".join(body), unsafe_allow_html=True)


def _format_snapshot_time(timestamp: float | None) -> str:
    if timestamp is None:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


base_settings = load_settings()
settings = merge_settings(base_settings, st.session_state.get("active_settings_payload"))
missing = missing_required_settings(settings)
init_db()
app_state = get_app_state()
latest_results = st.session_state.get("latest_results", [])

st.title("Chat History Compression Demo")
st.caption("Compare 5 prompt-construction methods on the same user turns, with token, latency, summary, compression, and RAG context.")

with st.sidebar:
    st.header("Config")
    current_snapshot_id = st.session_state.get("current_snapshot_id")
    current_snapshot_saved_at = st.session_state.get("current_snapshot_saved_at")
    if current_snapshot_id is not None:
        st.caption(
            f"Loaded snapshot id={current_snapshot_id} saved at {_format_snapshot_time(current_snapshot_saved_at)}"
        )
    st.write(f"Seed: `{settings.chat_seed}`")
    st.caption("Fixed seed helps reduce variation, but identical outputs are still best-effort rather than guaranteed.")
    st.write(f"Summary retrieval mode: `{settings.summary_retrieval_mode}`")
    st.write(f"Local embedding model: `{settings.local_embedding_model}`")
    st.write(f"RAG top-k: `{settings.rag_top_k}`")
    st.write(f"Summary top-k: `{settings.summary_top_k}`")
    st.write(f"LLMLingua rate: `{settings.llmlingua_rate}`")
    if st.button("Reset Conversation", width="stretch"):
        reset_state()
        st.rerun()

    st.header("Snapshot DB")
    if st.button("Save Current Snapshot", width="stretch"):
        snapshot_id = save_snapshot(app_state, latest_results, settings)
        st.session_state.last_saved_snapshot_id = snapshot_id
    last_saved_snapshot_id = st.session_state.get("last_saved_snapshot_id")
    if last_saved_snapshot_id is not None:
        st.success(f"Saved snapshot id={last_saved_snapshot_id}")

    snapshot_id_input = st.number_input(
        "Conversation ID",
        min_value=1,
        step=1,
        value=int(st.session_state.get("current_snapshot_id") or 1),
    )
    if st.button("Load Snapshot By ID", width="stretch"):
        try:
            snapshot = load_snapshot(int(snapshot_id_input))
            st.session_state.app_state = snapshot.app_state
            st.session_state.latest_results = snapshot.latest_results
            st.session_state.active_settings_payload = snapshot.settings_payload
            st.session_state.current_snapshot_id = snapshot.snapshot_id
            st.session_state.current_snapshot_saved_at = snapshot.saved_at
            st.rerun()
        except Exception as exc:
            st.error(f"Snapshot load failed: {exc}")

    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload `.txt`, `.md`, or `.pdf`",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    st.header("Upload Conversation")
    conversation_file = st.file_uploader(
        "Upload conversation JSON",
        type=["json"],
        accept_multiple_files=False,
    )

if missing:
    st.error("Missing required environment variables: " + ", ".join(missing))
    st.stop()

azure_client = AzureAIClient(settings)
embedding_client = get_embedding_client(settings.local_embedding_model)

if conversation_file is not None:
    conversation_file_key = conversation_file.file_id if hasattr(conversation_file, "file_id") else conversation_file.name
    last_import_key = st.session_state.get("last_conversation_import_key")
    if conversation_file_key != last_import_key:
        try:
            imported_count, estimated_turns = import_conversation_json(
                conversation_file.getvalue(),
                app_state,
                settings,
                azure_client,
                embedding_client,
            )
            st.session_state.last_conversation_import_key = conversation_file_key
            st.session_state.current_snapshot_id = None
            st.session_state.current_snapshot_saved_at = None
            message = f"Imported {imported_count} turns into all 5 methods."
            if estimated_turns:
                message += f" Backfilled estimated token charts for {estimated_turns} imported turns."
            if imported_count > estimated_turns:
                message += " Estimated charts currently cover the first 10 imported turns."
            st.sidebar.success(message)
        except Exception as exc:
            st.sidebar.error(f"Conversation import failed: {exc}")

if uploaded_files:
    existing_ids = {chunk.chunk_id for chunk in app_state.rag_chunks}
    for uploaded in uploaded_files:
        text = extract_text_from_upload(uploaded.name, uploaded.getvalue())
        chunks = chunk_text(text, uploaded.name)
        new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
        if not new_chunks:
            continue
        embeddings = embedding_client.embed_many(chunk.text for chunk in new_chunks)
        for chunk, embedding in zip(new_chunks, embeddings):
            chunk.embedding = embedding
            app_state.rag_chunks.append(chunk)
            existing_ids.add(chunk.chunk_id)

st.subheader("Conversation")
chat_input = st.chat_input("Type one message. All 5 methods will run.")

if chat_input:
    with st.spinner("Running all methods..."):
        try:
            latest_results = run_all_methods(app_state, chat_input, settings, azure_client, embedding_client)
            st.session_state.latest_results = latest_results
            st.session_state.current_snapshot_id = None
            st.session_state.current_snapshot_saved_at = None
        except Exception as exc:
            st.exception(exc)

latest_results = st.session_state.get("latest_results", [])

overview_tab, windows_tab, charts_tab, debug_tab = st.tabs(
    ["Overview", "5 Windows", "Charts", "Debug"]
)

with overview_tab:
    render_overview_cards(latest_results)

with windows_tab:
    if not app_state.method_states or not any(state.turns for state in app_state.method_states.values()):
        st.info("Upload a conversation JSON or submit a message to open the five method windows.")
    else:
        render_method_windows(app_state, latest_results)

with charts_tab:
    df = metrics_dataframe(app_state)
    if df.empty:
        st.info("No metrics yet.")
    else:
        top_row = st.columns(4)
        latest_df = df.sort_values("turn").groupby("method", as_index=False).tail(1)
        top_row[0].metric("Tracked Methods", len(latest_df))
        top_row[1].metric("Latest Avg Prompt", f"{latest_df['actual_input_tokens'].mean():.0f}")
        top_row[2].metric("Latest Avg Total", f"{latest_df['total_tokens'].mean():.0f}")
        top_row[3].metric("Latest Avg Latency", f"{latest_df['latency_seconds'].mean():.2f}s")

        chart_cols = st.columns(2)
        chart_specs = [
            ("actual_input_tokens", "Prompt Tokens Per Turn"),
            ("actual_output_tokens", "Completion Tokens Per Turn"),
            ("total_tokens", "Total Tokens Per Turn"),
            ("latency_seconds", "Latency Per Turn"),
            ("compression_ratio", "Compression Ratio Per Turn"),
        ]
        for index, (column, title) in enumerate(chart_specs):
            chart = line_chart(df, column, title)
            if chart is not None:
                with chart_cols[index % 2]:
                    st.plotly_chart(chart, width="stretch")
        with st.expander("Metrics Table", expanded=False):
            st.dataframe(df, width="stretch")

with debug_tab:
    st.write("Method summaries")
    latest_by_method = {result.method_key: result for result in latest_results}
    for method_key, state in app_state.method_states.items():
        with st.expander(METHOD_LABELS[method_key], expanded=False):
            st.write(f"Turns: {len(state.turns)}")
            st.write(f"Summaries: {len(state.summaries)}")
            st.write(f"Pending summary jobs: {len(state.pending_summary_blocks)}")
            latest_result = latest_by_method.get(method_key)
            if latest_result is not None:
                artifacts = latest_result.prompt_artifacts
                st.write("Latest prompt debug")
                st.json(
                    {
                        "compression_attempted": artifacts.compression_attempted,
                        "compression_applied": artifacts.compression_applied,
                        "compression_error": artifacts.compression_error,
                        "context_chars": len(artifacts.context_text),
                        "compressed_chars": (len(artifacts.compressed_context) if artifacts.compressed_context is not None else None),
                        "estimated_input_tokens": artifacts.estimated_input_tokens,
                        "compressed_input_tokens": artifacts.compressed_input_tokens,
                        "rag_chunks": len(artifacts.rag_chunks),
                        "retrieved_summaries": len(artifacts.retrieved_summaries),
                    }
                )
            for summary in state.summaries:
                st.code(
                    f"Block {summary.block_index + 1} turns {summary.start_turn}-{summary.end_turn}\n{summary.text}"
                )
