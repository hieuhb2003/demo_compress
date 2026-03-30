from __future__ import annotations

import html
import json
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from src.openai_client import OpenAIClient
from src.charts import line_chart, metrics_dataframe
from src.compressors import preload_llmlingua
from src.config import load_settings, missing_required_settings
from src.judge import load_judge_references, run_judge
from src.local_embeddings import OpenAIEmbeddingClient
from src.models import ChatTurn, MethodMetrics
from src.persistence import init_db, load_snapshot, merge_settings, save_snapshot
from src.prompt_builders import prepare_prompt
from src.rag import chunk_text, extract_text_from_upload
from src.runtime import METHOD_LABELS, build_initial_state, run_all_methods
from src.tokenizer import count_tokens


st.set_page_config(page_title="Prompt Compression Demo", layout="wide")


@st.cache_resource(show_spinner="Loading LLMLingua model...")
def _preload_llmlingua():
    preload_llmlingua()


@st.cache_resource(show_spinner=False)
def get_embedding_client(api_key: str, model: str, base_url: str = "") -> OpenAIEmbeddingClient:
    return OpenAIEmbeddingClient(api_key=api_key, model=model, base_url=base_url)


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


def backfill_imported_metrics(app_state, imported_turns, settings, openai_client, embedding_client) -> int:
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
                openai_client,
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


def import_conversation_json(raw_bytes: bytes, app_state, settings, openai_client, embedding_client) -> tuple[int, int]:
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
        openai_client,
        embedding_client,
    )

    st.session_state.pop("latest_results", None)
    return len(imported_turns), estimated_turns


def render_overview_cards(latest_results, app_state):
    if not latest_results:
        st.info("Submit a message to see outputs.")
        return

    cards = []
    for result in latest_results:
        state = app_state.method_states[result.method_key]
        cum_input = sum(m.actual_input_tokens for m in state.metrics_history)
        cum_output = sum(m.actual_output_tokens for m in state.metrics_history)
        cum_total = sum(m.total_tokens for m in state.metrics_history)
        avg_latency = (
            round(sum(m.latency_seconds for m in state.metrics_history) / len(state.metrics_history), 2)
            if state.metrics_history
            else 0
        )
        cards.append(
            {
                "Method": result.label,
                "Turns": len(state.turns),
                "Cumul. Prompt": cum_input,
                "Cumul. Output": cum_output,
                "Cumul. Total": cum_total,
                "Avg Latency (s)": avg_latency,
                "Last Compress": round(result.metrics.compression_ratio, 2),
                "Prep (s)": round(result.metrics.prep_time, 2),
                "Thread": result.metrics.thread_name,
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
        cum_total = sum(m.total_tokens for m in state.metrics_history)
        cum_input = sum(m.actual_input_tokens for m in state.metrics_history)
        latency = f"{latest_result.metrics.latency_seconds:.2f}s" if latest_result else "-"
        compression = f"{latest_result.metrics.compression_ratio:.2f}" if latest_result else "-"

        body = [
            '<div class="method-pane">',
            '<div class="method-head">',
            f'<div class="method-title">{html.escape(state.label)}</div>',
            '<div class="metric-grid">',
            f'<div class="metric-cell"><div class="metric-label">Turns</div><div class="metric-value">{len(state.turns)}</div></div>',
            f'<div class="metric-cell"><div class="metric-label">Summaries</div><div class="metric-value">{len(state.summaries)}</div></div>',
            f'<div class="metric-cell"><div class="metric-label">Cumul. Prompt</div><div class="metric-value">{cum_input}</div></div>',
            f'<div class="metric-cell"><div class="metric-label">Cumul. Total</div><div class="metric-value">{cum_total}</div></div>',
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


def build_responses_download(app_state) -> str:
    """Build JSON of all method responses for download."""
    first_state = next(iter(app_state.method_states.values()))
    turn_count = len(first_state.turns)
    output = []
    for turn_idx in range(turn_count):
        turn_data = {"turn_index": turn_idx + 1}
        for method_key, state in app_state.method_states.items():
            if turn_idx < len(state.turns):
                turn = state.turns[turn_idx]
                turn_data["question"] = turn.user_message
                turn_data[method_key] = turn.assistant_message
        output.append(turn_data)
    return json.dumps(output, ensure_ascii=False, indent=2)


def render_judge_tab(app_state, settings):
    st.subheader("LLM Judge Evaluation")
    st.caption(f"Judge model: `{settings.llm_judge_model}`")
    if settings.llm_judge_base_url:
        st.caption(f"Judge base URL: `{settings.llm_judge_base_url}`")

    ref_count = len(app_state.judge_references)
    first_state = next(iter(app_state.method_states.values()))
    turn_count = len(first_state.turns)

    st.write(f"References loaded: **{ref_count}** | Turns completed: **{turn_count}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run LLM Judge", disabled=(ref_count == 0 or turn_count == 0), use_container_width=True):
            with st.spinner(f"Judging {turn_count} turns x 5 methods..."):
                try:
                    scores = run_judge(app_state, settings)
                    app_state.judge_scores = scores
                    st.success(f"Judged {len(scores)} responses.")
                    st.rerun()
                except Exception as exc:
                    st.exception(exc)
    with col2:
        if ref_count == 0:
            st.warning("Upload QA References JSON in sidebar first.")

    if not app_state.judge_scores:
        st.info("No judge results yet. Load references and click 'Run LLM Judge'.")
        return

    # Build score matrix
    score_records = []
    for score in app_state.judge_scores:
        label = METHOD_LABELS.get(score.method_key, score.method_key)
        score_records.append({
            "Turn": score.turn_index,
            "Method": label,
            "Score": score.score,
            "Reasoning": score.reasoning,
        })
    score_df = pd.DataFrame(score_records)

    # Average scores per method
    st.subheader("Average Score by Method")
    avg_df = score_df.groupby("Method", as_index=False)["Score"].mean().sort_values("Score", ascending=False)
    avg_df["Score"] = avg_df["Score"].round(2)
    st.dataframe(avg_df, hide_index=True, use_container_width=True)

    # Bar chart
    import plotly.express as px
    fig = px.bar(avg_df, x="Method", y="Score", color="Method", title="Average Judge Score by Method")
    fig.update_yaxes(range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)

    # Score heatmap: turn x method
    st.subheader("Score per Turn")
    pivot_df = score_df.pivot_table(index="Turn", columns="Method", values="Score")
    st.dataframe(pivot_df.style.background_gradient(cmap="RdYlGn", vmin=1, vmax=10), use_container_width=True)

    # Line chart: scores over turns
    fig2 = px.line(score_df, x="Turn", y="Score", color="Method", markers=True, title="Judge Score per Turn")
    fig2.update_yaxes(range=[0, 10])
    st.plotly_chart(fig2, use_container_width=True)

    # Full details
    with st.expander("Detailed Judge Reasoning", expanded=False):
        st.dataframe(score_df, hide_index=True, use_container_width=True)


def _format_snapshot_time(timestamp: float | None) -> str:
    if timestamp is None:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


# ── Main App ───────────────────────────────────────────────────

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
    st.write(f"Model: `{settings.openai_model}`")
    if settings.openai_base_url:
        st.write(f"Base URL: `{settings.openai_base_url}`")
    st.write(f"Embedding: `{settings.openai_embedding_model}`")
    st.write(f"Seed: `{settings.chat_seed}`")
    st.caption("Fixed seed helps reduce variation, but identical outputs are still best-effort rather than guaranteed.")
    st.write(f"Summary retrieval mode: `{settings.summary_retrieval_mode}`")
    st.write(f"Summary block size: `6 turns`")
    st.write(f"RAG top-k: `{settings.rag_top_k}`")
    st.write(f"Summary top-k: `{settings.summary_top_k}`")
    st.write(f"LLMLingua rate: `{settings.llmlingua_rate}`")
    st.write(f"Judge model: `{settings.llm_judge_model}`")
    if st.button("Reset Conversation", use_container_width=True):
        reset_state()
        st.rerun()

    st.header("Snapshot DB")
    if st.button("Save Current Snapshot", use_container_width=True):
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
    if st.button("Load Snapshot By ID", use_container_width=True):
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

    st.header("QA References (for Judge)")
    qa_ref_file = st.file_uploader(
        "Upload QA JSON with `question` & `reference_answer`",
        type=["json"],
        accept_multiple_files=False,
        key="qa_ref_uploader",
    )

    st.header("Download")
    first_state = next(iter(app_state.method_states.values()), None)
    has_turns = first_state is not None and len(first_state.turns) > 0
    if has_turns:
        st.download_button(
            "Download All Responses (JSON)",
            data=build_responses_download(app_state),
            file_name="method_responses.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.caption("Chat first to enable download.")

if missing:
    st.error("Missing required environment variables: " + ", ".join(missing))
    st.stop()

openai_client = OpenAIClient(settings)
embedding_client = get_embedding_client(settings.openai_api_key, settings.openai_embedding_model, settings.embedding_base_url)
_preload_llmlingua()

# Handle QA reference upload
if qa_ref_file is not None:
    qa_ref_key = qa_ref_file.file_id if hasattr(qa_ref_file, "file_id") else qa_ref_file.name
    last_qa_key = st.session_state.get("last_qa_ref_import_key")
    if qa_ref_key != last_qa_key:
        try:
            raw = json.loads(qa_ref_file.getvalue().decode("utf-8"))
            refs = load_judge_references(raw)
            app_state.judge_references = refs
            st.session_state.last_qa_ref_import_key = qa_ref_key
            st.sidebar.success(f"Loaded {len(refs)} QA references for judge.")
        except Exception as exc:
            st.sidebar.error(f"QA reference import failed: {exc}")

if conversation_file is not None:
    conversation_file_key = conversation_file.file_id if hasattr(conversation_file, "file_id") else conversation_file.name
    last_import_key = st.session_state.get("last_conversation_import_key")
    if conversation_file_key != last_import_key:
        try:
            imported_count, estimated_turns = import_conversation_json(
                conversation_file.getvalue(),
                app_state,
                settings,
                openai_client,
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
            latest_results = run_all_methods(app_state, chat_input, settings, openai_client, embedding_client)
            st.session_state.latest_results = latest_results
            st.session_state.current_snapshot_id = None
            st.session_state.current_snapshot_saved_at = None
        except Exception as exc:
            st.exception(exc)

latest_results = st.session_state.get("latest_results", [])

overview_tab, windows_tab, charts_tab, judge_tab, debug_tab = st.tabs(
    ["Overview", "5 Windows", "Charts", "LLM Judge", "Debug"]
)

with overview_tab:
    render_overview_cards(latest_results, app_state)

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
        top_row[1].metric("Cumul. Avg Prompt", f"{latest_df['cumulative_input_tokens'].mean():.0f}")
        top_row[2].metric("Cumul. Avg Total", f"{latest_df['cumulative_total_tokens'].mean():.0f}")
        top_row[3].metric("Latest Avg Latency", f"{latest_df['latency_seconds'].mean():.2f}s")

        st.subheader("Cumulative Token Usage")
        chart_cols_cum = st.columns(2)
        cumulative_specs = [
            ("cumulative_input_tokens", "Cumulative Prompt Tokens"),
            ("cumulative_output_tokens", "Cumulative Completion Tokens"),
            ("cumulative_total_tokens", "Cumulative Total Tokens"),
        ]
        for index, (column, title) in enumerate(cumulative_specs):
            chart = line_chart(df, column, title)
            if chart is not None:
                with chart_cols_cum[index % 2]:
                    st.plotly_chart(chart, use_container_width=True)

        st.subheader("Per-Turn Metrics")
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
                    st.plotly_chart(chart, use_container_width=True)
        with st.expander("Metrics Table", expanded=False):
            st.dataframe(df, use_container_width=True)

with judge_tab:
    render_judge_tab(app_state, settings)

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
