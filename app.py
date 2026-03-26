from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.azure_client import AzureAIClient
from src.charts import line_chart, metrics_dataframe
from src.config import load_settings, missing_required_settings
from src.local_embeddings import LocalEmbeddingClient
from src.rag import chunk_text, extract_text_from_upload
from src.runtime import METHOD_LABELS, build_initial_state, run_all_methods


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


def _escape(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


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
    st.dataframe(pd.DataFrame(cards), use_container_width=True, hide_index=True)


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


settings = load_settings()
missing = missing_required_settings(settings)

st.title("Chat History Compression Demo")
st.caption("Compare 5 prompt-construction methods on the same user turns, with token, latency, summary, compression, and RAG context.")

with st.sidebar:
    st.header("Config")
    st.write(f"Seed: `{settings.chat_seed}`")
    st.caption("Fixed seed helps reduce variation, but identical outputs are still best-effort rather than guaranteed.")
    st.write(f"Summary retrieval mode: `{settings.summary_retrieval_mode}`")
    st.write(f"Local embedding model: `{settings.local_embedding_model}`")
    st.write(f"RAG top-k: `{settings.rag_top_k}`")
    st.write(f"Summary top-k: `{settings.summary_top_k}`")
    st.write(f"LLMLingua rate: `{settings.llmlingua_rate}`")
    if st.button("Reset Conversation", use_container_width=True):
        reset_state()
        st.rerun()

    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload `.txt`, `.md`, or `.pdf`",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

app_state = get_app_state()

if missing:
    st.error("Missing required environment variables: " + ", ".join(missing))
    st.stop()

azure_client = AzureAIClient(settings)
embedding_client = get_embedding_client(settings.local_embedding_model)

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
        except Exception as exc:
            st.exception(exc)

latest_results = st.session_state.get("latest_results", [])

overview_tab, windows_tab, charts_tab, debug_tab = st.tabs(
    ["Overview", "5 Windows", "Charts", "Debug"]
)

with overview_tab:
    render_overview_cards(latest_results)

with windows_tab:
    if not latest_results:
        st.info("Submit a message to open the five method windows.")
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
                    st.plotly_chart(chart, use_container_width=True)
        with st.expander("Metrics Table", expanded=False):
            st.dataframe(df, use_container_width=True)

with debug_tab:
    st.write("Method summaries")
    for method_key, state in app_state.method_states.items():
        with st.expander(METHOD_LABELS[method_key], expanded=False):
            st.write(f"Turns: {len(state.turns)}")
            st.write(f"Summaries: {len(state.summaries)}")
            st.write(f"Pending summary jobs: {len(state.pending_summary_blocks)}")
            for summary in state.summaries:
                st.code(
                    f"Block {summary.block_index + 1} turns {summary.start_turn}-{summary.end_turn}\n{summary.text}"
                )
