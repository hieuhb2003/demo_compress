from __future__ import annotations

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


settings = load_settings()
missing = missing_required_settings(settings)

st.title("Chat History Compression Demo")
st.caption("Compare 5 prompt-construction methods on the same user turns, with token, latency, summary, compression, and RAG context.")

with st.sidebar:
    st.header("Config")
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

history_tab, methods_tab, charts_tab, debug_tab = st.tabs(["Shared Turn View", "Method Outputs", "Charts", "Debug"])

with history_tab:
    turn_count = max((len(state.turns) for state in app_state.method_states.values()), default=0)
    if turn_count == 0:
        st.info("No turns yet.")
    else:
        for turn_index in range(1, turn_count + 1):
            st.markdown(f"**Turn {turn_index}**")
            user_messages = {
                state.label: next((turn.user_message for turn in state.turns if turn.turn_index == turn_index), "")
                for state in app_state.method_states.values()
            }
            assistant_messages = {
                state.label: next((turn.assistant_message for turn in state.turns if turn.turn_index == turn_index), "")
                for state in app_state.method_states.values()
            }
            baseline_user = next(iter(user_messages.values()), "")
            st.write(f"User: {baseline_user}")
            for label, assistant in assistant_messages.items():
                st.write(f"{label}: {assistant}")

with methods_tab:
    if not latest_results:
        st.info("Submit a message to see outputs.")
    else:
        cols = st.columns(2)
        for index, result in enumerate(latest_results):
            with cols[index % 2]:
                st.markdown(f"### {result.label}")
                st.write(result.assistant_message)
                st.caption(
                    f"Turn {result.metrics.turn_index} | input={result.metrics.actual_input_tokens} | "
                    f"output={result.metrics.actual_output_tokens} | total={result.metrics.total_tokens} | "
                    f"latency={result.metrics.latency_seconds:.2f}s | compression={result.metrics.compression_ratio:.2f}"
                )
                if result.prompt_artifacts.rag_chunks:
                    st.write("RAG chunks used:")
                    for chunk in result.prompt_artifacts.rag_chunks:
                        st.code(f"{chunk.source_name}: {chunk.text[:300]}")
                if result.prompt_artifacts.retrieved_summaries:
                    st.write("Retrieved summaries:")
                    for summary in result.prompt_artifacts.retrieved_summaries:
                        st.code(summary.text[:400])

with charts_tab:
    df = metrics_dataframe(app_state)
    if df.empty:
        st.info("No metrics yet.")
    else:
        for column, title in [
            ("actual_input_tokens", "Prompt Tokens Per Turn"),
            ("actual_output_tokens", "Completion Tokens Per Turn"),
            ("total_tokens", "Total Tokens Per Turn"),
            ("latency_seconds", "Latency Per Turn"),
            ("compression_ratio", "Compression Ratio Per Turn"),
        ]:
            chart = line_chart(df, column, title)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
        st.dataframe(df, use_container_width=True)

with debug_tab:
    st.write("Method summaries")
    for method_key, state in app_state.method_states.items():
        st.markdown(f"### {METHOD_LABELS[method_key]}")
        st.write(f"Turns: {len(state.turns)}")
        st.write(f"Summaries: {len(state.summaries)}")
        st.write(f"Pending summary jobs: {len(state.pending_summary_blocks)}")
        for summary in state.summaries:
            st.code(
                f"Block {summary.block_index + 1} turns {summary.start_turn}-{summary.end_turn}\n{summary.text}"
            )
