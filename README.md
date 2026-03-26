# Prompt Compression Demo

Demo Streamlit app to compare 5 chat-history strategies against the same user turns:

1. Summary window
2. Summary retrieval
3. Summary window + LLMLingua
4. Summary retrieval + LLMLingua
5. Full history

## Run

Create `.env` from `.env.example`, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- `1 turn = 1 user message + 1 assistant message`
- Chat and summary generation still use Azure OpenAI
- Embedding is local via `sentence-transformers`, not Azure
- Default local model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Summary jobs are created every 10 completed turns per method
- Turn 11 may continue with raw history if the summary job is still running
- Turn 12+ waits for the required summary block before prompt building
- LLMLingua only compresses the history/summary context, not the system prompt or current user message
- Document upload uses simple chunking and in-memory retrieval
