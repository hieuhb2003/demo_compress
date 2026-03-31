"""Log the full prompt and token usage for each method after every chat turn."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List

from src.models import MethodResult

LOG_DIR = Path("logs")


def log_turn_results(results: List[MethodResult], user_message: str) -> str:
    """Write a JSON log file for one chat turn across all methods.

    Returns the path to the log file.
    """
    LOG_DIR.mkdir(exist_ok=True)

    if not results:
        return ""

    turn_index = results[0].metrics.turn_index
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"turn_{turn_index:03d}_{timestamp}.json"
    filepath = LOG_DIR / filename

    entries = []
    for r in results:
        artifacts = r.prompt_artifacts
        m = r.metrics

        # Rebuild the actual messages sent to the API
        context = artifacts.compressed_context if artifacts.compressed_context is not None else artifacts.context_text
        content = context.strip() if context.strip() else "No prior history."
        user_payload = f"{content}\n\n[Current user message]\n{artifacts.user_message}"
        messages_sent = [
            {"role": "system", "content": artifacts.system_prompt},
            {"role": "user", "content": user_payload},
        ]

        entries.append({
            "method_key": r.method_key,
            "label": r.label,
            "messages_sent": messages_sent,
            "token_usage": {
                "estimated_input_tokens": m.estimated_input_tokens,
                "actual_input_tokens": m.actual_input_tokens,
                "actual_output_tokens": m.actual_output_tokens,
                "total_tokens": m.total_tokens,
            },
            "latency_seconds": round(m.latency_seconds, 3),
            "prep_time_seconds": round(m.prep_time, 3),
            "compression": {
                "attempted": artifacts.compression_attempted,
                "applied": artifacts.compression_applied,
                "ratio": round(m.compression_ratio, 4),
                "error": artifacts.compression_error,
            },
            "assistant_response": r.assistant_message,
        })

    log_data = {
        "turn_index": turn_index,
        "timestamp": timestamp,
        "user_message": user_message,
        "methods": entries,
    }

    filepath.write_text(json.dumps(log_data, ensure_ascii=False, indent=2))
    return str(filepath)


def list_log_files() -> list[Path]:
    """Return log files sorted by name (oldest first)."""
    if not LOG_DIR.exists():
        return []
    return sorted(LOG_DIR.glob("turn_*.json"))


def load_log_file(filepath: Path) -> dict | None:
    """Load and parse a single log JSON file."""
    try:
        return json.loads(filepath.read_text())
    except Exception:
        return None
