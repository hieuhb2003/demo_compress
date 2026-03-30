from __future__ import annotations

import pandas as pd
import plotly.express as px

from src.models import AppState


def metrics_dataframe(app_state: AppState) -> pd.DataFrame:
    records = []
    for state in app_state.method_states.values():
        cum_input = 0
        cum_output = 0
        cum_total = 0
        for metric in state.metrics_history:
            cum_input += metric.actual_input_tokens
            cum_output += metric.actual_output_tokens
            cum_total += metric.total_tokens
            records.append(
                {
                    "turn": metric.turn_index,
                    "method": state.label,
                    "estimated_input_tokens": metric.estimated_input_tokens,
                    "actual_input_tokens": metric.actual_input_tokens,
                    "actual_output_tokens": metric.actual_output_tokens,
                    "total_tokens": metric.total_tokens,
                    "cumulative_input_tokens": cum_input,
                    "cumulative_output_tokens": cum_output,
                    "cumulative_total_tokens": cum_total,
                    "latency_seconds": metric.latency_seconds,
                    "compression_ratio": metric.compression_ratio,
                    "prep_time": metric.prep_time,
                    "thread_name": metric.thread_name,
                }
            )
    return pd.DataFrame(records)


def line_chart(df: pd.DataFrame, value_column: str, title: str):
    if df.empty:
        return None
    return px.line(
        df,
        x="turn",
        y=value_column,
        color="method",
        markers=True,
        title=title,
    )
