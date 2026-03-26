from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ChatTurn:
    turn_index: int
    user_message: str
    assistant_message: str
    created_at: float


@dataclass
class SummaryRecord:
    block_index: int
    start_turn: int
    end_turn: int
    text: str
    embedding: Optional[List[float]] = None


@dataclass
class DocumentChunk:
    chunk_id: str
    source_name: str
    text: str
    embedding: Optional[List[float]] = None


@dataclass
class PromptArtifacts:
    system_prompt: str
    context_text: str
    user_message: str
    rag_chunks: List[DocumentChunk] = field(default_factory=list)
    retrieved_summaries: List[SummaryRecord] = field(default_factory=list)
    raw_prompt_preview: str = ""
    compressed_context: Optional[str] = None
    estimated_input_tokens: int = 0
    compressed_input_tokens: Optional[int] = None


@dataclass
class MethodMetrics:
    turn_index: int
    method_key: str
    estimated_input_tokens: int
    actual_input_tokens: int
    actual_output_tokens: int
    total_tokens: int
    latency_seconds: float
    compression_ratio: float


@dataclass
class MethodResult:
    method_key: str
    label: str
    assistant_message: str
    prompt_artifacts: PromptArtifacts
    metrics: MethodMetrics


@dataclass
class ConversationState:
    method_key: str
    label: str
    turns: List[ChatTurn] = field(default_factory=list)
    summaries: List[SummaryRecord] = field(default_factory=list)
    pending_summary_blocks: Dict[int, object] = field(default_factory=dict)
    metrics_history: List[MethodMetrics] = field(default_factory=list)


@dataclass
class AppState:
    method_states: Dict[str, ConversationState]
    rag_chunks: List[DocumentChunk] = field(default_factory=list)
