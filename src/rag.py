from __future__ import annotations

import hashlib
import io
from typing import List

from pypdf import PdfReader

from src.models import DocumentChunk


def extract_text_from_upload(file_name: str, raw_bytes: bytes) -> str:
    lowered = file_name.lower()
    if lowered.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return raw_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, source_name: str, chunk_size: int = 800, overlap: int = 150) -> List[DocumentChunk]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: List[DocumentChunk] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk_text_value = normalized[start:end]
        digest = hashlib.md5(f"{source_name}:{start}:{chunk_text_value}".encode("utf-8")).hexdigest()[:10]
        chunks.append(
            DocumentChunk(
                chunk_id=f"{source_name}-{digest}",
                source_name=source_name,
                text=chunk_text_value,
            )
        )
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return chunks
