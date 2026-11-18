"""Split document text into overlapping passages with source metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int
    start_char: int
    end_char: int

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_text(
    text: str,
    *,
    source: str,
    page: int,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """Create overlapping character windows from a single page/section."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}::p{page}::{index}",
                    text=piece,
                    source=source,
                    page=page,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1
        if end == len(cleaned):
            break
        start = end - overlap
    return chunks
