"""Load PDFs, chunk pages, and write a local chunk index."""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader

from grounded_ai.chunking import Chunk, chunk_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_DIR = ROOT / "data" / "pdfs"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
CHUNKS_PATH = DEFAULT_ARTIFACTS / "chunks.json"


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def ingest_pdfs(
    pdf_dir: Path = DEFAULT_PDF_DIR,
    *,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    chunks: list[Chunk] = []
    for pdf_path in pdf_paths:
        for page_num, text in extract_pages(pdf_path):
            chunks.extend(
                chunk_text(
                    text,
                    source=pdf_path.name,
                    page=page_num,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
    return chunks


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2),
        encoding="utf-8",
    )
    return path


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**row) for row in data]


def main() -> None:
    chunks = ingest_pdfs()
    out = save_chunks(chunks)
    print(f"Ingested {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()
