"""TF-IDF retrieval baseline over ingested chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from grounded_ai.chunking import Chunk
from grounded_ai.ingest import CHUNKS_PATH, DEFAULT_ARTIFACTS, load_chunks

INDEX_PATH = DEFAULT_ARTIFACTS / "tfidf_index.joblib"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {"score": self.score, **self.chunk.to_dict()}


@dataclass
class TfidfIndex:
    vectorizer: TfidfVectorizer
    matrix: object
    chunks: list[Chunk]


def build_index(chunks: list[Chunk] | None = None) -> TfidfIndex:
    if chunks is None:
        chunks = load_chunks(CHUNKS_PATH)
    if not chunks:
        raise ValueError("No chunks available to index")

    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return TfidfIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)


def save_index(index: TfidfIndex, path: Path = INDEX_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": index.vectorizer,
            "matrix": index.matrix,
            "chunks": [c.to_dict() for c in index.chunks],
        },
        path,
    )
    return path


def load_index(path: Path = INDEX_PATH) -> TfidfIndex:
    payload = joblib.load(path)
    chunks = [Chunk(**row) for row in payload["chunks"]]
    return TfidfIndex(
        vectorizer=payload["vectorizer"],
        matrix=payload["matrix"],
        chunks=chunks,
    )


def retrieve(
    question: str,
    index: TfidfIndex,
    *,
    top_k: int = 3,
) -> list[RetrievedChunk]:
    query_vec = index.vectorizer.transform([question])
    scores = cosine_similarity(query_vec, index.matrix).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    return [
        RetrievedChunk(chunk=index.chunks[i], score=float(scores[i]))
        for i in ranked
        if scores[i] > 0
    ]


def main() -> None:
    chunks = load_chunks()
    index = build_index(chunks)
    out = save_index(index)
    # Also keep a human-readable snapshot of chunk ids for debugging.
    meta = DEFAULT_ARTIFACTS / "index_meta.json"
    meta.write_text(
        json.dumps({"num_chunks": len(chunks), "path": str(out)}, indent=2),
        encoding="utf-8",
    )
    print(f"Built TF-IDF index over {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()
