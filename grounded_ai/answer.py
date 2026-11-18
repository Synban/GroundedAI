"""Grounded answering: return evidence-backed text or refuse."""

from __future__ import annotations

from dataclasses import dataclass

from grounded_ai.retrieve import RetrievedChunk, TfidfIndex, load_index, retrieve

REFUSAL = (
    "I cannot find enough support in the course PDFs to answer that. "
    "Try rephrasing or asking about syllabus policies, deadlines, or grading."
)


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    refused: bool
    evidence: list[RetrievedChunk]
    best_score: float


def answer_question(
    question: str,
    index: TfidfIndex | None = None,
    *,
    top_k: int = 3,
    min_score: float = 0.05,
) -> AnswerResult:
    if index is None:
        index = load_index()

    hits = retrieve(question, index, top_k=top_k)
    best = hits[0].score if hits else 0.0

    if not hits or best < min_score:
        return AnswerResult(
            question=question,
            answer=REFUSAL,
            refused=True,
            evidence=hits,
            best_score=best,
        )

    # Bare-minimum "answer": quote the strongest retrieved passage.
    # Later this can become LLM generation constrained to evidence.
    top = hits[0].chunk
    answer = (
        f"Based on {top.source} (page {top.page}):\n"
        f"{top.text}"
    )
    return AnswerResult(
        question=question,
        answer=answer,
        refused=False,
        evidence=hits,
        best_score=best,
    )


def format_result(result: AnswerResult) -> str:
    lines = [
        f"Q: {result.question}",
        f"Refused: {result.refused}",
        f"Best score: {result.best_score:.4f}",
        "",
        result.answer,
        "",
        "Evidence:",
    ]
    if not result.evidence:
        lines.append("  (none)")
    for i, hit in enumerate(result.evidence, start=1):
        lines.append(
            f"  [{i}] score={hit.score:.4f} "
            f"{hit.chunk.source} p{hit.chunk.page} :: {hit.chunk.chunk_id}"
        )
    return "\n".join(lines)
