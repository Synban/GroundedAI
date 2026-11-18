"""Offline evaluation: retrieval hit rate + refusal behavior."""

from __future__ import annotations

import json
from pathlib import Path

from grounded_ai.answer import answer_question
from grounded_ai.retrieve import load_index, retrieve

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "questions.json"
REPORT_PATH = ROOT / "artifacts" / "eval_report.json"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def passage_hit(evidence_texts: list[str], must_include: list[str]) -> bool:
    blob = " ".join(evidence_texts).lower()
    return all(token.lower() in blob for token in must_include)


def evaluate(
    questions_path: Path = QUESTIONS_PATH,
    *,
    top_k: int = 3,
    min_score: float = 0.05,
) -> dict:
    index = load_index()
    questions = load_questions(questions_path)

    retrieval_hits = 0
    answerable = 0
    unanswerable = 0
    correct_refusals = 0
    incorrect_refusals = 0
    false_answers = 0
    details: list[dict] = []

    for item in questions:
        q = item["question"]
        label = item["label"]  # "answerable" | "unanswerable"
        must_include = item.get("must_include", [])

        hits = retrieve(q, index, top_k=top_k)
        evidence_texts = [h.chunk.text for h in hits]
        hit = passage_hit(evidence_texts, must_include) if must_include else bool(hits)

        result = answer_question(q, index, top_k=top_k, min_score=min_score)

        if label == "answerable":
            answerable += 1
            if hit:
                retrieval_hits += 1
            if result.refused:
                incorrect_refusals += 1
        else:
            unanswerable += 1
            if result.refused:
                correct_refusals += 1
            else:
                false_answers += 1

        details.append(
            {
                "id": item.get("id"),
                "question": q,
                "label": label,
                "retrieval_hit": hit,
                "refused": result.refused,
                "best_score": result.best_score,
            }
        )

    report = {
        "num_questions": len(questions),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "retrieval_hit_rate_answerable": (
            retrieval_hits / answerable if answerable else None
        ),
        "refusal_recall_unanswerable": (
            correct_refusals / unanswerable if unanswerable else None
        ),
        "incorrect_refusal_rate_answerable": (
            incorrect_refusals / answerable if answerable else None
        ),
        "false_answer_rate_unanswerable": (
            false_answers / unanswerable if unanswerable else None
        ),
        "details": details,
    }
    return report


def main() -> None:
    report = evaluate()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
