from grounded_ai.answer import REFUSAL, answer_question
from grounded_ai.chunking import Chunk
from grounded_ai.retrieve import TfidfIndex, build_index


def _tiny_index() -> TfidfIndex:
    chunks = [
        Chunk(
            chunk_id="s::p1::0",
            text="The final project is worth 35 percent of the grade.",
            source="sample.pdf",
            page=1,
            start_char=0,
            end_char=50,
        ),
        Chunk(
            chunk_id="s::p1::1",
            text="Office hours are Tuesdays at 2:00 PM.",
            source="sample.pdf",
            page=1,
            start_char=51,
            end_char=90,
        ),
    ]
    return build_index(chunks)


def test_answers_when_evidence_exists():
    result = answer_question(
        "How much is the final project worth?",
        _tiny_index(),
        min_score=0.01,
    )
    assert result.refused is False
    assert "35 percent" in result.answer


def test_refuses_unrelated_question():
    result = answer_question(
        "What is the cafeteria soup of the day?",
        _tiny_index(),
        min_score=0.2,
    )
    assert result.refused is True
    assert result.answer == REFUSAL
