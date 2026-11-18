# GroundedAI

Bare-minimum starter for **grounded question answering over course PDFs**.

The system ingests a fixed PDF corpus, retrieves relevant passages with a TF-IDF baseline, and either returns evidence-backed text or **refuses** when support is weak.

This is a self-defined ML problem (not a Kaggle competition).

## Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev]"

# Optional: regenerate the sample syllabus PDF
python scripts/build_sample_pdf.py

# Pipeline
python -m grounded_ai.cli ingest
python -m grounded_ai.cli index
python -m grounded_ai.cli ask "What percent is the final project worth?"
python -m grounded_ai.cli eval
pytest
```

## Project layout

```
GroundedAI/
  grounded_ai/          # ingest → index → ask/evaluate
  data/pdfs/            # course PDF corpus
  data/questions.json   # labeled eval questions
  scripts/              # helpers (sample PDF builder)
  tests/                # unit tests
  artifacts/            # generated locally (gitignored)
```

## CLI

```bash
python -m grounded_ai.cli ingest
python -m grounded_ai.cli index
python -m grounded_ai.cli ask "When are office hours?"
python -m grounded_ai.cli eval
```

## Current scope

- PDF text extraction + overlapping chunks
- TF-IDF retrieval baseline
- Quote-the-top-passage answering with a score threshold refusal
- Offline metrics: retrieval hit rate on answerable questions, refusal behavior on unanswerable ones

## Next ideas

- Dense embeddings instead of / in addition to TF-IDF
- LLM answering constrained to retrieved context
- Stronger faithfulness checks and more course PDFs
