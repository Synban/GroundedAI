from grounded_ai.chunking import chunk_text


def test_chunk_text_overlap_and_metadata():
    text = "alpha " * 200
    chunks = chunk_text(text, source="demo.pdf", page=1, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[0].source == "demo.pdf"
    assert chunks[0].page == 1
    assert chunks[0].chunk_id.startswith("demo.pdf::p1::")


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ", source="x.pdf", page=1) == []
