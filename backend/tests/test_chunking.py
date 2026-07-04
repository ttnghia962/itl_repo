import pytest

from app.services.chunking import chunk_text


def test_chunk_text_returns_single_chunk_when_under_limit():
    assert chunk_text("hello world", max_tokens=200, overlap_tokens=20) == ["hello world"]


def test_chunk_text_returns_empty_list_for_empty_string():
    assert chunk_text("", max_tokens=200, overlap_tokens=20) == []


def test_chunk_text_splits_into_overlapping_chunks():
    text = " ".join(f"w{i}" for i in range(1, 10))  # "w1 w2 ... w9"
    chunks = chunk_text(text, max_tokens=4, overlap_tokens=1)
    assert chunks == ["w1 w2 w3 w4", "w4 w5 w6 w7", "w7 w8 w9"]


def test_chunk_text_raises_when_overlap_not_smaller_than_max_tokens():
    text = " ".join(f"w{i}" for i in range(1, 11))  # 10 words
    with pytest.raises(ValueError):
        chunk_text(text, max_tokens=4, overlap_tokens=4)
