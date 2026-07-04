import numpy as np

from app.services import embeddings


class FakeModel:
    def encode(self, texts, convert_to_numpy=True):
        return np.array([[float(len(t)), 0.0, 1.0] for t in texts])


def test_embed_texts_returns_list_of_vectors(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_model", lambda: FakeModel())
    result = embeddings.embed_texts(["hello", "hi"])
    assert result == [[5.0, 0.0, 1.0], [2.0, 0.0, 1.0]]


def test_embed_texts_handles_empty_list(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_model", lambda: FakeModel())
    assert embeddings.embed_texts([]) == []
