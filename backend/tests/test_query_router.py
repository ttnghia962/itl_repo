import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_repository, get_vector_store
from app.models.schemas import CVExtraction
from app.routers.query import router as query_router
from app.services import embeddings, llm_rag
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore


def _make_client(tmp_path, monkeypatch):
    settings = Settings(storage_dir=tmp_path / "storage")
    repository = CVRepository(settings)
    vector_store = VectorStore(settings, collection_name="query_router_test")

    repository.save_extraction(
        "cv1",
        CVExtraction(
            name="Alice",
            skills=["Python", "Machine Learning"],
            current_role="ML Engineer",
            domain="Information Technology",
        ),
    )
    vector_store.upsert_cv_chunks("cv1", ["python ml"], [[1.0, 0.0]], {"filename": "alice.pdf", "name": "Alice"})
    repository.save_raw_text("cv1", "Python developer with machine learning experience")

    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(
        llm_rag,
        "generate_answer",
        lambda question, contexts, client=None: {
            "answer": "Alice is the best fit.",
            "best_candidate_id": "cv1",
            "best_candidate_reason": "Alice has the strongest Python and ML background.",
        },
    )

    app = FastAPI()
    app.include_router(query_router)
    app.dependency_overrides[get_repository] = lambda: repository
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    return TestClient(app)


def test_query_returns_answer_and_retrieved_cvs(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    response = client.post("/api/query", json={"question": "Who knows Python and Machine Learning?", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["best_candidate_id"] == "cv1"
    assert "Alice" in body["answer"]
    assert body["retrieved"][0]["id"] == "cv1"
    assert body["retrieved"][0]["extraction"]["name"] == "Alice"


def test_query_score_is_a_reciprocal_rank_fusion_value(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    response = client.post("/api/query", json={"question": "Python", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    # RRF score for a single-CV corpus matched by both retrievers: 1/(60+1) + 1/(60+1)
    assert body["retrieved"][0]["score"] == pytest.approx(2 / 61)


def test_query_returns_best_candidate_reason(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    response = client.post("/api/query", json={"question": "Who knows Python and Machine Learning?", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["best_candidate_reason"] == "Alice has the strongest Python and ML background."
