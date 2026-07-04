from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_repository, get_vector_store
from app.main import app
from app.models.schemas import CVExtraction
from app.services import embeddings, llm_extraction, llm_rag, pdf_extractor
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore


def test_health_check():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_then_query_end_to_end(tmp_path, monkeypatch):
    settings = Settings(storage_dir=tmp_path / "storage")
    repository = CVRepository(settings)
    vector_store = VectorStore(settings, collection_name="main_test")
    app.dependency_overrides[get_repository] = lambda: repository
    app.dependency_overrides[get_vector_store] = lambda: vector_store

    monkeypatch.setattr(pdf_extractor, "extract_text", lambda path: "Python engineer, 6 years Machine Learning experience")
    monkeypatch.setattr(
        llm_extraction,
        "extract_cv_fields",
        lambda text: CVExtraction(
            name="Alice",
            skills=["Python", "Machine Learning"],
            current_role="ML Engineer",
            domain="Information Technology",
        ),
    )
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(
        llm_rag,
        "generate_answer",
        lambda question, contexts, client=None: {
            "answer": "Alice is the best fit for Python and ML.",
            "best_candidate_id": contexts[0]["candidate_id"] if contexts else None,
            "best_candidate_reason": "Alice has strong Python and ML experience.",
        },
    )

    client = TestClient(app)
    try:
        upload_response = client.post(
            "/api/cvs/upload", files={"file": ("alice.pdf", b"%PDF-1.4 fake", "application/pdf")}
        )
        assert upload_response.status_code == 200

        query_response = client.post(
            "/api/query", json={"question": "Who knows Python and Machine Learning?", "top_k": 3}
        )
        assert query_response.status_code == 200
        body = query_response.json()
        assert body["best_candidate_id"] == "alice"
        assert "Alice" in body["answer"]
    finally:
        app.dependency_overrides.clear()
