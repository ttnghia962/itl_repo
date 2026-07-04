from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_repository, get_vector_store
from app.models.schemas import CVExtraction
from app.routers.cvs import router as cvs_router
from app.services import embeddings, llm_extraction, pdf_extractor
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore


def _make_client(tmp_path, monkeypatch):
    settings = Settings(storage_dir=tmp_path / "storage")
    repository = CVRepository(settings)
    vector_store = VectorStore(settings, collection_name="router_test")

    monkeypatch.setattr(pdf_extractor, "extract_text", lambda path: "Python developer")
    monkeypatch.setattr(
        llm_extraction,
        "extract_cv_fields",
        lambda text: CVExtraction(name="Alice", skills=["Python"], current_role="Developer", domain="IT"),
    )
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])

    app = FastAPI()
    app.include_router(cvs_router)
    app.dependency_overrides[get_repository] = lambda: repository
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    return TestClient(app)


def test_upload_cv_returns_extraction(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    response = client.post(
        "/api/cvs/upload",
        files={"file": ("alice.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "alice"
    assert body["extraction"]["name"] == "Alice"


def test_list_and_get_cv(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    client.post("/api/cvs/upload", files={"file": ("alice.pdf", b"%PDF-1.4 fake", "application/pdf")})

    listing = client.get("/api/cvs")
    assert listing.status_code == 200
    assert listing.json() == [{"id": "alice", "name": "Alice", "current_role": "Developer"}]

    detail = client.get("/api/cvs/alice")
    assert detail.status_code == 200
    assert detail.json()["extraction"]["name"] == "Alice"


def test_get_missing_cv_returns_404(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    response = client.get("/api/cvs/does-not-exist")
    assert response.status_code == 404


def test_get_cv_file_returns_pdf_bytes(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    client.post("/api/cvs/upload", files={"file": ("alice.pdf", b"%PDF-1.4 fake", "application/pdf")})
    response = client.get("/api/cvs/alice/file")
    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 fake"
