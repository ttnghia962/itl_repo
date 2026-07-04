from app.config import Settings
from app.models.schemas import CVExtraction
from app.services import chunking, embeddings, llm_extraction, pdf_extractor, pipeline
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore


def test_process_cv_file_stores_original_extraction_and_chunk_embeddings(tmp_path, monkeypatch):
    settings = Settings(storage_dir=tmp_path / "storage")
    repository = CVRepository(settings)
    vector_store = VectorStore(settings, collection_name="pipeline_test")

    monkeypatch.setattr(pdf_extractor, "extract_text", lambda path: "Python engineer with ML experience")
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
    monkeypatch.setattr(
        chunking, "chunk_text", lambda text, max_tokens, overlap_tokens: ["chunk one", "chunk two"]
    )
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [[1.0, 0.0, 0.0] for _ in texts])

    cv_id, extraction = pipeline.process_cv_file("alice.pdf", b"%PDF-1.4 fake", repository, vector_store)

    assert cv_id == "alice"
    assert extraction.name == "Alice"
    assert repository.get_pdf_path(cv_id).exists()
    assert repository.load_extraction(cv_id).name == "Alice"

    all_chunks = vector_store._collection.get(where={"cv_id": "alice"})
    assert len(all_chunks["ids"]) == 3  # 1 summary chunk + 2 body chunks
    assert all_chunks["metadatas"][0]["years_experience"] == 0.0  # no experience entries in this fixture

    results = vector_store.query([1.0, 0.0, 0.0], top_k=1)
    assert results[0]["id"] == "alice"
    assert results[0]["metadata"]["name"] == "Alice"


def test_build_summary_text_joins_non_empty_extraction_fields():
    extraction = CVExtraction(name="Alice", current_role="ML Engineer", domain="IT", skills=["Python"])
    summary = pipeline.build_summary_text(extraction)
    assert summary == "Alice\nML Engineer\nIT\nPython"


def test_build_summary_text_skips_empty_fields():
    extraction = CVExtraction()
    assert pipeline.build_summary_text(extraction) == ""


def test_build_chunk_texts_prepends_summary_to_body_chunks(monkeypatch):
    monkeypatch.setattr(
        chunking, "chunk_text", lambda text, max_tokens, overlap_tokens: ["body chunk 1", "body chunk 2"]
    )
    extraction = CVExtraction(name="Alice")
    chunks = pipeline.build_chunk_texts(extraction, "raw resume text")
    assert chunks == ["Alice", "body chunk 1", "body chunk 2"]
