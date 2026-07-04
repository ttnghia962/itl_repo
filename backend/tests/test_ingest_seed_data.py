from pathlib import Path

from app.config import Settings
from app.ingestion import ingest_seed_data
from app.models.schemas import CVExtraction
from app.services import pipeline
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore


def test_ingest_directory_processes_every_pdf(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "10674770.pdf").write_bytes(b"%PDF-1.4 fake1")
    (source_dir / "11163645.pdf").write_bytes(b"%PDF-1.4 fake2")
    (source_dir / "notes.txt").write_text("ignore me")

    settings = Settings(storage_dir=tmp_path / "storage")
    repository = CVRepository(settings)
    vector_store = VectorStore(settings, collection_name="ingest_test")

    processed = []

    def fake_process(filename, content, repo, store):
        cv_id = Path(filename).stem
        processed.append(cv_id)
        return cv_id, CVExtraction(name=cv_id)

    monkeypatch.setattr(pipeline, "process_cv_file", fake_process)

    ids = ingest_seed_data.ingest_directory(source_dir, repository, vector_store)

    assert ids == ["10674770", "11163645"]
    assert processed == ["10674770", "11163645"]


def test_ingest_directory_skips_already_ingested_files(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "10674770.pdf").write_bytes(b"%PDF-1.4 fake1")
    (source_dir / "11163645.pdf").write_bytes(b"%PDF-1.4 fake2")

    settings = Settings(storage_dir=tmp_path / "storage")
    repository = CVRepository(settings)
    vector_store = VectorStore(settings, collection_name="ingest_skip_test")
    repository.save_extraction("10674770", CVExtraction(name="Already Ingested"))

    processed = []

    def fake_process(filename, content, repo, store):
        cv_id = Path(filename).stem
        processed.append(cv_id)
        return cv_id, CVExtraction(name=cv_id)

    monkeypatch.setattr(pipeline, "process_cv_file", fake_process)

    ids = ingest_seed_data.ingest_directory(source_dir, repository, vector_store)

    assert ids == ["11163645"]
    assert processed == ["11163645"]
