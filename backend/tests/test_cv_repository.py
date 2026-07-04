from app.config import Settings
from app.models.schemas import CVExtraction
from app.services.cv_repository import CVRepository


def _make_repo(tmp_path):
    settings = Settings(storage_dir=tmp_path / "storage")
    return CVRepository(settings)


def test_save_and_load_original(tmp_path):
    repo = _make_repo(tmp_path)
    cv_id = repo.save_original("10674770.pdf", b"%PDF-1.4 fake content")
    assert cv_id == "10674770"
    assert repo.get_pdf_path(cv_id).read_bytes() == b"%PDF-1.4 fake content"


def test_save_original_deduplicates_ids(tmp_path):
    repo = _make_repo(tmp_path)
    first = repo.save_original("cv.pdf", b"one")
    second = repo.save_original("cv.pdf", b"two")
    assert first == "cv"
    assert second == "cv-1"


def test_save_and_load_extraction_roundtrip(tmp_path):
    repo = _make_repo(tmp_path)
    extraction = CVExtraction(name="Jane", skills=["Python"], domain="Information Technology")
    repo.save_extraction("cv1", extraction)
    loaded = repo.load_extraction("cv1")
    assert loaded.name == "Jane"
    assert loaded.skills == ["Python"]


def test_list_cv_ids_returns_sorted_ids(tmp_path):
    repo = _make_repo(tmp_path)
    repo.save_extraction("b", CVExtraction(name="B"))
    repo.save_extraction("a", CVExtraction(name="A"))
    assert repo.list_cv_ids() == ["a", "b"]


def test_save_and_load_raw_text(tmp_path):
    repo = _make_repo(tmp_path)
    repo.save_raw_text("cv1", "hello world")
    assert repo.load_raw_text("cv1") == "hello world"


def test_load_extraction_missing_raises_file_not_found(tmp_path):
    repo = _make_repo(tmp_path)
    try:
        repo.load_extraction("does-not-exist")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
