from app.config import Settings
from app.models.schemas import CVExtraction
from app.services.cv_repository import CVRepository
from app.services.keyword_index import KeywordIndex, build_documents, tokenize


def test_tokenize_lowercases_and_splits_on_non_alphanumeric():
    assert tokenize("Python, C++ Developer!") == ["python", "c", "developer"]


def test_keyword_index_search_ranks_exact_term_match_highest():
    documents = {
        "cv1": "Experienced Python developer with Django background",
        "cv2": "Senior accountant with tax expertise",
    }
    index = KeywordIndex(documents)
    results = index.search("Python", top_k=2)
    assert results[0][0] == "cv1"


def test_keyword_index_search_returns_empty_list_when_no_documents_have_text():
    index = KeywordIndex({"cv1": "", "cv2": "   "})
    assert index.search("Python", top_k=5) == []


def test_build_documents_loads_raw_text_for_every_cv(tmp_path):
    settings = Settings(storage_dir=tmp_path / "storage")
    repository = CVRepository(settings)
    repository.save_extraction("cv1", CVExtraction(name="Alice"))
    repository.save_raw_text("cv1", "Python developer")

    documents = build_documents(repository)

    assert documents == {"cv1": "Python developer"}
