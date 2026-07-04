from app.config import Settings
from app.services.vector_store import VectorStore


def _make_store(tmp_path, name="test_cvs"):
    settings = Settings(storage_dir=tmp_path / "storage")
    return VectorStore(settings, collection_name=name)


def test_upsert_cv_chunks_and_query_returns_closest_cv(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_cv_chunks(
        "cv1",
        ["python developer", "built django apps"],
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        {"name": "Alice"},
    )
    store.upsert_cv_chunks(
        "cv2",
        ["accountant with tax expertise"],
        [[0.0, 1.0, 0.0]],
        {"name": "Bob"},
    )
    results = store.query([1.0, 0.0, 0.0], top_k=1)
    assert results[0]["id"] == "cv1"
    assert results[0]["metadata"]["name"] == "Alice"


def test_query_returns_one_result_per_cv_even_with_multiple_matching_chunks(tmp_path):
    store = _make_store(tmp_path, name="dedupe_test")
    store.upsert_cv_chunks(
        "cv1",
        ["python developer", "python django expert", "python ml engineer"],
        [[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.9, 0.1, 0.0]],
        {"name": "Alice"},
    )
    results = store.query([1.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "cv1"


def test_upsert_cv_chunks_replaces_previous_chunks_for_same_cv_id(tmp_path):
    store = _make_store(tmp_path, name="replace_test")
    store.upsert_cv_chunks("cv1", ["old chunk one", "old chunk two"], [[1.0, 0.0], [1.0, 0.0]], {"name": "Alice"})
    store.upsert_cv_chunks("cv1", ["new chunk"], [[0.0, 1.0]], {"name": "Alice Updated"})

    all_chunks = store._collection.get(where={"cv_id": "cv1"})
    assert len(all_chunks["ids"]) == 1
    assert all_chunks["documents"] == ["new chunk"]

    results = store.query([0.0, 1.0], top_k=5)
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "Alice Updated"


def test_query_returns_top_k_distinct_cvs_even_when_one_cv_dominates_chunks(tmp_path):
    store = _make_store(tmp_path, name="dominant_cv_test")
    # cv1 has many chunks, all closest to the query embedding, that could
    # otherwise crowd out other CVs if the search only considered a few closest matches.
    many_chunks = [f"python chunk {i}" for i in range(30)]
    many_embeddings = [[1.0, 0.0, 0.0] for _ in range(30)]
    store.upsert_cv_chunks("cv1", many_chunks, many_embeddings, {"name": "Alice"})
    # cv2..cv6 each have one chunk, less close but still relevant, and must
    # still be returned when top_k=5 asks for 5 distinct CVs.
    for i in range(2, 7):
        store.upsert_cv_chunks(f"cv{i}", [f"candidate {i}"], [[0.5, 0.5, 0.0]], {"name": f"Person{i}"})

    results = store.query([1.0, 0.0, 0.0], top_k=5)

    assert len(results) == 5
    returned_ids = {r["id"] for r in results}
    assert "cv1" in returned_ids
    assert len(returned_ids) == 5


def test_query_returns_fewer_than_top_k_when_collection_has_fewer_distinct_cvs(tmp_path):
    store = _make_store(tmp_path, name="fewer_cvs_test")
    store.upsert_cv_chunks("cv1", ["a", "b", "c"], [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], {"name": "Alice"})
    results = store.query([1.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "cv1"


def test_upsert_cv_chunks_raises_on_length_mismatch(tmp_path):
    store = _make_store(tmp_path, name="mismatch_test")
    try:
        store.upsert_cv_chunks("cv1", ["a", "b"], [[1.0, 0.0]], {"name": "Alice"})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_query_returns_deterministic_exact_distances_across_repeated_calls(tmp_path):
    store = _make_store(tmp_path, name="exact_search_test")
    store.upsert_cv_chunks("cv1", ["chunk"], [[1.0, 0.0, 0.0]], {"name": "Alice"})
    store.upsert_cv_chunks("cv2", ["chunk"], [[0.9, 0.1, 0.0]], {"name": "Bob"})
    store.upsert_cv_chunks("cv3", ["chunk"], [[0.0, 1.0, 0.0]], {"name": "Carol"})

    first = store.query([1.0, 0.0, 0.0], top_k=3)
    for _ in range(20):
        repeat = store.query([1.0, 0.0, 0.0], top_k=3)
        assert [r["id"] for r in repeat] == [r["id"] for r in first]
        assert [r["distance"] for r in repeat] == [r["distance"] for r in first]


def test_query_with_metadata_filter_returns_only_matching_predicate(tmp_path):
    store = _make_store(tmp_path, name="filter_test")
    store.upsert_cv_chunks("cv1", ["chunk"], [[1.0, 0.0]], {"name": "Alice", "years_experience": 2.0})
    store.upsert_cv_chunks("cv2", ["chunk"], [[1.0, 0.0]], {"name": "Bob", "years_experience": 7.0})
    store.upsert_cv_chunks("cv3", ["chunk"], [[1.0, 0.0]], {"name": "Carol", "years_experience": 10.0})

    results = store.query([1.0, 0.0], top_k=5, where={"years_experience": {"$gte": 5}})

    returned_ids = {r["id"] for r in results}
    assert returned_ids == {"cv2", "cv3"}


def test_update_cv_metadata_merges_into_existing_metadata(tmp_path):
    store = _make_store(tmp_path, name="update_meta_test")
    store.upsert_cv_chunks("cv1", ["a", "b"], [[1.0, 0.0], [1.0, 0.0]], {"name": "Alice"})

    store.update_cv_metadata("cv1", {"years_experience": 6.0})

    existing = store._collection.get(where={"cv_id": "cv1"})
    assert len(existing["metadatas"]) == 2
    for metadata in existing["metadatas"]:
        assert metadata["years_experience"] == 6.0
        assert metadata["name"] == "Alice"


def test_update_cv_metadata_is_a_noop_for_unknown_cv_id(tmp_path):
    store = _make_store(tmp_path, name="update_meta_noop_test")
    store.update_cv_metadata("does-not-exist", {"years_experience": 1.0})  # must not raise


def test_query_with_filter_supports_years_experience_gte_predicate(tmp_path):
    store = _make_store(tmp_path, name="years_experience_filter_test")
    candidates = {
        "junior1": 1.5,
        "junior2": 3.0,
        "senior1": 5.0,
        "senior2": 8.5,
        "senior3": 12.0,
    }
    for cv_id, years in candidates.items():
        store.upsert_cv_chunks(cv_id, ["resume text"], [[1.0, 0.0]], {"years_experience": years})

    results = store.query([1.0, 0.0], top_k=10, where={"years_experience": {"$gte": 5}})

    returned_ids = {r["id"] for r in results}
    assert returned_ids == {"senior1", "senior2", "senior3"}
