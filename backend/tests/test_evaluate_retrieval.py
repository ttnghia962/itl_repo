import math

from app.evaluation.evaluate_retrieval import (
    dcg_at_k,
    evaluate,
    ideal_dcg_at_k,
    ndcg_at_k,
    reciprocal_rank,
    success_at_k,
)


def test_reciprocal_rank_finds_first_relevant_position():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_dcg_at_k_sums_only_relevant_positions():
    # "a" relevant at position 0, "x" irrelevant at position 1
    assert dcg_at_k(["a", "x"], {"a"}, 2) == 1.0 / math.log2(2)
    # "a" relevant at position 1
    assert dcg_at_k(["x", "a"], {"a"}, 2) == 1.0 / math.log2(3)


def test_ideal_dcg_at_k_caps_at_min_of_relevant_count_and_k():
    assert ideal_dcg_at_k({"a", "b", "c"}, 2) == 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ideal_dcg_at_k({"a"}, 5) == 1.0 / math.log2(2)
    assert ideal_dcg_at_k(set(), 5) == 0.0


def test_ndcg_at_k_is_one_for_perfect_order():
    assert ndcg_at_k(["a", "x"], {"a"}, 2) == 1.0


def test_ndcg_at_k_penalizes_late_relevant_result():
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == (1.0 / math.log2(3)) / (1.0 / math.log2(2))


def test_ndcg_at_k_is_zero_when_no_relevant_ids():
    assert ndcg_at_k(["a", "b"], set(), 2) == 0.0


def test_success_at_k_true_if_any_relevant_in_top_k():
    assert success_at_k(["x", "y", "a"], {"a"}, 3) == 1.0
    assert success_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_evaluate_returns_macro_averages_and_per_query_breakdown():
    def fake_retrieve(query, k):
        return {"q1": ["a", "b"], "q2": ["x", "y"]}[query][:k]

    cases = [
        {"query": "q1", "relevant_ids": ["a"]},
        {"query": "q2", "relevant_ids": ["z"]},
    ]
    result = evaluate(cases, fake_retrieve, k=2)

    assert result["k"] == 2
    assert result["mrr"] == (1.0 + 0.0) / 2
    assert result["success_at_3"] == (1.0 + 0.0) / 2
    assert len(result["per_query"]) == 2
    assert result["per_query"][0]["retrieved_ids"] == ["a", "b"]
    assert result["per_query"][0]["num_relevant"] == 1
    assert result["per_query"][1]["mrr"] == 0.0
