import math
from typing import Callable, Dict, List, Set


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for i, cv_id in enumerate(retrieved_ids):
        if cv_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    total = 0.0
    for i, cv_id in enumerate(retrieved_ids[:k]):
        if cv_id in relevant_ids:
            total += 1.0 / math.log2(i + 2)
    return total


def ideal_dcg_at_k(relevant_ids: Set[str], k: int) -> float:
    n = min(len(relevant_ids), k)
    return sum(1.0 / math.log2(i + 2) for i in range(n))


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    idcg = ideal_dcg_at_k(relevant_ids, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved_ids, relevant_ids, k) / idcg


def success_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    return 1.0 if any(cv_id in relevant_ids for cv_id in retrieved_ids[:k]) else 0.0


def evaluate(cases: List[Dict], retrieve_fn: Callable[[str, int], List[str]], k: int = 5) -> Dict:
    per_query = []
    for case in cases:
        retrieved_ids = retrieve_fn(case["query"], k)
        relevant_ids = set(case["relevant_ids"])
        per_query.append(
            {
                "query": case["query"],
                "retrieved_ids": retrieved_ids,
                "num_relevant": len(relevant_ids),
                "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
                "ndcg_at_5": ndcg_at_k(retrieved_ids, relevant_ids, 5),
                "success_at_3": success_at_k(retrieved_ids, relevant_ids, 3),
                "success_at_5": success_at_k(retrieved_ids, relevant_ids, 5),
            }
        )
    n = len(per_query) or 1
    return {
        "k": k,
        "mrr": sum(q["mrr"] for q in per_query) / n,
        "ndcg_at_5": sum(q["ndcg_at_5"] for q in per_query) / n,
        "success_at_3": sum(q["success_at_3"] for q in per_query) / n,
        "success_at_5": sum(q["success_at_5"] for q in per_query) / n,
        "per_query": per_query,
    }
