from typing import Dict, List, Tuple

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, cv_id in enumerate(ranked_ids, start=1):
            scores[cv_id] = scores.get(cv_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
