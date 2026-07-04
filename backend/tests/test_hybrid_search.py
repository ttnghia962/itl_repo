import pytest

from app.services.hybrid_search import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_scores_items_by_combined_rank():
    vector_ranked = ["cv1", "cv2", "cv3"]
    keyword_ranked = ["cv2", "cv1", "cv4"]

    fused = reciprocal_rank_fusion([vector_ranked, keyword_ranked], k=60)
    fused_dict = dict(fused)

    assert fused_dict["cv1"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused_dict["cv2"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused_dict["cv3"] == pytest.approx(1 / 63)
    assert fused_dict["cv4"] == pytest.approx(1 / 63)

    fused_ids = [cv_id for cv_id, _ in fused]
    assert fused_ids[0] in {"cv1", "cv2"}


def test_reciprocal_rank_fusion_handles_single_list():
    fused = reciprocal_rank_fusion([["only"]])
    assert fused == [("only", pytest.approx(1 / 61))]


def test_reciprocal_rank_fusion_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
