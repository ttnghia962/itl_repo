from functools import lru_cache
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    torch.set_num_threads(1)
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(list(texts), convert_to_numpy=True)
    return vectors.tolist()
