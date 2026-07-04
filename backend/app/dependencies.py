from functools import lru_cache

from app.config import settings
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_repository() -> CVRepository:
    return CVRepository(settings)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore(settings)
