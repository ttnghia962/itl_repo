import re
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from app.services.cv_repository import CVRepository

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


def build_documents(repository: CVRepository) -> Dict[str, str]:
    return {cv_id: repository.load_raw_text(cv_id) for cv_id in repository.list_cv_ids()}


class KeywordIndex:
    def __init__(self, documents: Dict[str, str]):
        self._cv_ids: List[str] = []
        tokenized_corpus: List[List[str]] = []
        for cv_id, text in documents.items():
            tokens = tokenize(text)
            if not tokens:
                continue
            self._cv_ids.append(cv_id)
            tokenized_corpus.append(tokens)
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._cv_ids, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
