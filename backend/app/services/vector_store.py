from typing import Any, Dict, List, Optional

import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings


class VectorStore:
    def __init__(self, settings: Settings, collection_name: str = "cvs"):
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert_cv_chunks(
        self,
        cv_id: str,
        chunks: List[str],
        chunk_embeddings: List[List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        if len(chunks) != len(chunk_embeddings):
            raise ValueError("chunks and chunk_embeddings must have the same length")

        existing = self._collection.get(where={"cv_id": cv_id})
        existing_ids = existing.get("ids", [])
        if existing_ids:
            self._collection.delete(ids=existing_ids)

        if not chunks:
            return

        ids = [f"{cv_id}::{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "cv_id": cv_id, "chunk_index": i} for i in range(len(chunks))]
        self._collection.upsert(ids=ids, embeddings=chunk_embeddings, documents=chunks, metadatas=metadatas)

    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        got = self._collection.get(
            where=where,
            include=["embeddings", "metadatas", "documents"],
        )
        ids = got.get("ids", [])
        if not ids:
            return []

        query_vec = np.asarray(embedding, dtype=np.float64)
        chunk_embeddings = np.asarray(got["embeddings"], dtype=np.float64)
        distances = np.linalg.norm(chunk_embeddings - query_vec, axis=1)
        metadatas = got.get("metadatas", [])
        documents = got.get("documents", [])

        best_by_cv: Dict[str, Dict[str, Any]] = {}
        for i, chunk_id in enumerate(ids):
            cv_id = metadatas[i].get("cv_id", chunk_id)
            distance = float(distances[i])
            if cv_id not in best_by_cv or distance < best_by_cv[cv_id]["distance"]:
                best_by_cv[cv_id] = {
                    "id": cv_id,
                    "distance": distance,
                    "metadata": metadatas[i],
                    "document": documents[i],
                }

        ranked = sorted(best_by_cv.values(), key=lambda r: r["distance"])
        return ranked[:top_k]

    def update_cv_metadata(self, cv_id: str, updates: Dict[str, Any]) -> None:
        existing = self._collection.get(where={"cv_id": cv_id})
        ids = existing.get("ids", [])
        if not ids:
            return
        metadatas = existing.get("metadatas", [])
        updated_metadatas = [{**(metadata or {}), **updates} for metadata in metadatas]
        self._collection.update(ids=ids, metadatas=updated_metadatas)
