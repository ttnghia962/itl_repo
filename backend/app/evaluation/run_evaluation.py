import argparse
import json
from pathlib import Path

from app.config import settings
from app.services import embeddings
from app.services.vector_store import VectorStore
from app.evaluation.evaluate_retrieval import evaluate

LABELS_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "qa_labels.json"


def make_retrieve_fn(vector_store: VectorStore):
    def retrieve(query: str, k: int) -> list:
        [embedding] = embeddings.embed_texts([query])
        matches = vector_store.query(embedding, top_k=k)
        return [m["id"] for m in matches]

    return retrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval MRR/nDCG@5/success@K evaluation")
    parser.add_argument("--k", type=int, default=settings.top_k_default, help="K for retrieval (used by MRR/nDCG@5/success@K)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    vector_store = VectorStore(settings)
    result = evaluate(cases, make_retrieve_fn(vector_store), k=args.k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
