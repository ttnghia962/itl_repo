"""Phase 2, Task 6: staged latency measurement against the default Groq
model. Fires NUM_REQUESTS requests (cycling through the 4 use-case
queries), timing embed / Chroma retrieval / Groq generation separately.
If the Groq daily quota is hit partway through, stops and reports
whatever N was completed rather than blocking."""
import json
import time
from pathlib import Path

from app.config import settings
from app.services import embeddings, llm_rag
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore

LABELS_PATH = Path(__file__).resolve().parent.parent / "eval" / "qa_labels.json"
NUM_REQUESTS = 30


def percentile(values: list, p: float) -> float:
    values = sorted(values)
    idx = min(int(round(p / 100 * (len(values) - 1))), len(values) - 1)
    return values[idx]


def summarize(name: str, values: list) -> None:
    if not values:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: p50={percentile(values, 50):.3f}s "
        f"p95={percentile(values, 95):.3f}s "
        f"p99={percentile(values, 99):.3f}s (n={len(values)})"
    )


def main() -> None:
    cases = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    queries = [c["query"] for c in cases]
    repository = CVRepository(settings)
    vector_store = VectorStore(settings)

    embed_times, retrieval_times, generation_times, total_times = [], [], [], []
    completed = 0
    try:
        for i in range(NUM_REQUESTS):
            query = queries[i % len(queries)]
            t0 = time.perf_counter()

            t_embed = time.perf_counter()
            [query_embedding] = embeddings.embed_texts([query])
            embed_times.append(time.perf_counter() - t_embed)

            t_retrieval = time.perf_counter()
            matches = vector_store.query(query_embedding, top_k=5)
            retrieval_times.append(time.perf_counter() - t_retrieval)

            contexts = []
            for m in matches:
                cv_id = m["id"]
                extraction = repository.load_extraction(cv_id)
                contexts.append(
                    {
                        "candidate_id": cv_id,
                        "name": extraction.name,
                        "current_role": extraction.current_role,
                        "domain": extraction.domain,
                        "skills": extraction.skills,
                        "experience": [e.model_dump() for e in extraction.experience],
                        "education": [e.model_dump() for e in extraction.education],
                    }
                )

            t_gen = time.perf_counter()
            llm_rag.generate_answer(query, contexts)
            generation_times.append(time.perf_counter() - t_gen)

            total_times.append(time.perf_counter() - t0)
            completed += 1
            print(f"request {i + 1}/{NUM_REQUESTS} done")
    except Exception as err:
        print(f"Stopped after {completed} requests due to: {err}")

    print(f"\nCompleted {completed}/{NUM_REQUESTS} requests (model={settings.groq_model})")
    summarize("embed", embed_times)
    summarize("chroma_retrieval", retrieval_times)
    summarize("groq_generation", generation_times)
    summarize("end_to_end", total_times)


if __name__ == "__main__":
    main()
