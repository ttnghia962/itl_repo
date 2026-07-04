from pathlib import Path
from typing import List

from app.config import settings
from app.services import pipeline
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore


def ingest_directory(source_dir: Path, repository: CVRepository, vector_store: VectorStore) -> List[str]:
    already_ingested = set(repository.list_cv_ids())
    ingested_ids = []
    for pdf_path in sorted(source_dir.glob("*.pdf")):
        if pdf_path.stem in already_ingested:
            print(f"Already ingested {pdf_path.name}, skipping")
            continue
        content = pdf_path.read_bytes()
        try:
            cv_id, _ = pipeline.process_cv_file(pdf_path.name, content, repository, vector_store)
        except Exception as err:
            print(f"Skipped {pdf_path.name}: {err}")
            continue
        ingested_ids.append(cv_id)
        print(f"Ingested {pdf_path.name} -> {cv_id}")
    return ingested_ids


def main() -> None:
    repository = CVRepository(settings)
    vector_store = VectorStore(settings)
    ids = ingest_directory(settings.seed_data_dir, repository, vector_store)
    print(f"Done. Ingested {len(ids)} CVs from {settings.seed_data_dir}")


if __name__ == "__main__":
    main()
