"""One-off backfill (Phase 3, Task 12): add years_experience metadata to
every already-ingested CV's Chroma chunks, computed from the existing
extraction JSON in storage/extracted/ (no new Groq calls, no re-embedding
— this only updates metadata)."""
from datetime import date

from app.config import settings
from app.services import experience
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore


def main() -> None:
    repository = CVRepository(settings)
    vector_store = VectorStore(settings)
    current_year = date.today().year

    cv_ids = repository.list_cv_ids()
    print(f"Backfilling years_experience for {len(cv_ids)} CVs...")
    for cv_id in cv_ids:
        extraction = repository.load_extraction(cv_id)
        years = experience.estimate_years_experience(extraction, current_year=current_year)
        vector_store.update_cv_metadata(cv_id, {"years_experience": years})
        print(f"  {cv_id}: years_experience={years}")

    print("Done.")


if __name__ == "__main__":
    main()
