import logging
from datetime import date
from typing import List, Tuple

from app.config import settings
from app.models.schemas import CVExtraction
from app.services import chunking, embeddings, experience, llm_extraction, pdf_extractor
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def build_summary_text(extraction: CVExtraction) -> str:
    parts = [
        extraction.name or "",
        extraction.current_role or "",
        extraction.domain or "",
        " ".join(extraction.skills),
        " ".join(f"{e.title or ''} {e.company or ''} {e.description or ''}" for e in extraction.experience),
        " ".join(f"{e.degree or ''} {e.field_of_study or ''} {e.institution or ''}" for e in extraction.education),
    ]
    return "\n".join(p for p in parts if p)


def build_chunk_texts(extraction: CVExtraction, raw_text: str) -> List[str]:
    summary = build_summary_text(extraction)
    body_chunks = chunking.chunk_text(
        raw_text,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    chunks = ([summary] if summary else []) + body_chunks
    return chunks or [""]


def process_cv_file(
    filename: str,
    content: bytes,
    repository: CVRepository,
    vector_store: VectorStore,
) -> Tuple[str, CVExtraction]:
    cv_id = repository.save_original(filename, content)
    logger.info("Saved original file: cv_id=%s filename=%s", cv_id, filename)

    pdf_path = repository.get_pdf_path(cv_id)
    raw_text = pdf_extractor.extract_text(pdf_path)
    logger.info("Extracted text from PDF: cv_id=%s chars=%d", cv_id, len(raw_text))
    repository.save_raw_text(cv_id, raw_text)

    logger.info("Requesting Groq CV field extraction: cv_id=%s", cv_id)
    extraction = llm_extraction.extract_cv_fields(raw_text)
    logger.info(
        "Groq CV extraction done: cv_id=%s name=%s current_role=%s skills=%d",
        cv_id,
        extraction.name,
        extraction.current_role,
        len(extraction.skills),
    )
    repository.save_extraction(cv_id, extraction)

    chunks = build_chunk_texts(extraction, raw_text)
    chunk_embeddings = embeddings.embed_texts(chunks)
    logger.info("Computed %d chunk embedding(s): cv_id=%s", len(chunk_embeddings), cv_id)

    vector_store.upsert_cv_chunks(
        cv_id,
        chunks,
        chunk_embeddings,
        metadata={
            "filename": filename,
            "name": extraction.name or "",
            "current_role": extraction.current_role or "",
            "domain": extraction.domain or "",
            "years_experience": experience.estimate_years_experience(
                extraction, current_year=date.today().year
            ),
        },
    )
    logger.info("Indexed %d chunk(s) into vector store: cv_id=%s", len(chunks), cv_id)
    return cv_id, extraction
