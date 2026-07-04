import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.dependencies import get_repository, get_vector_store
from app.models.schemas import CVRecord
from app.services import pipeline
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cvs", tags=["cvs"])


@router.post("/upload", response_model=CVRecord)
async def upload_cv(
    file: UploadFile = File(...),
    repository: CVRepository = Depends(get_repository),
    vector_store: VectorStore = Depends(get_vector_store),
):
    content = await file.read()
    logger.info("Upload received: filename=%s size=%d bytes", file.filename, len(content))
    cv_id, extraction = pipeline.process_cv_file(file.filename, content, repository, vector_store)
    raw_text = repository.load_raw_text(cv_id)
    logger.info("Upload complete: cv_id=%s filename=%s name=%s", cv_id, file.filename, extraction.name)
    return CVRecord(id=cv_id, filename=file.filename, extraction=extraction, raw_text=raw_text)


@router.get("")
def list_cvs(repository: CVRepository = Depends(get_repository)):
    ids = repository.list_cv_ids()
    logger.info("Listing CVs: count=%d", len(ids))
    result = []
    for cv_id in ids:
        extraction = repository.load_extraction(cv_id)
        result.append({"id": cv_id, "name": extraction.name, "current_role": extraction.current_role})
    return result


@router.get("/{cv_id}", response_model=CVRecord)
def get_cv(cv_id: str, repository: CVRepository = Depends(get_repository)):
    try:
        extraction = repository.load_extraction(cv_id)
    except FileNotFoundError:
        logger.warning("CV not found: cv_id=%s", cv_id)
        raise HTTPException(status_code=404, detail="CV not found")
    raw_text = repository.load_raw_text(cv_id)
    return CVRecord(id=cv_id, filename=f"{cv_id}.pdf", extraction=extraction, raw_text=raw_text)


@router.get("/{cv_id}/file")
def get_cv_file(cv_id: str, repository: CVRepository = Depends(get_repository)):
    path = repository.get_pdf_path(cv_id)
    if not path.exists():
        logger.warning("CV file not found on disk: cv_id=%s path=%s", cv_id, path)
        raise HTTPException(status_code=404, detail="CV file not found")
    logger.info("Serving CV file: cv_id=%s path=%s", cv_id, path)
    return FileResponse(path, media_type="application/pdf", filename=path.name)
