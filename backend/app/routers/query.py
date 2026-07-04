import logging

from fastapi import APIRouter, Depends

from app.dependencies import get_repository, get_vector_store
from app.models.schemas import QueryRequest, QueryResponse, RetrievedCV
from app.services import embeddings, hybrid_search, keyword_index, llm_rag
from app.services.cv_repository import CVRepository
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_cvs(
    payload: QueryRequest,
    repository: CVRepository = Depends(get_repository),
    vector_store: VectorStore = Depends(get_vector_store),
):
    logger.info("Query received: question=%r top_k=%d", payload.question, payload.top_k)

    total_cvs = len(repository.list_cv_ids())
    overfetch_k = max(payload.top_k * 3, total_cvs) if total_cvs else payload.top_k

    [query_embedding] = embeddings.embed_texts([payload.question])
    vector_matches = vector_store.query(query_embedding, top_k=overfetch_k)
    vector_ranked_ids = [m["id"] for m in vector_matches]
    metadata_by_id = {m["id"]: m["metadata"] for m in vector_matches}

    documents = keyword_index.build_documents(repository)
    index = keyword_index.KeywordIndex(documents)
    keyword_matches = index.search(payload.question, top_k=overfetch_k)
    keyword_ranked_ids = [cv_id for cv_id, _ in keyword_matches]

    fused = hybrid_search.reciprocal_rank_fusion([vector_ranked_ids, keyword_ranked_ids])
    top_matches = fused[: payload.top_k]
    logger.info(
        "Hybrid search returned %d match(es): ids=%s",
        len(top_matches),
        [cv_id for cv_id, _ in top_matches],
    )
    if not top_matches:
        logger.warning("No matches found for question=%r", payload.question)

    retrieved: list[RetrievedCV] = []
    contexts = []
    for cv_id, fused_score in top_matches:
        extraction = repository.load_extraction(cv_id)
        filename = metadata_by_id.get(cv_id, {}).get("filename", f"{cv_id}.pdf")
        retrieved.append(
            RetrievedCV(
                id=cv_id,
                filename=filename,
                score=fused_score,
                extraction=extraction,
            )
        )
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

    result = llm_rag.generate_answer(payload.question, contexts)
    logger.info("Query answered: best_candidate_id=%s", result["best_candidate_id"])
    return QueryResponse(
        answer=result["answer"],
        best_candidate_id=result["best_candidate_id"],
        best_candidate_reason=result["best_candidate_reason"],
        retrieved=retrieved,
    )
