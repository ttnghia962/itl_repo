import json
import logging
from typing import Any, Dict, List, Optional

from groq import Groq

from app.config import settings
from app.services.llm_extraction import get_groq_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an HR assistant helping a recruiter evaluate candidates.
You will be given a recruiter question and a list of candidate profiles as JSON.
Answer the question using ONLY the information in the provided candidate profiles.
You MUST explicitly name exactly one candidate as the best fit by their candidate_id.
If no candidate is a reasonable fit, set best_candidate_id to null and explain why in the answer.
Additionally, provide a short (1-2 sentence) reason focused specifically on why that one
candidate was picked, separate from your general answer to the recruiter's question.

Respond with ONLY a JSON object of the form:
{"answer": "<your explanation, naming the candidate by name>", "best_candidate_id": "<one of the provided candidate_id values, or null>", "best_candidate_reason": "<1-2 sentence reason for this specific pick, or null>"}
"""


def generate_answer(
    question: str,
    contexts: List[Dict[str, Any]],
    client: Optional[Groq] = None,
) -> Dict[str, Any]:
    client = client or get_groq_client()
    user_content = json.dumps({"question": question, "candidates": contexts}, ensure_ascii=False)
    logger.info(
        "Groq RAG answer request: model=%s question=%r candidates=%d",
        settings.groq_model,
        question,
        len(contexts),
    )
    try:
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=2000,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )
    except Exception as err:
        logger.error("Groq RAG answer generation API call failed: %s", err)
        raise RuntimeError(f"Groq RAG answer generation API call failed: {err}") from err

    raw_content = completion.choices[0].message.content
    logger.debug("Groq RAG raw response: %s", raw_content)

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as err:
        logger.error("Groq returned invalid JSON for RAG answer generation: %s", err)
        raise ValueError(f"Groq returned invalid JSON for RAG answer generation: {err}") from err

    logger.info(
        "Groq RAG answer received: best_candidate_id=%s answer=%r",
        data.get("best_candidate_id"),
        data.get("answer", ""),
    )
    return {
        "answer": data.get("answer", ""),
        "best_candidate_id": data.get("best_candidate_id"),
        "best_candidate_reason": data.get("best_candidate_reason"),
    }
