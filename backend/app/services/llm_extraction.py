import itertools
import json
import logging
import threading
from typing import Optional

from groq import Groq

from app.config import settings
from app.models.schemas import CVExtraction

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert HR data-extraction assistant.
Read the candidate's resume text and extract structured information as a single JSON object.

Required JSON keys (always include all of them, use null or [] when unknown):
- name (string)
- email (string)
- phone (string)
- skills (array of strings)
- education (array of objects: institution, degree, field_of_study, start_date, end_date)
- experience (array of objects: company, title, start_date, end_date, description)
- current_role (string)
- domain (string, e.g. the industry/department such as "Information Technology")

Additional field (optional but encouraged):
- extra_fields: an object mapping any other useful field name you notice in the resume
  (for example "certifications", "languages", "projects") to a short string value.
  Only include fields you can support with text found in the resume.

Respond with ONLY the JSON object, no commentary.
"""

_client_lock = threading.Lock()
_client_cycle = None


def _build_client_cycle():
    if not settings.groq_api_keys:
        raise RuntimeError(
            "No Groq API keys configured (set GROQ_API_KEY1, GROQ_API_KEY2, ... in .env)"
        )
    logger.info("Initialized Groq client round-robin: keys=%d", len(settings.groq_api_keys))
    clients = [Groq(api_key=key) for key in settings.groq_api_keys]
    return itertools.cycle(clients)


def get_groq_client() -> Groq:
    global _client_cycle
    with _client_lock:
        if _client_cycle is None:
            _client_cycle = _build_client_cycle()
        return next(_client_cycle)


def extract_cv_fields(raw_text: str, client: Optional[Groq] = None) -> CVExtraction:
    client = client or get_groq_client()
    logger.info("Groq CV extraction request: model=%s input_chars=%d", settings.groq_model, len(raw_text))
    try:
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            temperature=0.2,
            max_tokens=4000,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )
    except Exception as err:
        logger.error("Groq CV extraction API call failed: %s", err)
        raise RuntimeError(f"Groq CV extraction API call failed: {err}") from err

    raw_content = completion.choices[0].message.content
    logger.info("Groq CV extraction response received: %d chars", len(raw_content))
    logger.debug("Groq CV extraction raw response: %s", raw_content)

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as err:
        logger.error("Groq returned invalid JSON for CV extraction: %s", err)
        raise ValueError(f"Groq returned invalid JSON for CV extraction: {err}") from err

    extra_fields = data.get("extra_fields")
    if isinstance(extra_fields, dict):
        data["extra_fields"] = {
            key: value for key, value in extra_fields.items() if isinstance(value, str)
        }

    try:
        return CVExtraction(**data)
    except Exception as err:
        raise ValueError(f"Groq CV extraction response did not match expected schema: {err}") from err
