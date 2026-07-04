import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
REPO_ROOT = BASE_DIR.parent

_GROQ_KEY_PATTERN = re.compile(r"^GROQ_API_KEY(\d+)$")


def _extract_groq_api_keys(env: Dict[str, Optional[str]]) -> List[str]:
    matches = []
    for name, value in env.items():
        m = _GROQ_KEY_PATTERN.match(name)
        if m and value:
            matches.append((int(m.group(1)), value))
    matches.sort(key=lambda pair: pair[0])
    return [value for _, value in matches]


def _load_groq_api_keys() -> List[str]:
    merged = {**dotenv_values(REPO_ROOT / ".env"), **os.environ}
    return _extract_groq_api_keys(merged)


class Settings(BaseSettings):
    groq_api_keys: List[str] = Field(default_factory=_load_groq_api_keys)
    groq_model: str = "llama-3.3-70b-versatile"
    storage_dir: Path = BASE_DIR / "storage"
    seed_data_dir: Path = REPO_ROOT / "data" / "INFORMATION-TECHNOLOGY"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    top_k_default: int = 5
    chunk_max_tokens: int = 200
    chunk_overlap_tokens: int = 20

    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    @property
    def cvs_dir(self) -> Path:
        d = self.storage_dir / "cvs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def extracted_dir(self) -> Path:
        d = self.storage_dir / "extracted"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def raw_text_dir(self) -> Path:
        d = self.storage_dir / "raw_text"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def chroma_dir(self) -> Path:
        d = self.storage_dir / "chroma"
        d.mkdir(parents=True, exist_ok=True)
        return d


settings = Settings()
