import json
from pathlib import Path
from typing import List

from app.config import Settings
from app.models.schemas import CVExtraction


class CVRepository:
    def __init__(self, settings: Settings):
        self._settings = settings

    def save_original(self, filename: str, content: bytes) -> str:
        stem = Path(filename).stem or "cv"
        candidate = stem
        i = 1
        while (self._settings.cvs_dir / f"{candidate}.pdf").exists():
            candidate = f"{stem}-{i}"
            i += 1
        (self._settings.cvs_dir / f"{candidate}.pdf").write_bytes(content)
        return candidate

    def get_pdf_path(self, cv_id: str) -> Path:
        return self._settings.cvs_dir / f"{cv_id}.pdf"

    def save_raw_text(self, cv_id: str, text: str) -> None:
        (self._settings.raw_text_dir / f"{cv_id}.txt").write_text(text, encoding="utf-8")

    def load_raw_text(self, cv_id: str) -> str:
        path = self._settings.raw_text_dir / f"{cv_id}.txt"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def save_extraction(self, cv_id: str, extraction: CVExtraction) -> None:
        path = self._settings.extracted_dir / f"{cv_id}.json"
        path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")

    def load_extraction(self, cv_id: str) -> CVExtraction:
        path = self._settings.extracted_dir / f"{cv_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return CVExtraction(**data)

    def list_cv_ids(self) -> List[str]:
        return sorted(p.stem for p in self._settings.extracted_dir.glob("*.json"))
