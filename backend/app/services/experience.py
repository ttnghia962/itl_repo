import re
from typing import Optional

from app.models.schemas import CVExtraction

_YEAR_PATTERN = re.compile(r"(19|20)\d{2}")
_CURRENT_PATTERN = re.compile(r"current|present", re.IGNORECASE)


def _parse_year(date_str: Optional[str], current_year: int) -> Optional[int]:
    if not date_str:
        return None
    if _CURRENT_PATTERN.search(date_str):
        return current_year
    match = _YEAR_PATTERN.search(date_str)
    return int(match.group()) if match else None


def estimate_years_experience(extraction: CVExtraction, current_year: int) -> float:
    start_years = []
    end_years = []
    for item in extraction.experience:
        start = _parse_year(item.start_date, current_year)
        end = _parse_year(item.end_date, current_year)
        if start is not None:
            start_years.append(start)
        if end is not None:
            end_years.append(end)
    if not start_years or not end_years:
        return 0.0
    return float(max(end_years) - min(start_years))
