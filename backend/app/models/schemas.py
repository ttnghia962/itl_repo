from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EducationItem(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ExperienceItem(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None


class CVExtraction(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    current_role: Optional[str] = None
    domain: Optional[str] = None
    extra_fields: Dict[str, str] = Field(default_factory=dict)


class CVRecord(BaseModel):
    id: str
    filename: str
    extraction: CVExtraction
    raw_text: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class RetrievedCV(BaseModel):
    id: str
    filename: str
    score: float
    extraction: CVExtraction


class QueryResponse(BaseModel):
    answer: str
    best_candidate_id: Optional[str] = None
    best_candidate_reason: Optional[str] = None
    retrieved: List[RetrievedCV]
