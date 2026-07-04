from app.models.schemas import CVExtraction


def test_cv_extraction_has_required_fields():
    data = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "555-0100",
        "skills": ["Python", "Machine Learning"],
        "education": [
            {
                "institution": "MIT",
                "degree": "BSc",
                "field_of_study": "CS",
                "start_date": "2016",
                "end_date": "2020",
            }
        ],
        "experience": [
            {
                "company": "Acme",
                "title": "ML Engineer",
                "start_date": "2020",
                "end_date": "2023",
                "description": "Built models",
            }
        ],
        "current_role": "ML Engineer",
        "domain": "Information Technology",
    }
    cv = CVExtraction(**data)
    assert cv.name == "Jane Doe"
    assert cv.skills == ["Python", "Machine Learning"]
    assert cv.education[0].institution == "MIT"
    assert cv.experience[0].company == "Acme"
    assert cv.extra_fields == {}


def test_cv_extraction_defaults_when_fields_missing():
    cv = CVExtraction()
    assert cv.name is None
    assert cv.skills == []
    assert cv.education == []
    assert cv.experience == []
    assert cv.extra_fields == {}


def test_cv_extraction_accepts_extra_fields():
    cv = CVExtraction(
        name="A",
        extra_fields={
            "certifications": "AWS Certified",
            "languages": "English, Vietnamese",
        },
    )
    assert cv.extra_fields["certifications"] == "AWS Certified"
