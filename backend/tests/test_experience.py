from app.models.schemas import CVExtraction, ExperienceItem
from app.services.experience import estimate_years_experience


def test_estimate_years_experience_spans_earliest_start_to_latest_end():
    extraction = CVExtraction(
        experience=[
            ExperienceItem(start_date="2010", end_date="2015"),
            ExperienceItem(start_date="2015", end_date="2020"),
        ]
    )
    assert estimate_years_experience(extraction, current_year=2026) == 10.0


def test_estimate_years_experience_treats_current_and_present_as_given_year():
    extraction = CVExtraction(
        experience=[ExperienceItem(start_date="2007-08", end_date="Current")]
    )
    assert estimate_years_experience(extraction, current_year=2026) == 19.0

    extraction2 = CVExtraction(
        experience=[ExperienceItem(start_date="2020", end_date="Present")]
    )
    assert estimate_years_experience(extraction2, current_year=2026) == 6.0


def test_estimate_years_experience_returns_zero_when_no_parseable_dates():
    extraction = CVExtraction(experience=[ExperienceItem(start_date=None, end_date=None)])
    assert estimate_years_experience(extraction, current_year=2026) == 0.0

    extraction_empty = CVExtraction(experience=[])
    assert estimate_years_experience(extraction_empty, current_year=2026) == 0.0
