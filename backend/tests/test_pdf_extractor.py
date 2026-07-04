from app.services import pdf_extractor


def _make_pdf(tmp_path, text):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, text)
    path = tmp_path / "sample.pdf"
    pdf.output(str(path))
    return path


def test_extract_text_returns_direct_text_when_present(tmp_path):
    path = _make_pdf(tmp_path, "Python Machine Learning Engineer with 6 years experience")
    result = pdf_extractor.extract_text(path)
    assert "Python" in result
    assert "Machine Learning" in result


def test_extract_text_falls_back_to_ocr_when_direct_text_too_short(tmp_path, monkeypatch):
    path = _make_pdf(tmp_path, "")
    monkeypatch.setattr(pdf_extractor, "_extract_with_ocr", lambda p: "OCR EXTRACTED TEXT")
    result = pdf_extractor.extract_text(path)
    assert result == "OCR EXTRACTED TEXT"
