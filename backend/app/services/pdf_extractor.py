from pathlib import Path
from typing import Union

import pdfplumber

MIN_TEXT_LENGTH = 20


def extract_text(pdf_path: Union[str, Path]) -> str:
    direct_text = _extract_with_pdfplumber(pdf_path)
    if len(direct_text.strip()) >= MIN_TEXT_LENGTH:
        return direct_text
    ocr_text = _extract_with_ocr(pdf_path)
    return ocr_text or direct_text


def _extract_with_pdfplumber(pdf_path: Union[str, Path]) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def _extract_with_ocr(pdf_path: Union[str, Path]) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        return ""
    try:
        images = convert_from_path(str(pdf_path))
        return "\n".join(pytesseract.image_to_string(img) for img in images).strip()
    except Exception:
        return ""
