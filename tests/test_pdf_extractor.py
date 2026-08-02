import pymupdf
import pytest

from career_match_agent.services.pdf_extractor import (
    InvalidPdfError,
    NoExtractableTextError,
    extract_pdf,
)


def create_pdf_bytes(page_texts: list[str]) -> bytes:
    """Create an in-memory PDF for testing."""
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()

            if text:
                page.insert_text((72, 72), text)

        return document.tobytes()


def test_extract_pdf_returns_text_and_metadata() -> None:
    pdf_bytes = create_pdf_bytes(
        [
            "Buggs Bunny",
            "Machine Learning Engineer with Python experience.",
        ]
    )

    result = extract_pdf(
        pdf_bytes,
        filename="cv.pdf",
        content_type="application/pdf",
    )

    assert result.filename == "cv.pdf"
    assert result.content_type == "application/pdf"
    assert result.page_count == 2
    assert result.size_bytes == len(pdf_bytes)
    assert result.character_count == len(result.text)
    assert result.word_count > 0
    assert len(result.sha256) == 64
    assert "Buggs Bunny" in result.text
    assert "Machine Learning Engineer" in result.text


def test_extract_pdf_rejects_non_pdf_bytes() -> None:
    with pytest.raises(
        InvalidPdfError,
        match="valid PDF header",
    ):
        extract_pdf(
            b"This is not a PDF.",
            filename="cv.pdf",
            content_type="application/pdf",
        )


def test_extract_pdf_rejects_pdf_without_text() -> None:
    pdf_bytes = create_pdf_bytes([""])

    with pytest.raises(
        NoExtractableTextError,
        match="No selectable text",
    ):
        extract_pdf(
            pdf_bytes,
            filename="cv.pdf",
            content_type="application/pdf",
        )
