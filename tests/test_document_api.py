import hashlib

import pymupdf
from fastapi.testclient import TestClient

from career_match_agent.api.main import app
from career_match_agent.services.pdf_extractor import MAX_PDF_SIZE_BYTES


client = TestClient(app)


def create_pdf_bytes(text: str) -> bytes:
    """Create an in-memory one-page PDF for API tests."""
    with pymupdf.open() as document:
        page = document.new_page()

        if text:
            page.insert_text((72, 72), text)

        return document.tobytes()


def test_cv_extraction_endpoint_returns_extracted_text() -> None:
    pdf_bytes = create_pdf_bytes(
        "Machine Learning Engineer with Python and PyTorch experience."
    )

    response = client.post(
        "/documents/cv/extract",
        files={
            "file": (
                "candidate_cv.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    response_payload = response.json()

    assert response_payload["filename"] == "candidate_cv.pdf"
    assert response_payload["content_type"] == "application/pdf"
    assert response_payload["size_bytes"] == len(pdf_bytes)
    assert response_payload["page_count"] == 1
    assert response_payload["word_count"] > 0
    assert response_payload["sha256"] == hashlib.sha256(pdf_bytes).hexdigest()
    assert "Machine Learning Engineer" in response_payload["text"]


def test_cv_extraction_endpoint_rejects_non_pdf_extension() -> None:
    response = client.post(
        "/documents/cv/extract",
        files={
            "file": (
                "candidate_cv.txt",
                b"This is not a PDF.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["detail"] == (
        "Only files with a .pdf extension are supported."
    )


def test_cv_extraction_endpoint_rejects_corrupted_pdf() -> None:
    response = client.post(
        "/documents/cv/extract",
        files={
            "file": (
                "candidate_cv.pdf",
                b"%PDF-1.7 corrupted content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422
    assert "could not be parsed" in response.json()["detail"]


def test_cv_extraction_endpoint_rejects_pdf_without_text() -> None:
    pdf_bytes = create_pdf_bytes("")

    response = client.post(
        "/documents/cv/extract",
        files={
            "file": (
                "scanned_cv.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422
    assert "No selectable text" in response.json()["detail"]


def test_cv_extraction_endpoint_rejects_oversized_file() -> None:
    oversized_content = b"%PDF-" + b"0" * MAX_PDF_SIZE_BYTES

    response = client.post(
        "/documents/cv/extract",
        files={
            "file": (
                "large_cv.pdf",
                oversized_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert "5 MB limit" in response.json()["detail"]
