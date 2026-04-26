"""
Redaction endpoints for the Budget AI Assistant API.

Endpoints for redacting PII from bank statement PDFs.
"""

import io
import zipfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.pdf_extractor.redactor import PiiRedactor


router = APIRouter()


@router.post("/redact")
async def redact_pdf(file: UploadFile = File(...)):
    """
    Redact personal identifiable information from a bank statement PDF.

    Uses PyMuPDF to detect and black-out names, credit card numbers,
    account numbers, mailing addresses, phone numbers, and email addresses.

    Args:
        file: PDF file to redact.

    Returns:
        Redacted PDF file as a download.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    redactor = PiiRedactor()

    try:
        redacted_bytes = redactor.redact(content)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Redaction failed: {str(e)}"
        )

    redacted_filename = f"redacted_{file.filename}"
    return StreamingResponse(
        io.BytesIO(redacted_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{redacted_filename}"'
        },
    )


@router.post("/redact/multiple")
async def redact_multiple_pdfs(files: list[UploadFile] = File(...)):
    """
    Redact PII from multiple bank statement PDFs.

    Returns a ZIP archive containing all redacted PDFs.

    Args:
        files: List of PDF files to redact.

    Returns:
        ZIP file containing the redacted PDFs.
    """
    redactor = PiiRedactor()
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue

            content = await file.read()
            try:
                redacted_bytes = redactor.redact(content)
                zf.writestr(f"redacted_{file.filename}", redacted_bytes)
            except Exception:
                # Skip files that fail redaction
                continue

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="redacted_statements.zip"'
        },
    )
