"""
Budget AI Assistant API

FastAPI endpoints for extracting transactions from bank statement PDFs.

Workflow: Upload PDF → Redact PII → Gemini extracts & categorises → Response
"""
import csv
import io
import uvicorn
from enum import Enum
from typing import Optional
 
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.categories import DEFAULT_BUDGET_CATEGORIES
from src.pdf_extractor.extractor import TransactionExtractor
from src.pdf_extractor.redactor import PiiRedactor


class OutputFormat(str, Enum):
    json = "json"
    csv = "csv"


class ExtractOptions(BaseModel):
    """Options for transaction extraction, sent as JSON in a form field."""
    categories: list[str] = DEFAULT_BUDGET_CATEGORIES
    context: Optional[str] = None
    format: OutputFormat = OutputFormat.json


def parse_options(options: Optional[str] = Form(None)) -> ExtractOptions:
    """Parse the 'options' form field as JSON into ExtractOptions."""
    if options is None:
        return ExtractOptions()
    return ExtractOptions.model_validate_json(options)


app = FastAPI(
    title="Budget AI Assistant",
    description="Extract transactions from bank statement PDFs",
    version="1.0.0",
)


def transactions_to_csv(transactions: list["TransactionResponse"]) -> str:
    """Convert a list of TransactionResponse objects to a CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "post_date", "description", "amount", "category", "transaction_source"])
    for txn in transactions:
        writer.writerow([txn.date, txn.post_date or "", txn.description, txn.amount, txn.category or "", txn.transaction_source or ""])
    return output.getvalue()


class TransactionResponse(BaseModel):
    """Response model for a single transaction."""
    date: str
    post_date: Optional[str]
    description: str
    amount: float
    category: Optional[str] = None
    transaction_source: Optional[str] = None


class FileResult(BaseModel):
    """Result for a single file extraction."""
    filename: str
    transactions: list[TransactionResponse]
    transaction_count: int
    error: Optional[str] = None


class Response(BaseModel):
    """Response model for extraction endpoint."""
    total_files: int
    successful: int
    failed: int
    results: list[FileResult]


@app.post("/extract", response_model=Response)
async def extract_transactions(
    files: list[UploadFile] = File(...),
    options: Optional[str] = Form(None),
):
    """
    Extract and categorise transactions from uploaded bank statement PDFs.

    Workflow per file:
        1. Redact PII from the PDF.
        2. Send the redacted PDF to the Gemini API which extracts all
           transactions and assigns a budget category to each one.

    Args:
        files: List of PDF files to process.
        options: JSON string with extraction options. Accepts:
            - categories: list of budget categories (default: DEFAULT_BUDGET_CATEGORIES)
            - context: optional context to refine categorisation
            - format: "json" or "csv" (default: "json")

    Returns:
        Response with transactions from all files, or a CSV download.
    """
    opts = parse_options(options)
    extractor = TransactionExtractor()
    results = []
    successful = 0
    failed = 0

    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            results.append(FileResult(
                filename=file.filename,
                transactions=[],
                transaction_count=0,
                error="File must be a PDF",
            ))
            failed += 1
            continue

        try:
            raw_transactions = await extractor.extract_from_upload(
                file, categories=opts.categories, context=opts.context,
            )

            transactions = [
                TransactionResponse(**txn) for txn in raw_transactions
            ]

            results.append(FileResult(
                filename=file.filename,
                transactions=transactions,
                transaction_count=len(transactions),
            ))
            successful += 1

        except Exception as e:
            results.append(FileResult(
                filename=file.filename,
                transactions=[],
                transaction_count=0,
                error=str(e),
            ))
            failed += 1

    if opts.format == OutputFormat.csv:
        all_transactions = [txn for result in results for txn in result.transactions]
        csv_content = transactions_to_csv(all_transactions)
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=transactions.csv"},
        )

    return Response(
        total_files=len(files),
        successful=successful,
        failed=failed,
        results=results,
    )


@app.post("/extract/single", response_model=FileResult)
async def extract_single_file(
    file: UploadFile = File(...),
    options: Optional[str] = Form(None),
):
    """
    Extract and categorise transactions from a single bank statement PDF.

    Args:
        file: PDF file to process.
        options: JSON string with extraction options. Accepts:
            - categories: list of budget categories
            - context: optional context for Gemini prompt
            - format: "json" or "csv" (default: "json")

    Returns:
        FileResult with extracted transactions, or a CSV download.
    """
    opts = parse_options(options)
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    extractor = TransactionExtractor()

    try:
        raw_transactions = await extractor.extract_from_upload(
            file, categories=opts.categories, context=opts.context,
        )

        transactions = [
            TransactionResponse(**txn) for txn in raw_transactions
        ]

        if opts.format == OutputFormat.csv:
            csv_content = transactions_to_csv(transactions)
            return StreamingResponse(
                io.StringIO(csv_content),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={file.filename.rsplit('.', 1)[0]}_transactions.csv"},
            )

        return FileResult(
            filename=file.filename,
            transactions=transactions,
            transaction_count=len(transactions),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/redact")
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


@app.post("/redact/multiple")
async def redact_multiple_pdfs(files: list[UploadFile] = File(...)):
    """
    Redact PII from multiple bank statement PDFs.

    Returns a ZIP archive containing all redacted PDFs.

    Args:
        files: List of PDF files to redact.

    Returns:
        ZIP file containing the redacted PDFs.
    """
    import zipfile

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


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
