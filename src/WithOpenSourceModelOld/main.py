"""
Budget AI Assistant API

FastAPI endpoints for extracting transactions from bank statement PDFs.

Workflow: Upload PDF → Redact PII → Gemini extracts & categorises → Response
"""
import io
import uvicorn
from typing import Optional
 
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import StreamingResponse

from src.pdf_extractor.extractor import TransactionExtractor
from src.schemas import (
    ExtractOptions,
    FileResult,
    OutputFormat,
    Response,
    TransactionResponse,
)
from src.helpers import parse_options, transactions_to_csv
from src.routes_redact import router as redact_router


app = FastAPI(
    title="Budget AI Assistant",
    description="Extract transactions from bank statement PDFs",
    version="1.0.0",
)

app.include_router(redact_router)


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
                transactions=[],
                transaction_count=0,
                error="File must be a PDF",
            ))
            failed += 1
            continue

        try:
            extraction_result = await extractor.extract_from_upload(
                file, categories=opts.categories, context=opts.context,
            )

            raw_transactions = extraction_result["transactions"]
            statement_year = extraction_result.get("statement_year")

            transactions = [
                TransactionResponse(**txn) for txn in raw_transactions
            ]

            results.append(FileResult(
                statement_year=statement_year,
                transactions=transactions,
                transaction_count=len(transactions),
            ))
            successful += 1

        except Exception as e:
            results.append(FileResult(
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
