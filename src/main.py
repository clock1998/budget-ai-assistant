"""
Budget AI Assistant API

FastAPI endpoints for extracting transactions from bank statement PDFs.
"""
import csv
import io
import uvicorn
from enum import Enum
from typing import Optional
 
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.categorizer.categories import DEFAULT_BUDGET_CATEGORIES
from src.google.client import export_transactions_to_sheet
from src.pdf_extractor.extractor import TransactionExtractor
from src.pdf_extractor.models import BankType
from src.categorizer.transaction_categorizer import TransactionCategorizer


class OutputFormat(str, Enum):
    json = "json"
    csv = "csv"
    google_sheets = "google_sheets"


class ExtractOptions(BaseModel):
    """Options for transaction extraction, sent as JSON in a form field."""
    categories: list[str] = DEFAULT_BUDGET_CATEGORIES
    context: Optional[str] = None
    format: OutputFormat = OutputFormat.json
    sheet_title: Optional[str] = None


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
    writer.writerow(["date", "post_date", "description", "amount", "category"])
    for txn in transactions:
        writer.writerow([txn.date, txn.post_date or "", txn.description, txn.amount, txn.category or ""])
    return output.getvalue()


class TransactionResponse(BaseModel):
    """Response model for a single transaction."""
    date: str
    post_date: Optional[str]
    description: str
    amount: float
    category: Optional[str] = None


class FileResult(BaseModel):
    """Result for a single file extraction."""
    filename: str
    bank_type: str
    transactions: list[TransactionResponse]
    transaction_count: int
    error: Optional[str] = None


class GoogleSheetResponse(BaseModel):
    """Response model when exporting to Google Sheets."""
    sheet_url: str
    transaction_count: int


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
    Extract transactions from uploaded bank statement PDFs.
    
    Args:
        files: List of PDF files to process.
        options: JSON string with extraction options. Accepts:
            - categories: list of budget categories (default: DEFAULT_BUDGET_CATEGORIES)
            - context: optional context to refine categorization (e.g. country, bank name)
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
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            results.append(FileResult(
                filename=file.filename,
                bank_type=BankType.UNKNOWN.value,
                transactions=[],
                transaction_count=0,
                error="File must be a PDF",
            ))
            failed += 1
            continue

        try:
            # Extract transactions using upload method
            df, bank_type = await extractor.extract_from_upload(file)

            if df.empty:
                transactions = []
            else:
                transactions = [
                    TransactionResponse(
                        date=row["date"],
                        post_date=row.get("post_date"),
                        description=row["description"],
                        amount=row["amount"],
                    )
                    for _, row in df.iterrows()
                ]
                categorizer = TransactionCategorizer(categories=opts.categories)
                categorizer.categorize(transactions, context=opts.context)

            results.append(FileResult(
                filename=file.filename,
                bank_type=bank_type.value,
                transactions=transactions,
                transaction_count=len(transactions),
            ))
            successful += 1

        except Exception as e:
            results.append(FileResult(
                filename=file.filename,
                bank_type=BankType.UNKNOWN.value,
                transactions=[],
                transaction_count=0,
                error=str(e),
            ))
            failed += 1

    all_transactions = [txn for result in results for txn in result.transactions]

    if opts.format == OutputFormat.csv:
        csv_content = transactions_to_csv(all_transactions)
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=transactions.csv"},
        )

    if opts.format == OutputFormat.google_sheets:
        txn_dicts = [txn.model_dump() for txn in all_transactions]
        try:
            sheet_url = export_transactions_to_sheet(
                txn_dicts, title=opts.sheet_title,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Google Sheets export failed: {e}")
        return GoogleSheetResponse(
            sheet_url=sheet_url,
            transaction_count=len(txn_dicts),
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
    Extract transactions from a single bank statement PDF.
    
    Args:
        file: PDF file to process.
        options: JSON string with extraction options. Accepts:
            - categories: list of budget categories (default: DEFAULT_BUDGET_CATEGORIES)
            - context: optional context to refine categorization (e.g. country, bank name)
            - format: "json" or "csv" (default: "json")
        
    Returns:
        FileResult with extracted transactions, or a CSV download.
    """
    opts = parse_options(options)
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    extractor = TransactionExtractor()

    try:
        df, bank_type = await extractor.extract_from_upload(file)

        if df.empty:
            transactions = []
        else:
            transactions = [
                TransactionResponse(
                    date=row["date"],
                    post_date=row.get("post_date"),
                    description=row["description"],
                    amount=row["amount"],
                )
                for _, row in df.iterrows()
            ]

            categorizer = TransactionCategorizer(categories=opts.categories)
            categorizer.categorize(transactions, context=opts.context)

        if opts.format == OutputFormat.csv:
            csv_content = transactions_to_csv(transactions)
            return StreamingResponse(
                io.StringIO(csv_content),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={file.filename.rsplit('.', 1)[0]}_transactions.csv"},
            )

        if opts.format == OutputFormat.google_sheets:
            txn_dicts = [txn.model_dump() for txn in transactions]
            try:
                sheet_url = export_transactions_to_sheet(
                    txn_dicts, title=opts.sheet_title,
                )
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Google Sheets export failed: {e}")
            return GoogleSheetResponse(
                sheet_url=sheet_url,
                transaction_count=len(txn_dicts),
            )

        return FileResult(
            filename=file.filename,
            bank_type=bank_type.value,
            transactions=transactions,
            transaction_count=len(transactions),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
