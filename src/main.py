"""
Budget AI Assistant API

FastAPI endpoints for extracting transactions from bank statement PDFs.
"""
import uvicorn
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from pdf_extractor.extractor import TransactionExtractor
from pdf_extractor.models import BankType


app = FastAPI(
    title="Budget AI Assistant",
    description="Extract transactions from bank statement PDFs",
    version="1.0.0",
)


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


class ExtractionResponse(BaseModel):
    """Response model for extraction endpoint."""
    total_files: int
    successful: int
    failed: int
    results: list[FileResult]


@app.post("/extract", response_model=ExtractionResponse)
async def extract_transactions(files: list[UploadFile] = File(...)):
    """
    Extract transactions from uploaded bank statement PDFs.
    
    Args:
        files: List of PDF files to process.
        
    Returns:
        ExtractionResponse with transactions from all files.
    """
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
                        category=row.get("category"),
                    )
                    for _, row in df.iterrows()
                ]

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

    return ExtractionResponse(
        total_files=len(files),
        successful=successful,
        failed=failed,
        results=results,
    )


@app.post("/extract/single", response_model=FileResult)
async def extract_single_file(file: UploadFile = File(...)):
    """
    Extract transactions from a single bank statement PDF.
    
    Args:
        file: PDF file to process.
        
    Returns:
        FileResult with extracted transactions.
    """
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
