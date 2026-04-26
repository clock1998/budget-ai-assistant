"""
Pydantic models and enums for the Budget AI Assistant API.
"""

from enum import Enum
from typing import Optional

from fastapi import Form
from pydantic import BaseModel

from src.categories import DEFAULT_BUDGET_CATEGORIES


class OutputFormat(str, Enum):
    json = "json"
    csv = "csv"


class ExtractOptions(BaseModel):
    """Options for transaction extraction, sent as JSON in a form field."""
    categories: list[str] = DEFAULT_BUDGET_CATEGORIES
    context: Optional[str] = None
    format: OutputFormat = OutputFormat.json
    # Google Sheets options (only used when format is "google_sheets")
    spreadsheet_id: Optional[str] = None
    worksheet_name: Optional[str] = None
    sheet_title: Optional[str] = None
    share_with: Optional[str] = None


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
    statement_year: Optional[int] = None
    transactions: list[TransactionResponse]
    transaction_count: int
    error: Optional[str] = None


class Response(BaseModel):
    """Response model for extraction endpoint."""
    total_files: int
    successful: int
    failed: int
    results: list[FileResult]
