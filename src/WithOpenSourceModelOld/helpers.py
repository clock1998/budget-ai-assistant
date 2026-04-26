"""
Helper / utility functions for the Budget AI Assistant API.
"""

import csv
import io
from typing import Optional

from fastapi import Form

from src.schemas import ExtractOptions, TransactionResponse


def parse_options(options: Optional[str] = Form(None)) -> ExtractOptions:
    """Parse the 'options' form field as JSON into ExtractOptions."""
    if options is None:
        return ExtractOptions()
    return ExtractOptions.model_validate_json(options)


def transactions_to_csv(transactions: list[TransactionResponse]) -> str:
    """Convert a list of TransactionResponse objects to a CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", "post_date", "description", "amount", "category", "transaction_source"])
    for txn in transactions:
        writer.writerow([txn.date, txn.post_date or "", txn.description, txn.amount, txn.category or "", txn.transaction_source or ""])
    return output.getvalue()
