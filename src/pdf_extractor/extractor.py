"""
Transaction Extractor Module

Orchestrates PII redaction and Gemini-based transaction extraction.

Workflow:
    1. Redact personal identifiers from the uploaded PDF.
    2. Send the redacted PDF to the Gemini API which extracts all
       transactions and assigns budget categories in a single pass.
"""

import io
from typing import Optional

from fastapi import UploadFile

from src.pdf_extractor.gemini_extractor import GeminiExtractor
from src.pdf_extractor.redactor import PiiRedactor


class TransactionExtractor:
    """
    Extracts and categorises transactions from bank statement PDFs.

    The pipeline:
        upload → redact PII → send to Gemini → structured transactions

    Usage:
        extractor = TransactionExtractor()
        transactions = await extractor.extract_from_upload(
            file, categories=[...], context="..."
        )
    """

    def __init__(self, *, gemini_model: str | None = None, api_key: str | None = None):
        """
        Args:
            gemini_model: Override the default Gemini model name.
            api_key: Google API key (falls back to GOOGLE_API_KEY env var).
        """
        self._redactor = PiiRedactor()
        self._gemini = GeminiExtractor(model=gemini_model, api_key=api_key)

    # ── Public API ───────────────────────────────────────────────────────

    async def extract_from_upload(
        self,
        file: UploadFile,
        *,
        categories: list[str] | None = None,
        context: Optional[str] = None,
    ) -> list[dict]:
        """
        Extract and categorise transactions from a single uploaded PDF.

        Args:
            file: FastAPI UploadFile (must be a PDF).
            categories: Budget categories to use for classification.
            context: Optional extra context for the Gemini prompt.

        Returns:
            List of transaction dicts with keys:
            date, post_date, description, amount, category.
        """
        content = await file.read()
        redacted_bytes = self._redactor.redact(content)
        return await self._gemini.extract_async(
            redacted_bytes, categories=categories, context=context,
        )

    async def extract_from_uploads(
        self,
        files: list[UploadFile],
        *,
        categories: list[str] | None = None,
        context: Optional[str] = None,
    ) -> list[tuple[str, list[dict]]]:
        """
        Extract transactions from multiple uploaded PDFs.

        Args:
            files: List of FastAPI UploadFile objects.
            categories: Budget categories.
            context: Optional extra context.

        Returns:
            List of (filename, transactions) tuples.
        """
        results = []
        for file in files:
            transactions = await self.extract_from_upload(
                file, categories=categories, context=context,
            )
            results.append((file.filename, transactions))
        return results

    def redact(self, pdf_bytes: bytes) -> bytes:
        """
        Redact PII from raw PDF bytes (convenience wrapper).

        Args:
            pdf_bytes: Original PDF content.

        Returns:
            Redacted PDF bytes.
        """
        return self._redactor.redact(pdf_bytes)
