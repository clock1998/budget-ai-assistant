"""
Gemini Transaction Extractor

Uses Google Gemini API to extract and categorize transactions from
redacted bank statement PDFs in a single pass.
"""

import json
import logging
import os

from google import genai
from google.genai import types

from src.categories import DEFAULT_BUDGET_CATEGORIES

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT_TEMPLATE = """\
You are an expert financial data extraction assistant.

Analyze the attached bank statement PDF and extract **every** transaction.

Return a JSON **object** with the following structure:
{{
  "statement_year": <int — the year the statement covers, e.g. 2025>,
  "transactions": [ ... ]
}}

Each item in the "transactions" array must have these fields:
- "date": transaction date in YYYY/MM/DD format
- "post_date": posting date in YYYY/MM/DD format (null if not available)
- "description": merchant / payee description exactly as it appears
- "amount": numeric amount (positive = debit/purchase, negative = credit/refund/payment)
- "category": one of the allowed categories listed below
- "transaction_source": the credit card name / product title shown on the statement (e.g. "DESJARDINS ODYSSEE WORLDELITE MASTERCARD", "Scotia Momentum VISA Infinite Card")

Allowed categories:
{categories}

Rules:
1. Use ONLY the categories listed above. Pick the single best match.
2. If the year is not explicitly shown on a transaction line, infer it from the statement date or surrounding context.
3. Payments to the credit card itself should be negative amounts with category "Insurance and Financial Services".
4. Return ONLY the JSON object described above — no markdown fences, no commentary.
5. "statement_year" must be an integer representing the primary year the statement covers (e.g. if the statement period is Dec 2024 – Jan 2025, use the year that most transactions fall in).
{context_instruction}
"""


class GeminiExtractor:
    """
    Extracts and categorizes transactions from a PDF using the Gemini API.

    The caller is expected to pass **redacted** PDF bytes so that no PII
    reaches the external API.

    Usage:
        extractor = GeminiExtractor()
        transactions = extractor.extract(redacted_pdf_bytes, categories=...)
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """
        Args:
            model: Gemini model name (default: gemini-2.5-flash).
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
        """
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Google API key is required. Pass api_key= or set "
                "the GOOGLE_API_KEY environment variable."
            )
        self._client = genai.Client(api_key=resolved_key)
        self._model = model or self.DEFAULT_MODEL

    def extract(
        self,
        pdf_bytes: bytes,
        *,
        categories: list[str] | None = None,
        context: str | None = None,
    ) -> dict:
        """
        Send a redacted PDF to Gemini and return extracted transactions.

        Args:
            pdf_bytes: Raw bytes of the **redacted** PDF.
            categories: Budget categories for classification.
                        Defaults to DEFAULT_BUDGET_CATEGORIES.
            context: Optional extra context for the prompt (e.g. country,
                     bank name, currency).

        Returns:
            Dict with keys:
            - statement_year: int
            - transactions: list of dicts with keys:
              date, post_date, description, amount, category.
        """
        cats = categories or DEFAULT_BUDGET_CATEGORIES
        category_list = "\n".join(f"- {c}" for c in cats)

        context_instruction = ""
        if context:
            context_instruction = (
                f"\nAdditional context provided by the user:\n{context}\n"
            )

        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            categories=category_list,
            context_instruction=context_instruction,
        )

        pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt, pdf_part],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        return self._parse_response(response)

    async def extract_async(
        self,
        pdf_bytes: bytes,
        *,
        categories: list[str] | None = None,
        context: str | None = None,
    ) -> dict:
        """
        Async version of extract(). Same interface, uses async Gemini client.

        Returns:
            Dict with keys: statement_year (int), transactions (list[dict]).
        """
        cats = categories or DEFAULT_BUDGET_CATEGORIES
        category_list = "\n".join(f"- {c}" for c in cats)

        context_instruction = ""
        if context:
            context_instruction = (
                f"\nAdditional context provided by the user:\n{context}\n"
            )

        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            categories=category_list,
            context_instruction=context_instruction,
        )

        pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[prompt, pdf_part],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        return self._parse_response(response)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(response) -> dict:
        """Parse the Gemini JSON response into a dict with statement_year and transactions."""
        raw = response.text.strip()

        # Strip markdown code fences if the model added them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]
            raw = raw.strip()

        data = json.loads(raw)

        # Extract statement_year from the top-level object
        statement_year: int | None = None
        if isinstance(data, dict):
            statement_year = data.get("statement_year")
            # Unwrap to the transactions list
            for key in ("transactions", "data", "results"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array of transactions, got {type(data).__name__}"
            )

        # Normalise each record
        transactions = []
        for item in data:
            transactions.append({
                "date": str(item.get("date", "")),
                "post_date": item.get("post_date") or None,
                "description": str(item.get("description", "")),
                "amount": float(item.get("amount", 0)),
                "category": item.get("category") or None,
                "transaction_source": item.get("transaction_source") or None,
            })

        return {
            "statement_year": int(statement_year) if statement_year is not None else None,
            "transactions": transactions,
        }
