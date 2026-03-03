"""
Google Sheets Integration

Exports transaction data to Google Sheets using a service account.
"""

import logging
import os
from datetime import datetime
from typing import Optional

import gspread
from google.oauth2.service_account import Credentials

logger = logging.getLogger(__name__)

# Scopes required for Google Sheets read/write
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Column headers written to the first row of each sheet
HEADERS = [
    "date",
    "post_date",
    "description",
    "amount",
    "category",
    "transaction_source",
]


class GoogleSheetsClient:
    """
    Thin wrapper around *gspread* for exporting transactions to Google Sheets.

    Authentication
    --------------
    Uses a **service-account** JSON key file.  The path is resolved in order:

    1. ``credentials_path`` argument
    2. ``GOOGLE_SHEETS_CREDENTIALS`` environment variable
    3. ``credentials.json`` in the working directory

    Usage::

        client = GoogleSheetsClient()

        # Create a brand-new spreadsheet and write data
        url = client.export_transactions(transactions, title="March 2026")

        # Append to an existing spreadsheet (by ID or full URL)
        url = client.export_transactions(
            transactions,
            spreadsheet_id="1BxiMVs0...",
            worksheet_name="Desjardins",
        )
    """

    def __init__(self, credentials_path: str | None = None):
        self._credentials_path = (
            credentials_path
            or os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
            or "credentials.json"
        )
        self._gc: gspread.Client | None = None

    # ── Public API ───────────────────────────────────────────────────────

    def export_transactions(
        self,
        transactions: list[dict],
        *,
        spreadsheet_id: str | None = None,
        title: str | None = None,
        worksheet_name: str | None = None,
        share_with: str | None = None,
    ) -> str:
        """
        Write transactions to a Google Sheet.

        Args:
            transactions: List of transaction dicts (keys must match HEADERS).
            spreadsheet_id: ID of an existing spreadsheet to write into.
                If ``None`` a new spreadsheet is created.
            title: Title for a newly created spreadsheet.
                Ignored when ``spreadsheet_id`` is provided.
                Defaults to ``"Transactions <timestamp>"``.
            worksheet_name: Name of the worksheet (tab) to write to.
                Created if it doesn't exist. Defaults to ``"Transactions"``.
            share_with: Optional email address to share the spreadsheet with
                (editor permission). Useful because service-account-owned
                sheets are not visible to the user by default.

        Returns:
            The URL of the spreadsheet.
        """
        gc = self._get_client()
        ws_name = worksheet_name or "Transactions"

        if spreadsheet_id:
            spreadsheet = gc.open_by_key(spreadsheet_id)
        else:
            sheet_title = title or f"Transactions {datetime.now():%Y-%m-%d %H:%M}"
            spreadsheet = gc.create(sheet_title)
            logger.info("Created spreadsheet: %s", spreadsheet.url)

        worksheet = self._get_or_create_worksheet(
            spreadsheet, ws_name, rows=len(transactions) + 1,
        )

        rows = self._transactions_to_rows(transactions)
        worksheet.update(
            range_name="A1",
            values=[HEADERS] + rows,
        )

        if share_with:
            spreadsheet.share(share_with, perm_type="user", role="writer")

        return spreadsheet.url

    def append_transactions(
        self,
        transactions: list[dict],
        *,
        spreadsheet_id: str,
        worksheet_name: str | None = None,
    ) -> str:
        """
        Append transactions to the end of an existing worksheet.

        Does **not** write headers — assumes they already exist.

        Returns:
            The URL of the spreadsheet.
        """
        gc = self._get_client()
        spreadsheet = gc.open_by_key(spreadsheet_id)
        ws_name = worksheet_name or "Transactions"

        worksheet = self._get_or_create_worksheet(
            spreadsheet, ws_name, rows=len(transactions) + 1,
        )

        rows = self._transactions_to_rows(transactions)
        worksheet.append_rows(rows, value_input_option="USER_ENTERED")

        return spreadsheet.url

    # ── Internal helpers ─────────────────────────────────────────────────

    def _get_client(self) -> gspread.Client:
        """Lazily authenticate and return the gspread client."""
        if self._gc is None:
            creds = Credentials.from_service_account_file(
                self._credentials_path, scopes=SCOPES,
            )
            self._gc = gspread.authorize(creds)
        return self._gc

    @staticmethod
    def _get_or_create_worksheet(
        spreadsheet: gspread.Spreadsheet,
        name: str,
        rows: int = 1000,
    ) -> gspread.Worksheet:
        """Return an existing worksheet by name, or create a new one."""
        try:
            return spreadsheet.worksheet(name)
        except gspread.WorksheetNotFound:
            return spreadsheet.add_worksheet(
                title=name, rows=max(rows, 100), cols=len(HEADERS),
            )

    @staticmethod
    def _transactions_to_rows(transactions: list[dict]) -> list[list]:
        """Convert transaction dicts to a list-of-lists matching HEADERS."""
        rows = []
        for txn in transactions:
            rows.append([
                txn.get("date", ""),
                txn.get("post_date") or "",
                txn.get("description", ""),
                txn.get("amount", 0),
                txn.get("category") or "",
                txn.get("transaction_source") or "",
            ])
        return rows
