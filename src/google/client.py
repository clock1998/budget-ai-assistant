"""
Google Sheets client for exporting transaction data.

Uses OAuth2 (user consent) authentication via gspread.
On first use, a browser window will open for authorization.
Subsequent calls reuse the stored refresh token.

Setup:
    1. Create OAuth2 credentials in Google Cloud Console (Desktop app type).
    2. Download the client secret JSON and save it as `credentials.json`
       in the project root (or set GOOGLE_CREDENTIALS_PATH env var).
    3. On first API call with google_sheets format, authorize in the browser.
       The token is cached at ~/.config/gspread/authorized_user.json.
"""

import os
from datetime import datetime

import gspread


# Default paths for OAuth2 credential files
DEFAULT_CREDENTIALS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "credentials.json"
)
CREDENTIALS_PATH = os.environ.get("GOOGLE_CREDENTIALS_PATH", DEFAULT_CREDENTIALS_PATH)
AUTHORIZED_USER_PATH = os.environ.get(
    "GOOGLE_AUTHORIZED_USER_PATH",
    os.path.join(os.path.expanduser("~"), ".config", "gspread", "authorized_user.json"),
)


def _get_client() -> gspread.Client:
    """
    Get an authenticated gspread client using OAuth2.

    Returns:
        Authenticated gspread Client.
    """
    return gspread.oauth(
        credentials_filename=CREDENTIALS_PATH,
        authorized_user_filename=AUTHORIZED_USER_PATH,
    )


def export_transactions_to_sheet(
    transactions: list[dict],
    title: str | None = None,
) -> str:
    """
    Create a new Google Sheet and populate it with transaction data.

    Args:
        transactions: List of transaction dicts with keys:
            date, post_date, description, amount, category.
        title: Optional spreadsheet title. Defaults to
            "Transactions <timestamp>".

    Returns:
        The URL of the newly created Google Sheet.
    """
    if title is None:
        title = f"Transactions {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    client = _get_client()
    spreadsheet = client.create(title)

    worksheet = spreadsheet.sheet1
    worksheet.update_title("Transactions")

    # Header row
    headers = ["Date", "Post Date", "Description", "Amount", "Category"]
    rows = [headers]

    for txn in transactions:
        rows.append([
            txn.get("date", ""),
            txn.get("post_date", "") or "",
            txn.get("description", ""),
            txn.get("amount", 0),
            txn.get("category", "") or "",
        ])

    # Write all data in a single batch
    worksheet.update(rows, value_input_option="USER_ENTERED")

    # Format header row bold
    worksheet.format("1:1", {"textFormat": {"bold": True}})

    return spreadsheet.url
