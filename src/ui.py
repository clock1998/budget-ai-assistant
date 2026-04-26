"""
Gradio UI for the Budget AI Assistant.

Provides a simple browser interface for uploading bank statement PDFs
and viewing extracted transactions.
"""

import csv
import io
import sys
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from categories import DEFAULT_BUDGET_CATEGORIES
from gemini_extractor import GeminiExtractor
from redactor import PiiRedactor


async def extract_transactions(
    files: list[str],
    context: str | None,
) -> tuple[pd.DataFrame, str | None]:
    """
    Process uploaded PDFs and return a DataFrame of transactions
    plus a downloadable CSV path.
    """
    if not files:
        return pd.DataFrame(), None

    redactor = PiiRedactor()
    gemini_extractor = GeminiExtractor()

    all_rows: list[dict] = []

    for filepath in files:
        path = Path(filepath)
        if path.suffix.lower() != ".pdf":
            continue

        pdf_bytes = path.read_bytes()
        redacted = redactor.redact(pdf_bytes)
        result = await gemini_extractor.extract_async(
            redacted,
            categories=DEFAULT_BUDGET_CATEGORIES,
            context=context or None,
        )

        statement_year = result.get("statement_year")
        for txn in result["transactions"]:
            txn["statement_year"] = statement_year
            all_rows.append(txn)

    if not all_rows:
        return pd.DataFrame(), None

    df = pd.DataFrame(all_rows)
    col_order = [
        "statement_year", "date", "post_date", "description",
        "amount", "category", "transaction_source",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # Write CSV to a temp file for download
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", prefix="transactions_",
    )
    df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_MINIMAL)

    return df, tmp.name


async def redact_pdfs(files: list[str]) -> list[str]:
    """Redact PII from uploaded PDFs and return paths to redacted files."""
    if not files:
        return []

    redactor = PiiRedactor()
    output_paths: list[str] = []

    for filepath in files:
        path = Path(filepath)
        if path.suffix.lower() != ".pdf":
            continue

        redacted_bytes = redactor.redact(path.read_bytes())
        out = Path(tempfile.mkdtemp()) / f"redacted_{path.name}"
        out.write_bytes(redacted_bytes)
        output_paths.append(str(out))

    return output_paths


# ── Build the Gradio app ────────────────────────────────────────────────

with gr.Blocks(title="Budget AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Budget AI Assistant\nUpload bank statement PDFs to extract and categorise transactions.")

    with gr.Tab("Extract Transactions"):
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(
                    label="Upload PDF(s)",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )
                context_input = gr.Textbox(
                    label="Extra Context (optional)",
                    placeholder="e.g. Canadian dollars, Desjardins credit card",
                    lines=2,
                )
                extract_btn = gr.Button("Extract", variant="primary")
            with gr.Column(scale=2):
                results_table = gr.Dataframe(
                    label="Transactions",
                    interactive=False,
                    wrap=True,
                )
                csv_download = gr.File(label="Download CSV")

        extract_btn.click(
            fn=extract_transactions,
            inputs=[upload, context_input],
            outputs=[results_table, csv_download],
        )

    with gr.Tab("Redact PII"):
        with gr.Row():
            with gr.Column(scale=1):
                redact_upload = gr.File(
                    label="Upload PDF(s)",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )
                redact_btn = gr.Button("Redact", variant="primary")
            with gr.Column(scale=2):
                redacted_files = gr.File(
                    label="Redacted PDFs",
                    file_count="multiple",
                )

        redact_btn.click(
            fn=redact_pdfs,
            inputs=[redact_upload],
            outputs=[redacted_files],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
