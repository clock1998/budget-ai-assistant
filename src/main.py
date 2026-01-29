from img2table.document import PDF
from img2table.ocr import TesseractOCR
import pandas as pd
import os
import re
from pathlib import Path
# Configure OCR languages: use single like ["eng"] or multiple like ["eng","spa"]
OCR_LANGS = ["eng","fra"]  # change to ["eng","spa"] etc. as needed (ensure traineddata installed)
ocr = TesseractOCR(n_threads=1, lang="+".join(OCR_LANGS))

def extract_tables_from_pdf(pdf_path, output_dir="extracted_tables"):
    """
    Extract tables from PDF using img2table

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save extracted tables
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize document with PDF
    try:
        doc = PDF(pdf_path, detect_rotation=False, pdf_text_extraction=True)
        print("Initialized PDF document with text extraction enabled")
    except Exception as e:
        print(f"Error initializing PDF without OCR: {e}")
        print("PDF likely requires OCR for text extraction")
        return None

    # Extract tables
    print(f"Extracting tables from {pdf_path}...")
    extracted_tables = doc.extract_tables(ocr=None, borderless_tables=True)
    # The extract_tables method returns a dict where keys are page numbers and values are lists of tables
    for page_num, tables_list in extracted_tables.items():
        print(f"Processing page {page_num+1}, found {len(tables_list)} tables")
        for table_idx, table in enumerate(tables_list):
            # Get the title and normalize it to uppercase for easier matching
            table_title = getattr(table, 'title', "") or ""
            
            if table is not None and hasattr(table, 'df'):
                df = table.df
                
                # Output table in markdown format
                print(f"\n### Page {page_num+1} - {table_title}\n")
                md_table = df.to_markdown(index=False)
                print(md_table)
                print()
                print(df.head())

    return extracted_tables

def promote_headers(df, row_count=1):
    """
    Promotes the first N rows to headers. 
    If row_count > 1, it joins them with a space.
    """
    if df.empty:
        return df

    if row_count == 1:
        new_columns = df.iloc[0].astype(str).str.strip()
    else:
        # Merges row 0 and row 1 for each column
        new_columns = [
            " ".join(df.iloc[0:row_count, i].dropna().astype(str)).strip()
            for i in range(df.shape[1])
        ]

    df.columns = new_columns
    df = df.iloc[row_count:].reset_index(drop=True)
    df.columns.name = None
    return df

# Usage in your script:
# table_df = promote_headers(table.df, row_count=2) # Use 2 if headers are stacked
# print(table_df.to_markdown(index=False))
if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = "./scotia_bank_december_2025.pdf"
    pdf_path = "./desjardins_december_2025.pdf"
    # Extract tables
    tables = extract_tables_from_pdf(pdf_path)