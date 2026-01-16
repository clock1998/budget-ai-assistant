from img2table.document import PDF
import pandas as pd
import os

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
    extracted_tables = doc.extract_tables()

    # The extract_tables method returns a dict where keys are page numbers and values are lists of tables
    for page_num, tables_list in extracted_tables.items():
        print(f"Processing page {page_num+1}, found {len(tables_list)} tables")
        for table_idx, table in enumerate(tables_list):
            if table is not None and hasattr(table, 'df'):
                # Convert to DataFrame
                df = table.df
                table_title = getattr(table, 'title', f'Table_{table_idx+1}')
                print(f"Table title: {table_title}")
                # Save as CSV
                output_file = os.path.join(output_dir, f"page_{page_num+1}_table_{table_idx+1}.csv")
                df.to_csv(output_file, index=False)
                print(f"Saved table {table_idx+1} from page {page_num+1} to {output_file}")

                # Also display the table content
                print(df.head())
            else:
                print(f"Table {table_idx+1} from page {page_num+1} is None or invalid")

    return extracted_tables

if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = "./december-2025.pdf"

    # Extract tables
    tables = extract_tables_from_pdf(pdf_path)