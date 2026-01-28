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
            # Get the title and normalize it to uppercase for easier matching
            table_title = getattr(table, 'title', "") or ""
            
            # Filter: Check if "TRANSACTIONS" is in the title
            if "TRANSACTION" in table_title.upper():
                print(f"🎯 Match found on page {page_num+1}: {table_title}")
                
                if table is not None and hasattr(table, 'df'):
                    df = table.df
                    
                    # Display original headers
                    print(f"\n**Original headers:** {list(df.columns)}")
                    
                    # Rename columns - customize this mapping as needed
                    column_mapping = {
                        df.columns[i]: new_name 
                        for i, new_name in enumerate(["Date", "Description", "Amount", "Balance"])
                        if i < len(df.columns)
                    }
                    df = df.rename(columns=column_mapping)
                    
                    # Select only the columns we need
                    desired_columns = ["Date", "Description", "Amount"]
                    df = df[[col for col in desired_columns if col in df.columns]]
                    
                    print(f"**Renamed headers:** {list(df.columns)}")
                    
                    # Output table in markdown format
                    print(f"\n### Page {page_num+1} - {table_title}\n")
                    print(df.to_markdown(index=False))
                    print()
            else:
                # Skip tables that don't match
                continue

    return extracted_tables

if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = "./scotia_bank_december_2025.pdf"

    # Extract tables
    tables = extract_tables_from_pdf(pdf_path)