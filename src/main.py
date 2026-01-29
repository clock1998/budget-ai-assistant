import pdfplumber
import pandas as pd
import os
import re


def extract_tables_from_pdf(pdf_path, output_dir="extracted_tables"):
    """
    Extract tables from bank statement PDFs using PDFPlumber.
    Handles tables without visible grid lines and multiline rows.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted tables as CSV
    
    Returns:
        List of pandas DataFrames containing extracted tables
    """
    os.makedirs(output_dir, exist_ok=True)
    extracted_tables = []
    
    # Table extraction settings optimized for bank statements
    # These settings help detect tables without visible grid lines
    table_settings = {
        "vertical_strategy": "text",      # Use text alignment to find columns
        "horizontal_strategy": "lines_strict",  # Use lines for rows when available
        "snap_tolerance": 4,              # Tolerance for snapping lines
        "snap_x_tolerance": 4,
        "snap_y_tolerance": 4,
        "join_tolerance": 4,              # Tolerance for joining lines
        "join_x_tolerance": 4,
        "join_y_tolerance": 4,
        "edge_min_length": 10,
        "min_words_vertical": 3,          # Min words to identify vertical line
        "min_words_horizontal": 1,        # Min words to identify horizontal line
        "intersection_tolerance": 5,
        "text_tolerance": 3,
        "text_x_tolerance": 3,
        "text_y_tolerance": 3,
    }
    
    # Alternative settings for tables without any grid lines
    text_only_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 5,
        "join_tolerance": 5,
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
        "text_tolerance": 3,
        "text_x_tolerance": 5,
        "text_y_tolerance": 3,
    }
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Processing PDF: {pdf_path}")
        print(f"Total pages: {len(pdf.pages)}")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            print(f"\n--- Page {page_num} ---")
            
            # Try multiple extraction strategies
            tables = []
            
            # Strategy 1: Try with lines-based detection first
            try:
                tables = page.extract_tables(table_settings)
            except Exception:
                pass
            
            # Strategy 2: If no tables found, try text-only detection
            if not tables:
                try:
                    tables = page.extract_tables(text_only_settings)
                except Exception:
                    pass
            
            # Strategy 3: Fall back to default settings
            if not tables:
                tables = page.extract_tables()
            
            for table_idx, table in enumerate(tables):
                if table and len(table) > 1:  # Ensure table has data
                    # Clean and process the table
                    df = process_table(table)
                    
                    if df is not None and not df.empty:
                        extracted_tables.append({
                            "page": page_num,
                            "table_index": table_idx,
                            "dataframe": df
                        })
                        
                        # Save to CSV
                        csv_filename = f"page{page_num}_table{table_idx}.csv"
                        csv_path = os.path.join(output_dir, csv_filename)
                        df.to_csv(csv_path, index=False)
                        print(f"Saved: {csv_filename} ({len(df)} rows)")
                        
                        # Display preview
                        print(df.head().to_string())
    
    print(f"\nTotal tables extracted: {len(extracted_tables)}")
    return extracted_tables


def extract_transactions_from_pdf(pdf_path, output_dir="extracted_tables"):
    """
    Extract transaction data from bank statements by parsing text lines.
    Better for statements where table detection fails.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted data
    
    Returns:
        DataFrame with extracted transactions
    """
    os.makedirs(output_dir, exist_ok=True)
    all_transactions = []
    
    # Detect bank type from filename
    is_desjardins = 'desjardins' in pdf_path.lower()
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Extracting transactions from: {pdf_path}")
        
        # First pass: collect all text to extract year
        all_text_lines = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text_lines.extend(text.split('\n'))
        
        # Second pass: extract transactions from each page
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            
            if is_desjardins:
                # Parse Desjardins format (pass all lines for year extraction)
                transactions = parse_desjardins_transactions(lines, all_text_lines)
            else:
                # Parse Scotia/generic format (pass all lines for year extraction)
                transactions = parse_generic_transactions(lines, all_text_lines)
            
            all_transactions.extend(transactions)
    
    if all_transactions:
        df = pd.DataFrame(all_transactions)
        
        # Save to CSV
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        csv_path = os.path.join(output_dir, f"{base_name}_transactions.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} transactions to {csv_path}")
        
        return df
    
    return pd.DataFrame()


def parse_desjardins_transactions(lines, all_lines=None):
    """
    Parse Desjardins bank statement format.
    
    Format: J M J M Description BONIDOLLARS% Amount
    Example: 26 11 26 11 BKG*BOOKING.COM HOTEL (888)850-3958NH 2,00 % 293,23
    
    Args:
        lines: Lines from current page
        all_lines: All lines from document (for year extraction)
    """
    transactions = []
    
    # Use all_lines for year extraction if provided, otherwise use current page lines
    year_search_lines = all_lines if all_lines else lines
    
    # Pattern for Desjardins transaction lines
    # Matches: DD MM DD MM Description [percentage%] Amount
    desjardins_pattern = re.compile(
        r'^(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+'  # J M J M (trans date, post date)
        r'(.+?)\s+'                                            # Description
        r'([\d,]+(?:,\d{2})?)(?:CR)?$'                         # Amount (with comma decimal)
    )
    
    # Alternative pattern for lines with percentage (BONIDOLLARS)
    desjardins_pattern_pct = re.compile(
        r'^(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+'  # J M J M
        r'(.+?)\s+'                                            # Description
        r'(\d+,\d{2})\s*%\s+'                                  # Percentage
        r'([\d\s,]+(?:,\d{2})?)(?:CR)?$'                       # Amount
    )
    
    # Pattern for continuation lines (multiline descriptions like EURO TX:)
    continuation_pattern = re.compile(r'^[\d,]+\s+(?:EURO|USD|CAD)\s+TX:\s*[\d.]+$')
    
    # Extract year from statement
    # Look for patterns like "Année 2025", "relevé 04 12 2025"
    current_year = None
    year_patterns = [
        r'Année\s+(\d{4})',                          # Année 2025
        r'relevé\s+\d{1,2}\s+\d{1,2}\s+(\d{4})',     # relevé 04 12 2025
        r'Date du relevé.*?(\d{4})',                  # Date du relevé ... 2025
        r'd\'échéance.*?(\d{4})',                     # d'échéance ... 2025
    ]
    
    for line in year_search_lines:
        if current_year:
            break
        for pattern in year_patterns:
            match = re.search(pattern, line)
            if match:
                year_candidate = match.group(1)
                # Validate it's a reasonable year (2000-2099)
                if year_candidate.startswith('20'):
                    current_year = year_candidate
                    break
    
    # Fallback to current year if not found
    if not current_year:
        from datetime import datetime
        current_year = str(datetime.now().year)
    
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip header/footer lines
        if not line or 'TRANSACTIONS' in line or 'Page :' in line or 'Relevé' in line:
            i += 1
            continue
        
        # Try pattern with percentage first
        match = desjardins_pattern_pct.match(line)
        if match:
            trans_day, trans_month = match.group(1), match.group(2)
            post_day, post_month = match.group(3), match.group(4)
            description = match.group(5).strip()
            amount_str = match.group(7).replace(' ', '').replace(',', '.')
            
            # Check for continuation line (currency conversion info)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^[\d,\.]+\s+(?:EURO|USD|CAD)', next_line):
                    description += ' ' + next_line
                    i += 1
            
            try:
                amount = float(amount_str)
                if line.endswith('CR'):
                    amount = -amount
                
                transactions.append({
                    'date': f"{trans_day}/{trans_month}/{current_year}",
                    'post_date': f"{post_day}/{post_month}/{current_year}",
                    'description': description,
                    'amount': amount
                })
            except ValueError:
                pass
            
            i += 1
            continue
        
        # Try basic pattern (without percentage)
        match = desjardins_pattern.match(line)
        if match:
            trans_day, trans_month = match.group(1), match.group(2)
            post_day, post_month = match.group(3), match.group(4)
            description = match.group(5).strip()
            amount_str = match.group(6).replace(' ', '').replace(',', '.')
            
            # Check if it's a credit
            is_credit = line.endswith('CR')
            
            # Check for continuation line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^[\d,\.]+\s+(?:EURO|USD|CAD)', next_line):
                    description += ' ' + next_line
                    i += 1
            
            try:
                amount = float(amount_str)
                if is_credit:
                    amount = -amount
                
                transactions.append({
                    'date': f"{trans_day}/{trans_month}/{current_year}",
                    'post_date': f"{post_day}/{post_month}/{current_year}",
                    'description': description,
                    'amount': amount
                })
            except ValueError:
                pass
        
        i += 1
    
    return transactions


def parse_generic_transactions(lines, all_lines=None):
    """
    Parse generic bank statement formats (Scotia, Rogers, etc.).
    
    Supports various formats:
    - Nov 14 Nov 16 Description Amount (Scotia)
    - Aug23 Aug25 Description Amount (Rogers - no space)
    - Nov 14 Description Amount (single date)
    - 001 Nov 14 Nov 16 Description Amount (with ref number - ignored)
    
    Args:
        lines: Lines from current page
        all_lines: All lines from document (for year extraction)
    """
    transactions = []
    
    # Use all_lines for year extraction if provided, otherwise use current page lines
    year_search_lines = all_lines if all_lines else lines
    
    # Extract year from statement
    current_year = None
    year_patterns = [
        r'Statement\s*Period.*?(\d{4})',          # Statement Period ... 2025
        r'StatementPeriod.*?(\d{4})',             # StatementPeriod ... 2025
        r'[A-Za-z]{3}\s+\d{1,2},?\s*(\d{4})',     # Nov 3, 2025 or Aug 17,2025
        r'Due\s*Date.*?(\d{4})',                  # Due Date ... 2025
        r'Paymentduedate.*?(\d{4})',              # Paymentduedate Oct7,2025
    ]
    
    for line in year_search_lines:
        if current_year:
            break
        for pattern in year_patterns:
            match = re.search(pattern, line)
            if match:
                year_candidate = match.group(1)
                # Validate it's a reasonable year (2000-2099)
                if year_candidate and year_candidate.startswith('20') and len(year_candidate) == 4:
                    current_year = year_candidate
                    break
    
    # Fallback to current year if not found
    if not current_year:
        from datetime import datetime
        current_year = str(datetime.now().year)
    
    # Pattern for dates with space: Nov 14, Dec 1
    # Pattern for dates without space: Aug23, Sep2 (Rogers format)
    
    # Main pattern: two dates with optional space between month and day
    transaction_pattern = re.compile(
        r'^'
        r'(?:\d{3}\s+)?'                           # Optional ref number (ignored)
        r'([A-Za-z]{3})\s*(\d{1,2})\s+'            # First date (transaction date)
        r'([A-Za-z]{3})\s*(\d{1,2})\s+'            # Second date (post date)
        r'(.+?)\s+'                                 # Description
        r'\$?([\d,]+\.\d{2})\s*'                   # Amount
        r'(CR|-)?'                                  # Optional credit indicator
        r'\s*$'
    )
    
    # Pattern for single date
    simple_pattern = re.compile(
        r'^'
        r'(?:\d{3}\s+)?'                           # Optional ref number (ignored)
        r'([A-Za-z]{3})\s*(\d{1,2})\s+'            # Date
        r'(.+?)\s+'                                 # Description
        r'\$?([\d,]+\.\d{2})\s*'                   # Amount
        r'(CR|-)?'                                  # Optional credit indicator
        r'\s*$'
    )
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip header/footer lines
        skip_keywords = [
            'TRANS', 'Statement', 'Page', 'DATE', 'AMOUNT', 'DESCRIPTION',
            'Balance', 'TOTAL', 'SUB-TOTAL', 'Interest', 'Account', 'Card',
            'Rate', 'PURCHASE', 'CASH', 'Daily', 'Annual', 'Charged'
        ]
        if any(kw in line for kw in skip_keywords):
            continue
        
        # Try two-date pattern first (transaction date + post date)
        match = transaction_pattern.match(line)
        if match:
            trans_month, trans_day = match.group(1), match.group(2)
            post_month, post_day = match.group(3), match.group(4)
            description = match.group(5).strip()
            amount_str = match.group(6).replace(',', '')
            is_credit = match.group(7) in ('CR', '-')
            
            try:
                amount = float(amount_str)
                if is_credit:
                    amount = -amount
                
                transactions.append({
                    'date': f"{trans_month} {trans_day}, {current_year}",
                    'post_date': f"{post_month} {post_day}, {current_year}",
                    'description': description,
                    'amount': amount
                })
            except ValueError:
                pass
            continue
        
        # Try single-date pattern
        match = simple_pattern.match(line)
        if match:
            trans_month, trans_day = match.group(1), match.group(2)
            description = match.group(3).strip()
            amount_str = match.group(4).replace(',', '')
            is_credit = match.group(5) in ('CR', '-')
            
            try:
                amount = float(amount_str)
                if is_credit:
                    amount = -amount
                
                transactions.append({
                    'date': f"{trans_month} {trans_day}, {current_year}",
                    'post_date': None,
                    'description': description,
                    'amount': amount
                })
            except ValueError:
                pass
    
    return transactions


def process_table(raw_table):
    """
    Process raw table data: handle multiline rows, clean cells, and create DataFrame.
    
    Args:
        raw_table: List of lists representing table rows
    
    Returns:
        Cleaned pandas DataFrame
    """
    if not raw_table:
        return None
    
    # Clean individual cells
    cleaned_rows = []
    for row in raw_table:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                # Handle multiline cells by joining with space
                cell_text = str(cell).strip()
                # Replace newlines with spaces for multiline content
                cell_text = " ".join(cell_text.split())
                cleaned_row.append(cell_text)
        cleaned_rows.append(cleaned_row)
    
    if not cleaned_rows:
        return None
    
    # Merge multiline rows (rows that are continuations of previous rows)
    merged_rows = merge_multiline_rows(cleaned_rows)
    
    # Create DataFrame
    if len(merged_rows) > 1:
        # Use first row as header
        df = pd.DataFrame(merged_rows[1:], columns=merged_rows[0])
    else:
        df = pd.DataFrame(merged_rows)
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all')
    df = df.loc[:, (df != '').any(axis=0)]
    
    return df


def merge_multiline_rows(rows):
    """
    Merge rows that appear to be continuations of previous rows.
    Bank statements often have transaction descriptions spanning multiple lines.
    
    Args:
        rows: List of row lists
    
    Returns:
        List of merged rows
    """
    if not rows:
        return rows
    
    merged = []
    current_row = None
    
    for row in rows:
        # Check if this row looks like a continuation
        # (e.g., first cell is empty but others have content, 
        # or row has fewer non-empty cells than expected)
        is_continuation = is_continuation_row(row, current_row)
        
        if is_continuation and current_row is not None:
            # Merge with previous row
            for i, cell in enumerate(row):
                if i < len(current_row) and cell:
                    if current_row[i]:
                        current_row[i] = current_row[i] + " " + cell
                    else:
                        current_row[i] = cell
        else:
            # Start new row
            if current_row is not None:
                merged.append(current_row)
            current_row = row.copy()
    
    # Don't forget the last row
    if current_row is not None:
        merged.append(current_row)
    
    return merged


def is_continuation_row(row, previous_row):
    """
    Determine if a row is a continuation of the previous row.
    
    Heuristics for bank statements:
    - First column (usually date) is empty
    - Last columns (usually amounts) are empty
    - Only middle columns (description) have content
    """
    if previous_row is None:
        return False
    
    if not row or len(row) < 2:
        return False
    
    non_empty_count = sum(1 for cell in row if cell and cell.strip())
    
    # If first cell is empty and we have some content, might be continuation
    first_empty = not row[0] or not row[0].strip()
    
    # Check if amount columns (typically last 1-2) are empty
    last_cols_empty = all(not cell or not cell.strip() for cell in row[-2:]) if len(row) >= 2 else False
    
    # Likely continuation if date column empty and amount columns empty
    # but there's some content in the middle (description)
    if first_empty and last_cols_empty and non_empty_count > 0:
        return True
    
    # Very sparse row compared to previous might be continuation
    if non_empty_count <= 1 and non_empty_count < len(row) // 2:
        return True
    
    return False

if __name__ == "__main__":
    # Path to the PDF files
    pdf_files = [
        "./desjardins_december_2025.pdf",
        "./scotia_bank_december_2025.pdf",
        "./rogers_december_2025.pdf"
    ]
    
    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path}")
        print('='*60)
        
        # Method 1: Extract tables using table detection
        print("\n--- Table Extraction ---")
        tables = extract_tables_from_pdf(pdf_path)
        
        # Method 2: Extract transactions using text parsing
        print("\n--- Transaction Extraction ---")
        transactions_df = extract_transactions_from_pdf(pdf_path)
        
        if not transactions_df.empty:
            print("\nExtracted Transactions:")
            print(transactions_df.to_string())
    