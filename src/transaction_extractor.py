"""
Transaction Extractor Module

Extracts transaction data from bank statement PDFs (Desjardins, Scotia, Rogers).
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd
import pdfplumber


class BankType(Enum):
    """Supported bank types."""
    DESJARDINS = "desjardins"
    SCOTIA = "scotia"
    ROGERS = "rogers"
    UNKNOWN = "unknown"


@dataclass
class Transaction:
    """Represents a single transaction."""
    date: str
    description: str
    amount: float
    post_date: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert transaction to dictionary."""
        return {
            "date": self.date,
            "post_date": self.post_date,
            "description": self.description,
            "amount": self.amount,
        }


class BankParser(ABC):
    """Abstract base class for bank-specific parsers."""

    @abstractmethod
    def parse(self, lines: list[str], all_lines: list[str]) -> list[Transaction]:
        """Parse transaction lines and return list of transactions."""
        pass

    @abstractmethod
    def extract_year(self, lines: list[str]) -> str:
        """Extract statement year from document lines."""
        pass

    def _get_fallback_year(self) -> str:
        """Return current year as fallback."""
        return str(datetime.now().year)


class DesjardinsParser(BankParser):
    """Parser for Desjardins bank statements."""

    # Pattern for transaction lines with percentage (BONIDOLLARS)
    PATTERN_WITH_PCT = re.compile(
        r'^(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+'  # J M J M
        r'(.+?)\s+'                                            # Description
        r'(\d+,\d{2})\s*%\s+'                                  # Percentage
        r'([\d\s,]+(?:,\d{2})?)(?:CR)?$'                       # Amount
    )

    # Pattern for basic transaction lines
    PATTERN_BASIC = re.compile(
        r'^(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+'  # J M J M
        r'(.+?)\s+'                                            # Description
        r'([\d,]+(?:,\d{2})?)(?:CR)?$'                         # Amount
    )

    # Currency continuation pattern
    CURRENCY_PATTERN = re.compile(r'^[\d,\.]+\s+(?:EURO|USD|CAD)')

    YEAR_PATTERNS = [
        r'Année\s+(\d{4})',
        r'relevé\s+\d{1,2}\s+\d{1,2}\s+(\d{4})',
        r'Date du relevé.*?(\d{4})',
        r'd\'échéance.*?(\d{4})',
    ]

    SKIP_KEYWORDS = ['TRANSACTIONS', 'Page :', 'Relevé']

    def extract_year(self, lines: list[str]) -> str:
        """Extract year from Desjardins statement."""
        for line in lines:
            for pattern in self.YEAR_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    year = match.group(1)
                    if year.startswith('20'):
                        return year
        return self._get_fallback_year()

    def parse(self, lines: list[str], all_lines: list[str]) -> list[Transaction]:
        """Parse Desjardins format transactions."""
        transactions = []
        year = self.extract_year(all_lines or lines)
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line or any(kw in line for kw in self.SKIP_KEYWORDS):
                i += 1
                continue

            transaction = self._try_parse_line(line, lines, i, year)
            if transaction:
                transactions.append(transaction[0])
                i = transaction[1]
            else:
                i += 1

        return transactions

    def _try_parse_line(
        self, line: str, lines: list[str], index: int, year: str
    ) -> Optional[tuple[Transaction, int]]:
        """Try to parse a single line as a transaction."""
        # Try pattern with percentage first
        match = self.PATTERN_WITH_PCT.match(line)
        if match:
            return self._create_transaction(
                match, lines, index, year, amount_group=7
            )

        # Try basic pattern
        match = self.PATTERN_BASIC.match(line)
        if match:
            return self._create_transaction(
                match, lines, index, year, amount_group=6
            )

        return None

    def _create_transaction(
        self, match, lines: list[str], index: int, year: str, amount_group: int
    ) -> tuple[Transaction, int]:
        """Create transaction from regex match."""
        trans_day, trans_month = match.group(1), match.group(2)
        post_day, post_month = match.group(3), match.group(4)
        description = match.group(5).strip()
        amount_str = match.group(amount_group).replace(' ', '').replace(',', '.')

        # Check for continuation line
        next_index = index + 1
        if next_index < len(lines):
            next_line = lines[next_index].strip()
            if self.CURRENCY_PATTERN.match(next_line):
                description += ' ' + next_line
                next_index += 1

        try:
            amount = float(amount_str)
            if lines[index].strip().endswith('CR'):
                amount = -amount

            return (
                Transaction(
                    date=f"{trans_day}/{trans_month}/{year}",
                    post_date=f"{post_day}/{post_month}/{year}",
                    description=description,
                    amount=amount,
                ),
                next_index,
            )
        except ValueError:
            return None


class GenericParser(BankParser):
    """Parser for generic bank statements (Scotia, Rogers, etc.)."""

    # Two-date pattern (transaction + post date)
    PATTERN_TWO_DATES = re.compile(
        r'^(?:\d{3}\s+)?'                           # Optional ref number
        r'([A-Za-z]{3})\s*(\d{1,2})\s+'             # First date
        r'([A-Za-z]{3})\s*(\d{1,2})\s+'             # Second date
        r'(.+?)\s+'                                  # Description
        r'\$?([\d,]+\.\d{2})\s*'                    # Amount
        r'(CR|-)?'                                   # Credit indicator
        r'\s*$'
    )

    # Single date pattern
    PATTERN_SINGLE_DATE = re.compile(
        r'^(?:\d{3}\s+)?'                           # Optional ref number
        r'([A-Za-z]{3})\s*(\d{1,2})\s+'             # Date
        r'(.+?)\s+'                                  # Description
        r'\$?([\d,]+\.\d{2})\s*'                    # Amount
        r'(CR|-)?'                                   # Credit indicator
        r'\s*$'
    )

    YEAR_PATTERNS = [
        r'Statement\s*Period.*?(\d{4})',
        r'StatementPeriod.*?(\d{4})',
        r'[A-Za-z]{3}\s+\d{1,2},?\s*(\d{4})',
        r'Due\s*Date.*?(\d{4})',
        r'Paymentduedate.*?(\d{4})',
    ]

    SKIP_KEYWORDS = [
        'TRANS', 'Statement', 'Page', 'DATE', 'AMOUNT', 'DESCRIPTION',
        'Balance', 'TOTAL', 'SUB-TOTAL', 'Interest', 'Account', 'Card',
        'Rate', 'PURCHASE', 'CASH', 'Daily', 'Annual', 'Charged'
    ]

    def extract_year(self, lines: list[str]) -> str:
        """Extract year from generic bank statement."""
        for line in lines:
            for pattern in self.YEAR_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    year = match.group(1)
                    if year and year.startswith('20') and len(year) == 4:
                        return year
        return self._get_fallback_year()

    def parse(self, lines: list[str], all_lines: list[str]) -> list[Transaction]:
        """Parse generic format transactions."""
        transactions = []
        year = self.extract_year(all_lines or lines)

        for line in lines:
            line = line.strip()
            if not line or any(kw in line for kw in self.SKIP_KEYWORDS):
                continue

            transaction = self._try_parse_line(line, year)
            if transaction:
                transactions.append(transaction)

        return transactions

    def _try_parse_line(self, line: str, year: str) -> Optional[Transaction]:
        """Try to parse a single line as a transaction."""
        # Try two-date pattern first
        match = self.PATTERN_TWO_DATES.match(line)
        if match:
            return self._create_two_date_transaction(match, year)

        # Try single-date pattern
        match = self.PATTERN_SINGLE_DATE.match(line)
        if match:
            return self._create_single_date_transaction(match, year)

        return None

    def _create_two_date_transaction(self, match, year: str) -> Optional[Transaction]:
        """Create transaction with two dates."""
        trans_month, trans_day = match.group(1), match.group(2)
        post_month, post_day = match.group(3), match.group(4)
        description = match.group(5).strip()
        amount_str = match.group(6).replace(',', '')
        is_credit = match.group(7) in ('CR', '-')

        try:
            amount = float(amount_str)
            if is_credit:
                amount = -amount

            return Transaction(
                date=f"{trans_month} {trans_day}, {year}",
                post_date=f"{post_month} {post_day}, {year}",
                description=description,
                amount=amount,
            )
        except ValueError:
            return None

    def _create_single_date_transaction(self, match, year: str) -> Optional[Transaction]:
        """Create transaction with single date."""
        trans_month, trans_day = match.group(1), match.group(2)
        description = match.group(3).strip()
        amount_str = match.group(4).replace(',', '')
        is_credit = match.group(5) in ('CR', '-')

        try:
            amount = float(amount_str)
            if is_credit:
                amount = -amount

            return Transaction(
                date=f"{trans_month} {trans_day}, {year}",
                post_date=None,
                description=description,
                amount=amount,
            )
        except ValueError:
            return None


class TransactionExtractor:
    """
    Main class for extracting transactions from bank statement PDFs.
    
    Usage:
        extractor = TransactionExtractor()
        df = extractor.extract("statement.pdf")
        
        # Or with custom output directory
        extractor = TransactionExtractor(output_dir="my_output")
        df = extractor.extract("statement.pdf", save_csv=True)
    """

    # Bank detection indicators
    BANK_INDICATORS = {
        BankType.DESJARDINS: [
            'desjardins', 'caisse populaire', 'mouvement desjardins',
            'bonidollars', 'relevé de compte', 'accès d',
        ],
        BankType.SCOTIA: [
            'scotiabank', 'scotia bank', 'bank of nova scotia',
            'scene+', 'scenepoints',
        ],
        BankType.ROGERS: [
            'rogers bank', 'rogersbank', 'rogers mastercard',
            'rogers world elite',
        ],
    }

    def __init__(self):
        """Initialize the transaction extractor."""
        self._parsers = {
            BankType.DESJARDINS: DesjardinsParser(),
            BankType.SCOTIA: GenericParser(),
            BankType.ROGERS: GenericParser(),
            BankType.UNKNOWN: GenericParser(),
        }

    def detect_bank_type(self, pdf_text: str) -> BankType:
        """
        Detect the bank type from PDF content.
        
        Args:
            pdf_text: Concatenated text from all pages of the PDF.
            
        Returns:
            BankType enum value.
        """
        text_lower = pdf_text.lower()
        scores = {}

        for bank_type, indicators in self.BANK_INDICATORS.items():
            scores[bank_type] = sum(
                1 for indicator in indicators if indicator in text_lower
            )

        # Find bank with highest score
        best_bank = max(scores, key=scores.get)
        if scores[best_bank] > 0:
            return best_bank

        return BankType.UNKNOWN

    def extract(self, pdf_path: str) -> pd.DataFrame:
        """
        Extract transactions from a bank statement PDF.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            DataFrame containing extracted transactions.
        """
        all_transactions = []

        with pdfplumber.open(pdf_path) as pdf:
            # First pass: collect all text
            all_lines, full_text = self._extract_text(pdf)

            # Detect bank type
            bank_type = self.detect_bank_type(full_text)
            parser = self._parsers[bank_type]

            # Second pass: extract transactions
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                lines = text.split('\n')
                transactions = parser.parse(lines, all_lines)
                all_transactions.extend(transactions)

        # Create DataFrame
        if all_transactions:
            return pd.DataFrame([t.to_dict() for t in all_transactions])

        return pd.DataFrame()

    def extract_with_info(self, pdf_path: str) -> tuple[pd.DataFrame, BankType]:
        """
        Extract transactions and return bank type information.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Tuple of (DataFrame, BankType).
        """
        all_transactions = []

        with pdfplumber.open(pdf_path) as pdf:
            all_lines, full_text = self._extract_text(pdf)
            bank_type = self.detect_bank_type(full_text)
            parser = self._parsers[bank_type]

            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                lines = text.split('\n')
                transactions = parser.parse(lines, all_lines)
                all_transactions.extend(transactions)

        if all_transactions:
            return pd.DataFrame([t.to_dict() for t in all_transactions]), bank_type

        return pd.DataFrame(), bank_type

    def _extract_text(self, pdf) -> tuple[list[str], str]:
        """Extract all text from PDF pages."""
        all_lines = []
        full_text = ""

        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_lines.extend(text.split('\n'))
                full_text += text + "\n"

        return all_lines, full_text


# Convenience function for quick extraction
def extract_transactions(pdf_path: str) -> pd.DataFrame:
    """
    Convenience function to extract transactions from a PDF.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        DataFrame containing extracted transactions.
    """
    extractor = TransactionExtractor()
    return extractor.extract(pdf_path)


if __name__ == "__main__":
    import os
    
    # Example usage
    pdf_files = [
        "./desjardins_december_2025.pdf",
        "./scotia_bank_december_2025.pdf",
        "./rogers_december_2025.pdf",
    ]

    extractor = TransactionExtractor()

    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"Skipping {pdf_path} (file not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path}")
        print('='*60)

        df, bank_type = extractor.extract_with_info(pdf_path)

        print(f"Detected bank: {bank_type.value}")
        print(f"Transactions found: {len(df)}")

        if not df.empty:
            print("\nExtracted Transactions:")
            print(df.to_string())
