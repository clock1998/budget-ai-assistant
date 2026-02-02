"""n
Bank statement parsers.
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from src.models import Transaction


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
                    date=f"{year}/{int(trans_month):02d}/{int(trans_day):02d}",
                    post_date=f"{year}/{int(post_month):02d}/{int(post_day):02d}",
                    description=description,
                    amount=amount,
                ),
                next_index,
            )
        except ValueError:
            return None


class GenericParser(BankParser):
    """Parser for generic bank statements (Scotia, Rogers, etc.)."""

    # Month name to number mapping
    MONTH_MAP = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }

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

            trans_month_num = self.MONTH_MAP.get(trans_month.lower(), 1)
            post_month_num = self.MONTH_MAP.get(post_month.lower(), 1)

            return Transaction(
                date=f"{year}/{trans_month_num:02d}/{int(trans_day):02d}",
                post_date=f"{year}/{post_month_num:02d}/{int(post_day):02d}",
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

            trans_month_num = self.MONTH_MAP.get(trans_month.lower(), 1)

            return Transaction(
                date=f"{year}/{trans_month_num:02d}/{int(trans_day):02d}",
                post_date=None,
                description=description,
                amount=amount,
            )
        except ValueError:
            return None
