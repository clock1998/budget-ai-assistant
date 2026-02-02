"""
Transaction Extractor Module

Main class for extracting transactions from bank statement PDFs.
"""

import io

import pandas as pd
import pdfplumber
from fastapi import UploadFile

from pdf_extractor.models import BankType, Transaction
from pdf_extractor.parsers import DesjardinsParser, GenericParser


class TransactionExtractor:
    """
    Main class for extracting transactions from bank statement PDFs.
    
    Usage:
        extractor = TransactionExtractor()
        df = extractor.extract("statement.pdf")
        
        # From uploaded files
        df, bank_type = await extractor.extract_from_upload(file)
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

    def _extract(self, pdf_path: str) -> pd.DataFrame:
        """
        Extract transactions from a bank statement PDF.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            DataFrame containing extracted transactions.
        """
        with pdfplumber.open(pdf_path) as pdf:
            transactions, _ = self._extract_from_pdf(pdf)

        if transactions:
            return pd.DataFrame([t.to_dict() for t in transactions])

        return pd.DataFrame()

    async def extract_from_upload(
        self, file: UploadFile
    ) -> tuple[pd.DataFrame, BankType]:
        """
        Extract transactions from an uploaded file.
        
        Args:
            file: FastAPI UploadFile object.
            
        Returns:
            Tuple of (DataFrame, BankType).
        """
        content = await file.read()
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            transactions, bank_type = self._extract_from_pdf(pdf)

        if transactions:
            return pd.DataFrame([t.to_dict() for t in transactions]), bank_type

        return pd.DataFrame(), bank_type

    async def extract_from_uploads(
        self, files: list[UploadFile]
    ) -> list[tuple[str, pd.DataFrame, BankType]]:
        """
        Extract transactions from multiple uploaded files.
        
        Args:
            files: List of FastAPI UploadFile objects.
            
        Returns:
            List of tuples (filename, DataFrame, BankType) for each file.
        """
        results = []
        for file in files:
            df, bank_type = await self.extract_from_upload(file)
            results.append((file.filename, df, bank_type))
        return results

    def _extract_from_pdf(self, pdf) -> tuple[list[Transaction], BankType]:
        """Extract transactions from an open PDF object."""
        all_transactions = []
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

        return all_transactions, bank_type

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
