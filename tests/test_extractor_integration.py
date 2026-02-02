"""
Integration tests for TransactionExtractor using actual PDF files.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_extractor.models import BankType
from pdf_extractor.extractor import TransactionExtractor


# Path to project root where PDF files are located
PROJECT_ROOT = Path(__file__).parent.parent


class TestTransactionExtractorIntegration:
    """Integration tests using real PDF files."""

    @pytest.fixture
    def extractor(self):
        """Create a TransactionExtractor instance."""
        return TransactionExtractor()

    @pytest.fixture
    def desjardins_pdf(self):
        """Path to Desjardins PDF file."""
        path = PROJECT_ROOT / "desjardins_december_2025.pdf"
        if not path.exists():
            pytest.skip(f"PDF file not found: {path}")
        return str(path)

    @pytest.fixture
    def scotia_pdf(self):
        """Path to Scotia Bank PDF file."""
        path = PROJECT_ROOT / "scotia_bank_december_2025.pdf"
        if not path.exists():
            pytest.skip(f"PDF file not found: {path}")
        return str(path)

    @pytest.fixture
    def rogers_pdf(self):
        """Path to Rogers PDF file."""
        path = PROJECT_ROOT / "rogers_december_2025.pdf"
        if not path.exists():
            pytest.skip(f"PDF file not found: {path}")
        return str(path)

    def test_extract_desjardins(self, extractor, desjardins_pdf):
        """Test extraction from Desjardins PDF."""
        df = extractor._extract(desjardins_pdf)

        assert not df.empty, "Should extract transactions from Desjardins PDF"
        assert "date" in df.columns
        assert "description" in df.columns
        assert "amount" in df.columns
        assert "post_date" in df.columns

        # Verify date format YYYY/MM/DD
        for date in df["date"]:
            assert len(date.split("/")) == 3, f"Invalid date format: {date}"
            year, month, day = date.split("/")
            assert len(year) == 4, f"Year should be 4 digits: {year}"
            assert len(month) == 2, f"Month should be 2 digits: {month}"
            assert len(day) == 2, f"Day should be 2 digits: {day}"

        print(f"\nDesjardins: Extracted {len(df)} transactions")

    def test_extract_scotia(self, extractor, scotia_pdf):
        """Test extraction from Scotia Bank PDF."""
        df = extractor._extract(scotia_pdf)

        assert not df.empty, "Should extract transactions from Scotia PDF"
        assert "date" in df.columns
        assert "description" in df.columns
        assert "amount" in df.columns

        print(f"\nScotia: Extracted {len(df)} transactions")

    def test_extract_rogers(self, extractor, rogers_pdf):
        """Test extraction from Rogers PDF."""
        df = extractor._extract(rogers_pdf)

        assert not df.empty, "Should extract transactions from Rogers PDF"
        assert "date" in df.columns
        assert "description" in df.columns
        assert "amount" in df.columns

        print(f"\nRogers: Extracted {len(df)} transactions")

    def test_detect_bank_type_desjardins(self, extractor, desjardins_pdf):
        """Test bank type detection for Desjardins."""
        import pdfplumber

        with pdfplumber.open(desjardins_pdf) as pdf:
            _, bank_type = extractor._extract_from_pdf(pdf)

        assert bank_type == BankType.DESJARDINS, f"Expected DESJARDINS, got {bank_type}"

    def test_detect_bank_type_scotia(self, extractor, scotia_pdf):
        """Test bank type detection for Scotia."""
        import pdfplumber

        with pdfplumber.open(scotia_pdf) as pdf:
            _, bank_type = extractor._extract_from_pdf(pdf)

        assert bank_type == BankType.SCOTIA, f"Expected SCOTIA, got {bank_type}"

    def test_detect_bank_type_rogers(self, extractor, rogers_pdf):
        """Test bank type detection for Rogers."""
        import pdfplumber

        with pdfplumber.open(rogers_pdf) as pdf:
            _, bank_type = extractor._extract_from_pdf(pdf)

        assert bank_type == BankType.ROGERS, f"Expected ROGERS, got {bank_type}"

    def test_all_amounts_are_numeric(self, extractor, desjardins_pdf, scotia_pdf, rogers_pdf):
        """Test that all amounts are valid numbers."""
        for pdf_path in [desjardins_pdf, scotia_pdf, rogers_pdf]:
            df = extractor._extract(pdf_path)
            
            for amount in df["amount"]:
                assert isinstance(amount, (int, float)), f"Amount should be numeric: {amount}"

    def test_extract_returns_dataframe(self, extractor, desjardins_pdf):
        """Test that _extract returns a pandas DataFrame."""
        import pandas as pd

        result = extractor._extract(desjardins_pdf)
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
