"""
Data models for transaction extraction.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
