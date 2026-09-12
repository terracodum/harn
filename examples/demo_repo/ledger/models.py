from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

Kind = Literal["payment", "refund"]


@dataclass(frozen=True)
class Transaction:
    """Public DTO: field names and order are part of the API."""
    tx_id: str
    kind: Kind
    amount: Decimal
    settled_at: datetime | None = None
