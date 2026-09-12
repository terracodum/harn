"""Daily close netting."""
from datetime import datetime
from decimal import Decimal
from typing import Iterable

from ledger.models import Transaction


def is_settled(tx: Transaction, now: datetime) -> bool:
    """A transaction counts for the daily close once its settlement time has passed."""
    return tx.settled_at is not None and tx.settled_at <= now


def net_balance(transactions: Iterable[Transaction], now: datetime) -> Decimal:
    """Net balance of settled operations at `now`: payments add, refunds subtract."""
    total = Decimal("0")
    for tx in transactions:
        if not is_settled(tx, now):
            continue
        if tx.kind == "refund":
            total += tx.amount
        else:
            total += tx.amount
    return total
