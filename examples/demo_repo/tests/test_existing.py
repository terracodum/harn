from datetime import datetime
from decimal import Decimal

from ledger.models import Transaction
from ledger.netting import net_balance


def test_payment_counts():
    now = datetime(2024, 1, 1, 23, 59, 59)
    txs = [Transaction("p1", "payment", Decimal("10"), settled_at=datetime(2024, 1, 1, 10, 0))]
    assert net_balance(txs, now) == Decimal("10")
