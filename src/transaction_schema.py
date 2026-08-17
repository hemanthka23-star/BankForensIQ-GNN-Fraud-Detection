"""
RECONSTRUCTED STAND-IN — NOT YOUR REAL transaction_schema.py

Your actual transaction_schema.py was never uploaded to this chat (only
counterparty_extractor.py and csv_adapter.py were). This file was inferred
purely from the exact set of keyword arguments csv_adapter.py passes into
Transaction(...), so the pipeline in this zip can actually run end-to-end
against your real data. Field names/order match; any validation, defaults,
or extra methods your real class has do not.

Replace this file with your real transaction_schema.py before treating any
output as production-accurate — see README.md, section "Known gaps".
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Transaction:
    transaction_id: Optional[str]
    account_id: Optional[str]
    date: Optional[str]
    time: Optional[str]
    amount: Optional[float]
    direction: Optional[str]
    balance: Optional[float]
    transaction_type: Optional[str]
    payment_method: Optional[str]
    description: Optional[str]
    counterparty_name: Optional[str] = None
    counterparty_label: Optional[str] = None
    counterparty_identifier: Optional[str] = None
    counterparty_account: Optional[str] = None
    counterparty_upi: Optional[str] = None
    counterparty_bank: Optional[str] = None
    transaction_reference: Optional[str] = None
    reference_id: Optional[str] = None
    is_reversal: bool = False
    source_file: Optional[str] = None
    source_format: Optional[str] = None

    def to_dict(self):
        return asdict(self)
