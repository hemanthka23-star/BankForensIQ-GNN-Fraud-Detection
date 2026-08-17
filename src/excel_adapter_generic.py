"""
GENERIC EXCEL ADAPTER — stand-in for a real excel_adapter.py

Your real excel_adapter.py (the one that produced the SOA/NEFT numbers
earlier in this project) was never uploaded to this chat — only
counterparty_extractor.py and csv_adapter.py were. This file is a
best-effort, independently-written replacement so the pipeline can run
end-to-end against your real .xlsx/.xls files.

It does NOT try to hardcode each bank's exact column layout. Instead, for
every sheet it scans the first `max_header_scan` rows, scores each row as
a candidate header by how many of {date, narration, debit, credit,
balance} columns it can identify via keyword matching, and parses from
the best-scoring row down. This is intentionally more general (and less
tuned) than a hand-written per-schema parser.

If you have your real excel_adapter.py, drop it into this same folder as
`excel_adapter.py` — run_pipeline.py imports that name FIRST and only
falls back to this generic version if it isn't present, so your real,
validated parser automatically takes over with no other changes needed.
"""

import re
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd

from transaction_schema import Transaction
from counterparty_extractor import extract_counterparty

warnings.filterwarnings("ignore")


DATE_KEYWORDS = [
    "TRAN DATE", "TRAN_DATE", "TXN_DATE", "TXN DT", "TRANSACTION DATE",
    "POST DATE", "DAT_TXN", "DATE",
]
DESC_KEYWORDS = [
    "NARRATION", "TRAN PARTICULAR", "TRAN_PARTICULAR", "PARTICULARS",
    "DESCRIPTION", "TRAN RMKS", "REMARKS", "TXT_TXN_DESC", "TXN_DESC",
    "DESC",
]
DEBIT_KEYWORDS = [
    "DEBIT AMOUNT", "DR_AMT", "WITHDRAWALS", "WITHDRAWAL", "DEBITS",
    "DEBIT", "WITHDRAWAL AMT",
]
CREDIT_KEYWORDS = [
    "CREDIT AMOUNT", "CR_AMT", "DEPOSITS", "DEPOSIT", "CREDITS",
    "CREDIT", "DEPOSIT AMT",
]
BALANCE_KEYWORDS = ["BALANCE AMOUNT", "BALANCE", "BAL"]
ACCOUNT_KEYWORDS = [
    "ACCOUNT NO", "ACCOUNT NUMBER", "AC_NO", "A/C NO", "ACC NO",
]
TXNID_KEYWORDS = [
    "TRAN_ID", "TRAN ID", "REF TXN NO", "REFERENCE NO", "CHEQUE NO",
    "CHQ", "REF CHQ NO", "CHEQUENO/REFERENCE", "REF_TXN_NO",
]
# Fallback for schemas with one combined amount column plus a separate
# debit/credit indicator column (e.g. AMT_TXN_LCY + COD_DRCR), instead
# of two separate debit/credit amount columns.
AMOUNT_KEYWORDS = ["AMT_TXN", "TXN_AMT", "TRAN AMT", "TXN AMOUNT", "AMOUNT"]
DRCR_KEYWORDS = ["COD_DRCR", "DR/CR", "CR/DR", "DRCR"]
SKIP_DESCRIPTIONS = {"OPENING BALANCE", "BROUGHT FORWARD", "CLOSING BALANCE"}

MIN_HEADER_SCORE = 3  # date + description + at least one of debit/credit


def _norm(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().upper())


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _number(value) -> Optional[float]:
    text = _clean(value)
    if text is None:
        return None
    text = re.sub(r"(CR|DR)$", "", text, flags=re.IGNORECASE).strip()
    text = text.replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _find_col(headers: List[str], keywords: List[str],
              exclude: Optional[List[str]] = None) -> Optional[str]:
    exclude = exclude or []
    for kw in keywords:
        for h in headers:
            if h == kw and not any(x in h for x in exclude):
                return h
    for kw in keywords:
        for h in headers:
            if kw in h and not any(x in h for x in exclude):
                return h
    return None


def _score_header_row(cells: List) -> Tuple[int, Dict[str, Optional[str]]]:
    headers = [_norm(c) for c in cells if c is not None and str(c).strip()]
    if len(headers) < 3:
        return 0, {}

    date_col = _find_col(headers, DATE_KEYWORDS, exclude=["VALUE"])
    if not date_col:
        date_col = _find_col(headers, DATE_KEYWORDS)

    cols = {
        "date": date_col,
        "desc": _find_col(headers, DESC_KEYWORDS),
        "debit": _find_col(headers, DEBIT_KEYWORDS),
        "credit": _find_col(headers, CREDIT_KEYWORDS),
        "balance": _find_col(headers, BALANCE_KEYWORDS),
        "account": _find_col(headers, ACCOUNT_KEYWORDS),
        "txnid": _find_col(headers, TXNID_KEYWORDS),
        "amount": _find_col(headers, AMOUNT_KEYWORDS),
        "drcr": _find_col(headers, DRCR_KEYWORDS),
    }
    has_split_amount = cols["debit"] is not None or cols["credit"] is not None
    has_combined_amount = cols["amount"] is not None and cols["drcr"] is not None

    score = sum(v is not None for k, v in cols.items() if k in ("date", "desc", "balance"))
    score += 1 if (has_split_amount or has_combined_amount) else 0
    return score, cols


def _account_from_preamble(raw: pd.DataFrame, header_row_idx: int) -> Optional[str]:
    for i in range(min(header_row_idx, 25)):
        row = raw.iloc[i]
        for j, val in enumerate(row.tolist()):
            if any(k in _norm(val) for k in ACCOUNT_KEYWORDS):
                for v in row.iloc[j + 1:j + 3].tolist():
                    v = _clean(v)
                    match = re.search(r"\d{6,}", v) if v else None
                    if match:
                        return match.group(0)
    return None


def read_excel_statement(filepath: str, max_header_scan: int = 60) -> List[Transaction]:
    path = Path(filepath)
    transactions: List[Transaction] = []

    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Could not open workbook ({type(e).__name__}: {e})")

    for sheet_name in xl.sheet_names:
        scan_rows = min(max_header_scan, 200)
        raw = pd.read_excel(
            xl, sheet_name=sheet_name, header=None, dtype=str,
            nrows=scan_rows,
        )
        if raw.empty:
            continue

        best_score, best_row, best_cols = 0, None, None
        for i in range(len(raw)):
            score, cols = _score_header_row(raw.iloc[i].tolist())
            if score > best_score:
                best_score, best_row, best_cols = score, i, cols

        if (
            best_row is None
            or best_score < MIN_HEADER_SCORE
            or not best_cols.get("date")
            or not best_cols.get("desc")
            or not (
                best_cols.get("debit") or best_cols.get("credit")
                or (best_cols.get("amount") and best_cols.get("drcr"))
            )
        ):
            continue

        account_id = _account_from_preamble(raw, best_row) or path.stem

        full = pd.read_excel(xl, sheet_name=sheet_name, header=best_row, dtype=str)
        full.columns = [_norm(c) for c in full.columns]

        date_col = best_cols["date"]
        desc_col = best_cols["desc"]
        debit_col = best_cols["debit"]
        credit_col = best_cols["credit"]
        balance_col = best_cols["balance"]
        txnid_col = best_cols.get("txnid")
        amount_col = best_cols.get("amount")
        drcr_col = best_cols.get("drcr")

        for index, row in full.iterrows():
            date = _clean(row.get(date_col))
            description = _clean(row.get(desc_col))

            if not date or not description:
                continue
            if description.upper() in SKIP_DESCRIPTIONS:
                continue

            debit = _number(row.get(debit_col)) if debit_col else None
            credit = _number(row.get(credit_col)) if credit_col else None

            if debit is not None and debit != 0:
                amount, direction = abs(debit), "DEBIT"
            elif credit is not None and credit != 0:
                amount, direction = abs(credit), "CREDIT"
            elif amount_col and drcr_col:
                combined = _number(row.get(amount_col))
                indicator = _clean(row.get(drcr_col))
                if combined is None or combined == 0 or not indicator:
                    continue
                indicator = indicator.strip().upper()
                if indicator.startswith("D"):
                    amount, direction = abs(combined), "DEBIT"
                elif indicator.startswith("C"):
                    amount, direction = abs(combined), "CREDIT"
                else:
                    continue
            else:
                continue

            balance = _number(row.get(balance_col)) if balance_col else None

            cp = extract_counterparty(description)

            txn_id = _clean(row.get(txnid_col)) if txnid_col else None
            if not txn_id or txn_id == "-":
                txn_id = f"{path.name}:{sheet_name}:{index}"

            transactions.append(
                Transaction(
                    transaction_id=txn_id,
                    account_id=account_id,
                    date=date,
                    time=None,
                    amount=amount,
                    direction=direction,
                    balance=balance,
                    transaction_type=None,
                    payment_method=cp["payment_method"],
                    description=description,
                    counterparty_name=cp["counterparty_name"],
                    counterparty_label=cp["counterparty_label"],
                    counterparty_identifier=cp["counterparty_identifier"],
                    counterparty_account=cp["counterparty_account"],
                    counterparty_upi=cp["counterparty_upi"],
                    counterparty_bank=cp["counterparty_bank"],
                    transaction_reference=cp["transaction_reference"],
                    reference_id=None,
                    is_reversal=cp["is_reversal"],
                    source_file=path.name,
                    source_format=f"xlsx_generic:{sheet_name}",
                )
            )

    if not transactions:
        raise ValueError("Could not find transaction header")

    return transactions


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/excel_adapter_generic.py <file.xlsx>")
        raise SystemExit(1)

    txns = read_excel_statement(sys.argv[1])
    print(f"Transactions: {len(txns)}")
    for t in txns[:5]:
        print(t)
