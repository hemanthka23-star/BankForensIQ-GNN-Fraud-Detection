import csv
import re
from pathlib import Path
from typing import List, Optional

from transaction_schema import Transaction
from counterparty_extractor import extract_counterparty


def clean(value):
    if value is None:
        return None

    value = str(value).strip()

    return value if value else None


def number(value):
    if value is None:
        return None

    value = str(value).strip().replace(",", "")

    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def detect_format(lines):
    """
    Detect the CSV statement format from its first few lines.
    """

    text = "\n".join(lines[:15]).upper()

    if "AC_NO,AC_NAME,TRAN_ID" in text:
        return "ICORE"

    if "TRAN_DATE,CHQNO,PARTICULARS,DR,CR,BAL,SOL" in text:
        return "BANK_METADATA"

    if "TRAN-DATE" in text and "TRAN_PARTICULAR" in text:
        return "TABULAR"

    return "UNKNOWN"


def extract_account_from_metadata(lines):
    for line in lines[:15]:

        match = re.search(
            r"STATEMENT OF ACCOUNT NO\s*-\s*(\d+)",
            line,
            re.IGNORECASE
        )

        if match:
            return match.group(1)

    return None


def parse_bank_metadata(path, lines):
    account_id = extract_account_from_metadata(lines)

    header_index = None

    for i, line in enumerate(lines):
        if line.strip().upper().startswith(
            "TRAN_DATE,CHQNO,PARTICULARS,DR,CR,BAL,SOL"
        ):
            header_index = i
            break

    if header_index is None:
        raise ValueError("Transaction header not found")

    transactions = []

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:

        reader = csv.DictReader(
            f,
            fieldnames=[
                "TRAN_DATE",
                "CHQNO",
                "PARTICULARS",
                "DR",
                "CR",
                "BAL",
                "SOL",
            ],
            skipinitialspace=True,
        )

        for _ in range(header_index + 1):
            next(reader)

        for index, row in enumerate(reader):

            date = clean(row["TRAN_DATE"])
            description = clean(row["PARTICULARS"])

            if not date or not description:
                continue

            debit = number(row["DR"])
            credit = number(row["CR"])
            balance = number(row["BAL"])

            if debit is not None and debit != 0:
                amount = abs(debit)
                direction = "DEBIT"
            elif credit is not None and credit != 0:
                amount = abs(credit)
                direction = "CREDIT"
            else:
                continue

            cp = extract_counterparty(description)

            transaction_id = clean(row["CHQNO"])

            if not transaction_id or transaction_id == "-":
                transaction_id = f"{path.name}:{index}"

            transactions.append(
                Transaction(
                    transaction_id=transaction_id,
                    account_id=account_id or path.stem,
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
                    source_format="csv",
                )
            )

    return transactions


def parse_tabular(path, lines):
    """
    Parse tab-separated bank statements where:

    Line 1 = account metadata
    Line 2 = transaction header
    Remaining lines = transactions
    """

    if len(lines) < 3:
        return []

    # Account number is the first field of the first line.
    account_id = clean(lines[0].split("\t")[0])

    # Locate transaction header.
    header_index = None

    for i, line in enumerate(lines):
        upper = line.upper()

        if (
            "TRAN-DATE" in upper
            and "TRAN_PARTICULAR" in upper
            and "WITHDRAWAL" in upper
            and "DEPOSIT" in upper
            and "BALANCE" in upper
        ):
            header_index = i
            break

    if header_index is None:
        raise ValueError(
            f"Tabular transaction header not found in {path.name}"
        )

    headers = [
        clean(x).replace("\\_", "_")
        for x in lines[header_index].rstrip("\n").split("\t")
    ]

    transactions = []

    for index, line in enumerate(
        lines[header_index + 1:],
        start=header_index + 1
    ):
        fields = line.rstrip("\n").split("\t")

        if not any(x.strip() for x in fields):
            continue

        row = dict(zip(headers, fields))

        date = clean(row.get("TRAN-DATE"))
        description = clean(row.get("TRAN_PARTICULAR"))

        if description:
            description = (
                description
                .replace("\\:", ":")
                .replace("\\_", "_")
            )

        if not date or not description:
            continue

        debit = number(row.get("WITHDRAWAL"))
        credit = number(row.get("DEPOSIT"))
        balance = number(row.get("BALANCE"))

        if debit is not None and debit != 0:
            amount = abs(debit)
            direction = "DEBIT"

        elif credit is not None and credit != 0:
            amount = abs(credit)
            direction = "CREDIT"

        else:
            continue

        cp = extract_counterparty(description)

        transaction_id = clean(row.get("CHQ-NUM"))

        if not transaction_id:
            transaction_id = f"{path.name}:{index}"

        transactions.append(
            Transaction(
                transaction_id=transaction_id,
                account_id=account_id or path.stem,
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
                source_format="csv",
            )
        )

    return transactions

def parse_icore(path):
    transactions = []

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:

        reader = csv.DictReader(f)

        for index, row in enumerate(reader):

            account_id = clean(row.get("Ac_No"))
            date = clean(row.get("Tran_Date"))
            description = clean(row.get("Narration"))

            if not account_id or not date or not description:
                continue

            debit = number(row.get("Dr_Amt"))
            credit = number(row.get("Cr_Amt"))
            balance = number(row.get("Balance"))

            if debit is not None and debit != 0:
                amount = abs(debit)
                direction = "DEBIT"
            elif credit is not None and credit != 0:
                amount = abs(credit)
                direction = "CREDIT"
            else:
                continue

            cp = extract_counterparty(description)

            transaction_id = clean(row.get("Tran_ID"))

            if not transaction_id:
                transaction_id = f"{path.name}:{index}"

            transactions.append(
                Transaction(
                    transaction_id=transaction_id,
                    account_id=account_id,
                    date=date,
                    time=clean(row.get("pstd_dt")),
                    amount=amount,
                    direction=direction,
                    balance=balance,
                    transaction_type=clean(row.get("Tran_Type")),
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
                    source_format="csv",
                )
            )

    return transactions


def read_csv_statement(filepath: str) -> List[Transaction]:

    path = Path(filepath)

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    fmt = detect_format(lines)

    if fmt == "BANK_METADATA":
        return parse_bank_metadata(path, lines)

    if fmt == "TABULAR":
        return parse_tabular(path, lines)

    if fmt == "ICORE":
        return parse_icore(path)

    raise ValueError(
        f"Unknown CSV format: {path.name}"
    )


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/csv_adapter.py <file.csv>")
        raise SystemExit(1)

    transactions = read_csv_statement(sys.argv[1])

    print(f"Transactions: {len(transactions)}")

    for transaction in transactions[:5]:
        print(transaction)
