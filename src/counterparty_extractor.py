import re
from typing import Optional, Dict


UPI_PATTERN = re.compile(
    r'([A-Za-z0-9._-]+@[A-Za-z0-9._-]+)',
    re.IGNORECASE
)

UPI_BANK_CODES = {
    "ybl": "YES_BANK",
    "ibl": "IDBI_BANK",
    "oksbi": "SBI",
    "okaxis": "AXIS_BANK",
    "okicici": "ICICI_BANK",
    "okhdfcbank": "HDFC_BANK",
    "okhdfc": "HDFC_BANK",
    "okpaytm": "PAYTM",
    "paytm": "PAYTM",
}


BANK_CODES = {
    "FDRL": "FEDERAL_BANK",
    "BARB": "BANK_OF_BARODA",
    "UTIB": "AXIS_BANK",
    "IDFB": "IDFC_FIRST_BANK",
    "ICIC": "ICICI_BANK",
    "KARB": "KARNATAKA_BANK",
    "IOBA": "INDIAN_OVERSEAS_BANK",
    "IDIB": "INDIAN_BANK",
    "HDFC": "HDFC_BANK",
    "YESB": "YES_BANK",
    "SBI": "SBI",
    "CNRB": "CANARA_BANK",
    "KKBK": "KOTAK_MAHINDRA_BANK",
    "AIRP": "AIRTEL_PAYMENTS_BANK",
    "DCBL": "DCB_BANK",
    "CBIN": "CENTRAL_BANK_OF_INDIA",
    "SBIN": "SBI",
    "KOMA": "KOTAK_MAHINDRA_BANK",
}


# Canonical full bank names, used to match the free-text (and often
# truncated) bank names seen in the generic UPI format, e.g.
# "State Bank Of" or "Bank of Barod" or "AU small Fina". Matched by
# prefix in either direction (see detect_bank_full_name) rather than
# substring, since these are truncated at arbitrary points.
UPI_BANK_FULL_NAMES = {
    "STATE BANK OF INDIA": "SBI",
    "HDFC BANK": "HDFC_BANK",
    "ICICI BANK": "ICICI_BANK",
    "AXIS BANK": "AXIS_BANK",
    "UCO BANK": "UCO_BANK",
    "BANK OF BARODA": "BANK_OF_BARODA",
    "BANDHAN BANK": "BANDHAN_BANK",
    "AU SMALL FINANCE BANK": "AU_SMALL_FINANCE_BANK",
    "KOTAK MAHINDRA BANK": "KOTAK_MAHINDRA_BANK",
    "IDBI BANK": "IDBI_BANK",
    "IDFC FIRST BANK": "IDFC_FIRST_BANK",
    "CANARA BANK": "CANARA_BANK",
    "INDIAN OVERSEAS BANK": "INDIAN_OVERSEAS_BANK",
    "INDIAN BANK": "INDIAN_BANK",
    "FEDERAL BANK": "FEDERAL_BANK",
    "YES BANK": "YES_BANK",
    "KARNATAKA BANK": "KARNATAKA_BANK",
    "AIRTEL PAYMENTS BANK": "AIRTEL_PAYMENTS_BANK",
    "PAYTM PAYMENTS BANK": "PAYTM",
    "DCB BANK": "DCB_BANK",
    "CENTRAL BANK OF INDIA": "CENTRAL_BANK_OF_INDIA",
    "UNION BANK OF INDIA": "UNION_BANK_OF_INDIA",
    "PUNJAB NATIONAL BANK": "PUNJAB_NATIONAL_BANK",
}


# Free-text bank name fragments seen in colon-delimited UPI/IMPS
# descriptions (e.g. "Kotak Mahi", "Union Bank"). These are truncated
# display names rather than IFSC codes, so they're matched separately
# by substring rather than looked up in BANK_CODES.
BANK_NAME_KEYWORDS = {
    "STATE BANK": "SBI",
    "KOTAK": "KOTAK_MAHINDRA_BANK",
    "UNION BANK": "UNION_BANK_OF_INDIA",
    "HDFC": "HDFC_BANK",
    "ICICI": "ICICI_BANK",
    "AXIS": "AXIS_BANK",
    "YES BANK": "YES_BANK",
    "FEDERAL": "FEDERAL_BANK",
    "CANARA": "CANARA_BANK",
    "INDIAN OVERSEAS": "INDIAN_OVERSEAS_BANK",
    "INDIAN BANK": "INDIAN_BANK",
    "BANK OF BARODA": "BANK_OF_BARODA",
    "IDFC": "IDFC_FIRST_BANK",
    "IDBI": "IDBI_BANK",
    "KARNATAKA": "KARNATAKA_BANK",
    "AIRTEL": "AIRTEL_PAYMENTS_BANK",
    "PAYTM": "PAYTM",
}


def extract_upi(text: str) -> Optional[str]:
    if not text:
        return None

    match = UPI_PATTERN.search(text)

    if match:
        return match.group(1).lower()

    return None


def detect_upi_bank(upi: Optional[str]) -> Optional[str]:
    if not upi or "@" not in upi:
        return None

    suffix = upi.rsplit("@", 1)[1].lower()

    return UPI_BANK_CODES.get(suffix)


def extract_transaction_reference(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r'UPI/(\d{8,20})',
        r'IMPS/P2A/(\d{8,20})',
        r'UPI:(?:PAY|REC):(\d{8,20})',
        r'IMPS:(?:PAY|REC):(\d{8,20})',
        r'NEFT/([A-Za-z0-9]+)(?=/)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1)

    return None


def detect_payment_method(text: str) -> Optional[str]:
    if not text:
        return None

    upper = text.upper()

    if "UPI/" in upper or "UPI:" in upper:
        return "UPI"

    if "IMPS/P2A/" in upper or "IMPS:" in upper:
        return "IMPS"

    if "NEFT" in upper:
        return "NEFT"

    if "RTGS" in upper:
        return "RTGS"

    if "LOAN DISBURSEMENT" in upper:
        return "LOAN"

    return None


def detect_reversal(text: str) -> bool:
    if not text:
        return False

    return text.strip().upper().startswith("REVERSED")


def extract_imps_counterparty(text: str) -> Optional[str]:
    """
    Extract the descriptive counterparty portion of an IMPS/P2A
    transaction.

    Examples:
        IMPS/P2A/509020012550//Nearby Technologies Private
        -> Nearby Technologies Private

        IMPS/P2A/514016777934/ICIC/SAVY INFOLINE PVT LT
        -> SAVY INFOLINE PVT LT
    """

    if not text:
        return None

    match = re.search(
        r'IMPS/P2A/\d{8,20}/([^/]*)/(.+)$',
        text,
        re.IGNORECASE
    )

    if not match:
        # Handle the case with an empty bank-code field.
        match = re.search(
            r'IMPS/P2A/\d{8,20}//(.+)$',
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip() or None

        return None

    value = match.group(2).strip()

    return value or None


def extract_upi_label(text: str) -> Optional[str]:
    """
    Extract the short participant label found between the UPI
    transaction reference/direction and the bank code.

    Example:
        UPI/546060109469/DR/HARI/YESB/vaishali@ybl/UPI
        -> HARI
    """

    if not text:
        return None

    clean = re.sub(
        r'^REVERSED\s*:\s*',
        '',
        text.strip(),
        flags=re.IGNORECASE
    )

    match = re.search(
        r'UPI/\d{8,20}/(?:CR|DR)/([^/]+)/([A-Za-z0-9]+)(?:/|$)',
        clean,
        re.IGNORECASE
    )

    if not match:
        return None

    label = match.group(1).strip()

    if not label:
        return None

    if label.upper() in {"MR", "MRS", "MS", "DR"}:
        return None

    return label.upper()


def detect_bank_code(text: str) -> Optional[str]:
    if not text:
        return None

    upper = text.upper()

    for code, bank in BANK_CODES.items():
        if f"/{code}/" in upper:
            return bank

    return None


def extract_neft_counterparty(text: str):
    """
    Extract the short sender code and the full counterparty name from
    an NEFT transaction.

    Examples:
        MB NEFT/DCBL427418943347/RITU/VIKRAM DUBEY/
        -> ("RITU", "VIKRAM DUBEY")

        NEFT/YESB42880670696/REKH/RESILIENT INNOVATIONS PR
        -> ("REKH", "RESILIENT INNOVATIONS PR")

    Returns a (code, name) tuple; either element may be None.
    """

    if not text:
        return None, None

    match = re.search(
        r'NEFT/[A-Za-z0-9]+/([A-Za-z]{2,10})/(.+)$',
        text,
        re.IGNORECASE
    )

    if not match:
        return None, None

    code = match.group(1).strip().upper() or None
    name = match.group(2).strip().rstrip("/").strip() or None

    return code, name


def extract_colon_transaction_details(text: str):
    """
    Extract counterparty details from colon-delimited UPI/IMPS
    descriptions.

    Examples:
        UPI:PAY:426036781143/NIDHI PILLAI/Kotak Mahi
        -> ("NIDHI PILLAI", "KOTAK_MAHINDRA_BANK", None)

        UPI:REC:427481674406/PALLAVI NAIR/Union Bank
        -> ("PALLAVI NAIR", "UNION_BANK_OF_INDIA", None)

        IMPS:PAY:427413593157/Vikram Da/KARB/XX7501/1 rupe
        -> ("Vikram Da", "KARNATAKA_BANK", "XX7501")

    Returns a (name, bank, account) tuple; any element may be None.
    Trailing free-text notes (e.g. "1 rupe") are ignored rather than
    guessed at.
    """

    if not text:
        return None, None, None

    match = re.search(
        r'(?:UPI|IMPS):(?:PAY|REC):\d{8,20}/([^/]+)(?:/(.*))?$',
        text,
        re.IGNORECASE
    )

    if not match:
        return None, None, None

    name = match.group(1).strip() or None
    remainder = match.group(2) or ""

    account = None
    bank = None

    for segment in (s.strip() for s in remainder.split("/")):
        if not segment:
            continue

        upper = segment.upper()

        if re.fullmatch(r'X{1,4}\d{3,6}', upper):
            account = segment
            continue

        if upper in BANK_CODES:
            bank = BANK_CODES[upper]
            continue

        keyword_bank = detect_bank_name_keyword(segment)

        if keyword_bank:
            bank = keyword_bank

    return name, bank, account


def detect_bank_name_keyword(segment: str) -> Optional[str]:
    if not segment:
        return None

    upper = segment.upper()

    for keyword, bank in BANK_NAME_KEYWORDS.items():
        if keyword in upper:
            return bank

    return None


def detect_bank_from_reference(reference: Optional[str]) -> Optional[str]:
    """
    Some NEFT UTR references lead with the sending bank's IFSC
    prefix (e.g. "YESB42880670696" -> YESB). Only matched against
    the known BANK_CODES table, so unrecognized prefixes return None
    rather than a guess.
    """

    if not reference or len(reference) < 4:
        return None

    return BANK_CODES.get(reference[:4].upper())


def detect_bank_full_name(segment: str) -> Optional[str]:
    """
    Match a free-text bank name fragment that may be truncated at an
    arbitrary point (e.g. "State Bank Of", "Bank of Barod", "AU small
    Fina") against known full bank names, by prefix in either
    direction. Requires at least 3 characters to avoid matching on
    near-empty fragments.
    """

    if not segment or len(segment) < 3:
        return None

    upper = segment.upper()

    best_match = None

    for canonical, bank in UPI_BANK_FULL_NAMES.items():
        if canonical.startswith(upper) or upper.startswith(canonical):
            if best_match is None or len(canonical) > len(best_match[0]):
                best_match = (canonical, bank)

    return best_match[1] if best_match else None


def extract_upi_participant(text: str):
    """
    Fallback UPI parser for descriptions that don't follow the CR/DR
    convention handled by extract_upi_label, e.g.:

        UPI/324743072944/Payment from Ph/9316670178@ybl/UC
        UPI/361351942465/UPI/Mrs ADITYA Neh/State Bank Of
        UPI/325090800106/UPI/dhokaiankit59-2/UCO Bank/HDF6

    The segment right after the reference/note field is either a
    full UPI handle (has "@" - returned as label, since extract_upi
    already captures the handle itself elsewhere), a bare/incomplete
    handle fragment (has "@" but no full domain - also returned as
    label, verbatim, not guessed at), or a human name/username
    (returned as name). The following segment, if present, is
    matched against known bank names (see detect_bank_full_name).
    Any further trailing segments (e.g. "HDF6" above) are ignored as
    unreliable truncation noise rather than guessed at.

    Returns (name, label, bank); any element may be None.
    """

    if not text:
        return None, None, None

    match = re.search(
        r'UPI/\d{8,20}/[^/]+/([^/]+)(?:/(.*))?$',
        text,
        re.IGNORECASE
    )

    if not match:
        return None, None, None

    segment = match.group(1).strip()
    remainder = match.group(2) or ""

    name = None
    label = None

    if segment:
        if "@" in segment:
            label = segment
        else:
            name = segment

    first_remainder_segment = remainder.split("/", 1)[0].strip()

    bank = (
        detect_bank_full_name(first_remainder_segment)
        if first_remainder_segment
        else None
    )

    return name, label, bank


def extract_counterparty(description: str) -> Dict[str, Optional[str]]:
    description = description or ""

    upi = extract_upi(description)

    payment_method = detect_payment_method(description)

    transaction_reference = extract_transaction_reference(description)

    reversal = detect_reversal(description)

    label = None
    counterparty_name = None
    account = None
    bank_from_details = None

    upper_description = description.upper()

    if payment_method == "IMPS":
        if "IMPS:" in upper_description:
            counterparty_name, bank_from_details, account = (
                extract_colon_transaction_details(description)
            )
        else:
            label = extract_imps_counterparty(description)

    elif payment_method == "UPI":
        if "UPI:" in upper_description:
            counterparty_name, bank_from_details, account = (
                extract_colon_transaction_details(description)
            )
        else:
            label = extract_upi_label(description)

            if not label:
                counterparty_name, label, bank_from_details = (
                    extract_upi_participant(description)
                )

    elif payment_method == "NEFT":
        label, counterparty_name = extract_neft_counterparty(description)

    bank = (
        detect_upi_bank(upi)
        or detect_bank_code(description)
        or bank_from_details
        or (
            detect_bank_from_reference(transaction_reference)
            if payment_method == "NEFT"
            else None
        )
    )

    # Prefer the strongest identifier available.
    if upi:
        identifier = upi
    elif label:
        identifier = label
    elif counterparty_name:
        identifier = counterparty_name
    else:
        identifier = None

    return {
        "payment_method": payment_method,
        "transaction_reference": transaction_reference,
        "counterparty_identifier": identifier,
        "counterparty_label": label,
        "counterparty_upi": upi,
        "counterparty_account": account,
        "counterparty_bank": bank,
        "counterparty_name": counterparty_name,
        "is_reversal": reversal,
    }


if __name__ == "__main__":

    examples = [
        "UPI/509031387885/CR/FARA/FDRL/rajesh@okicici/UPI",
        "UPI/546060109469/DR/HARI/YESB/vaishali@ybl/UPI",
        "UPI/546068389844/DR/Swat/AIRP/863714089@ikwik/A3Ff",
        "REVERSED : UPI/546068389844/DR/Swat/AIRP/863714089",
        "IMPS/P2A/509020012550//Nearby Technologies Private",
        "IMPS/P2A/514016777934/ICIC/SAVY INFOLINE PVT LT",
        "IMPS/P2A/515162601777/KARB/UPPUTHOLLA GOPAL",
        "Loan Disbursement/5115890003254264",
        "MB NEFT/DCBL427418943347/RITU/VIKRAM DUBEY/",
        "MB NEFT/DCBL427718958391/RITU/YASH DUBEY/",
        "MB NEFT/DCBL427718958669/AARA/ravish DUBEY/",
        "NEFT/YESB42880670696/REKH/RESILIENT INNOVATIONS PR",
        "NEFT/AXNFCN0777134239/NIKH/RESILIENT INNOVATIONS P",
        "IMPS:PAY:427413593157/Vikram Da/KARB/XX7501/1 rupe",
        "UPI:PAY:426036781143/NIDHI PILLAI/Kotak Mahi",
        "UPI:REC:427481674406/PALLAVI NAIR/Union Bank",
        "UPI/324743072944/Payment from Ph/9316670178@ybl/UC",
        "UPI/361351942465/UPI/Mrs ADITYA Neh/State Bank Of",
        "UPI/361312144126/UPI/Mrs ADITYA Neh/State Bank Of",
        "UPI/324822626850/Payment from Ph/JIOINAPPDIRECT@/Y",
        "UPI/324952707564/UPI/nidhi@okici/Bandhan Bank/",
        "UPI/325090800106/UPI/dhokaiankit59-2/UCO Bank/HDF6",
        "UPI/325055462449/UPI/Mrs ADITYA Neh/State Bank Of",
        "UPI/325173085593/UPI/Mrs ADITYA Neh/State Bank Of",
        "UPI/361719183400/UPI/dhokaiankit59-2/UCO Bank/ICI0",
        "UPI/325223016930/UPI/8200175382@payt/Axis Bank Ltd",
        "UPI/361865522410/UPI/rajagoswami1961/HDFC BANK LTD",
        "UPI/325265834780/UPI/dhokaiankit59-2/UCO Bank/ICIf",
        "UPI/325222341276/UPI/manjothiaslam12/AU small Fina",
        "UPI/361935848891/UPI/harshADITYA909/Bank of Barod",
        "UPI/361999558527/UPI/harshADITYA909/Bank of Barod",
    ]

    for text in examples:
        print("\nDESCRIPTION:")
        print(text)
        print("EXTRACTED:")
        print(extract_counterparty(text))
