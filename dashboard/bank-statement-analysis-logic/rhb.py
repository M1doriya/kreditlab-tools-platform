import re
import fitz  # PyMuPDF
import pdfplumber
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from core_utils import advance_year_on_rollover, finalize_parser_output


# ======================================================
# Helper: read PDF bytes safely (Streamlit / file / path)
# ======================================================
def _read_pdf_bytes(pdf_input: Any) -> bytes:
    """Return PDF bytes from bytes, Streamlit UploadedFile, file-like, or filesystem path."""
    if isinstance(pdf_input, (bytes, bytearray)):
        return bytes(pdf_input)

    # Streamlit UploadedFile
    if hasattr(pdf_input, "getvalue"):
        data = pdf_input.getvalue()
        if data:
            return data

    # file-like object
    if hasattr(pdf_input, "read"):
        try:
            pdf_input.seek(0)
        except Exception:
            pass
        data = pdf_input.read()
        if data:
            return data

    # path string
    if isinstance(pdf_input, str):
        with open(pdf_input, "rb") as f:
            return f.read()

    raise ValueError("Unable to read PDF bytes")


# -----------------------------
# Shared parsing helpers
# -----------------------------
_MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

# Money tokens in RHB statements often look like:
#   27,286.00
#   746,858.49-
#   0.00
_MONEY_TOKEN_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})*\.\d{2}[+-]?$|^[+-]?\d+\.\d{2}[+-]?$")


def _money_to_float(token: str) -> Optional[float]:
    if token is None:
        return None
    s = str(token).strip().replace(" ", "")
    if not s:
        return None

    # parenthesis negative
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()

    trailing = None
    if s.endswith("+"):
        trailing = "+"
        s = s[:-1]
    elif s.endswith("-"):
        trailing = "-"
        s = s[:-1]

    s = s.replace(",", "")
    try:
        v = float(s)
    except Exception:
        return None

    if trailing == "-":
        v = -abs(v)
    elif trailing == "+":
        v = abs(v)
    return float(v)


def _extract_year_from_statement_period(text: str) -> Optional[int]:
    """Extract statement year from common RHB header lines.

    Supports:
      - "Statement Period ... : 1 Jan 25 – 31 Jan 25"
      - "Statement Period" + "01 May 2025 31 May 2025" (sometimes without a dash, often across lines)
      - Any "DD Mon YYYY" occurrence near statement-period headers as fallback

    Returns the *ending* year where available.
    """
    if not text:
        return None

    # Normalize spacing so cross-line patterns work.
    t = re.sub(r"\s+", " ", text).strip()

    # Case 1: explicit range with dash
    m = re.search(
        r"Statement\s+Period.*?:\s*\d{1,2}\s+[A-Za-z]{3,9}\s+(?P<y1>\d{2,4})\s*[-–—]\s*"
        r"\d{1,2}\s+[A-Za-z]{3,9}\s+(?P<y2>\d{2,4})",
        t,
        re.IGNORECASE,
    )
    if m:
        y = m.group("y2") or m.group("y1")
        return int(y) if len(y) == 4 else 2000 + int(y)

    # Case 2: "01 May 2025 31 May 2025" (no dash)
    m = re.search(
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+(?P<y1>\d{4})\s+\d{1,2}\s+[A-Za-z]{3,9}\s+(?P<y2>\d{4})\b",
        t,
        re.IGNORECASE,
    )
    if m:
        return int(m.group("y2"))

    # Case 3: weaker fallback: first year-like token near Statement Period
    m = re.search(
        r"Statement\s+Period.*?\b\d{1,2}\s+[A-Za-z]{3,9}\s+(?P<y>\d{2,4})\b",
        t,
        re.IGNORECASE,
    )
    if m:
        y = m.group("y")
        return int(y) if len(y) == 4 else 2000 + int(y)

    return None


def _guess_bank_name(header_upper: str) -> str:
    if "ISLAMIC" in header_upper:
        return "RHB Islamic Bank"
    return "RHB Bank"


def _is_non_transaction_commodity_page(page_text: str) -> bool:
    """Detect commodity-trading certificate pages that are not account transactions."""
    if not page_text:
        return False
    t = page_text.upper()
    return (
        ("SELLER/PENJUAL" in t and "BUYER/PEMBELI" in t)
        or ("CERTIFICATE NO" in t and "NET DEPOSIT" in t and "SELLING PRICE" in t)
        or ("COMMODITY" in t and "TRADING" in t)
    )


# ======================================================
# 1) RHB ACCOUNT STATEMENT — text based (older layout)
# ======================================================
def _parse_rhb_account_statement_text(pdf_bytes: bytes, source_filename: str) -> List[Dict]:
    transactions: List[Dict] = []

    DATE_START_RE = re.compile(r"^(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3})\b\s+(?P<rest>.*)$")
    NOISE_LINE_RE = re.compile(
        r"^(?:"
        r"ACCOUNT\s+ACTIVITY|DEPOSIT\s+ACCOUNT|DEPOSIT\s+ACCOUNT\s+SUMMARY|STATEMENT\s+PERIOD|"
        r"IMPORTANT\s+NOTES|IMPORTANT\s+ANNOUNCEMENTS|PAGE\s+NO\.?|RHB\s+BANK|"
        r"MEMBER\s+OF\s+PIDM|PROTECTED\s+BY\s+PIDM|DILINDUNGI\s+OLEH\s+PIDM|"
        r"PRODUCT\s+NAME|ACCOUNT\s+NO\.?|CURRENCY|DATE\s+DESCRIPTION|"
        r"CHEQUE\s+\/\s+SERIAL|DEBIT|CREDIT|BALANCE"
        r")\b",
        re.IGNORECASE,
    )

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        header = pdf.pages[0].extract_text(x_tolerance=1) or ""
        header_up = header.upper()

        # Heuristic: only run this parser if the statement looks like the account-statement format
        if "ACCOUNT STATEMENT" not in header_up and "PENYATA" not in header_up:
            return []

        year = _extract_year_from_statement_period(header) or datetime.now().year
        bank_name = _guess_bank_name(header_up)

        prev_balance: Optional[float] = None
        last_tx: Optional[Dict] = None
        prev_date_iso: Optional[str] = None

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1) or ""
            if _is_non_transaction_commodity_page(text):
                continue
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]

            for line in lines:
                if NOISE_LINE_RE.match(line):
                    last_tx = None
                    continue

                # Totals / summary counters
                if re.match(r"^Total\s+Count\b", line, re.IGNORECASE):
                    last_tx = None
                    continue

                m = DATE_START_RE.match(line)
                if m:
                    dd = int(m.group("day"))
                    mon = m.group("mon").upper()
                    if mon not in _MONTH_MAP:
                        last_tx = None
                        continue

                    date_iso = f"{year:04d}-{_MONTH_MAP[mon]}-{dd:02d}"
                    date_iso = advance_year_on_rollover(date_iso, prev_date_iso)
                    prev_date_iso = date_iso

                    tokens = line.split()
                    rest_tokens = tokens[2:]  # drop day + mon

                    money_idx = [i for i, t in enumerate(rest_tokens) if _MONEY_TOKEN_RE.match(t)]
                    if not money_idx:
                        last_tx = None
                        continue

                    bal_token = rest_tokens[money_idx[-1]]
                    balance = _money_to_float(bal_token)
                    if balance is None:
                        last_tx = None
                        continue

                    # Description is everything before the numeric columns start
                    desc_tokens = rest_tokens[:money_idx[0]]
                    description = " ".join(desc_tokens).strip()

                    # Opening/closing balance lines (do not emit as transactions)
                    up_desc = description.upper()
                    if "B/F" in up_desc:
                        prev_balance = balance
                        last_tx = None
                        continue
                    if "C/F" in up_desc:
                        last_tx = None
                        continue

                    # Debit/credit from delta if possible
                    debit = credit = 0.0
                    if prev_balance is not None:
                        delta = round(balance - prev_balance, 2)
                        if delta < 0:
                            debit = abs(delta)
                        elif delta > 0:
                            credit = delta

                    tx = {
                        "date": date_iso,
                        "description": description[:200],
                        "debit": round(debit, 2),
                        "credit": round(credit, 2),
                        "balance": round(balance, 2),
                        "page": page_num,
                        "bank": bank_name,
                        "source_file": source_filename,
                    }
                    transactions.append(tx)
                    prev_balance = balance
                    last_tx = tx
                else:
                    # Continuation line
                    if last_tx is not None:
                        extra = line.strip()
                        if extra and not NOISE_LINE_RE.match(extra):
                            last_tx["description"] = (last_tx["description"] + " " + extra).strip()[:200]

    return transactions


# ======================================================
# 2) RHB ISLAMIC — older text-based format (kept, but guarded)
# ======================================================
def _parse_rhb_islamic_text(pdf_bytes: bytes, source_filename: str) -> List[Dict]:
    transactions: List[Dict] = []
    previous_balance: Optional[float] = None
    prev_date_iso: Optional[str] = None

    balance_re = re.compile(r"(?P<bal>[\d,]+\.\d{2}[+-]?)\s*$")
    date_re = re.compile(r"(?P<d>\d{1,2})\s+(?P<m>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        header = pdf.pages[0].extract_text(x_tolerance=1) or ""
        year = _extract_year_from_statement_period(header) or datetime.now().year

        header_up = header.upper()
        # Reflex Cash Management / Transaction Statement PDFs are handled by the layout-based parser.
        if ("REFLEX" in header_up) or ("CASH MANAGEMENT" in header_up) or ("DEPOSIT ACCOUNT SUMMARY" in header_up) or ("TRANSACTION STATEMENT" in header_up):
            return []

        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue
            if _is_non_transaction_commodity_page(text):
                continue

            for line in text.splitlines():
                bal_match = balance_re.search(line.strip())
                date_match = date_re.search(line)
                if not bal_match or not date_match:
                    continue

                balance = _money_to_float(bal_match.group("bal"))
                if balance is None:
                    continue

                if re.search(r"\bB/F\b|\bC/F\b", line):
                    previous_balance = balance
                    continue

                if previous_balance is None:
                    previous_balance = balance
                    continue

                day = int(date_match.group("d"))
                month = date_match.group("m")
                date_iso = datetime.strptime(f"{day:02d} {month} {year}", "%d %b %Y").strftime("%Y-%m-%d")
                date_iso = advance_year_on_rollover(date_iso, prev_date_iso)
                prev_date_iso = date_iso

                delta = round(balance - previous_balance, 2)
                debit = round(abs(delta), 2) if delta < 0 else 0.0
                credit = round(delta, 2) if delta > 0 else 0.0

                desc = balance_re.sub("", line)
                desc = desc.replace(date_match.group(0), "")
                desc = re.sub(r"\s+", " ", desc).strip()

                transactions.append(
                    {
                        "date": date_iso,
                        "description": desc,
                        "debit": debit,
                        "credit": credit,
                        "balance": round(balance, 2),
                        "page": page_index,
                        "bank": "RHB Islamic Bank",
                        "source_file": source_filename,
                    }
                )

                previous_balance = balance

    return transactions


# ======================================================
# 3) RHB CONVENTIONAL — older text-based format (kept, but guarded)
# ======================================================
def _parse_rhb_conventional_text(pdf_bytes: bytes, source_filename: str) -> List[Dict]:
    transactions: List[Dict] = []
    previous_balance: Optional[float] = None
    prev_date_iso: Optional[str] = None

    balance_re = re.compile(r"(?P<bal>[\d,]+\.\d{2}[+-]?)\s*$")
    # supports "05Jan" and "05 Jan"
    date_re = re.compile(r"(?P<d>\d{1,2})\s*(?P<m>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        header = pdf.pages[0].extract_text(x_tolerance=1) or ""
        year = _extract_year_from_statement_period(header) or datetime.now().year

        header_up = header.upper()
        # Reflex Cash Management / Transaction Statement PDFs are handled by the layout-based parser.
        if ("REFLEX" in header_up) or ("CASH MANAGEMENT" in header_up) or ("DEPOSIT ACCOUNT SUMMARY" in header_up) or ("TRANSACTION STATEMENT" in header_up):
            return []

        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue
            if _is_non_transaction_commodity_page(text):
                continue

            for line in text.splitlines():
                bal_m = balance_re.search(line.strip())
                date_m = date_re.search(line)
                if not bal_m or not date_m:
                    continue

                balance = _money_to_float(bal_m.group("bal"))
                if balance is None:
                    continue

                if previous_balance is None:
                    previous_balance = balance
                    continue

                day = int(date_m.group("d"))
                month = date_m.group("m")
                date_iso = datetime.strptime(f"{day:02d} {month} {year}", "%d %b %Y").strftime("%Y-%m-%d")
                date_iso = advance_year_on_rollover(date_iso, prev_date_iso)
                prev_date_iso = date_iso

                delta = round(balance - previous_balance, 2)
                debit = round(abs(delta), 2) if delta < 0 else 0.0
                credit = round(delta, 2) if delta > 0 else 0.0

                desc = balance_re.sub("", line)
                desc = desc.replace(date_m.group(0), "")
                desc = re.sub(r"\s+", " ", desc).strip()

                transactions.append(
                    {
                        "date": date_iso,
                        "description": desc,
                        "debit": debit,
                        "credit": credit,
                        "balance": round(balance, 2),
                        "page": page_index,
                        "bank": "RHB Bank",
                        "source_file": source_filename,
                    }
                )

                previous_balance = balance

    return transactions


# ======================================================
# 4) RHB REFLEX — layout based (kept as-is)
# ======================================================
def _parse_rhb_reflex_layout(pdf_bytes: bytes, source_filename: str) -> List[Dict]:
    transactions: List[Dict] = []

    DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    MONEY_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})*|\d)?\.\d{2}[+-]?")
    # Page furniture / non-table lines that must never be glued onto a
    # transaction's wrapped description rows.
    NOISE_RE = re.compile(
        r"(www\.rhbgroup\.com|For Any Enquiries|Statement Period|"
        r"TRANSACTION STATEMENT|Deposit Account Summary|Beginning Balance|"
        r"Ending Balance|Interest Paid|Deposits \(Plus\)|Withdraws \(Minus\)|"
        r"Reflex Cash Management|BranchDescription|Sender's Reference|"
        r"Beneficiary's|Recipient's|Other Payment|Name Reference Details|"
        r"Amount \(DR\)|Amount \(CR\)|IMPORTANT|Member of PIDM)",
        re.IGNORECASE,
    )
    # Wrapped rows per transaction observed up to 4; cap guards against an
    # unrecognised footer being swallowed wholesale.
    MAX_WRAP_LINES = 6

    def norm_date(text: str) -> str:
        return datetime.strptime(text, "%d-%m-%Y").strftime("%Y-%m-%d")

    def extract_opening_balance() -> Optional[float]:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if "Beginning Balance" in text:
                    m = re.search(r"([\d,]+\.\d{2})([+-])?", text)
                    if m:
                        amount = float(m.group(1).replace(",", ""))
                        if m.group(2) == "-":
                            amount = -amount
                        return amount
        return None

    previous_balance = extract_opening_balance()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        for page_index, page in enumerate(doc, start=1):
            words = page.get_text("words")
            rows = [
                {"x": w[0], "y": round(w[1], 1), "text": w[4].strip()}
                for w in words
                if w[4].strip()
            ]
            rows.sort(key=lambda r: (r["y"], r["x"]))

            # Cluster words into visual lines (y within 1.5pt), in page order.
            visual_lines: List[List[Dict]] = []
            for w in rows:
                if visual_lines and abs(visual_lines[-1][0]["y"] - w["y"]) <= 1.5:
                    visual_lines[-1].append(w)
                else:
                    visual_lines.append([w])

            # A transaction's description / beneficiary cells wrap onto the
            # rows BELOW the date-anchored line ("REFLEX- / PAYROLL / PYMT",
            # "INWARD INST TRF ... PUBLIC BANK TO RHB"). Capture those wrapped
            # rows onto the last emitted transaction; without them the engine
            # cannot see counterparties, own-account transfers or facility
            # markers (COMMITMENT FEE / INTEREST CHARGED).
            last_tx: Optional[Dict] = None  # last txn emitted on THIS page
            wrap_count = 0

            for line in visual_lines:
                line.sort(key=lambda w: w["x"])
                line_text = " ".join(w["text"] for w in line)
                date_word = next((w for w in line if DATE_RE.match(w["text"])), None)

                if date_word is None:
                    # Possible wrapped continuation of the current transaction.
                    if last_tx is None:
                        continue
                    if NOISE_RE.search(line_text):
                        continue
                    if wrap_count >= MAX_WRAP_LINES:
                        continue
                    extra_parts = [
                        w["text"]
                        for w in line
                        if not w["text"].isdigit() and not MONEY_RE.match(w["text"])
                    ]
                    extra = " ".join(extra_parts).strip()
                    if extra:
                        last_tx["description"] = (
                            (last_tx["description"] + " " + extra).strip()[:300]
                        )
                        wrap_count += 1
                    continue

                money = [w for w in line if MONEY_RE.match(w["text"])]
                if len(money) < 2:
                    continue

                bal_text = money[-1]["text"].replace(",", "")
                is_negative = bal_text.endswith("-")
                bal_val = float(bal_text.replace("-", "").replace("+", ""))

                if is_negative:
                    bal_val = -bal_val

                debit = credit = 0.0
                if previous_balance is not None:
                    delta = round(bal_val - previous_balance, 2)
                    if delta < 0:
                        debit = abs(delta)
                    elif delta > 0:
                        credit = delta

                description_parts = [
                    w["text"]
                    for w in line
                    if w not in money and not DATE_RE.match(w["text"]) and not w["text"].isdigit()
                ]

                transactions.append(
                    {
                        "date": norm_date(date_word["text"]),
                        "description": " ".join(description_parts)[:200],
                        "debit": round(debit, 2),
                        "credit": round(credit, 2),
                        "balance": round(bal_val, 2),
                        "page": page_index,
                        "bank": "RHB Bank",
                        "source_file": source_filename,
                    }
                )

                previous_balance = bal_val
                last_tx = transactions[-1]
                wrap_count = 0

    finally:
        doc.close()

    return transactions


def parse_transactions_rhb(pdf_input: Any, source_filename: str) -> List[Dict]:
    """Main entry used by app.py: returns list of canonical tx dicts.

    RHB has multiple PDF layouts. Some Reflex Cash Management PDFs contain month names in header
    summary lines (e.g., "31 May 2025") which can cause the older text-based parsers to emit
    bogus rows. For Reflex PDFs we therefore prefer the layout-based parser.
    """
    pdf_bytes = _read_pdf_bytes(pdf_input)

    header_up = ""
    header_text: Optional[str] = None
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            header = pdf.pages[0].extract_text(x_tolerance=1) or ""
            header_up = header.upper()
            # Sprint 4.5: truncate page-1 text at the RHB transaction-table
            # marker so footer disclaimers don't leak into determine_account_type.
            cut = header
            for marker in (
                "ACCOUNT ACTIVITY",
                "AKTIVITI AKAUN",
                "ACCOUNTACTIVITY",
                "AKTIVITIAKAUN",
                "Date Description Cheque",
                "Tarikh Diskripsi Cek",
                "Date Description",
                "Tarikh Diskripsi",
            ):
                idx = cut.find(marker)
                if idx != -1:
                    cut = cut[:idx]
                    break
            header_text = cut or None
    except Exception:
        header_up = ""

    def _finalize(rows: List[Dict]) -> List[Dict]:
        # Sprint 4.5: RHB opening/closing come from the layout-specific parsers,
        # which don't return them to this dispatcher. Pass None — row-math + the
        # page-1 header (truncated above) carry the OD signal (Clear Water Services
        # uses sustained-negative-balance convention).
        return finalize_parser_output(
            rows,
            header_text=header_text,
            opening_balance=None,
            closing_balance=None,
        )

    looks_like_reflex = (
        ("REFLEX" in header_up)
        or ("CASH MANAGEMENT" in header_up)
        or ("DEPOSIT ACCOUNT SUMMARY" in header_up)
        or ("TRANSACTION STATEMENT" in header_up)
    )

    # If it's a Reflex-style statement, try the layout-based parser first.
    if looks_like_reflex:
        try:
            tx = _parse_rhb_reflex_layout(pdf_bytes, source_filename)
            if tx:
                return _finalize(tx)
        except Exception:
            pass

    # Fallback order for other layouts
    for parser in (
        _parse_rhb_account_statement_text,
        _parse_rhb_islamic_text,
        _parse_rhb_conventional_text,
        _parse_rhb_reflex_layout,
    ):
        try:
            tx = parser(pdf_bytes, source_filename)
            if tx:
                return _finalize(tx)
        except Exception:
            continue

    return _finalize([])
