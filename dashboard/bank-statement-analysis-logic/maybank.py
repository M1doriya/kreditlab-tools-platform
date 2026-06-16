import re
from io import BytesIO
import fitz
import pdfplumber
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core_utils import advance_year_on_rollover, finalize_parser_output

# -----------------------------
# Regex patterns
# -----------------------------
DATE_DMY_SLASH_RE = re.compile(r"^(?P<d>\d{2})/(?P<m>\d{2})(?:/(?P<y>\d{2,4}))?$")
DATE_DMY_DASH_RE  = re.compile(r"^(?P<d>\d{2})-(?P<m>\d{2})(?:-(?P<y>\d{2,4}))?$")
STATEMENT_DATE_RE = re.compile(r"STATEMENT\s+DATE\s*:?\s*(\d{2})/(\d{2})/(\d{2,4})", re.I)

# Maybank Islamic OD facilities emit Banker's Acceptance settlement rows whose
# description is purely an internal reference code, e.g.
#   "9908BAZ7811435/L4698231/REF 811435"
# These are not counterparty names. Without canonicalisation the downstream
# classifier extracts the ref string as a counterparty, polluting top-parties
# and counterparty_ledger with what look like obscure entities.
BA_SETTLEMENT_RE = re.compile(r"\b9908BAZ\d+/L\d+/REF(?:\s+\d+)?\b", re.IGNORECASE)
BA_SETTLEMENT_LABEL = "BA SETTLEMENT (Maybank Islamic)"

# Amount tokens usually look like: 1,630.00-  or  9,576.40+
# OD (SME First Account-i / Cash Line-i) statements glue a DR/CR suffix onto
# the balance token, e.g. 215,324.73DR — match that too.
# Sub-RM1 amounts are printed with NO leading zero (e.g. a cheque processing
# fee of ".50-"); accept cents-only tokens but only with an explicit +/- sign
# so OCR-damaged balance tokens (".02", no sign) still take the
# BALANCE_CENTS_ONLY_RE repair path instead.
MONEY_RE = re.compile(
    r"^-?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}(?:[+-]|DR|CR)?$|^\.\d{2}[+-]$"
)
BALANCE_CENTS_ONLY_RE = re.compile(r"^\.\d{2}(?:[+-]|DR|CR)?$")

FOOTER_KEYWORDS = (
    "ENDING BALANCE",
    "LEDGER BALANCE",
    "TOTAL DEBIT",
    "TOTAL CREDIT",
    "TOTAL DEBITS",
    "TOTAL CREDITS",
    "END OF STATEMENT",
    "PROFIT OUTSTANDING",
    "BAKI LEGAR",
    "BAKI AKHIR",
    "MUKA/",
    "PAGE",
    "NOMBOR AKAUN",
    "NOT PROTECTED BY PIDM",
    "PLEASE BE REMINDED",
    "NOTICE:",
    "NOTIS",
)

MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _open_doc(inp: Any) -> fitz.Document:
    """Open a PDF input robustly for Streamlit, bytes, file-like, or path."""
    if isinstance(inp, (bytes, bytearray)):
        return fitz.open(stream=bytes(inp), filetype="pdf")

    # Streamlit UploadedFile often supports getvalue()
    if hasattr(inp, "getvalue"):
        try:
            b = inp.getvalue()
            return fitz.open(stream=b, filetype="pdf")
        except Exception:
            pass

    # file-like object
    if hasattr(inp, "read"):
        try:
            pos = inp.tell()
        except Exception:
            pos = None

        b = inp.read()

        if pos is not None:
            try:
                inp.seek(pos)
            except Exception:
                pass

        return fitz.open(stream=b, filetype="pdf")

    # path string
    return fitz.open(inp)


def _parse_year_and_bank(doc: fitz.Document) -> Tuple[str, int]:
    bank = "Maybank"
    year = None

    for i in range(min(2, doc.page_count)):
        txt = (doc[i].get_text("text") or "").upper()

        if "MAYBANK ISLAMIC" in txt:
            bank = "Maybank Islamic"
        elif "MAYBANK" in txt:
            bank = "Maybank"

        m = STATEMENT_DATE_RE.search(txt)
        if m:
            y = m.group(3)
            year = (2000 + int(y)) if len(y) == 2 else int(y)
            break

    if year is None:
        year = datetime.now().year

    return bank, year


def _is_footer_or_header(line_text: str) -> bool:
    up = line_text.upper()
    return any(k in up for k in FOOTER_KEYWORDS)


def _money_token_value(tok: str) -> Tuple[float, Optional[str]]:
    """
    Returns (value, sign) where sign is '+', '-', 'DR', or 'CR' if present at end.
    DR/CR are emitted by Maybank OD statements on the balance column.
    """
    s = tok.strip()
    sign = None

    if s.endswith("DR"):
        sign = "DR"
        s = s[:-2]
    elif s.endswith("CR"):
        sign = "CR"
        s = s[:-2]
    elif s.endswith("+"):
        sign = "+"
        s = s[:-1]
    elif s.endswith("-"):
        sign = "-"
        s = s[:-1]

    s = s.replace(",", "")
    return float(s), sign


def _parse_date_token(token: str, default_year: int) -> Optional[str]:
    """
    Supports:
      DD/MM
      DD/MM/YY
      DD/MM/YYYY
      DD-MM
      DD-MM-YY
      DD-MM-YYYY
    """
    t = token.strip().upper()

    m = DATE_DMY_SLASH_RE.match(t)
    if m:
        d, mo, y = m.group("d"), m.group("m"), m.group("y")
        yy = default_year
        if y:
            yy = int(y)
            if yy < 100:
                yy = 2000 + yy
        return f"{yy:04d}-{int(mo):02d}-{int(d):02d}"

    m = DATE_DMY_DASH_RE.match(t)
    if m:
        d, mo, y = m.group("d"), m.group("m"), m.group("y")
        yy = default_year
        if y:
            yy = int(y)
            if yy < 100:
                yy = 2000 + yy
        return f"{yy:04d}-{int(mo):02d}-{int(d):02d}"

    return None


def _parse_split_date_tokens(items: List[dict]) -> Optional[str]:
    """
    Supports:
      DD MON YYYY   (e.g., 2 FEB 2025 / 02 FEB 2025)
    Only used if the PDF actually emits those tokens separately.
    """
    if len(items) < 3:
        return None

    d = items[0]["text"]
    mon = items[1]["text"]
    y = items[2]["text"]

    if not d.isdigit() or not y.isdigit():
        return None

    mon_u = mon.upper()
    if mon_u not in MONTH_MAP:
        return None

    return f"{int(y):04d}-{int(MONTH_MAP[mon_u]):02d}-{int(d):02d}"


def _cluster_lines(word_items: List[dict], y_tol: float = 3.0) -> List[Tuple[float, List[dict]]]:
    """
    Cluster words into 'visual lines' using y proximity.
    This is critical for Maybank PDFs because date/desc and amount/balance
    can be slightly misaligned in y.
    """
    if not word_items:
        return []

    word_items.sort(key=lambda r: (r["y"], r["x0"]))
    clusters: List[dict] = []

    for it in word_items:
        placed = False
        for c in clusters:
            if abs(it["y"] - c["y"]) <= y_tol:
                c["items"].append(it)
                # update centroid
                c["y"] = (c["y"] * (len(c["items"]) - 1) + it["y"]) / len(c["items"])
                placed = True
                break
        if not placed:
            clusters.append({"y": it["y"], "items": [it]})

    clusters.sort(key=lambda c: c["y"])

    out: List[Tuple[float, List[dict]]] = []
    for c in clusters:
        c["items"].sort(key=lambda r: r["x0"])
        out.append((c["y"], c["items"]))

    return out


def parse_transactions_maybank(pdf_input: Any, source_filename: str = "") -> List[Dict]:
    """
    Maybank (Conventional + Islamic) statement parser.

    Output schema matches other banks:
      date (YYYY-MM-DD), description, debit, credit, balance, page, bank, source_file
    """
    doc = _open_doc(pdf_input)
    bank_name, default_year = _parse_year_and_bank(doc)

    # Sprint 4.5 / Sprint 7 #12: capture page-1 header text via pdfplumber.
    # PyMuPDF (`fitz`) reads multi-column Maybank PDFs in a non-linear order
    # — transaction rows end up before the page-header column, so cutting at
    # `URUSNIAGA AKAUN` discards the holder name and leaves transaction-row
    # `<NAME> SDN. BHD.*` strings looking like the holder. pdfplumber
    # preserves the visual top-to-bottom layout, so the actual header lines
    # (`PRINCIPAL GAS SDN. BHD`, `LSR AGENCY`, etc.) come out cleanly.
    header_text: Optional[str] = None
    try:
        pdf_bytes: Optional[bytes] = None
        if isinstance(pdf_input, (bytes, bytearray)):
            pdf_bytes = bytes(pdf_input)
        elif hasattr(pdf_input, "getvalue"):
            try:
                pdf_bytes = pdf_input.getvalue()
            except Exception:
                pdf_bytes = None
        if pdf_bytes:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pp:
                if pp.pages:
                    cut = pp.pages[0].extract_text() or ""
                    for marker in (
                        "URUSNIAGA AKAUN",
                        "ACCOUNT TRANSACTIONS",
                        "TARIKH MASUK",
                        "ENTRY DATE",
                    ):
                        idx = cut.find(marker)
                        if idx != -1:
                            cut = cut[:idx]
                            break
                    header_text = cut or None
    except Exception:
        header_text = None
    if not header_text:
        try:
            if doc.page_count > 0:
                header_text = doc[0].get_text("text") or None
        except Exception:
            header_text = None

    txs: List[Dict] = []
    prev_balance: Optional[float] = None

    # Capture BEGINNING BALANCE (no date, just the value with DR/CR suffix).
    # Required so downstream consumers (classifier, monthly aggregator) can
    # anchor period-start opening for the first month of the analysis window
    # without having to back-derive it from the first transaction row.
    opening_balance_value: Optional[float] = None
    opening_balance_page: Optional[int] = None

    # Context: Maybank sometimes omits the date for subsequent rows with the same date
    carry_date_iso: Optional[str] = None

    # Context: a date-only line may appear (rare) followed by the actual data row below
    pending_date_iso: Optional[str] = None
    pending_date_x_end: Optional[float] = None

    # For multi-line description continuation
    last_tx: Optional[Dict] = None
    last_desc_left: Optional[float] = None
    last_money_left: Optional[float] = None

    # Thresholds (tuned for Maybank PDFs)
    DATE_COL_RIGHT_FALLBACK = 85.0  # if date cell is blank, description starts after this

    def append_desc(line_items: List[dict]):
        nonlocal last_tx
        if not last_tx or last_money_left is None:
            return

        # BUG-001 fix (v3.3.1 cross-bank): continuation lines carry the counterparty name on
        # wrapped rows (`TRANSFER FR A/C` / `TRANSFER TO A/C` / `PAYMENT FR A/C` / etc. on line 1,
        # then `SITI NURUL AMIRA* Comm` on line 2). The left bound was previously last_desc_left
        # (the x-end of the date column on the transaction row) — but continuation lines often
        # indent slightly differently and their tokens fall LEFT of that boundary, so every
        # token got filtered out and the name was silently lost. 278 transactions in the
        # MYTUTOR ACADEMY run ended up with bare "TRANSFER FR A/C" descriptions because of this.
        #
        # Fix: on continuation lines, drop the left bound entirely — continuation lines have no
        # date, so nothing needs to be skipped on the left. The right bound (last_money_left)
        # is retained so we never accidentally grab balance digits from the money column.
        # A minimal left bound of 10.0 is applied only to skip page-edge artefacts.
        CONTINUATION_LEFT_FLOOR = 10.0

        parts = []
        for it in line_items:
            if it["is_money"]:
                continue
            if it["x0"] < CONTINUATION_LEFT_FLOOR:
                continue
            if it["x0"] >= last_money_left:
                continue
            parts.append(it["text"])

        if parts:
            last_tx["description"] = _norm_spaces(
                (last_tx.get("description", "") + " " + " ".join(parts)).strip()
            )

    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            words = page.get_text("words")

            word_items = []
            for w in words:
                txt = str(w[4]).strip()
                if not txt:
                    continue
                word_items.append({"y": float(w[1]), "x0": float(w[0]), "text": txt})

            for _, line_items in _cluster_lines(word_items, y_tol=5.0):
                line_text = _norm_spaces(" ".join(i["text"] for i in line_items))
                if not line_text:
                    continue

                # BEGINNING BALANCE row: capture the value before the footer/header
                # filter consumes it (FOOTER_KEYWORDS doesn't include "BEGINNING",
                # but the line has no date so it would otherwise just be ignored).
                # Negate DR-suffixed values so the synthetic row carries the
                # signed balance under the bank-wide convention.
                if opening_balance_value is None and "BEGINNING BALANCE" in line_text.upper():
                    for it in line_items:
                        if MONEY_RE.match(it["text"]):
                            val, sgn = _money_token_value(it["text"])
                            if sgn == "DR":
                                val = -val
                            opening_balance_value = val
                            opening_balance_page = page_index + 1
                            break
                    continue

                if _is_footer_or_header(line_text):
                    pending_date_iso = None
                    pending_date_x_end = None
                    last_tx = None
                    last_desc_left = None
                    last_money_left = None
                    continue

                # Mark money tokens + find money column start
                money_positions = []
                for it in line_items:
                    it["is_money"] = bool(MONEY_RE.match(it["text"]))
                    if it["is_money"]:
                        money_positions.append(it["x0"])

                money_left = min(money_positions) if money_positions else None

                # Detect date token at start (DD/MM...) OR split date tokens (DD MON YYYY)
                date_iso: Optional[str] = None
                date_x_end: Optional[float] = None
                start_after_date = 0

                split = _parse_split_date_tokens(line_items[:3]) if len(line_items) >= 3 else None
                if split:
                    date_iso = split
                    date_x_end = line_items[2]["x0"] + 20.0
                    start_after_date = 3
                else:
                    if line_items:
                        maybe = _parse_date_token(line_items[0]["text"], default_year)
                        if maybe:
                            maybe = advance_year_on_rollover(maybe, carry_date_iso)
                            date_iso = maybe
                            date_x_end = line_items[0]["x0"] + 20.0
                            start_after_date = 1

                if date_iso:
                    carry_date_iso = date_iso  # always update date context

                # Date-only line (no money) -> pending date for the next line
                if date_iso and money_left is None:
                    # Keep only if it is "mostly just a date"
                    if len(line_items) <= start_after_date + 3:
                        pending_date_iso = date_iso
                        pending_date_x_end = date_x_end
                        last_tx = None
                        continue

                # Determine date for a transaction row
                effective_date = date_iso or pending_date_iso

                # Fix: blank-date rows (same date as previous transaction),
                # where date column is empty but amount/balance exist.
                if effective_date is None and money_left is not None and carry_date_iso:
                    first_x = line_items[0]["x0"] if line_items else 0.0
                    # If the first token is far right, date cell is likely blank
                    if first_x > 70.0:
                        effective_date = carry_date_iso
                        date_x_end = DATE_COL_RIGHT_FALLBACK

                # Transaction row
                if effective_date and money_left is not None:
                    money_tokens = [it["text"] for it in line_items if it["is_money"]]
                    if len(money_tokens) >= 2 or (len(money_tokens) == 1 and prev_balance is not None):
                        bal_val: Optional[float] = None

                        # Normal case: balance is the last money token.
                        # OD accounts (SME First Account-i / Cash Line-i) suffix the
                        # balance with `DR` to signal an overdrawn magnitude. Store
                        # those as negative so the existing delta-vs-prev_balance
                        # OCR-correction logic below stays correct without per-row
                        # sign branching (Ambank convention, per CLAUDE.md).
                        if len(money_tokens) >= 2:
                            bal_val, bal_sign = _money_token_value(money_tokens[-1])
                            if bal_sign == "DR":
                                bal_val = -bal_val

                        # Choose transaction amount: prefer last signed token before balance
                        amt_val: Optional[float] = None
                        amt_sign: Optional[str] = None

                        if len(money_tokens) >= 2:
                            for t in reversed(money_tokens[:-1]):
                                v, sgn = _money_token_value(t)
                                if sgn in ("+", "-"):
                                    amt_val, amt_sign = v, sgn
                                    break

                            if amt_val is None:
                                v, sgn = _money_token_value(money_tokens[-2])
                                amt_val, amt_sign = v, sgn
                        else:
                            # OCR fallback: some lines lose integer digits on the balance token
                            # (e.g. ".02"), leaving only one parseable money token.
                            amt_val, amt_sign = _money_token_value(money_tokens[0])

                        # Description tokens: use token-index to skip date (avoids
                        # filtering out description words that sit close to the date
                        # column, e.g. "02/04 TRANSFER FR AC 16,000.00 ...").
                        # X-coord filter retained only as fallback when no date is on
                        # this line (blank-date row inheriting carry_date_iso).
                        desc_left = date_x_end if date_x_end is not None else pending_date_x_end
                        if desc_left is None:
                            desc_left = DATE_COL_RIGHT_FALLBACK

                        desc_parts = []
                        for idx, it in enumerate(line_items):
                            if it["is_money"]:
                                continue
                            if it["x0"] >= money_left:
                                continue
                            # Skip actual date tokens by INDEX, not by x-coordinate
                            if date_iso and idx < start_after_date:
                                continue
                            # Only apply x-coord filter when there's no date on this
                            # line (otherwise TRANSFER/CASH DEPOSIT/etc adjacent to
                            # the date column get wrongly dropped)
                            if not date_iso and it["x0"] < desc_left:
                                continue
                            desc_parts.append(it["text"])

                        description = _norm_spaces(" ".join(desc_parts))

                        # Canonicalise Banker's Acceptance settlement refs so
                        # downstream counterparty extractors don't treat them
                        # as entity names.
                        if BA_SETTLEMENT_RE.search(description):
                            description = BA_SETTLEMENT_LABEL

                        debit = 0.0
                        credit = 0.0

                        # Primary rule: sign on the transaction amount
                        if amt_sign == "+":
                            credit = float(amt_val)
                        elif amt_sign == "-":
                            debit = float(amt_val)
                        else:
                            # Fallback: balance delta if sign missing
                            if prev_balance is not None and bal_val is not None:
                                delta = round(bal_val - prev_balance, 2)
                                if delta > 0:
                                    credit = abs(delta)
                                elif delta < 0:
                                    debit = abs(delta)
                            else:
                                debit = float(amt_val)

                        # Infer missing balance for one-money-token rows.
                        if bal_val is None and prev_balance is not None:
                            if amt_sign == "+":
                                bal_val = round(prev_balance + float(amt_val), 2)
                            elif amt_sign == "-":
                                bal_val = round(prev_balance - float(amt_val), 2)

                            # If a cents-only token exists in the balance column, enforce cents.
                            for it in reversed(line_items):
                                if it["is_money"]:
                                    continue
                                if it["x0"] <= money_left:
                                    continue
                                if BALANCE_CENTS_ONLY_RE.match(it["text"]):
                                    cents = int(it["text"][-2:])
                                    if bal_val is not None:
                                        bal_val = round(int(bal_val) + (cents / 100.0), 2)
                                    break

                        if bal_val is None:
                            continue

                        # Conservative OCR correction using balance delta
                        # (Fix digit swaps / missing decimals, without breaking correct values)
                        if prev_balance is not None:
                            delta = round(bal_val - prev_balance, 2)
                            expected = abs(delta)
                            parsed = credit if credit > 0 else debit
                            if expected > 0 and parsed > 0 and abs(expected - parsed) <= 500:
                                if delta > 0:
                                    credit = expected
                                    debit = 0.0
                                elif delta < 0:
                                    debit = expected
                                    credit = 0.0

                        tx = {
                            "date": effective_date,
                            "description": description,
                            "debit": round(float(debit), 2),
                            "credit": round(float(credit), 2),
                            "balance": round(float(bal_val), 2),
                            "page": page_index + 1,
                            "bank": bank_name,
                            "source_file": source_filename,
                        }
                        txs.append(tx)

                        # Continuation boundaries
                        last_tx = tx
                        last_desc_left = desc_left
                        last_money_left = money_left

                        prev_balance = bal_val

                        # Clear pending date after use
                        if date_iso is None and pending_date_iso is not None:
                            pending_date_iso = None
                            pending_date_x_end = None

                        continue

                # Description continuation lines
                if last_tx is not None and money_left is None and date_iso is None:
                    append_desc(line_items)

    finally:
        doc.close()

    # Dedupe exact duplicates only.
    # Important: some statements legitimately contain multiple transactions
    # with the same date/debit/credit/balance on the same page, so we must
    # keep rows when descriptions differ.
    seen = set()
    out: List[Dict] = []
    for t in txs:
        key = (
            t["date"],
            t["description"],
            t["debit"],
            t["credit"],
            t["balance"],
            t["page"],
            t["source_file"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(t)

    # Emit synthetic OPENING BALANCE row when the PDF's BEGINNING BALANCE line
    # was captured. Anchors first-month opening for downstream aggregation
    # (classifier / monthly_summary) so it does not have to back-derive from
    # the first transaction row. CIMB / RHB / others follow the same pattern.
    if opening_balance_value is not None and out:
        first_date = out[0].get("date")
        if first_date:
            anchor_date = first_date
            opening_row = {
                "date": anchor_date,
                "description": "OPENING BALANCE",
                "debit": 0.0,
                "credit": 0.0,
                "balance": round(float(opening_balance_value), 2),
                "page": opening_balance_page,
                "bank": bank_name,
                "source_file": source_filename,
                "is_opening_balance": True,
                "opening_balance_source": "page_1",
            }
            out.insert(0, opening_row)

    # Sprint 4.5: per-PDF account_type determination + statutory stamping.
    # opening_balance is forwarded so determine_account_type can run a proper
    # CR/OD trail; falls back to row-math when missing.
    return finalize_parser_output(
        out,
        header_text=header_text,
        opening_balance=opening_balance_value,
        closing_balance=None,
    )
