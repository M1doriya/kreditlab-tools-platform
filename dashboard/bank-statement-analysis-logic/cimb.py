# cimb.py - CIMB Bank Parser (robust)
#
# CIMB quirks handled:
# - Statement table is usually reverse chronological (latest is #1).
# - "Opening Balance" often appears without a date and is printed on page 1.
# - "Closing Balance / Baki Penutup" appears near end of PDF -> scan full doc text.
# - Extraction can duplicate rows with wrapped descriptions -> dedupe ignoring description.
#
# Output:
# - Standard transaction rows
# - Synthetic OPENING BALANCE (PAGE 1) row if detected
# - Synthetic CLOSING BALANCE / BAKI PENUTUP row if detected
#   (plus optional statement totals metadata on the closing row)

import re
from datetime import datetime

from core_utils import clean_description as _clean_description_core
from core_utils import finalize_parser_output


# -----------------------------
# Regex
# -----------------------------

_MONEY_TOKEN_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$")

_STMT_DATE_RE = re.compile(
    r"(?:STATEMENT\s+DATE|TARIKH\s+PENYATA)\s*[:\s]+(\d{1,2})/(\d{1,2})/(\d{2,4})",
    re.IGNORECASE,
)

_CLOSING_RE = re.compile(
    r"CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUP\s+(-?[\d,]+\.\d{2})",
    re.IGNORECASE,
)

_OPENING_LINE_RE = re.compile(r"^\s*OPENING\s+BALANCE\b", re.IGNORECASE)

# Mirror of _CLOSING_RE — captures the numeric value on the OPENING BALANCE
# line. Needed because pdfplumber's table extraction renders this line as a
# short (2-cell) row that the table-mode loop skips via its `len(row) < 6`
# guard, silently losing the opening figure on every PDF. Confirmed against
# Huahub CIMB OD Oct'25 (opening -877,598.70 was dropped, engine back-derived
# -949,417.44 from row 0 and Oct reconciliation failed by +71,818.74).
_OPENING_RE = re.compile(
    r"OPENING\s+BALANCE\s+(-?[\d,]+\.\d{2})",
    re.IGNORECASE,
)


# -----------------------------
# Basic helpers
# -----------------------------

def parse_float(value):
    """Convert string like '1,234.56' or '-1,234.56' to float. Return 0.0 if invalid."""
    if value is None:
        return 0.0
    s = str(value).replace("\n", " ").strip()
    s = s.replace(" ", "").replace(",", "")
    if not s:
        return 0.0
    if not re.match(r"^-?\d+(\.\d+)?$", s):
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def clean_text(text):
    if not text:
        return ""
    return str(text).replace("\n", " ").strip()


# CIMB-specific ref patterns (in addition to cross-bank patterns in core_utils).
# Add patterns here that appear ONLY in CIMB statements; cross-bank FPX / long-digit /
# generic alphanumeric refs are handled by core_utils.clean_description.
_CIMB_EXTRA_REF_PATTERNS = [
    # CIMB ITF rail — "ITF/202" or "ITF 202", any digit count
    re.compile(r'\bITF[/\s]\d+\b', re.IGNORECASE),
    # CIMB SFTP batch-payment refs: U2025082902838, RTB2508280299700842.TXT
    re.compile(r'\bU\d{11,}\b'),
    re.compile(r'\bRTB\d{10,}(?:\.TXT)?\b', re.IGNORECASE),
]


def _clean_description(desc):
    """CIMB-scoped description cleanup: applies core cross-bank rules + CIMB extras."""
    return _clean_description_core(desc, extra_ref_patterns=_CIMB_EXTRA_REF_PATTERNS)


# Re-exported for tests and tooling that still reference the private names.
def _collapse_duplicated_segments(desc):
    from core_utils import collapse_duplicated_segments
    return collapse_duplicated_segments(desc)


def _strip_reference_numbers(desc):
    from core_utils import strip_reference_numbers
    return strip_reference_numbers(desc, extra_patterns=_CIMB_EXTRA_REF_PATTERNS)


def format_date(date_str, year):
    """
    Convert 'DD/MM/YYYY' or 'DD/MM' into 'YYYY-MM-DD'.
    """
    if not date_str:
        return None
    s = clean_text(date_str)

    m = re.match(r"(\d{2})/(\d{2})/(\d{4})$", s)
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"

    m = re.match(r"(\d{2})/(\d{2})$", s)
    if m:
        dd, mm = m.groups()
        return f"{year}-{mm}-{dd}"

    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    return None


def extract_year_from_text(text):
    if not text:
        return None
    m = re.search(
        r"(?:STATEMENT\s+DATE|TARIKH\s+PENYATA)\s*[:\s]+\d{1,2}/\d{1,2}/(\d{2,4})",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    y = m.group(1)
    return y if len(y) == 4 else str(2000 + int(y))


def extract_closing_balance_from_text(text):
    if not text:
        return None
    m = _CLOSING_RE.search(text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def extract_opening_balance_from_text(text):
    if not text:
        return None
    m = _OPENING_RE.search(text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def _extract_statement_totals_from_text(full_text):
    """
    Extract TOTAL WITHDRAWAL (debit) and TOTAL DEPOSITS (credit) from footer block.
    Layout often includes counts first, then two amounts:
      <no_wd> <no_dep> <total_withdrawal> <total_deposits>
    Returns (td, tc) or (None, None).
    """
    if not full_text:
        return (None, None)

    up = full_text.upper()
    if "TOTAL WITHDRAWAL" not in up or "TOTAL DEPOSITS" not in up:
        return (None, None)

    idx = up.rfind("TOTAL WITHDRAWAL")
    window = full_text[idx: idx + 900] if idx != -1 else full_text

    m = re.search(r"\b\d{1,6}\s+\d{1,6}\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})\b", window)
    if m:
        return (parse_float(m.group(1)), parse_float(m.group(2)))

    money = re.findall(r"-?[\d,]+\.\d{2}", window)
    if len(money) >= 2:
        return (parse_float(money[-2]), parse_float(money[-1]))

    return (None, None)


def _prev_month(yyyy: int, mm: int):
    if mm == 1:
        return (yyyy - 1, 12)
    return (yyyy, mm - 1)


def _infer_statement_month_from_statement_date(full_text):
    """
    Two CIMB conventions exist in the wild:
      - Statement Date = period END   (e.g. 31/10/2025 for the October period).
      - Statement Date = period END + 1..few days (e.g. 02/11/2025 / 03/11/2025
        / 01/11/2025 for the October period; bank issued the statement at the
        start of the following month).

    Day-of-month disambiguates: day 1-7 -> previous month is the period;
    day 8-31 -> the statement-date month IS the period.

    Returns 'YYYY-MM' or None.
    """
    m = _STMT_DATE_RE.search(full_text or "")
    if not m:
        return None
    dd = int(m.group(1))
    mm = int(m.group(2))
    yy_raw = m.group(3)
    yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)
    if not (1 <= mm <= 12 and 2000 <= yy <= 2100):
        return None
    if 1 <= dd <= 7:
        # Statement issued in the few days following month-end -> period was previous month.
        py, pm = _prev_month(yy, mm)
        return f"{py:04d}-{pm:02d}"
    # Statement Date is end-of-period -> same month.
    return f"{yy:04d}-{mm:02d}"


def _dedupe_cimb(rows):
    """
    CIMB-specific dedupe:
    ignore description differences (wrapping/spacing).
    Key by (date, debit, credit, balance, ref_no).

    ref_no is included to disambiguate same-date / same-amount cheques
    (e.g. CLRG CHQ DR 1459 5,000.00 and CLRG CHQ DR 1460 5,000.00 on
    the same day) that would otherwise collide if balance fails to
    distinguish them.
    """
    seen = set()
    out = []
    for r in rows:
        key = (
            str(r.get("date") or "").strip(),
            round(parse_float(r.get("debit", 0.0)), 2),
            round(parse_float(r.get("credit", 0.0)), 2),
            None if r.get("balance") is None else round(parse_float(r.get("balance")), 2),
            str(r.get("ref_no") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _chronological_sort(rows):
    """
    CIMB table is reverse chronological (latest first).
    Convert to chronological (oldest first):
      sort by (date asc, extracted_index desc)
    so within same date we also reverse the order.
    """
    def key(r):
        return (r.get("date") or "9999-99-99", -int(r.get("__idx", 0)))
    return sorted(rows, key=key)


def _extract_last_balance_token(line):
    """
    Return (balance_float, first_money_index)
    """
    toks = line.split()
    last_idx = None
    for i in range(len(toks) - 1, -1, -1):
        if _MONEY_TOKEN_RE.match(toks[i]):
            last_idx = i
            break
    if last_idx is None:
        return None, None

    bal = parse_float(toks[last_idx])

    first_money_idx = None
    for i, t in enumerate(toks):
        if t == "0" or _MONEY_TOKEN_RE.match(t):
            first_money_idx = i
            break

    return bal, first_money_idx


# -----------------------------
# Text fallback parser (if tables fail)
# -----------------------------

def _parse_transactions_cimb_text(pdf, source_filename, detected_year, bank_name, closing_balance):
    """
    Text parser:
    - collect rows with date/desc/balance (raw order)
    - reorder to chronological
    - infer debit/credit by balance delta (fallback only)
    - capture opening balance line (no date) and emit synthetic opening row
    """
    raw = []
    idx = 0
    prev_balance = None
    latest_tx_date = None

    opening_balance_value = None
    opening_balance_page = None

    cur = None  # {"date":..., "parts":[...], "page":...}

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for ln in lines:
            up = ln.upper()

            # Opening balance line (no date)
            if _OPENING_LINE_RE.match(ln):
                bal, _ = _extract_last_balance_token(ln)
                if bal is not None:
                    opening_balance_value = bal
                    opening_balance_page = page_num
                    prev_balance = bal
                continue

            # ignore closing balance line here
            if "CLOSING BALANCE" in up and "BAKI" in up:
                continue

            # Start of transaction
            m = re.match(r"^(\d{2}/\d{2}/\d{4})\s+(.*)$", ln)
            if m:
                cur = {"date": m.group(1), "parts": [m.group(2)], "page": page_num}

                # sometimes includes balance same line
                bal, first_money_idx = _extract_last_balance_token(ln)
                if bal is not None:
                    toks = ln.split()
                    desc = " ".join(toks[1:first_money_idx]) if first_money_idx is not None else " ".join(toks[1:])
                    date_iso = format_date(cur["date"], detected_year)
                    if date_iso:
                        idx += 1
                        desc_raw = clean_text(desc)
                        desc_clean = _clean_description(desc_raw)
                        raw_row = {
                            "date": date_iso,
                            "description": desc_clean,
                            "balance": round(bal, 2),
                            "page": page_num,
                            "__idx": idx,
                        }
                        if desc_clean != desc_raw:
                            raw_row["_raw_description"] = desc_raw
                        raw.append(raw_row)
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso
                    cur = None
                continue

            # Continuation
            if cur is not None:
                bal, first_money_idx = _extract_last_balance_token(ln)
                if bal is not None:
                    toks = ln.split()
                    cur["parts"].append(" ".join(toks[:first_money_idx]) if first_money_idx is not None else ln)
                    date_iso = format_date(cur["date"], detected_year)
                    if date_iso:
                        idx += 1
                        desc_raw = clean_text(" ".join(cur["parts"]))
                        desc_clean = _clean_description(desc_raw)
                        raw_row = {
                            "date": date_iso,
                            "description": desc_clean,
                            "balance": round(bal, 2),
                            "page": cur["page"],
                            "__idx": idx,
                        }
                        if desc_clean != desc_raw:
                            raw_row["_raw_description"] = desc_raw
                        raw.append(raw_row)
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso
                    cur = None
                else:
                    cur["parts"].append(ln)

    # Full-doc closing fallback if needed
    if closing_balance is None:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        closing_balance = extract_closing_balance_from_text(full_text)

    # reorder before delta inference
    raw = _chronological_sort(raw)

    txs = []
    for r in raw:
        bal = parse_float(r.get("balance"))
        debit = credit = 0.0
        if prev_balance is not None:
            delta = round(bal - prev_balance, 2)
            if delta > 0:
                credit = delta
            elif delta < 0:
                debit = -delta

        txs.append({
            "date": r.get("date"),
            "description": r.get("description"),
            "debit": round(debit, 2),
            "credit": round(credit, 2),
            "balance": round(bal, 2),
            "page": r.get("page"),
            "source_file": source_filename,
            "bank": bank_name,
            "__idx": r.get("__idx", 0),
        })
        prev_balance = bal

    # Emit synthetic opening row (labeled clearly)
    if opening_balance_value is not None:
        anchor = latest_tx_date or (txs[0]["date"] if txs else f"{detected_year}-01-01")
        opening_date = f"{anchor[:8]}01" if re.match(r"^\d{4}-\d{2}-\d{2}$", anchor) else f"{detected_year}-01-01"
        txs.insert(0, {
            "date": opening_date,
            "description": "OPENING BALANCE (PAGE 1)",
            "debit": 0.0,
            "credit": 0.0,
            "balance": round(float(opening_balance_value), 2),
            "page": opening_balance_page,
            "source_file": source_filename,
            "bank": bank_name,
            "is_opening_balance": True,
            "opening_balance_source": "page_1",
            "__idx": -1,
        })

    # Emit synthetic closing row
    if closing_balance is not None:
        cb_date = latest_tx_date or (txs[-1]["date"] if txs else f"{detected_year}-01-01")
        txs.append({
            "date": cb_date,
            "description": "CLOSING BALANCE / BAKI PENUTUP",
            "debit": 0.0,
            "credit": 0.0,
            "balance": round(float(closing_balance), 2),
            "page": None,
            "source_file": source_filename,
            "bank": bank_name,
            "is_statement_balance": True,
            "__idx": 10**12,
        })

    txs = _dedupe_cimb(txs)
    for t in txs:
        t.pop("__idx", None)
        t.setdefault("ref_no", None)
        t.setdefault("is_statement_balance", False)
        t.setdefault("is_opening_balance", False)
        t.setdefault("opening_balance_source", None)
        t.setdefault("statement_month", None)
        t.setdefault("statement_total_debit", None)
        t.setdefault("statement_total_credit", None)
    return txs


# -----------------------------
# Main parser
# -----------------------------

def parse_transactions_cimb(pdf, source_filename=""):
    """
    Parse CIMB statement using pdfplumber.
    Prefer extract_table; fallback to text parsing if tables missing.
    """
    bank_name = "CIMB Bank"
    detected_year = None

    # quick branding + year
    for page in pdf.pages[:2]:
        text = page.extract_text() or ""
        if "CIMB ISLAMIC BANK" in text.upper():
            bank_name = "CIMB Islamic Bank"
        if not detected_year:
            detected_year = extract_year_from_text(text)

    if not detected_year:
        detected_year = str(datetime.now().year)

    # Full PDF text (critical for closing + totals + statement month)
    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    closing_balance = extract_closing_balance_from_text(full_text)
    stmt_total_debit, stmt_total_credit = _extract_statement_totals_from_text(full_text)
    stmt_month = _infer_statement_month_from_statement_date(full_text)

    # Page-1 header for determine_account_type, truncated at the CIMB
    # transaction-table marker ("Date Description Cheque / Ref No ...").
    # Keeping it to the pre-table region avoids the footer disclaimer and
    # embedded transactions from being scanned for facility-limit keywords.
    page1_text = pdf.pages[0].extract_text() if pdf.pages else ""
    header_cut = page1_text or ""
    for _marker in (
        "Date Description Cheque",
        "Date Description",
        "Tarikh Diskripsi",
    ):
        _i = header_cut.find(_marker)
        if _i != -1:
            header_cut = header_cut[:_i]
            break
    header_text = header_cut or None

    # Extract opening balance if present in table rows (often no date)
    opening_balance_value = None
    opening_balance_page = None

    rows = []
    idx = 0
    latest_tx_date = None

    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table()
        if not table:
            continue

        for row in table:
            # Expected: [Date, Desc, Ref, Withdrawal, Deposit, (Tax,) Balance]
            # Modern CIMB statements include a Tax column (7 cols);
            # legacy layouts without Tax are 6 cols. Read balance from the
            # last column to handle both safely.
            if not row or len(row) < 6:
                continue

            first_col = str(row[0]).lower() if row[0] else ""
            if "date" in first_col or "tarikh" in first_col:
                continue

            desc_raw = clean_text(row[1])
            desc = _clean_description(desc_raw)
            desc_l = desc.lower()

            # opening balance row may appear here; capture balance but do not treat as tx
            if "opening balance" in desc_l:
                ob = parse_float(row[-1])
                if ob != 0.0:
                    opening_balance_value = ob
                    opening_balance_page = page_num
                continue

            # require balance (last column)
            if row[-1] is None:
                continue

            date_iso = format_date(row[0], detected_year)
            if not date_iso:
                continue

            debit_val = parse_float(row[3])
            credit_val = parse_float(row[4])

            # skip rows without amounts (continuations)
            if debit_val == 0.0 and credit_val == 0.0:
                continue

            bal = parse_float(row[-1])

            if latest_tx_date is None or date_iso > latest_tx_date:
                latest_tx_date = date_iso

            idx += 1
            row_out = {
                "date": date_iso,
                "description": desc,
                "ref_no": clean_text(row[2]),
                "debit": round(debit_val, 2),
                "credit": round(credit_val, 2),
                "balance": round(bal, 2),
                "page": page_num,
                "source_file": source_filename,
                "bank": bank_name,
                "__idx": idx,  # extraction order
            }
            if desc != desc_raw:
                row_out["_raw_description"] = desc_raw
            rows.append(row_out)

    # If table mode failed, fallback to text mode (also labels opening row)
    if not rows:
        text_rows = _parse_transactions_cimb_text(
            pdf,
            source_filename=source_filename,
            detected_year=detected_year,
            bank_name=bank_name,
            closing_balance=closing_balance,
        )
        # Text-mode helper emits its own synthetic opening row, so it carries
        # the opening balance internally — pass None so determine_account_type
        # relies on row-math + header signals only.
        return finalize_parser_output(
            text_rows,
            header_text=header_text,
            opening_balance=None,
            closing_balance=closing_balance,
        )

    # Deduplicate then reorder to chronological
    rows = _dedupe_cimb(rows)
    rows = _chronological_sort(rows)

    # Text-layer fallback for the OPENING BALANCE line. pdfplumber's table
    # extraction renders this row as 2 cells (label + amount), which the
    # `len(row) < 6` guard above drops before the table-mode handler can see
    # it. Without this fallback, OD accounts whose first transaction isn't a
    # clean baseline reconcile incorrectly (see Huahub Oct'25 OD).
    if opening_balance_value is None:
        opening_balance_value = extract_opening_balance_from_text(full_text)

    # Emit synthetic opening row if we captured it (labeled clearly)
    if opening_balance_value is not None:
        anchor = latest_tx_date or (rows[0]["date"] if rows else f"{detected_year}-01-01")
        opening_date = f"{anchor[:8]}01" if re.match(r"^\d{4}-\d{2}-\d{2}$", anchor) else f"{detected_year}-01-01"
        rows.insert(0, {
            "date": opening_date,
            "description": "OPENING BALANCE (PAGE 1)",
            "ref_no": "",
            "debit": 0.0,
            "credit": 0.0,
            "balance": round(float(opening_balance_value), 2),
            "page": opening_balance_page,
            "source_file": source_filename,
            "bank": bank_name,
            "is_opening_balance": True,
            "opening_balance_source": "page_1",
            "__idx": -1,
        })

    # Emit synthetic closing row from footer
    if closing_balance is not None:
        cb_date = latest_tx_date or (rows[-1]["date"] if rows else f"{detected_year}-01-01")
        rows.append({
            "date": cb_date,
            "description": "CLOSING BALANCE / BAKI PENUTUP",
            "ref_no": "",
            "debit": 0.0,
            "credit": 0.0,
            "balance": round(float(closing_balance), 2),
            "page": None,
            "source_file": source_filename,
            "bank": bank_name,
            "is_statement_balance": True,
            # optional metadata
            "statement_month": stmt_month,
            "statement_total_debit": None if stmt_total_debit is None else round(float(stmt_total_debit), 2),
            "statement_total_credit": None if stmt_total_credit is None else round(float(stmt_total_credit), 2),
            "__idx": 10**12,
        })

    # Final dedupe after adding synthetic rows
    rows = _dedupe_cimb(rows)

    # Remove internal field and default missing metadata
    for r in rows:
        r.pop("__idx", None)
        r.setdefault("ref_no", None)
        r.setdefault("is_statement_balance", False)
        r.setdefault("is_opening_balance", False)
        r.setdefault("opening_balance_source", None)
        r.setdefault("statement_month", None)
        r.setdefault("statement_total_debit", None)
        r.setdefault("statement_total_credit", None)

    # Sprint 4.5: per-PDF account_type determination + statutory stamping.
    # CIMB statements use CR convention universally in the corpus (34/37 CR,
    # 3/37 UNDETERMINED on OCR-only PDFs, 0 OD). The wiring is still required
    # so the classifier receives a locked verdict per PDF instead of re-
    # deriving it.
    return finalize_parser_output(
        rows,
        header_text=header_text,
        opening_balance=opening_balance_value,
        closing_balance=closing_balance,
    )
