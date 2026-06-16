"""
core_utils.py

Project-wide utilities used by Streamlit apps and bank parsers.

Goals:
1) Standardize input handling (PDF bytes)
2) Standardize transaction schema and types
3) Make date/amount parsing resilient across banks
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# PDF INPUT
# -----------------------------
def read_pdf_bytes(pdf_input: Any) -> bytes:
    """Return PDF bytes from:
    - bytes / bytearray
    - Streamlit UploadedFile (has getvalue)
    - file-like objects (has read)
    - filesystem path (str)
    """
    if isinstance(pdf_input, (bytes, bytearray)):
        return bytes(pdf_input)

    # Streamlit UploadedFile
    if hasattr(pdf_input, "getvalue"):
        data = pdf_input.getvalue()
        if data:
            return data

    # file-like
    if hasattr(pdf_input, "read"):
        try:
            pdf_input.seek(0)
        except Exception:
            pass
        data = pdf_input.read()
        if data:
            return data

    # path
    if isinstance(pdf_input, str):
        with open(pdf_input, "rb") as f:
            return f.read()

    raise ValueError("Unable to read PDF bytes from the provided input")


def bytes_to_pdfplumber(pdf_bytes: bytes, password: str | None = None):
    """Helper to open pdfplumber using bytes, with optional password for encrypted PDFs."""
    import pdfplumber  # local import to keep utils lightweight
    kwargs = {}
    if password:
        kwargs["password"] = password
    return pdfplumber.open(BytesIO(pdf_bytes), **kwargs)


# -----------------------------
# NORMALIZATION
# -----------------------------
_WS_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    return _WS_RE.sub(" ", str(text or "")).strip()


def safe_float(value: Any) -> float:
    """Convert numeric strings to float safely.

    Handles:
    - None / empty
    - commas
    - parentheses negatives: (1,234.56)
    - trailing +/-: 123.45- / 123.45+
    - currency symbols and stray text
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return 0.0

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # trailing sign
    trailing_sign = None
    if s.endswith("+"):
        trailing_sign = "+"
        s = s[:-1].strip()
    elif s.endswith("-"):
        trailing_sign = "-"
        s = s[:-1].strip()

    s = s.replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", "."}:
        return 0.0

    try:
        f = float(s)
    except Exception:
        return 0.0

    if neg or trailing_sign == "-":
        f = -abs(f)
    elif trailing_sign == "+":
        f = abs(f)
    return float(f)


def normalize_date(date_value: Any, default_year: Optional[int] = None) -> Optional[str]:
    """Normalize many common bank-statement date formats to ISO YYYY-MM-DD.
    Returns None if parsing fails.
    """
    if date_value is None:
        return None

    s = normalize_text(date_value)
    if not s:
        return None

    # already ISO
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s

    # common patterns (day-first)
    patterns: List[Tuple[str, str]] = [
        (r"^\d{1,2}/\d{1,2}/\d{4}$", "%d/%m/%Y"),
        (r"^\d{1,2}-\d{1,2}-\d{4}$", "%d-%m-%Y"),
        (r"^\d{1,2}/\d{1,2}/\d{2}$", "%d/%m/%y"),
        (r"^\d{1,2}-\d{1,2}-\d{2}$", "%d-%m-%y"),
        (r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$", "%d %b %Y"),
        (r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{2}$", "%d %b %y"),
        (r"^\d{1,2}\s+[A-Za-z]{3}$", "%d %b"),
        (r"^\d{1,2}/\d{1,2}$", "%d/%m"),
        (r"^\d{1,2}-\d{1,2}$", "%d-%m"),
    ]

    for rx, fmt in patterns:
        if not re.fullmatch(rx, s):
            continue
        try:
            if fmt in {"%d %b", "%d/%m", "%d-%m"}:
                if default_year is None:
                    return None
                dt = datetime.strptime(f"{s} {default_year}", fmt + " %Y")
            else:
                dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # last-resort: dateutil
    try:
        from dateutil import parser as dateparser

        dt = dateparser.parse(
            s,
            dayfirst=True,
            yearfirst=False,
            default=datetime(default_year or 2000, 1, 1),
        )
        # if no explicit year and default_year is None, dt will use 2000 and likely be wrong -> reject
        if default_year is None and dt.year == 2000 and not re.search(r"\b\d{4}\b", s):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def infer_default_year(transactions: Iterable[Dict[str, Any]]) -> Optional[int]:
    """Infer a reasonable default year from any transaction that already contains a year."""
    for tx in transactions:
        d = normalize_text(tx.get("date"))
        if re.search(r"\b\d{4}\b", d):
            iso = normalize_date(d)
            if iso:
                return int(iso[:4])
    return None


def advance_year_on_rollover(new_iso: str, previous_iso: Optional[str]) -> str:
    if not previous_iso:
        return new_iso
    try:
        new_d = datetime.strptime(new_iso, "%Y-%m-%d").date()
        prev_d = datetime.strptime(previous_iso, "%Y-%m-%d").date()
    except Exception:
        return new_iso
    if (prev_d - new_d).days > 30:
        return new_d.replace(year=new_d.year + 1).strftime("%Y-%m-%d")
    return new_iso


def ensure_transaction_schema(
    tx: Dict[str, Any],
    *,
    default_bank: str,
    default_source_file: str,
    default_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a sanitized transaction dict with consistent keys and types."""
    raw_date = tx.get("date")
    date_iso = normalize_date(raw_date, default_year=default_year)

    description = normalize_text(tx.get("description"))
    debit = safe_float(tx.get("debit", 0))
    credit = safe_float(tx.get("credit", 0))

    # Some parsers store negative values; normalize to non-negative debit/credit where possible
    if debit < 0 and credit == 0:
        credit = abs(debit)
        debit = 0.0
    if credit < 0 and debit == 0:
        debit = abs(credit)
        credit = 0.0

    balance_raw = tx.get("balance", None)
    balance = safe_float(balance_raw) if balance_raw is not None and str(balance_raw).strip() != "" else None

    page_raw = tx.get("page")
    try:
        page = int(page_raw) if page_raw is not None and str(page_raw).strip() != "" else None
    except Exception:
        page = None

    bank = normalize_text(tx.get("bank")) or default_bank
    source_file = normalize_text(tx.get("source_file")) or default_source_file

    out: Dict[str, Any] = {
        "date": date_iso or normalize_text(raw_date),
        "description": description,
        "debit": round(float(debit), 2),
        "credit": round(float(credit), 2),
        "balance": round(float(balance), 2) if isinstance(balance, (int, float)) else None,
        "page": page,
        "bank": bank,
        "source_file": source_file,
    }

    # retain raw date if normalization changed it
    if date_iso and normalize_text(raw_date) and normalize_text(raw_date) != date_iso:
        out["_raw_date"] = normalize_text(raw_date)

    # Preserve additional parser-provided metadata (scalar JSON-friendly fields only).
    # This prevents accidental loss of useful fields like: seq, account_no, company_name,
    # is_statement_balance, transaction_date, time, etc.
    for k, v in (tx or {}).items():
        if k in out or k.startswith("_"):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v

    # Harmonize common metadata keys without changing calculations.
    # Many banks use account_no/account_number inconsistently; keep both when present.
    if out.get("account_no") and not out.get("account_number"):
        out["account_number"] = normalize_text(out.get("account_no"))
    if out.get("account_number") and not out.get("account_no"):
        out["account_no"] = normalize_text(out.get("account_number"))
    if out.get("company_name"):
        out["company_name"] = normalize_text(out.get("company_name"))

    # PATRONYMIC GUARD METADATA
    # If description contains 2-letter banking-acronym tokens (BA, TL, TF, FD, CT) that
    # appear directly after a patronymic marker (BIN, BINTI, BT, BTE, A/L, A/P, S/O, D/O),
    # emit them here. Downstream classifier (CLASSIFICATION_RULES v3.4+) consults this
    # field and SKIPS C10 / FD / CT banking-acronym rules when the token is a name
    # fragment — killing the "BINTI BA -> Banker's Acceptance" false positive class.
    #
    # Empty-list / no-field semantics: absence means "no ambiguity detected, classifier
    # can apply banking-acronym rules normally".
    try:
        amb = strip_patronymic_ambiguity(description)
        if amb:
            out["_patronymic_ambiguous_tokens"] = amb
    except Exception:
        # Never fail transaction emission on metadata computation
        pass

    # Per-PDF account_type determination rides on the first row only (attached by
    # finalize_parser_output). Preserve dict-valued metadata through the scalar-only
    # passthrough filter above. App.py harvests this into a top-level full_report key.
    det = tx.get("_account_type_determination")
    if isinstance(det, dict):
        out["_account_type_determination"] = det

    return out


def normalize_transactions(
    transactions: List[Dict[str, Any]],
    *,
    default_bank: str,
    source_file: str,
) -> List[Dict[str, Any]]:
    """Normalize a list of transactions and infer year if needed."""
    year = infer_default_year(transactions)
    return [
        ensure_transaction_schema(
            tx,
            default_bank=default_bank,
            default_source_file=source_file,
            default_year=year,
        )
        for tx in transactions
    ]


def transaction_fingerprint(tx: Dict[str, Any]) -> str:
    """Create a stable fingerprint suitable for de-duplication."""
    parts = [
        normalize_text(tx.get("date")),
        normalize_text(tx.get("description")),
        f"{safe_float(tx.get('debit', 0)):.2f}",
        f"{safe_float(tx.get('credit', 0)):.2f}",
        "" if tx.get("balance") is None else f"{safe_float(tx.get('balance')):.2f}",
        normalize_text(tx.get("bank")),
    ]
    blob = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(blob).hexdigest()


def dedupe_transactions(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for tx in transactions:
        fp = transaction_fingerprint(tx)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(tx)
    return out


# =========================================================
# Affin-specific fixes (DO NOT affect other banks unless called)
# =========================================================
def dedupe_transactions_affin(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Affin statements are frequently OCR-based; description strings vary across files.
    De-dupe must NOT depend on description/page/source_file, or overlap PDFs will inflate totals.

    Key:
      (date, debit, credit, balance, bank)
    """
    seen = set()
    out = []
    for tx in transactions:
        date = normalize_text(tx.get("date"))
        bank = normalize_text(tx.get("bank"))
        debit = round(safe_float(tx.get("debit", 0.0)), 2)
        credit = round(safe_float(tx.get("credit", 0.0)), 2)
        bal_raw = tx.get("balance")
        balance = None if bal_raw is None else round(safe_float(bal_raw), 2)

        key = (date, debit, credit, balance, bank)
        if key in seen:
            continue
        seen.add(key)
        out.append(tx)
    return out


def filter_affin_balance_outliers(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop rows whose balance is a clear OCR outlier (e.g. extra digit -> millions),
    which causes massive delta-based phantom debits/credits.

    Method:
      - compute median balance
      - keep balances within +/- 1.5M of median
      - keep rows with balance=None unchanged
    """
    bals = [safe_float(t.get("balance")) for t in transactions if t.get("balance") is not None]
    if len(bals) < 10:
        return transactions

    bals_sorted = sorted(bals)
    median = bals_sorted[len(bals_sorted) // 2]

    lo = median - 1_500_000
    hi = median + 1_500_000

    out = []
    for t in transactions:
        b = t.get("balance")
        if b is None:
            out.append(t)
            continue
        bf = safe_float(b)
        if lo <= bf <= hi:
            out.append(t)

    return out


# =========================================================
# Monthly Summary - PRESENTATION STANDARDIZATION ONLY
# =========================================================
def compute_swing(highest_balance: Any, lowest_balance: Any) -> Optional[float]:
    """Compute swing = highest - lowest safely."""
    if highest_balance is None or lowest_balance is None:
        return None
    try:
        return round(float(safe_float(highest_balance) - safe_float(lowest_balance)), 2)
    except Exception:
        return None


def present_monthly_summary_standard(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert an existing monthly summary (any bank-specific schema) into the standard schema:

      opening_balance, total_debit, total_credit, highest_balance, lowest_balance,
      swing, ending_balance, source_files

    This is intentionally "presentation-only":
    - It does NOT recalculate debit/credit/opening/ending.
    - It only maps fields and computes swing from existing high/low.
    """
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        highest = r.get("highest_balance")
        lowest = r.get("lowest_balance")
        out.append(
            {
                "month": r.get("month"),
                "opening_balance": r.get("opening_balance"),
                "total_debit": r.get("total_debit"),
                "total_credit": r.get("total_credit"),
                "highest_balance": highest,
                "lowest_balance": lowest,
                "swing": compute_swing(highest, lowest),
                "ending_balance": r.get("ending_balance"),
                "source_files": r.get("source_files"),
            }
        )
    return out


# =========================================================
# ACCOUNT TYPE DETERMINATION
# Universal, bank-agnostic detection: CR / OD / UNDETERMINED.
# One source of truth so the classifier doesn't re-detect.
#
# Design (per user direction, Sprint 4.5):
#   - Single binary outcome (CR vs OD). Islamic Cashline-i / CAP-i / SAP-i
#     is treated as OD (functionally identical revolving credit). No
#     CASH_LINE enum value in schema.
#   - Evidence-based detection from balance and header signals only:
#       1. Header-disclosed non-zero Overdraft / Cashline-i limit.
#       2. DR-suffix majority on balances (Alliance / Ambank positive-debt
#          convention) — 50% threshold to avoid flagging temporary drawdowns.
#       3. Sustained negative balance (Ambank / UOB negated-OD convention) —
#          same 50% threshold.
#       4. OD row-math reconciles cleanly while CR math does not (fallback).
#   - NO transaction-description keyword scanning. Keywords like
#     `OVERDRAFT INTEREST`, `CASHLINE-i PROFIT CHARGED`, `DR Interest` all
#     turned out to generate false positives when they appeared in transfer
#     memos (Instant Transfer / RFLX / DuitNow) settling OTHER accounts,
#     or in one-off "temporary drawdown" events on plain CR accounts.
#     The balance convention and header limit are the authoritative signals.
#   - Header CR keyword is NOT a lock signal. "CURRENT A/C" in the header
#     doesn't rule out an attached OD being drawn; rely on balance instead.
#   - 50% threshold on DR-suffix / negative-balance distinguishes:
#       * OD facility account: sustained DR or sustained negative (>= 50%)
#       * CA with temporary drawdown: brief dip (< 50%)
# =========================================================

BALANCE_TOLERANCE = 1.00  # ringgit — used for balance trail reconciliation


# --- Header facility-limit extractors (disclosed facility limits = OD lock) ---
#
# Matches lines like:
#   Overdraft Kemudahan Tunai: 1,848,047.51 DR        (Alliance)
#   Overdraft Limit: RM 500,000.00                    (generic)
#   OD Facility : RM 100,000.00                       (generic)
#   Cashline-i Limit : RM 250,000.00                  (Bank Rakyat)
# Zero-amount limits are ignored (facility not active).
_HDR_OD_LIMIT_RE = re.compile(
    r"(?:Overdraft(?:\s+Kemudahan\s+Tunai)?|OD\s+(?:Facility|Limit))"
    r"\s*[:]?\s*(?:RM\s*)?([0-9,]+\.\d{2})",
    re.IGNORECASE,
)
_HDR_CASHLINE_LIMIT_RE = re.compile(
    r"Cash\s*line(?:-?i)?[\s-]*Limit"
    r"\s*[.:]?\s*(?:RM\s*)?([0-9,]+\.\d{2})",
    re.IGNORECASE,
)


def _extract_facility_limits(header_text: Optional[str]) -> Dict[str, List[float]]:
    """Extract non-zero facility limits from page-1 header text.

    Returns {"overdraft": [...], "cashline": [...]} with amounts > 0 only.
    Either list being non-empty is a definitive OD signal.
    """
    out: Dict[str, List[float]] = {"overdraft": [], "cashline": []}
    if not header_text:
        return out
    for m in _HDR_OD_LIMIT_RE.finditer(header_text):
        try:
            v = float(m.group(1).replace(",", ""))
            if v > 0:
                out["overdraft"].append(v)
        except Exception:
            pass
    for m in _HDR_CASHLINE_LIMIT_RE.finditer(header_text):
        try:
            v = float(m.group(1).replace(",", ""))
            if v > 0:
                out["cashline"].append(v)
        except Exception:
            pass
    return out


def _scan_dr_suffix_ratio(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count transactions whose `balance_sign` field was set to "DR" by the
    parser (Alliance / Ambank positive-debt-magnitude convention).

    Parsers that don't track DR/CR suffix leave `balance_sign` unset — this
    returns 0/0 for those banks and the decision tree falls through to
    the negative-balance / row-math signals.
    """
    total = 0
    dr = 0
    for t in transactions:
        sign = t.get("balance_sign")
        if sign is None:
            continue
        total += 1
        if str(sign).upper() == "DR":
            dr += 1
    return {
        "dr_rows": dr,
        "total_with_sign": total,
        "dr_ratio": round(dr / total, 4) if total else 0.0,
    }


def _row_level_formula_match(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For each consecutive row pair with balances, test whether the delta matches
    CR math (credit - debit) or OD math (debit - credit).

    Returns counts + ratios so caller can decide.
    """
    rows = [t for t in transactions if t.get("balance") is not None]
    cr_matches = 0
    od_matches = 0
    total_checks = 0
    for i in range(1, len(rows)):
        prev_bal = safe_float(rows[i - 1].get("balance"))
        curr_bal = safe_float(rows[i].get("balance"))
        credit = safe_float(rows[i].get("credit", 0))
        debit = safe_float(rows[i].get("debit", 0))
        actual_delta = curr_bal - prev_bal

        cr_expected = credit - debit
        od_expected = debit - credit

        cr_hit = abs(actual_delta - cr_expected) < BALANCE_TOLERANCE
        od_hit = abs(actual_delta - od_expected) < BALANCE_TOLERANCE

        # Skip trivially ambiguous rows (credit == debit == 0): would match both and tell us nothing
        if credit == 0 and debit == 0:
            continue

        total_checks += 1
        if cr_hit:
            cr_matches += 1
        if od_hit:
            od_matches += 1

    return {
        "cr_match_count": cr_matches,
        "od_match_count": od_matches,
        "total_checks": total_checks,
        "cr_match_ratio": round(cr_matches / total_checks, 4) if total_checks else 0.0,
        "od_match_ratio": round(od_matches / total_checks, 4) if total_checks else 0.0,
    }


def _negative_balance_stats(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count rows with negative balance (indicator of sustained overdraft)."""
    rows_with_bal = [t for t in transactions if t.get("balance") is not None]
    total = len(rows_with_bal)
    if total == 0:
        return {"negative_rows": 0, "total_rows_with_balance": 0, "negative_ratio": 0.0}
    negative = sum(1 for t in rows_with_bal if safe_float(t.get("balance")) < 0)
    return {
        "negative_rows": negative,
        "total_rows_with_balance": total,
        "negative_ratio": round(negative / total, 4),
    }


def determine_account_type(
    transactions: List[Dict[str, Any]],
    *,
    opening_balance: Optional[float] = None,
    closing_balance: Optional[float] = None,
    header_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Universal CR / OD detector. Bank-agnostic. Single-label outcome
    (CR / OD / UNDETERMINED). Islamic revolving credit (Cashline-i, CAP-i,
    SAP-i, etc.) is classified as OD — same facility type, different label.

    Evidence-based detection — any one of these signals locks OD:
      1. Header-disclosed non-zero Overdraft / Cashline-i limit
      2. DR-suffix on >= 50% of balance rows (Alliance/Ambank positive-debt)
      3. Sustained negative balance on >= 50% of rows (Ambank/UOB negated)
      4. OD row-math fits >= 90% while CR math fits less (Alliance fallback)

    50% threshold on (2) and (3) distinguishes an OD facility from a plain
    CA that briefly dipped into debit on a single cheque — a brief dip does
    NOT change account type.

    Transaction-description keywords are DELIBERATELY NOT scanned. Every
    historic keyword (OVERDRAFT INTEREST, CASHLINE-i PROFIT CHARGED, DR
    Interest, etc.) generated false positives when it appeared in a transfer
    memo settling some other account, or on a plain CA during a brief
    overdraw event. Balance convention + header limit are authoritative.

    Returns an account_type_determination dict suitable for emission into
    the final JSON.
    """
    # ---- Gather evidence ----
    facility_limits = _extract_facility_limits(header_text)
    dr_sfx = _scan_dr_suffix_ratio(transactions)
    neg_stats = _negative_balance_stats(transactions)
    row_test = _row_level_formula_match(transactions)

    cr_ratio = row_test["cr_match_ratio"]
    od_ratio = row_test["od_match_ratio"]
    neg_ratio = neg_stats["negative_ratio"]
    dr_ratio = dr_sfx["dr_ratio"]

    cr_math_fits = cr_ratio >= 0.90
    od_math_fits = od_ratio >= 0.90
    has_sustained_negative = neg_ratio >= 0.50
    has_dr_suffix_majority = dr_ratio >= 0.50 and dr_sfx["total_with_sign"] > 0

    # Opening/closing corroboration (informational — not a lock driver)
    cr_trail: Optional[Dict[str, Any]] = None
    od_trail: Optional[Dict[str, Any]] = None
    if opening_balance is not None and closing_balance is not None:
        total_d = sum(safe_float(t.get("debit", 0)) for t in transactions)
        total_c = sum(safe_float(t.get("credit", 0)) for t in transactions)
        open_f = safe_float(opening_balance)
        close_f = safe_float(closing_balance)
        cr_close = open_f + total_c - total_d
        od_close = open_f + total_d - total_c
        cr_trail = {
            "computed_closing": round(cr_close, 2),
            "actual_closing": round(close_f, 2),
            "delta": round(abs(cr_close - close_f), 2),
            "reconciles": abs(cr_close - close_f) < BALANCE_TOLERANCE,
        }
        od_trail = {
            "computed_closing": round(od_close, 2),
            "actual_closing": round(close_f, 2),
            "delta": round(abs(od_close - close_f), 2),
            "reconciles": abs(od_close - close_f) < BALANCE_TOLERANCE,
        }

    # ---- Decision (priority cascade — first match wins) ----
    locked_type = "UNDETERMINED"
    confidence = "LOW"
    rationale_parts: List[str] = []

    if facility_limits["overdraft"] or facility_limits["cashline"]:
        locked_type = "OD"
        confidence = "HIGH"
        bits = []
        if facility_limits["overdraft"]:
            bits.append(f"Overdraft limit(s) {facility_limits['overdraft']}")
        if facility_limits["cashline"]:
            bits.append(f"Cashline-i limit(s) {facility_limits['cashline']}")
        rationale_parts.append("Header discloses non-zero " + ", ".join(bits))
    elif has_dr_suffix_majority:
        locked_type = "OD"
        confidence = "HIGH"
        rationale_parts.append(
            f"DR-suffix on {dr_sfx['dr_rows']}/{dr_sfx['total_with_sign']} balance rows "
            f"({dr_ratio:.0%}) — Alliance/Ambank positive-debt-magnitude convention"
        )
    elif has_sustained_negative:
        locked_type = "OD"
        confidence = "HIGH"
        rationale_parts.append(
            f"{neg_ratio:.0%} of balance rows negative "
            f"({neg_stats['negative_rows']}/{neg_stats['total_rows_with_balance']}) "
            f"— Ambank/UOB negated-OD convention"
        )
    elif od_math_fits and not cr_math_fits:
        locked_type = "OD"
        confidence = "MEDIUM"
        rationale_parts.append(
            f"OD row-math reconciles {od_ratio:.0%} while CR math only {cr_ratio:.0%} "
            f"— OD facility with positive-stored debt magnitude"
        )
    elif cr_math_fits:
        locked_type = "CR"
        confidence = "HIGH" if row_test["total_checks"] >= 10 else "MEDIUM"
        rationale_parts.append(
            f"No OD signal in header limit, balance-sign, or sustained-negative; "
            f"CR row-math reconciles {cr_ratio:.0%} "
            f"({row_test['cr_match_count']}/{row_test['total_checks']} checkable rows)"
        )
    elif row_test["total_checks"] == 0 and len(transactions) == 0:
        locked_type = "UNDETERMINED"
        confidence = "LOW"
        rationale_parts.append(
            "No transactions extracted (likely OCR-only PDF with empty text layer)"
        )
    elif row_test["total_checks"] == 0 and len(transactions) > 0:
        # Some transactions extracted but none had checkable balance pairs
        # (e.g. single-row statement, or balance column missing). Default CR
        # unless there's any negative or DR-suffix signal.
        locked_type = "CR"
        confidence = "LOW"
        rationale_parts.append(
            f"No checkable balance-trail rows; default CR (no OD evidence from "
            f"header limit, DR-suffix, or negative balance)"
        )
    elif cr_ratio > od_ratio and not has_sustained_negative and not has_dr_suffix_majority:
        # Parser-gap case: CR math is the dominant reconciling formula but
        # doesn't cross the 90% threshold — typically a small number of
        # row-pairs where the parser missed a balance or a split-line row.
        # None of the OD lock signals fire (no limit, no DR-suffix majority,
        # no sustained negative). Low OD-math % here is parser noise, not
        # a real OD signal. Default CR with LOW confidence rather than
        # mask as UNDETERMINED.
        locked_type = "CR"
        confidence = "LOW"
        rationale_parts.append(
            f"No OD signal fires (header limit, DR-suffix, negative-balance all "
            f"clear); CR row-math dominant ({cr_ratio:.0%}) vs OD ({od_ratio:.0%}) "
            f"but below 90% threshold — parser-gap suspected; default CR"
        )
    else:
        locked_type = "UNDETERMINED"
        confidence = "LOW"
        rationale_parts.append(
            f"Ambiguous: no OD signals, neither CR ({cr_ratio:.0%}) nor OD "
            f"({od_ratio:.0%}) math reconciles cleanly on "
            f"{row_test['total_checks']} checkable rows"
        )

    return {
        "tested_formulas": ["CR", "OD"],
        "facility_limits_in_header": facility_limits,       # {overdraft: [...], cashline: [...]}
        "dr_suffix_stats": dr_sfx,                          # {dr_rows, total_with_sign, dr_ratio}
        "row_level_test": row_test,
        "cr_trail": cr_trail,
        "od_trail": od_trail,
        "negative_balance_rows": neg_stats["negative_rows"],
        "total_rows_with_balance": neg_stats["total_rows_with_balance"],
        "negative_ratio": neg_stats["negative_ratio"],
        "locked_type": locked_type,
        "confidence": confidence,
        "locked_rationale": "; ".join(rationale_parts),
    }


def stamp_account_type_once(
    transactions: List[Dict[str, Any]],
    account_type: str,
) -> List[Dict[str, Any]]:
    """Stamp `account_type` on every transaction IN PLACE. Use after determine_account_type().

    Enforces: account_type is per-PDF-file, not per-row. Never re-evaluate per row based on
    balance magnitude. Fixes the KYDN BUG-002 class of bugs where low-positive-balance rows
    got wrongly flipped to OD.
    """
    for t in transactions:
        t["account_type"] = account_type
    return transactions


# =========================================================
# STATUTORY BUCKETING — truncation-tolerant, cross-bank
# One source of truth. Classifier trusts this field.
# =========================================================

# Ordered most-specific first. Patterns tolerate FPX 20-char truncation.
# RHB uses a tighter "FPX B2B <KEYWORD> -" format that strips beyond a single
# word — handled by the FPX-B2B-anchored alternates on KWSP, LHDN, and SOCSO below.
_STATUTORY_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("LHDN", re.compile(
        r"\b(?:LHDN|LEMBAGA\s+HASIL(?:\s+DAL(?:AM)?(?:\s+NEGER[I]?)?)?|FPX\s*B2B\s+LEMBAGA(?=\s|$|/|-))\b",
        re.IGNORECASE,
    )),
    ("HRDF", re.compile(
        r"\b(?:HRDF|PSMB|PEMBANGUNAN\s+SUMBER\s+MANU(?:SIA)?)\b",
        re.IGNORECASE,
    )),
    ("KWSP", re.compile(
        r"\b(?:KWSP|EPF|KUMPULAN\s+WANG\s+SIMP(?:A|AN|ANAN)?(?:\s+PEKERJA)?|FPX\s*B2B\s+KUMPULAN(?=\s|$|/|-))\b",
        re.IGNORECASE,
    )),
    ("SOCSO", re.compile(
        r"(?:\b(?:SOCSO|PERKESO|PERTUBUHAN\s+KESELAM(?:AT(?:AN)?|A)(?:\s+SOSIAL)?|EIS)\b"
        r"|\bFPX\s*B2B\s+PERTUBUH(?=\s|$|/|-)"
        r"|\bPERTUBUH(?:AN)?\s+CP(?=[_\s]|$))",
        re.IGNORECASE,
    )),
]


def statutory_bucket_for(description: Any) -> Optional[str]:
    """Return 'KWSP' / 'SOCSO' / 'LHDN' / 'HRDF' or None for a transaction description.

    Truncation-tolerant: handles FPX 20-char truncated forms
    (e.g. 'KUMPULAN WANG SIMPAN' matches KWSP even without '...PEKERJA').

    Cross-bank: same rules for every parser. Classifier trusts this bucket and does NOT
    re-regex the raw description for statutory keywords.
    """
    s = normalize_text(description)
    if not s:
        return None
    for bucket, pattern in _STATUTORY_PATTERNS:
        if pattern.search(s):
            return bucket
    return None


_OWN_PARTY_SUFFIX_RE = re.compile(
    r"\b(?:SDN\.?\s*BHD\.?|SDN\.?|BHD\.?|BERHAD|&\s*CO\.?|\(M\)|PTY|LTD\.?)\b",
    re.IGNORECASE,
)


def _normalize_for_own_party_check(text: Any) -> str:
    if not text:
        return ""
    s = str(text).upper()
    s = _OWN_PARTY_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def stamp_statutory_buckets(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stamp `statutory_bucket` on every transaction IN PLACE.

    Sets the field to 'KWSP' / 'SOCSO' / 'LHDN' / 'HRDF' when matched, else None.
    Classifier trusts this field and does NOT re-regex the description.

    Side-gate (BUG-001 fix): never stamp on CR rows (statutory contributions
    are always outbound) and never stamp when the company's own normalised
    name appears in the description (own-party transfer earmarked for a
    future statutory run is not the payment itself — e.g. MUHAFIZ Feb 2026
    'DUITNOW TO ACCOUNT EPF PAYMENT MUHAFIZ SECURITY SDN' RM 600K CR).
    """
    for t in transactions:
        try:
            credit = float(t.get("credit") or 0)
        except (TypeError, ValueError):
            credit = 0.0
        if credit > 0:
            t["statutory_bucket"] = None
            continue

        own_name = t.get("own_party_name") or t.get("company_name")
        if own_name:
            own_norm = _normalize_for_own_party_check(own_name)
            desc_norm = _normalize_for_own_party_check(t.get("description"))
            if len(own_norm) >= 4 and own_norm in desc_norm:
                t["statutory_bucket"] = None
                continue

        t["statutory_bucket"] = statutory_bucket_for(t.get("description"))
    return transactions


# =========================================================
# ACCOUNT-HOLDER EXTRACTION (Sprint 7 #12)
# =========================================================
# Bank-agnostic helpers that pull the statement holder's name from page-1
# header text. Used by parsers that don't have their own bespoke extractor
# (alliance.py / bank_islam.py / ocbc.py keep theirs because they handle
# bank-specific quirks). Output is stamped on every transaction's
# `company_name` and `own_party_name` so the verification harnesses (which
# build company_roots from per-row metadata) can fire C01/C02 own-party.

_HEADER_HOLDER_SUFFIX_RE = re.compile(
    r"\b(?:SDN\.?\s*BHD\.?|BERHAD|ENTERPRISE|TRADING|SERVICES|HOLDINGS|RESOURCES|CONSULTING)\b",
    re.IGNORECASE,
)
_HEADER_HOLDER_SKIP_PREFIXES = (
    "TRANSACTION STATEMENT", "STATEMENT OF ACCOUNT", "PENYATA AKAUN",
    "ACCOUNT STATEMENT", "TARIKH", "STATEMENT DATE", "MUKA", "PAGE",
    "BRANCH", "CAWANGAN", "PROTECTED BY", "DILINDUNGI",
    "PIDM", "PERBADANAN", "MEMBER OF", "TEL:", "PHONE",
    "NOMBOR", "ACCOUNT NO", "A/C NO", "CIF", "STATEMENT PERIOD",
    "REFLEX CASH", "JOMPAY", "DUITNOW",
    "MALAYAN BANKING", "PUBLIC BANK", "CIMB BANK", "RHB BANK",
    "BANK ISLAM", "ALLIANCE BANK", "OCBC BANK", "UOB",
    "AMBANK", "AGRO BANK", "BANK MUAMALAT", "BANK RAKYAT",
    "HONG LEONG", "AFFIN",
)
_HEADER_HOLDER_BAD_TOKENS_RE = re.compile(
    r"^(?:NO\.?\s+\d|LOT\s|JLN|JALAN|TAMAN|KAMPUNG|TMN|BLOK|BLOCK|"
    r"\d+(?:ST|ND|RD|TH)?\s+FLOOR|FLOOR|UNIT|LEVEL)\b",
    re.IGNORECASE,
)

# Sprint 7 #13 — concatenated-form holder match for Bank Rakyat DATAPOS
# layout (and similar) where pdfplumber loses inter-word spacing and the
# corporate-suffix anchor lacks a leading word boundary
# (e.g. `KOPERASIKAKITANGANFELCRA(M)BERHAD`,
# `AZLANBOUTIQUEENTERPRISE`).
_HEADER_HOLDER_CONCAT_RE = re.compile(
    r"([A-Z][A-Z0-9.&()\-]*"
    r"(?:BERHAD|SDNBHD|ENTERPRISE|TRADING|SERVICES|HOLDINGS|RESOURCES))\b",
    re.IGNORECASE,
)
# Skip bank-self lines (which contain BERHAD / BHD as a property of the
# bank itself, not the account holder). Patterns tolerate concatenated
# forms via `\s*` (zero-or-more whitespace).
_HEADER_HOLDER_BANK_SELF_RE = re.compile(
    r"(?:MALAYAN\s*BANKING|BANK\s*RAKYAT|PUBLIC\s*BANK|CIMB\s*BANK|"
    r"RHB\s*BANK|BANK\s*ISLAM|ALLIANCE\s*BANK|HONG\s*LEONG|AMBANK|"
    r"AGRO\s*BANK|BANK\s*MUAMALAT|AFFIN\s*BANK|OCBC|\bUOB\b)",
    re.IGNORECASE,
)


def extract_account_holder_from_header(
    header_text: Optional[str],
    *,
    relaxed: bool = False,
) -> Optional[str]:
    """Pull the account-holder name from page-1 header text.

    Strict mode (relaxed=False): require a corporate suffix anchor (SDN BHD /
    BERHAD / ENTERPRISE / TRADING / SERVICES / HOLDINGS / RESOURCES /
    CONSULTING). Best for Maybank, Public Bank, CIMB headers where the
    company line always carries SDN BHD.

    Relaxed mode (relaxed=True): accept the first plausible all-caps line
    that isn't a known header label, address fragment, or postcode line.
    Required for RHB's `JATI WAJA QUALITY SERVICES` / similar names that
    skip the corporate suffix.
    """
    if not header_text:
        return None
    seen_skip = False
    for raw_line in header_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or len(line) < 6:
            continue
        up = line.upper()
        # Strip leading 3-digit branch / sequence codes that some banks prepend
        line = re.sub(r"^\d{1,3}\s+", "", line)
        up = line.upper()
        if any(up.startswith(pref) for pref in _HEADER_HOLDER_SKIP_PREFIXES):
            seen_skip = True
            continue
        if line != up:
            continue
        if re.search(r"\b\d{5}\b", line):
            continue
        if _HEADER_HOLDER_BAD_TOKENS_RE.match(line):
            continue
        if not relaxed:
            m = _HEADER_HOLDER_SUFFIX_RE.search(line)
            if not m:
                continue
            # Truncate at the suffix end so trailing labels (Chinese statement-
            # date markers, 'PENYATA AKAUN', etc.) don't leak into the holder
            # name.
            line = line[: m.end()]
        else:
            tokens = line.split()
            if len(tokens) < 2 or len(tokens) > 8:
                continue
            if not all(re.fullmatch(r"[A-Z][A-Z0-9.&\-/']*", t) for t in tokens):
                continue
        cleaned = re.sub(r"\.", "", line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\bSDN\s*BHD\b", "SDN BHD", cleaned)
        return cleaned

    # Pass 2 — concatenated-form fallback (Bank Rakyat DATAPOS, older RHB).
    # The strict pass requires a `\b<suffix>\b` anchor which fails when
    # pdfplumber loses inter-word spacing. Scan for `<UPPERCASE-RUN><suffix>`
    # without leading word-boundary, while skipping bank-self lines.
    for raw_line in header_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or len(line) < 8:
            continue
        if _HEADER_HOLDER_BANK_SELF_RE.search(line):
            continue
        m = _HEADER_HOLDER_CONCAT_RE.search(line)
        if not m:
            continue
        cand = m.group(1).upper()
        if _HEADER_HOLDER_BANK_SELF_RE.search(cand):
            continue
        if len(cand) < 8:
            continue
        cand = re.sub(r"\.", "", cand)
        cand = re.sub(r"\s+", " ", cand).strip()
        return cand
    return None


def stamp_account_holder(
    transactions: List[Dict[str, Any]],
    holder_name: Optional[str],
) -> None:
    """Stamp `company_name` + `own_party_name` on every tx IN PLACE if the
    holder name is set. Uses setdefault so parsers that already stamp these
    fields (alliance / bank_islam / ocbc / uob) aren't overwritten."""
    if not holder_name:
        return
    name_norm = re.sub(r"\s+", " ", holder_name).strip().upper()
    if not name_norm:
        return
    for t in transactions:
        t.setdefault("company_name", name_norm)
        t.setdefault("own_party_name", name_norm)


def finalize_parser_output(
    transactions: List[Dict[str, Any]],
    *,
    header_text: Optional[str] = None,
    opening_balance: Optional[float] = None,
    closing_balance: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Post-parse finalization. Call this from each bank parser's entry point right
    before returning rows. Does two things:

    1. Locks account_type per-PDF via `determine_account_type` and stamps it on every
       row via `stamp_account_type_once`. Overwrites any bank-specific heuristic the
       parser may have stamped earlier.
    2. Stamps `statutory_bucket` on every row via `stamp_statutory_buckets`.

    The full determination dict (locked_type, confidence, rationale, formula ratios,
    negative-balance stats, etc.) is attached to the first row under the preserved
    metadata key `_account_type_determination`, where `ensure_transaction_schema`
    explicitly passes it through normalization. `app.py` harvests this per source_file
    for emission into `full_report.json`'s top-level `account_type_determinations` list.
    """
    if not transactions:
        return transactions

    determination = determine_account_type(
        transactions,
        opening_balance=opening_balance,
        closing_balance=closing_balance,
        header_text=header_text,
    )
    stamp_account_type_once(transactions, determination["locked_type"])
    stamp_statutory_buckets(transactions)

    # Sprint 7 #12 — bank-agnostic account-holder stamping for parsers that
    # don't extract it themselves. setdefault preserves whatever a parser-
    # specific helper already stamped (alliance / bank_islam / ocbc / uob).
    holder = extract_account_holder_from_header(header_text, relaxed=False)
    stamp_account_holder(transactions, holder)

    transactions[0]["_account_type_determination"] = determination
    return transactions


# =========================================================
# STOP-WORDS — names that should never survive as counterparty
# =========================================================

_MONTH_TOKENS: set = {
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "SEPT", "OCT", "NOV", "DEC",
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
}

_PURPOSE_STOP_TOKENS: set = {
    "BULK", "SALARY", "PAYMENT", "PAY", "TRANSFER", "TRF",
    "FUND", "SETTLEMENT", "SETTLE",
    "LOAN", "REPAYMENT", "ADVANCE",
    "IBG", "ADVICE", "CREDIT", "DEBIT", "ACCOUNT",
    "HUB", "MISC",
    "DUITNOW", "FPX", "INSTANT",
    "CR", "DR", "TO", "FROM", "FR",
    "TRANSACTION", "TRANSACTIONS", "TRANS",
    "DEPOSIT", "WITHDRAWAL", "CASH",
    "INWARD", "OUTWARD",
}


def should_drop_as_counterparty(name: Any) -> bool:
    """Return True if a cleaned counterparty name is actually just a purpose fragment /
    month / stop word and should be dropped (transaction has no real counterparty).

    Examples that return True:
      "JAN FEB", "BULK", "LOAN REPAYMENT", "CR ADVICE", "PAYMENT TRANSFER",
      "IBG TRANSACTION", "" (empty), "X" (too short), "123" (no letters).
    """
    s = normalize_text(name).upper()
    if not s:
        return True
    if not re.search(r"[A-Z]", s):
        return True
    if len(s) <= 2:
        return True
    toks = s.split()
    # all tokens are months (covers "JAN FEB" / "FEB MAR" etc.)
    if toks and all(t in _MONTH_TOKENS for t in toks):
        return True
    # all tokens are purpose stop-words or months
    if toks and all(t in _PURPOSE_STOP_TOKENS or t in _MONTH_TOKENS for t in toks):
        return True
    return False


# =========================================================
# 2026-05-02 BUG-003 — Malaysian company-suffix normaliser.
# PDF column-width clipping truncates "SDN BHD" to variants like "SB",
# "SDN", "SDN.", "SDN B", "SDN BH", "SDN. B", "SDN. BH" across many
# banks (Kay R RHB was the trigger; same pattern observed wherever a
# long company name sits at the right edge of a transaction-description
# column). Restoring the canonical "SDN BHD" form before downstream
# normalisation lets fragmented buckets merge and lets classifier rules
# match on a single literal suffix.
#
# Tail-only by design: only fires when the truncation is at the very
# end of the candidate name. Mid-string "SB" tokens (e.g. "ABC SB DUMMY")
# are left alone. Applied to EXTRACTED counterparty names (post
# _extract_counterparty), not raw descriptions — this enforces the
# entity-name boundary before the regex sees the string.
#
# Alternation order matters: longest forms (SDN BHD, SDN BH, SDN B) are
# tried before bare SDN so "WILMAR ... SDN BHD" matches the full form
# rather than back-tracking to "SDN" and corrupting the trailing BHD.
# =========================================================
_COMPANY_SUFFIX_NORM_RE = re.compile(
    r"\b(SB|SDN\s*\.?\s*BHD\.?|SDN\s*\.?\s*BH|SDN\s*\.?\s*B|SDN\.?)\s*$",
    re.IGNORECASE,
)


def normalize_company_suffix(name: Any) -> str:
    """Restore truncated Malaysian "SDN BHD" tails to the canonical form.

    Variants handled at end-of-string only:
      "SB", "SDN", "SDN.", "SDN B", "SDN. B", "SDN B.",
      "SDN BH", "SDN. BH", "SDN BHD", "SDN BHD.",
      "SDN. BHD.", "SDN. BHD"  →  "SDN BHD"

    Returns the original (stripped) string when no tail variant matches.
    Empty / non-string input is coerced to "".
    """
    if not name:
        return ""
    s = str(name)
    return _COMPANY_SUFFIX_NORM_RE.sub("SDN BHD", s).strip()


# =========================================================
# PATRONYMIC GUARD
# 2-letter banking acronyms (BA, TL, TF, FD, CT) immediately following a Malay / Indian
# patronymic marker (BIN, BINTI, A/L, A/P, BT, BTE, S/O, D/O) are ALWAYS parts of a
# person's name, NEVER banking acronyms. This kills a whole class of C10 false positives
# on truncated long names.
# =========================================================

_PATRONYMIC_MARKERS: List[str] = [
    "BIN", "BINTI", "BT", "BTE",
    "A/L", "A/P", "S/O", "D/O",
]

_AMBIGUOUS_BANKING_TOKENS: set = {"BA", "TL", "TF", "FD", "CT"}

_PATRONYMIC_MARKER_RE = re.compile(
    r"(?:^|\s)(?:" + "|".join(re.escape(p) for p in _PATRONYMIC_MARKERS) + r")\s+([A-Z]{2})\b",
    re.IGNORECASE,
)


def is_patronymic_fragment(text: Any, token: str) -> bool:
    """Return True iff `token` (a 2-letter string) appears in `text` directly after a
    patronymic marker — meaning it's part of a truncated person's name, not a banking acronym.

    Example:
      is_patronymic_fragment("NOR FAIZAH BINTI BA", "BA")        -> True
      is_patronymic_fragment("TERM LOAN BA DRAWDOWN", "BA")      -> False
      is_patronymic_fragment("AHMAD A/L TL", "TL")               -> True
    """
    t = normalize_text(text).upper()
    tok = str(token or "").upper().strip()
    if not t or not tok or len(tok) != 2:
        return False
    for m in _PATRONYMIC_MARKER_RE.finditer(t):
        if m.group(1) == tok:
            return True
    return False


def strip_patronymic_ambiguity(text: Any) -> List[str]:
    """Return the list of 2-letter banking-acronym tokens in `text` that are actually
    patronymic fragments and should NOT be treated as banking acronyms.

    Used as a quick disambiguation helper before running C10 / FD / CT rules.
    """
    t = normalize_text(text).upper()
    if not t:
        return []
    out: List[str] = []
    for m in _PATRONYMIC_MARKER_RE.finditer(t):
        tok = m.group(1)
        if tok in _AMBIGUOUS_BANKING_TOKENS:
            out.append(tok)
    return out


# =========================================================
# DESCRIPTION CLEANING — cross-bank, reusable by all 14 parsers
# =========================================================

# A 2-6 token phrase immediately repeated (same tokens, same order).
# Requires >=2 tokens to avoid collapsing legitimate single-word repeats like
# "PAYMENT PAYMENT FOR SERVICE".
_DUP_PHRASE_RE = re.compile(
    r'(?:^|(?<=\s))(\S{2,}(?:\s+\S+){1,5})\s+\1(?=\s|$)',
)


def collapse_duplicated_segments(desc: Any) -> str:
    """Collapse immediate consecutive duplicated phrases (2-6 tokens) in a description.

    Cross-bank: no knowledge of customer or bank required. Handles patterns like:

      "IBG CREDIT MUHAFIZ SECURITY SDN MUHAFIZ SECURITY SDN SUBALIPACK (M) SDN."
        -> "IBG CREDIT MUHAFIZ SECURITY SDN SUBALIPACK (M) SDN."

      "DUITNOW TO ACCOUNT EPF PAYMENT EPF PAYMENT MUHAFIZ SECURITY SDN"
        -> "DUITNOW TO ACCOUNT EPF PAYMENT MUHAFIZ SECURITY SDN"

    Iterates to fixed point — handles nested / chained duplications like
    "PAYMENT LT GOLF SHOP PAYMENT LT GOLF SHOP LT GOLF SHOP SD AMFB"
      -> "PAYMENT LT GOLF SHOP SD AMFB".
    """
    s = str(desc or "")
    if not s or len(s) < 10:
        return s
    prev: Optional[str] = None
    current = s
    while current != prev:
        prev = current
        current = _DUP_PHRASE_RE.sub(r'\1', current)
    return current


# Cross-bank reference-number patterns. These appear in any bank's statement because
# they come from shared payment rails (FPX, DuitNow, JomPay, RPP).
_COMMON_REF_PATTERNS: List["re.Pattern[str]"] = [
    # FPX / interbank rail prefixes with 6+ digit body
    re.compile(r'\b(?:AOBB2B|AOBIFT|AOBFTR|AOBBY|RPP)\d{6,}\b', re.IGNORECASE),
    # Alliance "I<digits>" FPX ref (10+ digits mandatory so it doesn't eat "I-" words)
    re.compile(r'\bI\d{10,}\b'),
    # Generic alphanumeric ref: 1-4 letters + 6+ digits, optional dash/slash/dot continuation
    # Covers CIMB BR25110153-1, generic PO123456, etc. — rarely meaningful beyond the letter prefix.
    re.compile(r'\b[A-Z]{1,4}\d{6,}(?:[-./]\w+)*\b'),
    # Long digit runs (>=10 digits) — always transaction IDs / timestamp refs / account numbers
    re.compile(r'\b\d{10,}(?:[-./]\S+)*\b'),
]


def strip_reference_numbers(
    desc: Any,
    extra_patterns: Optional[List["re.Pattern[str]"]] = None,
) -> str:
    """Strip digit-heavy reference numbers that leak into descriptions.

    Cross-bank patterns are always applied (FPX rails, 10+ digit runs, generic
    alphanumeric refs). Bank-specific extras can be passed via `extra_patterns`
    — each pattern must be a compiled re.Pattern and will have its matches
    replaced with whitespace.

    Conservative: short digit runs (<10 chars) stay, so years, postal codes, and
    4-digit sub-account IDs are preserved.
    """
    s = str(desc or "")
    if not s:
        return s
    result = s
    patterns = list(_COMMON_REF_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)
    for pattern in patterns:
        result = pattern.sub(' ', result)
    return re.sub(r'\s+', ' ', result).strip()


def cleanup_trailing_artifacts(desc: Any) -> str:
    """Remove obvious truncation-artifact junk at the END of a description.

    Banks with 20-char description columns (CIMB, Maybank FPX) often truncate
    mid-token, leaving stray opening parens, separators, or dangling dashes.
    This helper removes ONLY those obvious artifacts — never invents missing
    content, never strips valid abbreviation punctuation (`SDN.`, `BHD.`).

    Stripped (loops until stable):
      "SCHENKER LOGISTICS ("          -> "SCHENKER LOGISTICS"
      "ACME TRADING ,"                -> "ACME TRADING"
      "ACME CO -"                     -> "ACME CO"
      "ACME  ;"                       -> "ACME"

    Preserved (no invention of content):
      "PERBENA EMAS SDN. BH"          -> unchanged (could be BHD, could be something else)
      "DAMINA SECURITY & D"           -> unchanged (could be DEFENCE, could be something else)
      "ACME SDN."                     -> unchanged ('.' is a valid abbreviation marker)
      "(M) SDN. BHD."                 -> unchanged (')' is legitimate paren close)
    """
    s = str(desc or "").rstrip()
    if not s:
        return s
    while True:
        prev = s
        if s.endswith("("):
            s = s[:-1].rstrip()
        elif s.endswith((",", ";")):
            s = s[:-1].rstrip()
        elif s.endswith("-"):
            s = s[:-1].rstrip()
        if s == prev:
            break
    return s


def clean_description(
    desc: Any,
    extra_ref_patterns: Optional[List["re.Pattern[str]"]] = None,
) -> str:
    """Standard cross-bank description cleanup pipeline.

    Steps (order matters):
      1. collapse_duplicated_segments — removes repeated 2-6 token phrases while
         the duplication signal is still intact
      2. strip_reference_numbers — removes FPX rails, long digit runs, and any
         bank-specific extras passed via extra_ref_patterns
      3. cleanup_trailing_artifacts — drops stray opening parens, commas, dashes
         that became dangling after stripping refs or after cell truncation

    Returns empty string for empty / None input.
    """
    s = str(desc or "")
    if not s:
        return s
    s = collapse_duplicated_segments(s)
    s = strip_reference_numbers(s, extra_patterns=extra_ref_patterns)
    s = cleanup_trailing_artifacts(s)
    return s
