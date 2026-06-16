import json
import math
import os
import re
import secrets
from datetime import datetime
from io import BytesIO
from typing import Callable, Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

from core_utils import (
    bytes_to_pdfplumber,
    dedupe_transactions,
    determine_account_type,
    normalize_company_suffix,
    normalize_transactions,
    safe_float,
    should_drop_as_counterparty,
)

from maybank import parse_transactions_maybank
from public_bank import parse_transactions_pbb
from rhb import parse_transactions_rhb
from cimb import parse_transactions_cimb
from bank_islam import parse_bank_islam
from bank_rakyat import parse_bank_rakyat
from hong_leong import parse_hong_leong
from ambank import parse_ambank, extract_ambank_statement_totals
from bank_muamalat import parse_transactions_bank_muamalat
from affin_bank import parse_affin_bank, extract_affin_statement_totals
from agro_bank import parse_agro_bank
from ocbc import parse_transactions_ocbc

# ✅ UOB Bank parser
from uob import parse_transactions_uob

# ✅ Alliance Bank parser
from alliance import parse_transactions_alliance

# ✅ PDF password support
from pdf_security import is_pdf_encrypted, decrypt_pdf_bytes

# ✅ PDF integrity / fraud detection (8-layer + batch comparison)
from pdf_fraud_detector import analyze_pdf as analyze_pdf_integrity, compare_batch as compare_pdf_batch


# Track 2 deterministic classifier wire-through. Off by default; flip with
# USE_TRACK_2=1 to surface a "Download Track 2 Analysis (JSON)" button next
# to the existing Full Report exports. Conditional import keeps cold-start
# unchanged when the flag is off — Track 1 frozen-equivalence at module load.
_USE_TRACK_2 = os.getenv("USE_TRACK_2", "").strip().lower() in (
    "1", "true", "yes", "on",
)
if _USE_TRACK_2:
    from kredit_lab_classify_track2 import (
        account_meta_from_determinations as _track2_account_meta,
        build_track2_result as _build_track2_result,
        validate_track2_result as _validate_track2_result,
    )
    # === TRACK 2 RENDERER INTEGRATION (added — remove this block to revert) ===
    # Shared, presentation-only render core (byte-identical extract of the
    # standalone HTML renderer). Lets the parser turn the in-memory v6.3.5
    # engine result straight into the analyst HTML/Excel deliverable, and
    # also render a claude.ai-enriched analysis JSON. No classification logic
    # lives here — engine/rulebook remains the single source of truth.
    try:
        import renderer_core as _t2_render
    except Exception:  # pragma: no cover - renderer_core is optional
        _t2_render = None
    # === END TRACK 2 RENDERER INTEGRATION ===

# Facility-keyword regexes for the build_counterparty_ledger own-party
# memo-echo guard (keeps a loan/facility row whose memo echoes the holder
# name out of the OWN-PARTY bucket). Single source of truth is the engine;
# fall back to no-op matchers if the engine module is unavailable so the
# ledger still builds.
try:
    from kredit_lab_classify_track2 import (
        LOAN_DISBURSEMENT_RE as _LOAN_DISBURSEMENT_RE,
        LOAN_REPAYMENT_RE as _LOAN_REPAYMENT_RE,
    )
except Exception:  # pragma: no cover
    import re as _re_fallback
    _LOAN_DISBURSEMENT_RE = _re_fallback.compile(r"(?!)")
    _LOAN_REPAYMENT_RE = _re_fallback.compile(r"(?!)")


def require_basic_auth() -> None:
    """Gate the app behind credentials loaded from environment variables."""
    configured_user = os.getenv("BASIC_AUTH_USER")
    configured_pass = os.getenv("BASIC_AUTH_PASS")

    if not configured_user or not configured_pass:
        st.error(
            "Missing BASIC_AUTH_USER or BASIC_AUTH_PASS environment variables. "
            "Set both to use this app."
        )
        st.stop()

    if st.session_state.get("is_authenticated"):
        return

    st.subheader("🔐 Login required")

    with st.form("basic_auth_form"):
        entered_user = st.text_input("Username")
        entered_pass = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        is_valid = secrets.compare_digest(entered_user, configured_user) and secrets.compare_digest(
            entered_pass,
            configured_pass,
        )
        if is_valid:
            st.session_state.is_authenticated = True
            st.rerun()
        st.error("Invalid username or password.")

    st.stop()


st.set_page_config(page_title="Bank Statement Parser", layout="wide")
require_basic_auth()
st.title("📄 Bank Statement Parser (Multi-File Support)")
st.write("Upload one or more bank statement PDFs to extract transactions.")


# -----------------------------
# Session state init
# -----------------------------
if "status" not in st.session_state:
    st.session_state.status = "idle"

if "results" not in st.session_state:
    st.session_state.results = []

if "affin_statement_totals" not in st.session_state:
    st.session_state.affin_statement_totals = []

if "affin_file_transactions" not in st.session_state:
    st.session_state.affin_file_transactions = {}

if "ambank_statement_totals" not in st.session_state:
    st.session_state.ambank_statement_totals = []

if "ambank_file_transactions" not in st.session_state:
    st.session_state.ambank_file_transactions = {}

if "cimb_statement_totals" not in st.session_state:
    st.session_state.cimb_statement_totals = []

if "cimb_file_transactions" not in st.session_state:
    st.session_state.cimb_file_transactions = {}

if "rhb_statement_totals" not in st.session_state:
    st.session_state.rhb_statement_totals = []

if "rhb_file_transactions" not in st.session_state:
    st.session_state.rhb_file_transactions = {}

if "bank_islam_file_month" not in st.session_state:
    st.session_state.bank_islam_file_month = {}

# ✅ file_uploader reset counter — a file_uploader can only be cleared by
# changing its widget key, so the uploader keys off this int and Reset bumps it.
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ✅ password + company name tracking
if "pdf_password" not in st.session_state:
    st.session_state.pdf_password = ""

if "company_name_override" not in st.session_state:
    st.session_state.company_name_override = ""

if "file_company_name" not in st.session_state:
    st.session_state.file_company_name = {}

if "file_account_no" not in st.session_state:
    st.session_state.file_account_no = {}

# ✅ PDF integrity results (fraud detection)
if "pdf_integrity_results" not in st.session_state:
    st.session_state.pdf_integrity_results = {}
if "pdf_raw_bytes" not in st.session_state:
    st.session_state.pdf_raw_bytes = {}  # {filename: bytes} for batch comparison

# ✅ Sprint 4.5: per-file account_type determination (parser-locked, AI trusts)
if "account_type_determinations" not in st.session_state:
    st.session_state.account_type_determinations = []


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_any_date_for_summary(x) -> pd.Timestamp:
    if x is None:
        return pd.NaT
    s = str(x).strip()
    if not s:
        return pd.NaT
    if _ISO_RE.match(s):
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _parse_with_pdfplumber(parser_func: Callable, pdf_bytes: bytes, filename: str) -> List[dict]:
    with bytes_to_pdfplumber(pdf_bytes) as pdf:
        return parser_func(pdf, filename)


# -----------------------------
# Company name extraction (FIXED)
# -----------------------------
# Strong signals
_COMPANY_NAME_PATTERNS = [
    r"(?:ACCOUNT\s+NAME|A\/C\s+NAME|CUSTOMER\s+NAME|NAMA\s+AKAUN|NAMA\s+PELANGGAN|NAMA)\s*[:\-]\s*(.+)",
    r"(?:ACCOUNT\s+HOLDER|PEMEGANG\s+AKAUN)\s*[:\-]\s*(.+)",
]

# Lines we should NOT treat as a company name
_EXCLUDE_LINE_REGEX = re.compile(
    r"(A\/C\s*NO|AC\s*NO|ACCOUNT\s*NO|ACCOUNT\s*NUMBER|NO\.?\s*AKAUN|NO\s+AKAUN|"
    r"STATEMENT\s+DATE|TARIKH\s+PENYATA|DATE\s+FROM|DATE\s+TO|CURRENCY|BRANCH|SWIFT|IBAN|PAGE\s+\d+)",
    re.IGNORECASE,
)

# If a candidate contains a long digit run, it’s usually not a company name.
_LONG_DIGITS_RE = re.compile(r"\d{6,}")
# Money amounts never appear in a real company-name line; they mark transaction
# rows (e.g. "02Jan INWARD IBG, MUSHTARI MAINTENANCE 15,020.00 306,910.44").
_MONEY_TOKEN_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}")
# Transaction-table start markers: the company name always lives in the page
# header ABOVE these, so fallback scans must never read past the first one.
_TABLE_START_RE = re.compile(
    r"(DATE\s+TRANSACTION|TARIKH\s+TRANSAKSI|TARIKH\s+URUS\s*NIAGA|"
    r"ENTRY\s+DATE\s+VALUE\s+DATE|BAKI\s+BAWA\s+KE\s+HADAPAN|BALANCE\s+B/F|"
    r"ACCOUNT\s+SUMMARY\s*/\s*RINGKASAN)",
    re.IGNORECASE,
)
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(SDN\.?\s*BHD\.?|BHD\.?|ENTERPRISE|PERNIAGAAN|AGENCY|RESOURCES|HOLDINGS|TRADING|SERVICES|TECHNOLOGY|VENTURES|INDUSTRIES|GLOBAL|GROUP|CORPORATION|PLT)\b",
    re.IGNORECASE,
)
_COMPANY_BAD_WORDS_RE = re.compile(
    r"\b(STATEMENT|ACCOUNT\s+STATEMENT|CURRENT\s+ACCOUNT|PAGE\b|BALANCE\b|SUMMARY\b|TRANSACTION|ENQUIRIES|BRANCH|PIDM|DATE\b|MUKA\b|HALAMAN\b|結單日期|结单日期)\b",
    re.IGNORECASE,
)


def _clean_candidate_name(s: str) -> str:
    s = (s or "").strip()
    # stop at common trailing fields
    s = re.split(
        r"\s{2,}|ACCOUNT\s+NO|A\/C\s+NO|NO\.\s*AKAUN|NO\s+AKAUN|STATEMENT|PENYATA|DATE|TARIKH|CURRENCY|BRANCH|PAGE|HALAMAN|TAX\s+INVOICE|INVOIS\s+CUKAI|結單日期|结单日期",
        s,
        flags=re.IGNORECASE,
    )[0].strip()
    # remove weird leading bullets/colons
    s = s.lstrip(":;-• ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# Lines that are address / branch / mail-sort noise: never a name prefix to
# merge with the company-suffix line below them (e.g. "TAMAN TUN DR. ISMAIL -
# 258" + "CLEAR WATER SERVICES SDN. BHD.", or PBB's bare "168" sort code).
_JUNK_PREFIX_RE = re.compile(
    r"(\b(TAMAN|JALAN|JLN|LORONG|LRG|WISMA|MENARA|PLAZA|TINGKAT|FLOOR|BANDAR|"
    r"KAMPUNG|PERSIARAN|LEBUH|BLOK|SEKSYEN)\b|-\s*\d+\s*$)",
    re.IGNORECASE,
)


def _junk_merge_prefix(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    if re.fullmatch(r"[\d\s/.-]+", s):  # digits-only mail-sort / routing codes
        return True
    return bool(_JUNK_PREFIX_RE.search(s))


def _standalone_company_name(s: str) -> bool:
    """Company-like AND carries a real name beyond the bare legal suffix."""
    if not _looks_like_company_name(s):
        return False
    leftover = _COMPANY_SUFFIX_RE.sub("", _clean_candidate_name(s)).strip(" .,&-")
    return len(re.sub(r"[^A-Za-z]", "", leftover)) >= 3


def _looks_like_account_number_line(s: str) -> bool:
    if not s:
        return True
    up = s.upper()
    if _EXCLUDE_LINE_REGEX.search(up):
        return True
    if _LONG_DIGITS_RE.search(s):
        # long digit run strongly suggests account number/reference, not company name
        return True
    # too short is suspicious
    if len(s.strip()) < 3:
        return True
    return False


def _looks_like_company_name(s: str) -> bool:
    if not s:
        return False

    cand = _clean_candidate_name(s)
    if not cand:
        return False
    if _looks_like_account_number_line(cand):
        return False
    if _COMPANY_BAD_WORDS_RE.search(cand):
        return False
    if re.search(r"https?://|www\.", cand, flags=re.IGNORECASE):
        return False
    if len(cand) < 6:
        return False
    if re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", cand):
        return False
    # transaction-row shapes: "02Jan INWARD IBG, ..." date starts or any line
    # carrying a money amount can never be the account holder's name
    if re.match(r"^\d{1,2}[A-Za-z]{3}\b", cand):
        return False
    if _MONEY_TOKEN_RE.search(cand):
        return False
    return bool(_COMPANY_SUFFIX_RE.search(cand))


def extract_company_name(pdf, max_pages: int = 2) -> Optional[str]:
    """
    Extract company/account holder name from statement.
    Strategy:
      1) Search explicit labels (Account Name / Customer Name / Nama...) on first N pages
      2) Fallback: choose first plausible line that is NOT account-number-ish
    """
    texts: List[str] = []
    try:
        for i in range(min(max_pages, len(pdf.pages))):
            texts.append((pdf.pages[i].extract_text() or "").strip())
    except Exception:
        pass

    texts = [t for t in texts if t]
    if not texts:
        return None

    full = "\n".join(texts)

    # 0) UOB "Account Activities" export style
    # Example block:
    #   Company / Account Account Balance
    #   Company Available Balance
    #   UPELL CORPORATION SDN. BHD. MYR 55,744.04
    m_uob = re.search(
        r"Company\s*/\s*Account.*?\bCompany\b.*?\n\s*([A-Z0-9 &().,'\/-]{3,})",
        full,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_uob:
        cand = _clean_candidate_name(m_uob.group(1))
        # strip appended currency/balance if present
        cand = re.split(r"\bMYR\b", cand, maxsplit=1, flags=re.IGNORECASE)[0].strip() or cand
        if cand and not _looks_like_account_number_line(cand):
            return cand

    # 0.5) Maybank bilingual header style (company line with statement-date markers)
    # Examples:
    #   LSR AGENCY 結單日期 : 31/03/25
    #   PERNIAGAAN SEPAKAT ABADI 結單日期 : 31/01/25
    maybank_lines = [ln.strip() for ln in full.splitlines() if ln.strip()]

    # Maybank often places the company around "TARIKH PENYATA", sometimes split:
    #   QUATTRO FRATELLI
    #   TARIKH PENYATA
    #   ENERGY SDN. BHD.
    for i, ln in enumerate(maybank_lines[:80]):
        if not re.search(r"^TARIKH\s+PENYATA$", ln, flags=re.IGNORECASE):
            continue

        prev_ln = _clean_candidate_name(maybank_lines[i - 1]) if i - 1 >= 0 else ""
        next_ln = _clean_candidate_name(maybank_lines[i + 1]) if i + 1 < len(maybank_lines) else ""

        if prev_ln and next_ln and not _looks_like_account_number_line(prev_ln):
            if re.search(r"(MUKA|PAGE|MAYBANK|IBS\s|BRANCH)", prev_ln, flags=re.IGNORECASE):
                prev_ln = ""

        if prev_ln and next_ln:
            merged = _clean_candidate_name(f"{prev_ln} {next_ln}")
            if merged and not _looks_like_account_number_line(merged):
                if _looks_like_company_name(merged) or re.search(
                    r"\b(SDN\.?\s*BHD\.?|PERNIAGAAN|AGENCY)\b",
                    merged,
                    flags=re.IGNORECASE,
                ):
                    return merged

        if next_ln and not _looks_like_account_number_line(next_ln):
            if _looks_like_company_name(next_ln):
                return next_ln

    for i, ln in enumerate(maybank_lines[:80]):
        m_maybank_line = re.match(
            r"^([A-Z][A-Z0-9 &().,\'\/-]{2,}?)\s+(?:結單日期|结单日期|STATEMENT\s+DATE)\s*:?\s*\d{2}/\d{2}/\d{2,4}\s*$",
            ln,
            flags=re.IGNORECASE,
        )
        if not m_maybank_line:
            continue

        cand = _clean_candidate_name(m_maybank_line.group(1))

        # Some Maybank statements split the name over 2 lines, e.g.:
        #   QUATTRO FRATELLI ENERGY
        #   SDN. BHD. 結單日期 : 31/07/2025
        if re.fullmatch(r"SDN\.?\s*BHD\.?", cand, flags=re.IGNORECASE):
            # In some files the line right above is "TARIKH PENYATA", so walk
            # backward to find the nearest plausible company prefix.
            for j in range(i - 1, max(-1, i - 4), -1):
                if j < 0:
                    break
                prefix = _clean_candidate_name(maybank_lines[j])
                if not prefix:
                    continue
                if re.search(r"^(TARIKH\s+PENYATA|STATEMENT\s+DATE|MUKA|PAGE)\b", prefix, flags=re.IGNORECASE):
                    continue
                merged = _clean_candidate_name(f"{prefix} {cand}")
                if merged and not _looks_like_account_number_line(merged):
                    return merged

        if cand and not _looks_like_account_number_line(cand):
            return cand

    # 1) label-based extraction
    for pat in _COMPANY_NAME_PATTERNS:
        m = re.search(pat, full, flags=re.IGNORECASE)
        if m:
            cand = _clean_candidate_name(m.group(1))
            if cand and not _looks_like_account_number_line(cand):
                return cand

    # 2) fallback: scan lines — but never past the transaction-table start.
    # Scanning into the table is how a transaction row ("02Jan INWARD IBG,
    # MUSHTARI MAINTENANCE ... SERVICES SDN. BHD.") gets mistaken for the
    # account holder; the real name always sits in the header above the table.
    lines: List[str] = []
    for t in texts:
        page_lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        for ln in page_lines:
            if _TABLE_START_RE.search(ln):
                break
            lines.append(ln)

    # 2) context-aware: line before account label often contains company name
    for i, ln in enumerate(lines[:80]):
        if re.search(r"A\/C|ACCOUNT\s*NO|ACCOUNT\s*NUMBER|NOMBOR\s+AKAUN|NO\.?\s*AKAUN", ln, flags=re.IGNORECASE):
            if i > 0:
                prev = _clean_candidate_name(lines[i - 1])
                if _looks_like_company_name(prev):
                    return prev

    # 3) suffix-aware scan (most reliable for Malaysian company names)
    for i, ln in enumerate(lines[:80]):
        cand = _clean_candidate_name(ln)
        if _looks_like_company_name(cand):
            return cand

        # handle split names e.g. "CLEAR WATER SERVICES" + "SDN. BHD." —
        # but never glue an address/branch/sort-code line onto the name.
        if i + 1 < len(lines):
            nxt = _clean_candidate_name(lines[i + 1])
            if _junk_merge_prefix(ln):
                if _standalone_company_name(nxt):
                    return nxt
                continue
            merged = _clean_candidate_name(f"{ln} {lines[i + 1]}")
            if _looks_like_company_name(merged) and len(merged) <= 120:
                return merged

    return None


# -----------------------------
# Account number extraction (NEW)
# -----------------------------
_ACCOUNT_NO_PATTERNS = [
    r"(?:A\/C\s*NO|AC\s*NO|ACC(?:OUNT)?\s*NO\.?|ACCOUNT\s*NUMBER|NOMBOR\s+AKAUN|NO\.?\s*AKAUN|NO\s+AKAUN)\s*[:\-]?\s*([\d][\d\- ]{4,36}\d)",
    # UOB export: "Account Ledger Balance" then the account number on the next line
    r"Account\s+Ledger\s+Balance\s*\n\s*([\d][\d\- ]{4,36}\d)",
]

_ACCOUNT_LABEL_RE = re.compile(
    r"(A\/C\s*NO|AC\s*NO|ACC(?:OUNT)?\s*NO\.?|ACCOUNT\s*NUMBER|NOMBOR\s+AKAUN|NO\.?\s*AKAUN|NO\s+AKAUN)",
    re.IGNORECASE,
)

_ACCOUNT_NUM_RE = re.compile(r"\b\d(?:[\d\-]{4,28}\d)\b")


def _normalize_account_no(raw: str) -> Optional[str]:
    if not raw:
        return None
    cleaned = re.sub(r"\s+", "", str(raw).strip())
    digits_only = re.sub(r"\D", "", cleaned)
    if 6 <= len(digits_only) <= 16:
        return digits_only
    return None


def _candidate_account_numbers(text: str) -> List[str]:
    if not text:
        return []

    out: List[str] = []
    for m in _ACCOUNT_NUM_RE.finditer(text):
        num = _normalize_account_no(m.group(0) or "")
        if not num:
            continue
        # avoid date-like fragments accidentally captured from labels/windows
        if re.fullmatch(r"\d{8}", num):
            yyyy = int(num[:4])
            mm = int(num[4:6])
            dd = int(num[6:8])
            if 1900 <= yyyy <= 2100 and 1 <= mm <= 12 and 1 <= dd <= 31:
                continue
        out.append(num)
    return out


def extract_account_number(pdf, max_pages: int = 2) -> Optional[str]:
    texts: List[str] = []
    try:
        for i in range(min(max_pages, len(pdf.pages))):
            texts.append((pdf.pages[i].extract_text() or "").strip())
    except Exception:
        pass

    texts = [t for t in texts if t]
    if not texts:
        return None

    full = "\n".join(texts)
    lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
    full_upper = full.upper()

    # Bank-specific hardening: RHB Reflex headers usually print the account number directly
    # after "Reflex Cash Management ...", often on the next line.
    if ("REFLEX CASH MANAGEMENT" in full_upper) and ("DEPOSIT ACCOUNT SUMMARY" in full_upper):
        reflex_candidates: List[str] = []
        for m in re.finditer(r"REFLEX\s+CASH\s+MANAGEMENT[^\n\r]{0,120}[\n\r]+\s*([0-9][0-9\-\s]{9,20})\b", full, re.IGNORECASE):
            num = _normalize_account_no(m.group(1) or "")
            if num and len(num) >= 10:
                reflex_candidates.append(num)
        if reflex_candidates:
            # pick the most repeated, then the longest (stable across pages/months)
            uniq = sorted(set(reflex_candidates), key=lambda x: (-reflex_candidates.count(x), -len(x), x))
            return uniq[0]

    # Bank-specific hardening: RHB deposit-account summary pages often place the account number
    # in compact rows such as "ORDINARYCURRENTACCOUNT21406200114180".
    full_compact = re.sub(r"\s+", "", full_upper)
    if "DEPOSITACCOUNTSUMMARY" in full_compact or "RINGKASANAKAUNDEPOSIT" in full_compact:
        # Prefer summary rows: account number followed by balance columns.
        for ln in lines[:140]:
            m = re.search(
                r"(?:CURRENT\s*ACCOUNT(?:-I)?|ACCOUNT(?:-I)?)\s*([0-9]{10,16})\s+\d{1,3}(?:,\d{3})*\.\d{2}\s+\d{1,3}(?:,\d{3})*\.\d{2}",
                ln,
                re.IGNORECASE,
            )
            if m:
                num = _normalize_account_no(m.group(1) or "")
                if num:
                    return num

        # Fallback for compact rows like "...CURRENTACCOUNT21406200114180".
        for ln in lines[:140]:
            if len(ln) > 60:
                continue
            m = re.search(r"(?:CURRENT\s*ACCOUNT(?:-I)?|ACCOUNT(?:-I)?)\s*([0-9]{10,16})\b", ln, re.IGNORECASE)
            if m:
                num = _normalize_account_no(m.group(1) or "")
                if num:
                    return num

    scored: Dict[str, int] = {}

    def _add(num: Optional[str], points: int) -> None:
        if not num:
            return
        scored[num] = scored.get(num, 0) + points

    # 1) Strong patterns with account labels.
    for pat in _ACCOUNT_NO_PATTERNS:
        m = re.search(pat, full, flags=re.IGNORECASE | re.DOTALL)
        if m:
            num = _normalize_account_no(m.group(1) or "")
            if num:
                _add(num, 120)

    # Bonus for candidates that appear repeatedly in the document.
    for cand in {c for c in _candidate_account_numbers(full)}:
        repeats = len(re.findall(rf"\b{re.escape(cand)}\b", re.sub(r"\D", " ", full)))
        if repeats >= 2:
            _add(cand, repeats * 10)

    # 2) Label-aware scan on individual lines and short windows.
    for i, ln in enumerate(lines[:180]):
        if not _ACCOUNT_LABEL_RE.search(ln):
            continue

        for cand in _candidate_account_numbers(ln):
            _add(cand, 100)

        window = " ".join(lines[i : min(i + 3, len(lines))])
        for cand in _candidate_account_numbers(window):
            _add(cand, 60)

    if scored:
        return sorted(scored.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[0][0]

    # 4) Fallback: standalone account-number-like lines.
    for ln in lines[:120]:
        raw = (ln or "").strip()
        if re.fullmatch(r"\d{10,16}", raw):
            return raw

    return None

# -----------------------------
# Bank Islam: statement month for zero-transaction months
# -----------------------------
_BANK_ISLAM_STMT_DATE_RE = re.compile(
    r"(?:STATEMENT\s+DATE|TARIKH\s+PENYATA)\s*:?\s*(\d{1,2})/(\d{1,2})/(\d{2,4})",
    re.IGNORECASE,
)


def extract_bank_islam_statement_month(pdf) -> Optional[str]:
    try:
        t = (pdf.pages[0].extract_text() or "")
    except Exception:
        return None

    m = _BANK_ISLAM_STMT_DATE_RE.search(t)
    if not m:
        return None

    mm = int(m.group(2))
    yy_raw = m.group(3)
    yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)

    if 1 <= mm <= 12 and 2000 <= yy <= 2100:
        return f"{yy:04d}-{mm:02d}"
    return None


# -----------------------------
# CIMB totals extractor (existing)
# -----------------------------
_CIMB_STMT_DATE_RE = re.compile(
    r"(?:STATEMENT\s+DATE|TARIKH\s+PENYATA)\s*:?\s*(\d{1,2})/(\d{1,2})/(\d{2,4})",
    re.IGNORECASE,
)
_CIMB_CLOSING_RE = re.compile(
    r"CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUP\s+(-?[\d,]+\.\d{2})",
    re.IGNORECASE,
)


def _prev_month(yyyy: int, mm: int) -> Tuple[int, int]:
    if mm == 1:
        return (yyyy - 1, 12)
    return (yyyy, mm - 1)


def extract_cimb_statement_totals(pdf, source_file: str) -> dict:
    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    up = full_text.upper()

    page_opening_balance = None
    try:
        first_text = pdf.pages[0].extract_text() or ""
        mo = re.search(r"Opening\s+Balance\s+(-?[\d,]+\.\d{2})", first_text, re.IGNORECASE)
        if mo:
            page_opening_balance = float(mo.group(1).replace(",", ""))
    except Exception:
        page_opening_balance = None

    stmt_month = None
    m = _CIMB_STMT_DATE_RE.search(full_text)
    if m:
        mm = int(m.group(2))
        yy_raw = m.group(3)
        yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)
        if 1 <= mm <= 12 and 2000 <= yy <= 2100:
            # CIMB's "Statement Date / Tarikh Penyata" is the period-END date
            # (last day of the statement month), not an issue date the day after.
            # Earlier code rolled this back one month via _prev_month, which
            # produced an off-by-one (Oct'25 statement tagged 2025-09 etc.).
            # Use the matched month directly; calculate_monthly_summary will
            # still override with the dominant transaction month when
            # transactions are available (defense-in-depth against future
            # CIMB format changes or non-calendar cycles).
            stmt_month = f"{yy:04d}-{mm:02d}"

    closing_balance = None
    m = _CIMB_CLOSING_RE.search(full_text)
    if m:
        closing_balance = float(m.group(1).replace(",", ""))

    total_debit = None
    total_credit = None
    if "TOTAL WITHDRAWAL" in up and "TOTAL DEPOSITS" in up:
        idx = up.rfind("TOTAL WITHDRAWAL")
        window = full_text[idx : idx + 900] if idx != -1 else full_text

        mm2 = re.search(r"\b\d{1,6}\s+\d{1,6}\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})\b", window)
        if mm2:
            total_debit = float(mm2.group(1).replace(",", ""))
            total_credit = float(mm2.group(2).replace(",", ""))
        else:
            money = re.findall(r"-?[\d,]+\.\d{2}", window)
            if len(money) >= 2:
                total_debit = float(money[-2].replace(",", ""))
                total_credit = float(money[-1].replace(",", ""))

    return {
        "bank": "CIMB Bank",
        "source_file": source_file,
        "statement_month": stmt_month,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "ending_balance": closing_balance,
        "page_opening_balance": page_opening_balance,
        "opening_balance": None,
    }



def extract_rhb_statement_totals(pdf, source_file: str) -> dict:
    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    full_text_norm = re.sub(r"\s+", " ", full_text).strip()

    def _signed_money(token: str) -> Optional[float]:
        if not token:
            return None
        s = token.strip().replace(",", "")
        sign = 1.0
        if s.endswith("-"):
            sign = -1.0
            s = s[:-1]
        elif s.endswith("+"):
            s = s[:-1]
        try:
            return round(sign * float(s), 2)
        except Exception:
            return None

    period_match = re.search(
        r"Statement\s+Period.*?:\s*\d{1,2}\s+([A-Za-z]{3})\s+(\d{2,4})",
        full_text,
        re.IGNORECASE,
    )
    statement_month = None
    if period_match:
        month_map = {
            "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
            "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
        }
        mon = period_match.group(1).upper()
        yy = period_match.group(2)
        if mon in month_map:
            year = int(yy) if len(yy) == 4 else (2000 + int(yy))
            statement_month = f"{year:04d}-{month_map[mon]}"
    else:
        # Reflex-style: "Statement Period 01 August 2025 To 31 August 2025"
        period_match2 = re.search(
            r"Statement\s+Period\s+\d{1,2}\s+([A-Za-z]{3,9})\s+(\d{4})\s+To\s+\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
            full_text_norm,
            re.IGNORECASE,
        )
        if period_match2:
            mon = period_match2.group(1).upper()[:3]
            yy = int(period_match2.group(2))
            month_map = {
                "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
                "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
            }
            if mon in month_map:
                statement_month = f"{yy:04d}-{month_map[mon]}"

    opening_balance = None
    ending_balance = None
    total_debit = None
    total_credit = None

    bfm = re.search(r"\b\d{1,2}\s+[A-Za-z]{3}\s+B/F\s+BALANCE\s+(-?[\d,]+\.\d{2})", full_text, re.IGNORECASE)
    if bfm:
        opening_balance = float(bfm.group(1).replace(",", ""))

    cfm = re.search(r"\b\d{1,2}\s+[A-Za-z]{3}\s+C/F\s+BALANCE\s+(-?[\d,]+\.\d{2})", full_text, re.IGNORECASE)
    if cfm:
        ending_balance = float(cfm.group(1).replace(",", ""))

    tm = re.search(r"\(RM\)\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})", full_text, re.IGNORECASE)
    if tm:
        total_debit = float(tm.group(1).replace(",", ""))
        total_credit = float(tm.group(2).replace(",", ""))

    # Reflex summary fallback
    if opening_balance is None:
        m = re.search(
            r"Beginning\s+Balance\s+as\s+of\s+\d{1,2}\s+[A-Za-z]{3,9}(?:\s+\d{2,4})?\s+([\d,]+\.\d{2}[+-]?)",
            full_text_norm,
            re.IGNORECASE,
        )
        opening_balance = _signed_money(m.group(1)) if m else None

    if ending_balance is None:
        m = re.search(
            r"Ending\s+Balance\s+as\s+of\s+\d{1,2}\s+[A-Za-z]{3,9}(?:\s+\d{2,4})?\s+([\d,]+\.\d{2}[+-]?)",
            full_text_norm,
            re.IGNORECASE,
        )
        ending_balance = _signed_money(m.group(1)) if m else None

    if total_credit is None:
        m = re.search(r"\b\d+\s+Deposits\s*\(Plus\)\s+([\d,]+\.\d{2})", full_text_norm, re.IGNORECASE)
        if m:
            total_credit = float(m.group(1).replace(",", ""))

    if total_debit is None:
        m = re.search(r"\b\d+\s+Withdraws\s*\(Minus\)\s+([\d,]+\.\d{2})", full_text_norm, re.IGNORECASE)
        if m:
            total_debit = float(m.group(1).replace(",", ""))

    return {
        "bank": "RHB Bank",
        "source_file": source_file,
        "statement_month": statement_month,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "ending_balance": ending_balance,
        "opening_balance": opening_balance,
    }

# -----------------------------
# Bank parsers
# -----------------------------
PARSERS: Dict[str, Callable[[bytes, str], List[dict]]] = {
    "Affin Bank": lambda b, f: _parse_with_pdfplumber(parse_affin_bank, b, f),
    "Agro Bank": lambda b, f: _parse_with_pdfplumber(parse_agro_bank, b, f),
    "Alliance Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_alliance, b, f),
    "Ambank": lambda b, f: _parse_with_pdfplumber(parse_ambank, b, f),
    "Bank Islam": lambda b, f: _parse_with_pdfplumber(parse_bank_islam, b, f),
    "Bank Muamalat": lambda b, f: _parse_with_pdfplumber(parse_transactions_bank_muamalat, b, f),
    "Bank Rakyat": lambda b, f: _parse_with_pdfplumber(parse_bank_rakyat, b, f),
    "CIMB Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_cimb, b, f),
    "Hong Leong": lambda b, f: _parse_with_pdfplumber(parse_hong_leong, b, f),
    "Maybank": lambda b, f: parse_transactions_maybank(b, f),
    "Public Bank (PBB)": lambda b, f: _parse_with_pdfplumber(parse_transactions_pbb, b, f),
    "RHB Bank": lambda b, f: parse_transactions_rhb(b, f),
    "OCBC Bank": lambda b, f: parse_transactions_ocbc(b, f),
    "UOB Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_uob, b, f),
}


bank_choice = st.selectbox("Select Bank Format", list(PARSERS.keys()))

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"pdf_uploader_{st.session_state.uploader_key}",
)
if uploaded_files:
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)

# Manual company name override
st.text_input("Company Name (optional override)", key="company_name_override")

# Detect encrypted files
encrypted_files: List[str] = []
if uploaded_files:
    for uf in uploaded_files:
        try:
            if is_pdf_encrypted(uf.getvalue()):
                encrypted_files.append(uf.name)
        except Exception:
            encrypted_files.append(uf.name)

    if encrypted_files:
        st.warning(
            "🔒 Encrypted PDF(s) detected. Enter the password once and it will be used for all encrypted files:\n\n"
            + "\n".join([f"- {n}" for n in encrypted_files])
        )
        st.text_input("PDF Password", type="password", key="pdf_password")


def _reset_app_state():
    """Reset callback — runs at the start of the rerun, BEFORE any widget is
    instantiated, so clearing widget-bound keys (``pdf_password`` /
    ``company_name_override``) is permitted here. Assigning those keys from an
    inline button handler instead raises StreamlitAPIException (the widgets at
    L814/L831 are already instantiated by then), which is why the old Reset
    button silently failed."""
    st.session_state.status = "idle"
    st.session_state.results = []
    st.session_state.affin_statement_totals = []
    st.session_state.affin_file_transactions = {}
    st.session_state.ambank_statement_totals = []
    st.session_state.ambank_file_transactions = {}
    st.session_state.cimb_statement_totals = []
    st.session_state.rhb_statement_totals = []
    st.session_state.cimb_file_transactions = {}
    st.session_state.rhb_file_transactions = {}
    st.session_state.bank_islam_file_month = {}
    st.session_state.file_company_name = {}
    st.session_state.file_account_no = {}
    st.session_state.pdf_password = ""
    st.session_state.company_name_override = ""
    st.session_state.pdf_integrity_results = {}
    st.session_state.pdf_raw_bytes = {}
    st.session_state.account_type_determinations = []
    # Bump the uploader key so the file_uploader is re-created empty —
    # the only way to clear uploaded files in Streamlit.
    st.session_state.uploader_key += 1


col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Start Processing"):
        st.session_state.status = "running"
        st.session_state.affin_statement_totals = []
        st.session_state.affin_file_transactions = {}
        st.session_state.ambank_statement_totals = []
        st.session_state.ambank_file_transactions = {}
        st.session_state.cimb_statement_totals = []
        st.session_state.rhb_statement_totals = []
        st.session_state.cimb_file_transactions = {}
        st.session_state.rhb_file_transactions = {}
        st.session_state.bank_islam_file_month = {}
        st.session_state.file_company_name = {}
        st.session_state.file_account_no = {}
        st.session_state.pdf_integrity_results = {}
        st.session_state.pdf_raw_bytes = {}
        st.session_state.account_type_determinations = []

with col2:
    if st.button("⏹️ Stop"):
        st.session_state.status = "stopped"

with col3:
    # on_click callback (not an inline body): the callback fires before the
    # script reruns, so resetting the widget-bound keys is allowed. Streamlit
    # reruns automatically after a callback, so no st.rerun() is needed.
    st.button("🔄 Reset", on_click=_reset_app_state)

st.write(f"### ⚙️ Status: **{st.session_state.status.upper()}**")


all_tx: List[dict] = []

if uploaded_files and st.session_state.status == "running":
    bank_display_box = st.empty()
    progress_bar = st.progress(0)

    total_files = len(uploaded_files)
    parser = PARSERS[bank_choice]

    for file_idx, uploaded_file in enumerate(uploaded_files):
        if st.session_state.status == "stopped":
            st.warning("⏹️ Processing stopped by user.")
            break

        st.write(f"### 🗂️ Processing File: **{uploaded_file.name}**")
        bank_display_box.info(f"📄 Processing {bank_choice}: {uploaded_file.name}...")

        try:
            pdf_bytes = uploaded_file.getvalue()

            # decrypt if encrypted — try pypdf first, fall back to passing
            # password directly to pdfplumber (handles more encryption types)
            pdf_pw = st.session_state.pdf_password or None
            needs_password = is_pdf_encrypted(pdf_bytes)
            if needs_password:
                try:
                    pdf_bytes = decrypt_pdf_bytes(pdf_bytes, pdf_pw)
                    pdf_pw = None  # successfully decrypted, no need to pass pw downstream
                except Exception:
                    # pypdf couldn't decrypt — keep original bytes and let
                    # pdfplumber try with the password directly
                    pass

            # ✅ PDF integrity check (8-layer fraud detection) — flag only, never blocks
            try:
                integrity_result = analyze_pdf_integrity(
                    pdf_bytes, uploaded_file.name, bank_hint=bank_choice
                )
                st.session_state.pdf_integrity_results[uploaded_file.name] = integrity_result
                st.session_state.pdf_raw_bytes[uploaded_file.name] = pdf_bytes

                risk = integrity_result.get("overall_risk", "LOW")
                high_c = integrity_result.get("high_count", 0)
                med_c = integrity_result.get("medium_count", 0)

                if risk == "HIGH":
                    st.error(
                        f"🚨 **PDF INTEGRITY ALERT — HIGH RISK**: {uploaded_file.name} "
                        f"({high_c} high, {med_c} medium findings)"
                    )
                elif risk == "MEDIUM":
                    st.warning(
                        f"⚠️ **PDF Integrity Warning**: {uploaded_file.name} "
                        f"({med_c} medium findings)"
                    )
            except Exception as e:
                st.caption(f"ℹ️ PDF integrity check skipped for {uploaded_file.name}: {e}")

            # extract company name (FIXED)
            company_name = None
            try:
                with bytes_to_pdfplumber(pdf_bytes, password=pdf_pw) as meta_pdf:
                    company_name = extract_company_name(meta_pdf, max_pages=2)
            except Exception:
                company_name = None

            # extract account number (NEW)
            account_no = None
            try:
                with bytes_to_pdfplumber(pdf_bytes, password=pdf_pw) as meta_pdf:
                    account_no = extract_account_number(meta_pdf, max_pages=2)
            except Exception:
                account_no = None

            # manual override wins
            if (st.session_state.company_name_override or "").strip():
                company_name = st.session_state.company_name_override.strip()

            st.session_state.file_company_name[uploaded_file.name] = company_name
            st.session_state.file_account_no[uploaded_file.name] = account_no

            # Parse transactions (existing logic)
            if bank_choice == "Affin Bank":
                with bytes_to_pdfplumber(pdf_bytes, password=pdf_pw) as pdf:
                    totals = extract_affin_statement_totals(pdf, uploaded_file.name)
                    st.session_state.affin_statement_totals.append(totals)
                    tx_raw = parse_affin_bank(pdf, uploaded_file.name) or []

            elif bank_choice == "Ambank":
                with bytes_to_pdfplumber(pdf_bytes, password=pdf_pw) as pdf:
                    totals = extract_ambank_statement_totals(pdf, uploaded_file.name)
                    st.session_state.ambank_statement_totals.append(totals)
                    tx_raw = parse_ambank(pdf, uploaded_file.name) or []

            elif bank_choice == "CIMB Bank":
                with bytes_to_pdfplumber(pdf_bytes, password=pdf_pw) as pdf:
                    totals = extract_cimb_statement_totals(pdf, uploaded_file.name)
                    st.session_state.cimb_statement_totals.append(totals)
                    tx_raw = parse_transactions_cimb(pdf, uploaded_file.name) or []

            elif bank_choice == "RHB Bank":
                with bytes_to_pdfplumber(pdf_bytes, password=pdf_pw) as pdf:
                    totals = extract_rhb_statement_totals(pdf, uploaded_file.name)
                    st.session_state.rhb_statement_totals.append(totals)
                tx_raw = parser(pdf_bytes, uploaded_file.name) or []

            elif bank_choice == "Bank Islam":
                with bytes_to_pdfplumber(pdf_bytes, password=pdf_pw) as pdf:
                    tx_raw = parse_bank_islam(pdf, uploaded_file.name) or []
                    stmt_month = extract_bank_islam_statement_month(pdf)
                    if stmt_month:
                        st.session_state.bank_islam_file_month[uploaded_file.name] = stmt_month

            else:
                tx_raw = parser(pdf_bytes, uploaded_file.name) or []

            # Normalize then attach company_name
            tx_norm = normalize_transactions(
                tx_raw,
                default_bank=bank_choice,
                source_file=uploaded_file.name,
            )
            for t in tx_norm:
                t["company_name"] = company_name
                t["account_no"] = account_no

            # Sprint 4.5: harvest parser-locked account_type determination.
            # finalize_parser_output stamps it on row[0]; ensure_transaction_schema
            # preserves the `_account_type_determination` key through normalization.
            # Pop it off every row here so it doesn't leak into full_report.transactions[].
            # Zero-row files (OCR-only) get an UNDETERMINED sentinel.
            _det = None
            for t in tx_norm:
                _payload = t.pop("_account_type_determination", None)
                if _det is None and isinstance(_payload, dict):
                    _det = _payload
            if _det is None:
                _det = determine_account_type([])
            st.session_state.account_type_determinations.append({
                "source_file": uploaded_file.name,
                "bank": bank_choice,
                "company_name": company_name,
                "account_no": account_no,
                **_det,
            })

            if bank_choice == "Affin Bank":
                st.session_state.affin_file_transactions[uploaded_file.name] = tx_norm
            if bank_choice == "Ambank":
                st.session_state.ambank_file_transactions[uploaded_file.name] = tx_norm
            if bank_choice == "CIMB Bank":
                st.session_state.cimb_file_transactions[uploaded_file.name] = tx_norm
            if bank_choice == "RHB Bank":
                st.session_state.rhb_file_transactions[uploaded_file.name] = tx_norm

            if tx_norm:
                st.success(f"✅ Extracted {len(tx_norm)} transactions from {uploaded_file.name}")
                all_tx.extend(tx_norm)
            else:
                st.warning(f"⚠️ No transactions found in {uploaded_file.name}")

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            st.exception(e)

        progress_bar.progress((file_idx + 1) / total_files)

    bank_display_box.success(f"🏦 Completed processing: **{bank_choice}**")

    # ✅ Batch comparison: cross-file outlier detection
    try:
        if len(st.session_state.pdf_raw_bytes) >= 2:
            batch_extra = compare_pdf_batch(
                st.session_state.pdf_integrity_results,
                st.session_state.pdf_raw_bytes,
            )
            for fname, extra_findings in batch_extra.items():
                if extra_findings and fname in st.session_state.pdf_integrity_results:
                    res = st.session_state.pdf_integrity_results[fname]
                    res["all_findings"].extend(extra_findings)
                    res.setdefault("layer_results", {}).setdefault("batch_comparison", []).extend(extra_findings)
                    res["finding_count"] = len(res["all_findings"])
                    res["high_count"] = sum(1 for f in res["all_findings"] if f.get("severity") == "HIGH")
                    res["medium_count"] = sum(1 for f in res["all_findings"] if f.get("severity") == "MEDIUM")
                    res["low_count"] = sum(1 for f in res["all_findings"] if f.get("severity") == "LOW")
                    # Recalculate overall risk
                    severities = [f.get("severity", "LOW") for f in res["all_findings"]]
                    if "HIGH" in severities:
                        res["overall_risk"] = "HIGH"
                    elif "MEDIUM" in severities:
                        res["overall_risk"] = "MEDIUM"
                    if any(f.get("severity") == "HIGH" for f in extra_findings):
                        st.error(
                            f"🚨 **BATCH OUTLIER — HIGH RISK**: {fname} has a different "
                            "generation profile than other files from the same bank."
                        )
    except Exception:
        pass  # batch comparison is best-effort
    finally:
        # Free raw bytes from memory after comparison
        st.session_state.pdf_raw_bytes = {}

    all_tx = dedupe_transactions(all_tx)

    # Stable ordering
    for idx, t in enumerate(all_tx):
        if "__row_order" not in t:
            t["__row_order"] = idx

    def _sort_key(t: dict) -> Tuple:
        dt = parse_any_date_for_summary(t.get("date"))
        page = t.get("page")
        try:
            page_i = int(page) if page is not None else 10**9
        except Exception:
            page_i = 10**9

        seq = t.get("seq", None)
        try:
            seq_i = int(seq) if seq is not None else 10**9
        except Exception:
            seq_i = 10**9

        row_order = t.get("__row_order", 10**12)
        try:
            row_order_i = int(row_order)
        except Exception:
            row_order_i = 10**12

        return (
            dt if pd.notna(dt) else pd.Timestamp.max,
            page_i,
            seq_i,
            row_order_i,
        )

    all_tx = sorted(all_tx, key=_sort_key)
    st.session_state.results = all_tx


# =========================================================
# Monthly Summary Calculation (same logic, adds company_name)
# =========================================================
def calculate_monthly_summary(transactions: List[dict]) -> List[dict]:
    # Affin-only
    if bank_choice == "Affin Bank" and st.session_state.affin_statement_totals:
        rows: List[dict] = []
        for t in st.session_state.affin_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)

            opening = t.get("opening_balance")
            ending = t.get("ending_balance")

            txs = st.session_state.affin_file_transactions.get(fname, []) if fname else []
            tx_count = int(len(txs)) if txs else None

            # BUG-001 fix (2026-05-02): sum transactions instead of footer-parsed totals.
            td = round(sum(float(safe_float(x.get("debit") or 0)) for x in txs), 2) if txs else 0.0
            tc = round(sum(float(safe_float(x.get("credit") or 0)) for x in txs), 2) if txs else 0.0

            opening_balance = round(float(safe_float(opening)), 2) if opening is not None else None
            ending_balance = round(float(safe_float(ending)), 2) if ending is not None else None

            balances: List[float] = []
            for x in txs:
                b = x.get("balance")
                if b is None:
                    continue
                try:
                    balances.append(float(safe_float(b)))
                except Exception:
                    pass

            if ending_balance is None and balances:
                ending_balance = round(float(balances[-1]), 2)

            lowest_balance = round(min(balances), 2) if balances else None
            highest_balance = round(max(balances), 2) if balances else None

            net_change = None
            if td is not None and tc is not None:
                net_change = round(float(tc - td), 2)

            if opening_balance is None and ending_balance is not None and td is not None and tc is not None:
                opening_balance = round(float(ending_balance - (tc - td)), 2)

            rows.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": tx_count,
                    "opening_balance": opening_balance,
                    "total_debit": td,
                    "total_credit": tc,
                    "net_change": net_change,
                    "ending_balance": ending_balance,
                    "lowest_balance": lowest_balance,
                    "lowest_balance_raw": lowest_balance,
                    "highest_balance": highest_balance,
                    "od_flag": bool(lowest_balance is not None and float(lowest_balance) < 0),
                    "source_files": fname,
                }
            )
        return sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

    # Ambank-only
    if bank_choice == "Ambank" and st.session_state.ambank_statement_totals:
        rows: List[dict] = []
        for t in st.session_state.ambank_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)

            opening = t.get("opening_balance")
            ending = t.get("ending_balance")

            txs = st.session_state.ambank_file_transactions.get(fname, []) if fname else []
            tx_count = int(len(txs)) if txs else None

            # BUG-001 fix (2026-05-02): sum transactions instead of footer-parsed totals.
            td = round(sum(float(safe_float(x.get("debit") or 0)) for x in txs), 2) if txs else 0.0
            tc = round(sum(float(safe_float(x.get("credit") or 0)) for x in txs), 2) if txs else 0.0

            opening_balance = round(float(safe_float(opening)), 2) if opening is not None else None
            ending_balance = round(float(safe_float(ending)), 2) if ending is not None else None

            balances: List[float] = []
            for x in txs:
                b = x.get("balance")
                if b is None:
                    continue
                try:
                    balances.append(float(safe_float(b)))
                except Exception:
                    pass

            lowest_balance = round(min(balances), 2) if balances else None
            highest_balance = round(max(balances), 2) if balances else None

            net_change = None
            if td is not None and tc is not None:
                net_change = round(float(tc - td), 2)

            if opening_balance is None and ending_balance is not None and td is not None and tc is not None:
                opening_balance = round(float(ending_balance - (tc - td)), 2)

            rows.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": tx_count,
                    "opening_balance": opening_balance,
                    "total_debit": td,
                    "total_credit": tc,
                    "net_change": net_change,
                    "ending_balance": ending_balance,
                    "lowest_balance": lowest_balance,
                    "lowest_balance_raw": lowest_balance,
                    "highest_balance": highest_balance,
                    "od_flag": bool(lowest_balance is not None and float(lowest_balance) < 0),
                    "source_files": fname,
                }
            )
        return sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

    # CIMB-only
    if bank_choice == "CIMB Bank" and st.session_state.cimb_statement_totals:
        rows: List[dict] = []
        for t in st.session_state.cimb_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)

            ending = t.get("ending_balance")

            txs = st.session_state.cimb_file_transactions.get(fname, []) if fname else []
            tx_count = int(len(txs)) if txs else None

            # BUG-001 fix (2026-05-02): sum transactions instead of footer-parsed totals.
            td = round(sum(float(safe_float(x.get("debit") or 0)) for x in txs), 2) if txs else 0.0
            tc = round(sum(float(safe_float(x.get("credit") or 0)) for x in txs), 2) if txs else 0.0
            ending_balance = round(float(safe_float(ending)), 2) if ending is not None else None

            net_change = round(float(tc - td), 2)
            opening_balance = None
            if ending_balance is not None:
                opening_balance = round(float(ending_balance - (tc - td)), 2)

            # Prefer the dominant transaction-date month over the parsed
            # Statement Date. Robust against statements whose period spans
            # calendar months and against any future CIMB format change that
            # might mis-label the header date. Falls through to the Statement
            # Date when transactions are absent (zero-tx month).
            if txs:
                tx_months: List[str] = []
                for x in txs:
                    d = str(x.get("date") or "")
                    if len(d) >= 7 and d[4] == "-":
                        tx_months.append(d[:7])
                if tx_months:
                    from collections import Counter
                    month = Counter(tx_months).most_common(1)[0][0]

            balances: List[float] = []
            for x in txs:
                desc = str(x.get("description") or "")
                if re.search(r"CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUP", desc, flags=re.IGNORECASE):
                    continue
                b = x.get("balance")
                if b is None:
                    continue
                try:
                    balances.append(float(safe_float(b)))
                except Exception:
                    pass

            lowest_balance = round(min(balances), 2) if balances else None
            highest_balance = round(max(balances), 2) if balances else None

            rows.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": tx_count,
                    "opening_balance": opening_balance,
                    "total_debit": td,
                    "total_credit": tc,
                    "net_change": net_change,
                    "ending_balance": ending_balance,
                    "lowest_balance": lowest_balance,
                    "lowest_balance_raw": lowest_balance,
                    "highest_balance": highest_balance,
                    "od_flag": bool(lowest_balance is not None and float(lowest_balance) < 0),
                    "source_files": fname,
                }
            )
        return sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

    # RHB-only
    # The per-file branch assumes one PDF = one statement month. RHB Reflex
    # multi-month combined exports break that — fall through to the default
    # groupby below when any single uploaded file spans >1 transaction month.
    rhb_file_months: Dict[str, set] = {}
    if bank_choice == "RHB Bank":
        for tx in transactions:
            fname = str(tx.get("source_file") or "")
            d = str(tx.get("date") or "")
            if not fname or len(d) < 7 or d[4] != "-":
                continue
            rhb_file_months.setdefault(fname, set()).add(d[:7])
    rhb_multi_month_file = any(len(months) > 1 for months in rhb_file_months.values())

    if (
        bank_choice == "RHB Bank"
        and st.session_state.rhb_statement_totals
        and not rhb_multi_month_file
    ):
        rows: List[dict] = []
        for t in st.session_state.rhb_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)

            opening = t.get("opening_balance")
            ending = t.get("ending_balance")

            txs = st.session_state.rhb_file_transactions.get(fname, []) if fname else []
            tx_count = int(len(txs)) if txs else None

            # BUG-001 fix (2026-05-02): derive totals from the transactions array
            # rather than the PDF footer parse. Footer-parsed Withdraws/Deposits
            # totals were inflated by an identical amount on both sides each month
            # (Kay R Aug 2025: +RM 177,122 on each side). Transaction-level data is
            # authoritative; balance trail reconciles 100%.
            td = round(sum(float(safe_float(x.get("debit") or 0)) for x in txs), 2) if txs else 0.0
            tc = round(sum(float(safe_float(x.get("credit") or 0)) for x in txs), 2) if txs else 0.0
            opening_balance = round(float(safe_float(opening)), 2) if opening is not None else None
            ending_balance = round(float(safe_float(ending)), 2) if ending is not None else None

            balances: List[float] = []
            for x in txs:
                b = x.get("balance")
                if b is None:
                    continue
                try:
                    balances.append(float(safe_float(b)))
                except Exception:
                    pass

            lowest_balance = round(min(balances), 2) if balances else None
            highest_balance = round(max(balances), 2) if balances else None

            net_change = None
            if td is not None and tc is not None:
                net_change = round(float(tc - td), 2)

            if opening_balance is None and ending_balance is not None and td is not None and tc is not None:
                opening_balance = round(float(ending_balance - (tc - td)), 2)

            rows.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": tx_count,
                    "opening_balance": opening_balance,
                    "total_debit": td,
                    "total_credit": tc,
                    "net_change": net_change,
                    "ending_balance": ending_balance,
                    "lowest_balance": lowest_balance,
                    "lowest_balance_raw": lowest_balance,
                    "highest_balance": highest_balance,
                    "od_flag": bool(lowest_balance is not None and float(lowest_balance) < 0),
                    "source_files": fname,
                }
            )
        return sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

    # Default banks
    if not transactions:
        if bank_choice == "Bank Islam" and getattr(st.session_state, "bank_islam_file_month", {}):
            rows: List[dict] = []
            for fname, month in sorted(st.session_state.bank_islam_file_month.items(), key=lambda x: x[1]):
                company_name = st.session_state.file_company_name.get(fname)
                account_no = st.session_state.file_account_no.get(fname)
                rows.append(
                    {
                        "month": month,
                        "company_name": company_name,
                        "account_no": account_no,
                        "transaction_count": 0,
                        "opening_balance": None,
                        "total_debit": 0.0,
                        "total_credit": 0.0,
                        "net_change": 0.0,
                        "ending_balance": None,
                        "lowest_balance": None,
                        "lowest_balance_raw": None,
                        "highest_balance": None,
                        "od_flag": False,
                        "source_files": fname,
                    }
                )
            return rows
        return []

    df = pd.DataFrame(transactions)
    if df.empty:
        return []

    df = df.reset_index(drop=True)
    if "__row_order" not in df.columns:
        df["__row_order"] = range(len(df))

    df["date_parsed"] = df.get("date").apply(parse_any_date_for_summary)
    df = df.dropna(subset=["date_parsed"])
    if df.empty:
        st.warning("⚠️ No valid transaction dates found.")
        return []

    df["month_period"] = df["date_parsed"].dt.strftime("%Y-%m")
    df["debit"] = df.get("debit", 0).apply(safe_float)
    df["credit"] = df.get("credit", 0).apply(safe_float)
    df["balance"] = df.get("balance", None).apply(lambda x: safe_float(x) if x is not None else None)

    if "page" in df.columns:
        df["page"] = pd.to_numeric(df["page"], errors="coerce").fillna(0).astype(int)
    else:
        df["page"] = 0

    has_seq = "seq" in df.columns
    if has_seq:
        df["seq"] = pd.to_numeric(df["seq"], errors="coerce").fillna(0).astype(int)

    df["__row_order"] = pd.to_numeric(df["__row_order"], errors="coerce").fillna(0).astype(int)

    monthly_summary: List[dict] = []
    for period, group in df.groupby("month_period", sort=True):
        sort_cols = ["date_parsed", "page"]
        if has_seq:
            sort_cols.append("seq")
        sort_cols.append("__row_order")

        group_sorted = group.sort_values(sort_cols, na_position="last")

        balances = group_sorted["balance"].dropna()
        # highest/lowest stay over the union of all account balances within
        # the month — semantics preserved (worst-case dip across accounts).
        highest_balance = round(float(balances.max()), 2) if not balances.empty else None
        lowest_balance_raw = round(float(balances.min()), 2) if not balances.empty else None
        lowest_balance = lowest_balance_raw
        od_flag = bool(lowest_balance is not None and float(lowest_balance) < 0)

        company_vals = [
            x for x in group_sorted.get("company_name", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist()
            if x.strip()
        ]
        company_name = company_vals[0] if company_vals else None

        acct_vals = [
            x for x in group_sorted.get("account_no", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist() if x.strip()
        ]
        account_no = acct_vals[0] if len(acct_vals) == 1 else (", ".join(acct_vals) if acct_vals else None)

        # Alliance DR-balance (overdraft) accounts invert the opening-balance formula.
        is_dr_balance = False
        if "account_type" in group_sorted.columns:
            at = group_sorted["account_type"].dropna().astype(str).str.upper().unique().tolist()
            is_dr_balance = "OD" in at

        # BUG-002 (2026-05-05) — multi-account ending/opening must SUM across
        # accounts, not pick one account's last row. When the bundle contains
        # N accounts for a given month, ending_balance was picking whichever
        # account sorted last by (date, page, seq) — single-account value
        # instead of the sum. Same shape for the opening-row seed. Per-account
        # balance trails are already correct (validator PASSES); only the
        # roll-up was wrong.
        #
        # Opening-row description match covers both Alliance's "BEGINNING
        # BALANCE" and Maybank's synthetic "OPENING BALANCE" row (maybank.py
        # emits is_opening_balance=True). Without recognising the Maybank
        # form, OD first months fall through to the is_dr_balance formula
        # below, which encodes Alliance's positive-debt-magnitude convention
        # and produces the wrong sign for Maybank's pre-negated OD convention
        # (HUAHUB Oct 2025 was off by RM 8,582.30 = 2 * net_change).
        OPENING_DESC_RE = r"(?:BEGINNING|OPENING)\s+BALANCE"
        per_acct_endings: List[float] = []
        per_acct_seeds: List[float] = []
        if "account_no" in group_sorted.columns and group_sorted["account_no"].notna().any():
            for _acct, acct_group in group_sorted.groupby("account_no", sort=False):
                acct_balances = acct_group["balance"].dropna()
                if not acct_balances.empty:
                    per_acct_endings.append(float(acct_balances.iloc[-1]))
                if "description" in acct_group.columns:
                    bb_acct = acct_group[
                        acct_group["description"].astype(str).str.upper()
                        .str.contains(OPENING_DESC_RE, na=False, regex=True)
                    ]
                    if not bb_acct.empty:
                        bal0 = bb_acct.iloc[0].get("balance")
                        if bal0 is not None and not pd.isna(bal0):
                            per_acct_seeds.append(float(bal0))
        else:
            # No account_no field — fall back to single-account behaviour.
            if not balances.empty:
                per_acct_endings.append(float(balances.iloc[-1]))
            if "description" in group_sorted.columns:
                bb = group_sorted[
                    group_sorted["description"].astype(str).str.upper()
                    .str.contains(OPENING_DESC_RE, na=False, regex=True)
                ]
                if not bb.empty:
                    bal0 = bb.iloc[0].get("balance")
                    if bal0 is not None and not pd.isna(bal0):
                        per_acct_seeds.append(float(bal0))

        ending_balance = round(sum(per_acct_endings), 2) if per_acct_endings else None
        seed_opening = round(sum(per_acct_seeds), 2) if per_acct_seeds else None


        monthly_summary.append(
            {
                "month": period,
                "company_name": company_name,
                "account_no": account_no,
                "transaction_count": int(len(group_sorted)),
                "opening_balance": seed_opening,
                "is_dr_balance": is_dr_balance,
                "total_debit": round(float(group_sorted["debit"].sum()), 2),
                "total_credit": round(float(group_sorted["credit"].sum()), 2),
                "net_change": round(float(group_sorted["credit"].sum() - group_sorted["debit"].sum()), 2),
                "ending_balance": ending_balance,
                "lowest_balance": lowest_balance,
                "lowest_balance_raw": lowest_balance_raw,
                "highest_balance": highest_balance,
                "od_flag": od_flag,
                "source_files": ", ".join(sorted(set(group_sorted.get("source_file", []))))
                if "source_file" in group_sorted.columns
                else "",
            }
        )

    # Bank Islam ensure statement months with zero tx still appear
    if bank_choice == "Bank Islam" and getattr(st.session_state, "bank_islam_file_month", {}):
        existing_months = {r.get("month") for r in monthly_summary}
        for fname, month in st.session_state.bank_islam_file_month.items():
            if month in existing_months:
                continue
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)
            monthly_summary.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": 0,
                    "opening_balance": None,
                    "total_debit": 0.0,
                    "total_credit": 0.0,
                    "net_change": 0.0,
                    "ending_balance": None,
                    "lowest_balance": None,
                    "lowest_balance_raw": None,
                    "highest_balance": None,
                    "od_flag": False,
                    "source_files": fname,
                }
            )

    # RHB multi-month fall-through path: seed the earliest month's opening_balance
    # from extract_rhb_statement_totals when default's OPENING/BEGINNING-row detection
    # can't (RHB Reflex emits B/F BALANCE in headers, not as transaction rows).
    if (
        bank_choice == "RHB Bank"
        and monthly_summary
        and st.session_state.rhb_statement_totals
        and rhb_multi_month_file
    ):
        first_row = min(monthly_summary, key=lambda x: x["month"])
        if first_row.get("opening_balance") is None:
            for t in st.session_state.rhb_statement_totals:
                ob = t.get("opening_balance")
                if ob is not None:
                    first_row["opening_balance"] = round(float(safe_float(ob)), 2)
                    break

    # Fill opening_balance for default banks using prior month's ending_balance when possible.
    monthly_summary_sorted = sorted(monthly_summary, key=lambda x: x["month"])
    prev_end = None
    for r in monthly_summary_sorted:
        if r.get("opening_balance") is None:
            if prev_end is not None:
                r["opening_balance"] = round(float(prev_end), 2)
            else:
                # best-effort fallback: opening = ending - net_change (CR account)
                # For DR-balance accounts (overdraft): opening = ending + net_change
                eb = r.get("ending_balance")
                nc = r.get("net_change")
                if eb is not None and nc is not None:
                    try:
                        if r.get("is_dr_balance"):
                            r["opening_balance"] = round(float(safe_float(eb) + safe_float(nc)), 2)
                        else:
                            r["opening_balance"] = round(float(safe_float(eb) - safe_float(nc)), 2)
                    except Exception:
                        r["opening_balance"] = None

        # update prev_end for next month
        if r.get("ending_balance") is not None:
            prev_end = safe_float(r.get("ending_balance"))

    return monthly_summary_sorted


# =========================================================
# Presentation-only Monthly Summary Standardization
# =========================================================
def present_monthly_summary_standard(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows or []:
        highest = r.get("highest_balance")
        lowest = r.get("lowest_balance")

        swing = None
        try:
            if highest is not None and lowest is not None:
                swing = round(float(safe_float(highest) - safe_float(lowest)), 2)
        except Exception:
            swing = None

        out.append(
            {
                "month": r.get("month"),
                "company_name": r.get("company_name"),
                "account_no": r.get("account_no"),
                "opening_balance": r.get("opening_balance"),
                "total_debit": r.get("total_debit"),
                "total_credit": r.get("total_credit"),
                "highest_balance": highest,
                "lowest_balance": lowest,
                "swing": swing,
                "ending_balance": r.get("ending_balance"),
                "source_files": r.get("source_files"),
            }
        )
    return out


# ---------------------------------------------------
# COUNTERPARTY LEDGER (Change 6 — Part A, parser-side)
# ---------------------------------------------------
# Deterministic per-counterparty grouping. No classification logic.
# Extraction patterns align with Classification Rules v3 CP1–CP11.
# Renderer (HTML repo) cross-references related_parties[] for RP badges.

_CP_STOP_KEYWORDS = (
    r"(?:SALARY|GAJI|MONTHLY INSTALMENT|INSTALMENT|HOUSE INSTALMENT|HOUSING LOAN|"
    r"CREDIT CARD|CC CIMB|CC RHB|CC PAYMENT|CC [A-Z]{2,}|"
    r"PETTY CASH|TERM LOAN|ADVANCE|REFUND|VISA|TICKETS?|TRAINS?|CLAIMS?|"
    r"ACCOMMODATION|UNIFORM|PETROL|SITE VISIT|HP MONTHLY|HP SETTLEMENT|REPAYMENT|"
    r"BONUS|INCENTIVE|OVERTIME|ALLOWANCE|PERUNTKN|PERUNTUKAN|"
    r"EPF PAYMENT|EPF|SOCSO|EIS|LHDN|PCB|ZAKAT|"
    r"INSURANCE|ELECTRICITY|ELECTRICTY|WATER BILL|UTILITIES|TELEKOM|ASTRO|TNB|"
    r"GOLF|PESONA GOLF|CAR SERVICE|CAR LOAN|"
    r"STAFF CLAIM|STAFF BONUS|STAFF ADVANCE|STAFF INCENTIVE|STAFF OUTSTATION|STAFF OVERTIME|CLAIM|"
    r"DIRECTOR FEE|DIRECTORS FEE|SHARE CAPITAL|SHARE CAP|MTSB SHARE CAP|"
    r"TRANSFER BACK|MONTH END|MTH END|MMU FEES|"
    r"TRAVEL|TRIP|LAPTOP|RESIDENT|KUCHING|ADDITIONAL PACKAGE|"
    r"OPENING CA|BALANCE PAYMENT|PAID INVOICE|CLOSE ACC|"
    r"ONLINE TRANSFER|FUND TRANSFER|GUARD|SECURITY GUARD|SECURITY CHARGE|SERVICE CHARGE|"
    r"RENTAL|DEPOSIT|SUMBANGAN|AUDIT FEE|FORM C|SETTLEMENT|MILESTONE|"
    r"TROPHY|RAMADAN|KEMMAS|PAYMENT|CASH|DEVICE|DONATION|FOOD|TUNTUTAN|"
    r"13TH EXTRA|PRESTRO|BAOFEN|SAFETY|STOCK HQ|STAMPING|DUTI SETEM|"
    r"TRADE LICENSE|TAJAAN|TOKEN AWARD|TOPUP|PERMIT|BUFFET|ANNUAL DINNER|"
    r"DUITNOW \(TRANSFER\))"
)
_CP_BANK_SUFFIX = r"(?:RHB|CIMB|MBB|HLB|AMB|BIMB|ABB|PBB|OCBC|UOB|HLBB|BSN|MBSB|BKRM|AFFIN|AFFN|BIBM|AMFB|ABMB)"

# Vehicle plate e.g. "QRT8957", "VJS8957", "G8957"; quarter/period codes; generic numeric refs
_CP_STOP_TOKEN_RE = re.compile(
    r"^(?:[A-Z]{1,3}\d{3,5}|QTR\d+|QT\d+|CP_\d+_\d+|F\d{2}[A-Z]\d+|\d{5,})$"
)


def _strip_trailing_refs(s: str) -> str:
    # Remove trailing digit/slash runs and currency-like codes
    s = re.sub(r"\s+[\d/]{3,}.*$", "", s).strip()
    return s


def _strip_stop_tokens(s: str) -> str:
    """Drop trailing tokens that match vehicle-plate / numeric-ref patterns."""
    if not s:
        return s
    tokens = s.split()
    while tokens and _CP_STOP_TOKEN_RE.match(tokens[-1]):
        tokens.pop()
    return " ".join(tokens).strip()


def _dedupe_duplicated_prefix(s: str) -> str:
    """Collapse 'X X Y' → 'Y' where X is a duplicated purpose phrase (1-8 words)."""
    if not s:
        return s
    tokens = s.split()
    n = len(tokens)
    # Try longest repeated prefix first
    for k in range(min(8, n // 2), 0, -1):
        if tokens[:k] == tokens[k:2 * k]:
            return " ".join(tokens[2 * k:]).strip()
    return s


def _find_duplicated_block_end(tokens: List[str]) -> int:
    """Return index just after the second occurrence of a duplicated block.

    Finds (a, k, b) such that tokens[a:a+k] == tokens[b:b+k] (with a+k <= b,
    non-overlapping) and returns b+k. Prefers the longest k; among ties
    prefers the rightmost b. At least one token in the block must contain
    a digit or be a reference-like token (avoid matching common words).
    """
    n = len(tokens)
    if n < 4:
        return 0

    best_end = 0
    best_k = 0
    # Try larger k first so longest blocks win
    for k in range(min(n // 2, 6), 0, -1):
        for a in range(0, n - 2 * k + 1):
            block = tokens[a:a + k]
            # Skip trivial/noisy single-char blocks (k=1 only)
            if k == 1 and (len(block[0]) <= 1 or block[0] in (".", ",", "-")):
                continue
            for b in range(a + k, n - k + 1):
                if tokens[b:b + k] == block:
                    end = b + k
                    if k > best_k or (k == best_k and end > best_end):
                        best_end, best_k = end, k
        if best_k == k and best_end > 0:
            return best_end
    return best_end


_CP_NAME_ANCHORS = ("BIN", "BINTI", "BT", "BTE", "ANAK", "A/L", "A/P")
_CP_PURPOSE_WORDS = re.compile(
    r"^(?:FUND|TRANSFER|REFUND|BAKI|SUMBANGAN|PERUNTUKAN|PAYMENT|BALANCE|"
    r"PK|PMB|INV|IV|QT|QTR|SDR|YE|I|FEE|SEC|MILESTONE|COMPLET|"
    r"JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC|"
    r"JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)$"
)


def _strip_purpose_prefix_tokens(tokens: List[str]) -> List[str]:
    """Drop leading tokens that are obviously purpose/date words (not entity names)."""
    out = list(tokens)
    stripped_something = False
    while out:
        t = out[0].strip("().,")
        if not t:
            out.pop(0)
            continue
        # Skip AND/& connector only if we've already stripped a purpose/date
        # token (e.g. "JAN AND FEB 2026 <name>"). Prevents dropping a legit
        # entity name that happens to start with "AND" or "&".
        if stripped_something and t.upper() in ("AND", "&"):
            out.pop(0)
            continue
        if _CP_PURPOSE_WORDS.match(t.upper()):
            out.pop(0)
            stripped_something = True
            continue
        if re.fullmatch(r"\d+", t) or re.fullmatch(r"\d{4}", t):  # year/number
            out.pop(0)
            stripped_something = True
            continue
        if re.fullmatch(r"[A-Z0-9/.-]+", t) and any(c.isdigit() for c in t):
            out.pop(0)
            stripped_something = True
            continue
        break
    return out


def _tail_alpha_run(tokens: List[str], min_len: int = 2) -> str:
    """Return the longest trailing run of purely-alphabetic tokens (len ≥ 2 each).

    Allows parenthesised or ampersand tokens like '(MALAYSI' or '&' to be part
    of the entity (will be stripped in normalisation). Returns '' if no run.
    """
    out: List[str] = []
    for t in reversed(tokens):
        clean = t.strip("().,&")
        if clean.isalpha() and len(clean) >= min_len:
            out.append(t)
        elif t in ("&", "(M)", "(SARAWAK)", "(MALAYSI", "(SAR", "(L"):
            out.append(t)
        else:
            break
    return " ".join(reversed(out)).strip() if len(out) >= min_len else ""


def _has_real_word(s: str) -> bool:
    """True if s contains a ≥3-char alphabetic run that isn't just a masked token.

    Used to distinguish a real entity substring ("MAYBANK VISA CARD") from a
    masked card / ref / digit string ("XXXX-XXXX-XXXX-1386", "210257120002").
    """
    if not s:
        return False
    for word in re.findall(r"[A-Z]+", s):
        if len(word) >= 3 and word != "X" * len(word):
            return True
    return False


def _hlb_extract_entity(body: str) -> str:
    """Sprint 6 #4 — HLB entity extraction from a stripped body.

    Strips boilerplate / trailing-noise tokens, then walks back from the end
    collecting name-shaped tokens (letters, no embedded digits). Returns the
    trailing alpha run, or "" if none found. Tolerates Malay name punctuation
    (apostrophes), Malaysian legal-suffix variants (SDN BHD / S/B / (M) /
    (KLANG)), and ampersand bridges. Per Q2 (audit), accepts truncated names
    so the analyst gets a clue rather than UNCATEGORIZED.
    """
    if not body:
        return ""
    body = body.strip()
    # Drop trailing 'Hong Leong (Islamic) Bank Berhad ...' boilerplate that
    # leaks into some rows from the PDF page footer.
    body = re.sub(
        r"\s+Hong\s+Leong\s+(?:Islamic\s+)?Bank.*$",
        "",
        body,
        flags=re.IGNORECASE,
    )
    tokens = body.split()
    # Strip trailing noise: pure-numeric tokens, lone underscores / single
    # punctuation chars, dangling open paren.
    while tokens:
        t = tokens[-1]
        if t == "_" or re.fullmatch(r"[\d,.]+", t):
            tokens.pop()
            continue
        if len(t) == 1 and not t.isalpha():
            tokens.pop()
            continue
        break
    # Walk right-to-left collecting name-shaped tokens. A name-shaped token
    # has at least one letter and no embedded digits. Bare ampersand acts
    # as a name-bridge (HABLEM OIL & GAS) but only when sandwiched between
    # name tokens — never as the leading/trailing token.
    out: List[str] = []
    for t in reversed(tokens):
        has_letter = any(c.isalpha() for c in t)
        has_digit = any(c.isdigit() for c in t)
        if has_letter and not has_digit:
            out.append(t)
            continue
        if t == "&" and out:
            out.append(t)
            continue
        break
    if not out:
        return ""
    return " ".join(reversed(out)).strip().rstrip(".,;:&")


def _hlb_extract_uppercase_tail(body: str) -> str:
    """2026-05-08 — HLB retail-format entity extraction.

    Variant of `_hlb_extract_entity` that only walks back over tokens whose
    alphabetic characters are all UPPERCASE. Used for HLConnect retail
    descriptions where lowercase Malay purpose text ('Bayaran balik modal',
    'gaji pekerja', 'katering', 'Fund transfer') sits between the amount
    prefix and the all-caps entity. Stops at the first non-uppercase or
    digit-bearing token.
    """
    if not body:
        return ""
    body = body.strip()
    body = re.sub(
        r"\s+Hong\s+Leong\s+(?:Islamic\s+)?Bank.*$",
        "",
        body,
        flags=re.IGNORECASE,
    )
    tokens = body.split()
    while tokens:
        t = tokens[-1]
        if t == "_" or re.fullmatch(r"[\d,.]+", t):
            tokens.pop(); continue
        if len(t) == 1 and not t.isalpha():
            tokens.pop(); continue
        break
    out: List[str] = []
    for t in reversed(tokens):
        if any(c.isdigit() for c in t):
            break
        if t == "&" and out:
            out.append(t); continue
        # Token must be all-uppercase across its alpha chars (e.g. BHD., M.,
        # SDN. all qualify because str.isupper() ignores non-cased chars).
        if t and t.isupper():
            out.append(t); continue
        break
    if not out:
        return ""
    return " ".join(reversed(out)).strip().rstrip(".,;:&")


def _clip_at_stop_keyword(rest: str) -> str:
    """Clip at first purpose-keyword match (anchored at word boundary, not pos 0)."""
    stop = re.search(rf"\b{_CP_STOP_KEYWORDS}\b", rest, flags=re.IGNORECASE)
    if stop and stop.start() > 0:
        return rest[:stop.start()].strip()
    return rest


# ── Sprint 6 #9 — Bank Rakyat entity extraction ───────────────────────────
# Felcra-style Bank Rakyat PDFs strip ALL spaces from continuation lines, so
# entity names arrive as concatenated blobs (`AHMADJAWWADBINYAHAYA` rather
# than `AHMAD JAWWAD BIN YAHAYA`). Per Q2=C: take the first token from the
# continuation, after stripping known noise (sub-account tags like AGROBIZ /
# TRADING / KKF, staff IDs, ref codes like KZ\d / IV-\d / WFV\d). The
# concatenated name is acceptable per the HLB Q2 directive — analyst clue
# beats UNCATEGORIZED.
_BR_NOISE_LITERAL_UPPER = {
    "AGROBIZ", "TRADING", "KKF", "AHMAD", "REKUP",
    "PREMIUMHOSPITALPES", "INSURANCE", "KKFINSURANS", "KKFINS", "KAW.",
}

_BR_NOISE_PREFIX_RE = re.compile(
    r"^(?:"
    r"STAFFID\d+"             # StaffID029
    r"|INSHOSP[\w/]*"         # Inshosp2023/24
    r"|GAJI[A-Z]*\d*"         # GAJINOV2023
    r"|WFV\d+|VAX\d+|WMH\d+|WRK\d+"  # Felcra ref tokens
    r"|KZ\d[\w/]*"            # KZ2611, KZ2611/BAY1...
    r"|IV-?\d[\w/-]*"         # invoice refs
    r"|INV-?[\w/-]+"          # INV0001511
    r"|PBWPHG/[\w/-]+"        # PBWPHG/PK3047
    r"|BAY\d[\w/-]*"          # BAY7SIJIL...
    r"|KMIWK[\w/-]*"          # KMIWK119/23
    r"|CFC\d+|RTB\d+"
    r"|\d{5,}"                # bare long numeric refs
    r")$",
    re.IGNORECASE,
)


# Sprint 7 #2 (V3-A) — corporate-suffix terminators for the multi-token
# entity case (MTCEC-style PDFs preserve spaces in the continuation, so the
# entity arrives as "MTC ENGINEERING CONSULTANCY SDN BHD" rather than the
# Felcra-style concatenated blob).
_BR_CORP_TERMINATORS = {
    "BHD", "BERHAD", "ENTERPRISE", "TRADING", "CORPORATION", "CORP",
    "GROUP", "HOLDINGS", "INDUSTRIES",
}
_BR_ENTITY_LOOKAHEAD = 6

# Sprint 7 #3 (V3-A continued) — opcode-spacing normalization. MTCEC-style
# Bank Rakyat PDFs emit opcodes with spaces ("DUITNOW TRANSFER", "CIB CR
# ADVICE") while Felcra-style concatenates them ("DUITNOWTRANSFER"). All
# downstream BR-section regexes assume the concatenated form. Collapsing
# the spaces inside known opcodes at the start of body_after_code lets the
# existing regexes match either format without rewriting them.
_BR_OPCODE_NORMALIZE_RE = re.compile(
    r"^("
    r"DUITNOW\s+(?:TRANSFER|FEE)"
    r"|CIB\s+(?:CR|DR)\s+ADVICE(?:\s*\([A-Z]+\))?"
    r"|CIB\s+SMS\s+FEE"
    r"|CIB\s+DR\s+CHARGES"
    r"|CIB\s+COMMISSION"
    r"|PROFIT\s+CHARGED"
    r"|CDM\s+CASH\s+DEPOSIT"
    r"|CASH\s+DEPOSIT"
    r"|CASH\s+WITHDRAWAL"
    r"|CREDIT\s+PROFIT(?:\s*/\s*HIBAH)?"
    r"|IBG\s+INWARD\s+RETURN"
    r"|IBG\s+CREDIT"
    r"|REVERSAL\s+CR"
    r"|LOCAL\s+CHQ\s+RTN"
    r"|BILL\s+PAYMENT\s+TO\s+FIN"
    r"|TRFR\s+SHARE\s+MEMBER"
    r"|ATM\s+TRANSFER\s+CR"
    r"|2D\s+LOCAL\s+CHQ"
    r"|REMITTANCE\s+CR(?:\s*-\s*[A-Z]+)?"
    r"|TRANSFER\s+FROM\s+SA"
    r"|TR\s+FROM\s+SA"
    r"|TR\s+TO\s+SAVINGS"
    r"|ATM\s+MEPS\s+IBFT\s+CR"
    r"|CREDIT\s+ADV"
    r")",
    re.IGNORECASE,
)


def _br_normalize_opcode(body_after_code: str) -> str:
    """Collapse internal whitespace in a known MTCEC-style opcode at the start
    of body_after_code, producing the Felcra-style concatenated form.
    No-op for descriptions that already start with a concatenated opcode or
    that don't begin with a known opcode at all."""
    return _BR_OPCODE_NORMALIZE_RE.sub(
        lambda m: re.sub(r"\s+", "", m.group(0).upper()),
        body_after_code,
        count=1,
    )


def _br_extract_entity(body: str) -> str:
    """Pick the entity name from Bank Rakyat continuation tokens.

    Two surface formats coexist in the wild:
      - Felcra-style: continuation lines have all spaces stripped, so the
        entity arrives as ONE concatenated token (`AHMADJAWWADBINYAHAYA`).
      - MTCEC-style: continuation preserves spaces, so the entity is
        multi-word (`MTC ENGINEERING CONSULTANCY SDN BHD ITB TRF ...`).

    Strategy: skip noise, pick the first valid starter (≥3 letters, not a
    ref/staff/code prefix). Then look ahead up to _BR_ENTITY_LOOKAHEAD
    tokens for a corporate suffix (BHD / BERHAD / ENTERPRISE / TRADING /
    CORPORATION / CORP / GROUP / HOLDINGS / INDUSTRIES). If found, return
    everything from the starter through the terminator. Otherwise return
    just the starter — preserves the intentional Felcra single-blob output.
    """
    if not body:
        return ""
    # Some Felcra PDFs render non-Latin glyphs as literal '?' between name
    # fragments (e.g. MD?YUSOF?BIN?HASHIM). Strip so the same person doesn't
    # appear as both ABDUL?MALEK?BIN?HASHIM and ABDMALEKBHASHIM.
    body = body.replace("?", "")
    tokens: list[str] = []
    for tok in body.split():
        clean = tok.rstrip(",.;:")
        if clean:
            tokens.append(clean)
    if not tokens:
        return ""
    start = -1
    for i, clean in enumerate(tokens):
        upper = clean.upper()
        if upper in _BR_NOISE_LITERAL_UPPER:
            continue
        if _BR_NOISE_PREFIX_RE.match(clean):
            continue
        if clean[0].isdigit():
            continue
        if len(re.sub(r"[^A-Za-z]", "", clean)) < 3:
            continue
        start = i
        break
    if start < 0:
        return ""
    window_end = min(len(tokens), start + 1 + _BR_ENTITY_LOOKAHEAD)
    for j in range(start + 1, window_end):
        if tokens[j].upper() in _BR_CORP_TERMINATORS:
            return " ".join(tokens[start:j + 1])
    return tokens[start]


# ── Alliance Bank (ABMB) — patterns AB1-AB11 ────────────────────────────────
# Alliance statements duplicate the description text (entity appears twice)
# and truncate entity names at ~20 chars. Handled here before the generic
# extractor since prefixes are distinct enough not to collide with other banks.

_AB_TRUNCATION_FIXES = [
    (re.compile(r"\bSDN\s*BH\b(?!D)"), "SDN BHD"),
    (re.compile(r"\bSD\s*$"), "SDN BHD"),
    (re.compile(r"\bBH\s*$"), "BHD"),
    (re.compile(r"\bELECTRI\s*$"), "ELECTRICAL"),
    (re.compile(r"\bTECHNOLOG\s*$"), "TECHNOLOGIES"),
    (re.compile(r"\bKESELAMAT\s*$"), "KESELAMATAN"),
]


def _ab_fix_truncation(name: str) -> str:
    if not name:
        return name
    for pat, repl in _AB_TRUNCATION_FIXES:
        name = pat.sub(repl, name)
    return re.sub(r"\s+", " ", name).strip()


def _ab_dedupe_halves(text: str) -> str:
    """Alliance duplicates the entity text. Two cases:
    (A) truncated-prefix duplicate — first half ends with a truncated token
        whose full form appears in the second half
        ("KODENKI ELECTRICAL C KODENKI ELECTRICAL CABLE SDN BHD" → keep 2nd)
    (B) full duplicate — second half largely repeats the first (keep 1st)."""
    if not text:
        return text
    toks = text.split()
    n = len(toks)
    if n < 4:
        return text

    # (A) Truncated-prefix duplicate: for k = 2..n//2, check if toks[:k]
    # matches toks[k:2k] exactly on all-but-last token AND toks[k-1] is a
    # prefix of toks[2k-1] (or equal). Keep the fuller second half + tail.
    for k in range(2, n // 2 + 1):
        if 2 * k > n:
            break
        first_part = toks[:k]
        second_part = toks[k:2 * k]
        if first_part[:-1] != second_part[:-1]:
            continue
        a, b = first_part[-1], second_part[-1]
        if a == b or (len(a) >= 1 and b.upper().startswith(a.upper())):
            return " ".join(toks[k:]).strip()

    # (B) Fallback: overlap-based full duplicate detection.
    mid = n // 2
    first = set(t for t in toks[:mid] if len(t) >= 3)
    second = [t for t in toks[mid:] if len(t) >= 3]
    if not second or not first:
        return text
    overlap = sum(1 for t in second if t in first)
    if overlap / len(second) >= 0.7:
        return " ".join(toks[:mid]).strip()
    return text


_AB_MONTH_RE = r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"

# Strips payer-reference prefixes of the form
#   "<WORD>[-/ ]<MONTH>'YY[-/ <MONTH>'YY]*"  e.g. BESTLITE-MAY'25, BESTLITE-MAY'25-JUN'25,
#   BESTLITE-APR'25/JUN'25, FOO MAY25, etc. Also strips stray trailing month
#   fragments left behind by description truncation ("/AUG'", "-JUN'").
# Generic — the leading word is not hardcoded, so this works for any customer.
_AB_PAYER_MONTH_REF_RE = re.compile(
    rf"\b[A-Z][A-Z0-9]{{1,}}[-\s/]+{_AB_MONTH_RE}'?\d{{0,2}}"
    rf"(?:[-/\s]+{_AB_MONTH_RE}'?\d{{0,2}})*",
    re.IGNORECASE,
)
_AB_STRAY_MONTH_RE = re.compile(
    rf"(?:^|\s)[-/]{_AB_MONTH_RE}'?\d{{0,2}}(?=\s|$)",
    re.IGNORECASE,
)


def _ab_strip_month_refs(s: str) -> str:
    if not s:
        return s
    s = _AB_PAYER_MONTH_REF_RE.sub(" ", s)
    s = _AB_STRAY_MONTH_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# Leading purpose/action keywords that precede the entity in Alliance DuitNow
# descriptions ("payment NEXUS GEMILANG", "fund transfer SUMMER HEALTHCARE",
# "Transfer Chua Yi Tung"). Stripped AFTER half-dedupe so duplicated prefixes
# collapse first, then this peels the purpose word.
# v3.3.1: added invoice/fee/settlement purposes with optional month prefix
# ("JAN INVOICES", "FEB 2025 INVOICE", "MARCH FEE"), and billing/repayment
# keywords — for cases where payer stamps purpose BEFORE entity name.
_AB_MONTH_KW = (
    r"JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH|ET|AC)?|APR(?:IL)?|MAY|MEI|"
    r"JUN(?:E)?|JUL(?:Y|AI)?|AUG(?:UST|OS)?|OGOS|SEP(?:T|TEMBER)?|"
    r"OCT(?:OBER)?|OKT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?|DIS(?:EMBER)?"
)
_AB_LEADING_PURPOSE_RE = re.compile(
    rf"^(?:"
    rf"(?:{_AB_MONTH_KW})(?:\s+\d{{2,4}})?\s+(?:INVOICES?|FEES?|SETTLEMENT|PAYMENT|BILLING)|"
    rf"PAYMENT|PAYMT|PAY|FUND\s+TRANSFER|FUND\s+TRF|TRANSFER|TRF|"
    rf"BAYARAN|BAYAR|REFUND|CLAIM|DEPOSIT(?:\s+FOR)?|"
    rf"INVOICES?|FEES?|SETTLEMENT|BILLING|REPAYMENT|REIMBURSEMENT"
    rf")\s+",
    re.IGNORECASE,
)


def _ab_strip_leading_purpose(s: str) -> str:
    if not s:
        return s
    return _AB_LEADING_PURPOSE_RE.sub("", s, count=1).strip()


def _ab_strip_trailing_refs(s: str) -> str:
    """Drop trailing invoice/reference junk: INV..., INVOICE NO..., OUR REF..., PV-..., BA\\d+."""
    s = re.sub(
        r"\s+(?:INVOICE\s+NO|INV(?:OICE)?|OUR\s+REF|PYMT\s+OF\s+INV|PV[-\s]\d+|BL[-\s]\d+|BA\d+)\b.*$",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    # Trailing pure numeric / ref run — only strip if nothing but digits/slashes/
    # dashes/spaces follows (a mid-string year like " 2025 ENTITY SDN BHD" must
    # NOT be stripped, which the old greedy ".*$" incorrectly consumed).
    s = re.sub(r"\s+[\d/-][\d/\s-]*$", "", s).strip()
    return s


# v3.3.1 cross-bank: generic voucher-code pattern. Matches multi-separator
# internal voucher references like PV/YN/2507-094, IV-YN-00856, BL/XXX/1234.
# These are customer-side voucher systems (not bank-specific) so they can
# appear in any Malaysian bank statement. Stripping them before entity extraction
# prevents the "PV/YN/2507-094 MUDAH HEALTHCARE" → spurious unique counterparty
# problem that inflated KDYN's counterparty count from ~400-600 real entities
# to 1,742 parser rows.
_VOUCHER_CODE_RE = re.compile(
    r"\b[A-Z]{1,4}[/\-][A-Z]{1,4}[/\-]\d+[-\w]*\b",
    re.IGNORECASE,
)


def _strip_voucher_codes(s: str) -> str:
    """Strip generic multi-separator voucher codes (PV/YN/..., IV-YN-..., etc).
    Cross-bank safe — matches only tokens with TWO separators (/ or -) between
    letter-runs and digits, which virtually never appears in real entity names.
    """
    if not s:
        return s
    s = _VOUCHER_CODE_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# v3.3.1: drop leading alphanumeric reference tokens. Alliance sometimes prefixes
# a transaction description with an internal system ID like "I202507010110653"
# or "IF01200011839125NPYT" (1-4 letters + 5+ digits + optional alphanumeric tail),
# which isn't a counterparty. Kept conservative — must be a token of ≥6 chars,
# start with letters, contain at least 5 consecutive digits. Will NOT match
# normal entity words (no 5-digit run in a real name).
_LEADING_REF_TOKEN_RE = re.compile(r"^[A-Z]{1,4}\d{5,}[A-Z0-9]*\b", re.IGNORECASE)


# BUG-001 (2026-05-05) — UOB counterparty extractor support regexes.
# Bank-routing anchors observed in UOB BIBPlus descriptions; the canonical
# counterparty name follows the LAST anchor in the BEFORE-pipe segment.
# Excludes RPP/RFLX/IBG (rail labels per memory feedback — never used as
# counterparty anchors).
_UOB_BANK_ANCHOR_RE = re.compile(
    r"\b(GEB|HLB|MBB|RHB|UOB|PSM|CIMB|HSBC|OCBC|PBB|AMB|AMBG|ABB|ABMB|"
    r"BIMB|BSN|MBSB|CTBC|BSB|BNPP|MPSB)\b"
)
# UOB factoring drawdown reference (per-invoice, must NOT disambiguate counterparty).
_UOB_MSF_REF_RE = re.compile(r"\|MSF-GW\d+\|")
# UOB time-stamp leak: standalone AM/PM between uppercase tokens within a name.
_UOB_AMPM_LEAK_RE = re.compile(r"(?<=[A-Z])\s+(AM|PM)\s+(?=[A-Z])")


def _strip_leading_ref_token(s: str) -> str:
    if not s:
        return s
    return _LEADING_REF_TOKEN_RE.sub("", s, count=1).strip()


def _extract_counterparty_alliance(
    desc: str, direction: Optional[str] = None
) -> Optional[Tuple[str, str]]:
    """Alliance-specific extraction. Returns None if no pattern matches
    (caller falls through to the generic extractor).

    v3.3.1: pre-strip generic voucher codes (PV/YN/... etc.) before pattern
    matching, so they can't contaminate extracted counterparty names.

    Sprint 6 #11a: `direction` ("CR" | "DR") — passed by caller so genuinely
    nameless rows bucket into rail-agnostic UNNAMED ALLIANCE TRANSFER (CR|DR)
    rather than rail-named labels. Defaults to "CR" for back-compat.
    """
    desc = _strip_voucher_codes(desc)
    u = desc.upper()

    # Sprint 6 #11a — direction-aware UNNAMED label. Replaces the 7 rail-
    # named UNNAMEDs (INSTANT TRANSFER, CR ADVICE, IB2G FUND TRANSFER, IB2G
    # DEBIT, FPX PAYMENT, RENTAS CREDIT, DD CASA) with one rail-agnostic,
    # direction-aware label. Rail mechanism is implementation detail
    # irrelevant to credit underwriting.
    direction_norm = direction if direction in ("CR", "DR") else "CR"
    _unnamed_alliance = f"UNNAMED ALLIANCE TRANSFER ({direction_norm})"

    # AB0 — Bare "DuitNow CR Trf CA" with no entity body (optionally with a
    # lone RPP ref). No counterparty to extract; bail to UNIDENTIFIED so the
    # generic CP3 path doesn't scrape "TRF CA" or the raw RPP ref as a name.
    if re.match(r"^DUITNOW\s+CR\s+TRF\s+CA(?:\s+RPP\d+)?\s*$", u):
        return "UNIDENTIFIED", "pattern"

    # Sprint 6 #11a — 7 rail-named buckets renamed to rail-agnostic
    # UNNAMED ALLIANCE TRANSFER (CR|DR). Regexes preserved verbatim; only
    # the return label changes. Each AB-BARE sits BEFORE its AB-full
    # counterpart; the full pattern still runs when the description has
    # extractable body text.
    #
    # NOTE: an entity-extractor pre-pass was attempted here (catching CR
    # ADVICE - IBG <entity-with-legal-suffix>) and reverted. It intercepted
    # rows that the CP3 generic extractor was already cleaning correctly
    # (AIA / SIRIM BERHAD / HEALTH CONNECT / FOMEMA / PM CARE / COMPUMED),
    # and re-routed them through Alliance helpers that are LESS aggressive
    # at leading-ref / multi-token trailing-ref strip. Net effect was -188
    # labels removed (good CP3 forms) + 206 worse labels added (with
    # leading invoice/policy refs re-glued on). Keep CP3 as the path for
    # CR ADVICE rows with body.
    if re.match(r"^INSTANT\s+TRANSFER(?:\s+AOBIFT\d+)?\s*$", u):
        return _unnamed_alliance, "pattern"
    if re.match(r"^CR\s+ADVICE\s*-\s*IBG(?:\s+[A-Z0-9/-]{2,}){0,3}\s*$", u):
        return _unnamed_alliance, "pattern"
    if re.match(r"^IB2G\s+FND\s+TRF\s+CA\s*-\s*CA(?:\s+AOBFTR\d+)?\s*$", u):
        return _unnamed_alliance, "pattern"
    if re.match(r"^IB2G\s+IBG\s+DR\s+CA(?:\s+AOBIBG\d+)?\s*$", u):
        return _unnamed_alliance, "pattern"
    if re.match(r"^FPX\s+ABB\s+AS\s+BYR\s*\(CA\)(?:\s+AOBB2B\d+)?\s*$", u):
        return _unnamed_alliance, "pattern"
    if re.match(r"^RENTAS\s+CA\s+CREDIT\s*$", u):
        return _unnamed_alliance, "pattern"
    if re.match(r"^DD\s+CASA\s+PYMT\s*$", u):
        return _unnamed_alliance, "pattern"

    # AB3 — CA IMPORT DR {ref} — trade finance, counterparty not extractable.
    if re.match(r"^CA\s+IMPORT\s+DR\b", u):
        return "Trade Collections (CA Import)", "pattern"

    # AB9 — ACH INCLEARING-CHEQUE {cheque_no} — cheque clearing OUT (debit)
    if re.match(r"^ACH\s+INCLEARING[-\s]CHEQUE\b", u):
        return "Unidentified (Cheque Issue)", "pattern"

    # AB8 — LOCAL CHQ DEP/MISC ... — cheque deposit IN (credit)
    if re.match(r"^LOCAL\s+CHQ\s+DEP", u):
        return "Unidentified (Cheque Deposit)", "pattern"

    # AB10 — Bank charges & fees (order matters: more specific first)
    if re.search(r"\bODP\s+INT\b|\bCLF\s+PFT\b", u):
        return "Overdraft Interest / Cashline Profit", "special"
    if re.search(r"CA\s+DR\s+CHQ\s+PRO\s+FEE", u):
        return "BANK FEES", "special"
    if re.search(r"STAMP\s+DUTY\s+ON\s+CHEQUE\s+BOOK", u):
        return "BANK FEES", "special"
    if re.search(r"COMMITMENT\s+FEE", u):
        return "BANK FEES", "special"
    if re.search(r"LA-SJPP\s+ANN\s+GTEE\s+FEE", u):
        return "BANK FEES", "special"
    if re.search(r"CA\s+MISC\s+(?:DR|CR)\b.*(?:FORCED\s+POST|REPRESENTMENT|AUDCHG|REPR\s+CHQ|RVSL\s+REPCHQ)", u):
        return "BANK FEES", "special"
    if re.search(r"MISCELLANEOUS\s+DEBIT\b.*INS\s+PREMIUM", u):
        return "BANK FEES", "special"

    # AB6b — IB2G Dr CA Cr LN → loan drawdown
    if re.match(r"^IB2G\s+DR\s+CA\s+CR\s+LN\b", u):
        return "Loan Account", "special"

    # AB6d — IB2G Cr Card Pymt CA → credit card payment
    if re.match(r"^IB2G\s+CR\s+CARD\s+PYMT\s+CA\b", u):
        return "Credit Card Payment", "special"

    # AB6e — IB2G BLKTRF DR CA(M) AOBBY{ref} ... → Alliance bulk payroll.
    # v3.3.1: relaxed from requiring SALARY keyword to also accepting a
    # month/period suffix (e.g. "MARCH 2025" or "GAJI MAC"), since IB2G BLKTRF
    # DR CA is ALWAYS a bulk transfer and the month reference confirms payroll
    # timing. Fixes the KDYN March 2025 case where the description was just
    # "IB2G BLKTRF Dr CA(m) AOBBY25032025025516 MARCH 2025" with no SALARY keyword.
    if re.match(r"^IB2G\s+BLKTRF\s+DR\s+CA", u):
        _has_salary_kw = re.search(r"\bSALARY\b|\bGAJI\b|\bPAYROLL\b|\bWAGES\b", u)
        _has_month_kw = re.search(
            r"\b(?:JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH|ET|AC)?|APR(?:IL)?|MAY|MEI|"
            r"JUN(?:E)?|JUL(?:Y|AI)?|AUG(?:UST|OS)?|OGOS|SEP(?:T|TEMBER)?|OCT(?:OBER)?|"
            r"OKT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?|DIS(?:EMBER)?)\b",
            u,
        )
        if _has_salary_kw or _has_month_kw:
            return "BULK SALARY", "special"

    # AB2b — Instant Transfer ... BACK PAY SALARY ... → single-person back-pay salary
    if re.match(r"^INSTANT\s+TRANSFER\b", u) and re.search(r"\bBACK\s+PAY\s+SALARY\b", u):
        return "BULK SALARY", "special"

    # Helper: take text after the given prefix, dedupe halves, extract tail entity.
    def _tail_entity(body: str) -> str:
        body = _ab_dedupe_halves(body)
        body = _ab_strip_month_refs(body)
        # Strip known reference prefixes leaving uppercase entity
        body = _ab_strip_trailing_refs(body)
        return _ab_fix_truncation(body)

    # AB1 — DuitNow CR Trf CA RPP{ref} {body}
    m = re.match(r"^DUITNOW\s+CR\s+TRF\s+CA\s+RPP\d+\s+(.+)$", u)
    if m:
        body = m.group(1).strip()
        # Own-party transfer: "FUND TRF" or "...TO AB BESTLITE..."
        if "FUND TRF" in body and "BESTLITE" in body:
            return "BESTLITE ELECTRICAL (own-party)", "pattern"
        body = _ab_dedupe_halves(body)
        body = _ab_strip_month_refs(body)
        body = _ab_strip_leading_purpose(body)
        # Drop intermediary bank mentions
        body = re.sub(
            rf"\b(?:{_CP_BANK_SUFFIX})\s+BANK\s+BERHAD\b", "", body, flags=re.IGNORECASE
        ).strip()
        body = re.sub(r"\bD\d{5,}\b", "", body).strip()  # drop reference codes like D886666
        body = _ab_strip_trailing_refs(body)
        # Prefer the alpha tail (entity typically at end)
        tail = _tail_alpha_run(body.split(), min_len=2)
        name = _ab_fix_truncation(tail or body)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # AB2 — Instant Transfer AOBIFT{datetime} {body}
    m = re.match(r"^INSTANT\s+TRANSFER\s+AOBIFT\d+\s+(.+)$", u)
    if m:
        body = m.group(1).strip()
        body_dedup = _ab_dedupe_halves(body)
        # Own-party: "Transfer From ABMB TO {BANK} BESTLITE ELECTRICAL"
        if re.search(r"TRANSFER\s+FROM\s+ABMB\s+TO?\s+\w+\s+BESTLITE", body):
            return "BESTLITE ELECTRICAL (own-party)", "pattern"
        # Strip payer-ref month prefixes (generic: <WORD>-MMM'YY[-MMM'YY]*)
        body_clean = _ab_strip_month_refs(body_dedup)
        # Strip "Transfer From ABMB" (generic incoming marker)
        body_clean = re.sub(
            r"\bTRANSFER\s+FROM\s+ABMB(?:\s+TO?\s+[A-Z]{2,5})?\b",
            "",
            body_clean,
            flags=re.IGNORECASE,
        ).strip()
        body_clean = _ab_strip_leading_purpose(body_clean)
        body_clean = _ab_strip_trailing_refs(body_clean)
        name = _ab_fix_truncation(body_clean)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # AB4 — CR ADVICE - IBG {body}
    m = re.match(r"^CR\s+ADVICE\s*-\s*IBG\s+(.+)$", u)
    if m:
        body = m.group(1).strip()
        if "FUND TRF" in body and "BESTLITE" in body:
            return "BESTLITE ELECTRICAL (own-party)", "pattern"
        # v3.3.1: drop leading numeric OR alphanumeric reference tokens (e.g.
        # "I202507010110653", "IF01200011839125NPYT" — 1-4 letters + 5+ digits
        # + optional alphanumeric tail). Previously only pure-numeric tokens
        # were stripped, so KDYN leaked "I202507010110653 I2 TNB HQ PAYMENT
        # ACCOU" as a counterparty name. Also drop short 1-3 char mixed
        # letter+digit fragments that follow (like "I2") since those are parser
        # leftovers from the same reference system — real entity names never
        # start with a mixed short token.
        toks = body.split()
        while toks and (
            re.fullmatch(r"[\d/-]{3,}", toks[0])
            or re.fullmatch(r"[A-Z]{1,4}\d{5,}[A-Z0-9]*", toks[0])
        ):
            toks.pop(0)
        # Post-ref fragment cleanup — a short mixed letter+digit token after a
        # long ref is almost certainly a fragment of the same ref system.
        while toks and len(toks[0]) <= 3 and re.fullmatch(
            r"[A-Z]+\d+[A-Z]*|\d+[A-Z]+", toks[0]
        ):
            toks.pop(0)
        body = " ".join(toks)
        body = _ab_strip_leading_purpose(body)
        body = _ab_strip_trailing_refs(body)
        name = _ab_fix_truncation(body)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # AB5 — RENTAS CA CREDIT {bank_ref} {entity} ...
    m = re.match(r"^RENTAS\s+CA\s+CREDIT\s+(.+)$", u)
    if m:
        body = m.group(1).strip()
        toks = body.split()
        # Skip the first alphanumeric reference token (typically 10+ chars with digits)
        if toks and re.fullmatch(r"[A-Z0-9]{8,}", toks[0]) and any(c.isdigit() for c in toks[0]):
            toks = toks[1:]
        body = " ".join(toks)
        body = _ab_strip_trailing_refs(body)
        name = _ab_fix_truncation(body)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # AB6a — IB2G FND TRF CA - CA AOBFTR{ref} {body}
    m = re.match(r"^IB2G\s+FND\s+TRF\s+CA\s*-\s*CA\s+AOBFTR\d+\s+(.+)$", u)
    if m:
        body = m.group(1).strip()
        body = _ab_dedupe_halves(body)
        body = _ab_strip_trailing_refs(body)
        tail = _tail_alpha_run(body.split(), min_len=2)
        name = _ab_fix_truncation(tail or body)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # AB6c — IB2G IBG Dr CA AOBIBG{ref} {body}
    m = re.match(r"^IB2G\s+IBG\s+DR\s+CA\s+AOBIBG\d+\s+(.+)$", u)
    if m:
        body = m.group(1).strip()
        body = _ab_dedupe_halves(body)
        body = _ab_strip_month_refs(body)
        body = _ab_strip_trailing_refs(body)
        name = _ab_fix_truncation(body)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # AB7 — FPX ABB as Byr (CA) AOBB2B{ref} {maybe ref} {entity}
    m = re.match(r"^FPX\s+ABB\s+AS\s+BYR\s*\(CA\)\s+AOBB2B\d+\s+(.+)$", u)
    if m:
        body = m.group(1).strip()
        # Drop a second numeric reference if present
        body = re.sub(r"^\d+\s+", "", body).strip()
        body = _ab_dedupe_halves(body)
        body = _ab_strip_trailing_refs(body)
        # Known FPX billers
        if "LEMBAGA HASIL" in body:
            return "LHDN", "pattern"
        if "KUMPULAN WANG SIMPAN" in body:
            return "KWSP", "pattern"
        if "PERTUBUHAN KESELAMAT" in body:
            return "SOCSO", "pattern"
        name = _ab_fix_truncation(body)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # AB11 — DD CASA PYMT {entity} ...
    m = re.match(r"^DD\s+CASA\s+PYMT\s+(.+)$", u)
    if m:
        body = m.group(1).strip()
        body = _ab_dedupe_halves(body)
        body = _ab_strip_trailing_refs(body)
        name = _ab_fix_truncation(body)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # Sprint 6 #8 — AB12 NBPS IBG Dr CA AOBJOM{date} {ref} {biller_id} {entity}
    # NBPS rows fell through every existing handler and CP3 generic, leaving
    # 137 corpus rows in UNIDENTIFIED. Shape: fixed prefix + 8-digit date,
    # C-style ref token, 9-12 digit biller ID, then entity tokens + trailing
    # refs. Strip the leading biller ID then reuse the standard Alliance
    # body-cleanup pipeline. Empty body falls back to the rail-agnostic
    # UNNAMED ALLIANCE TRANSFER (DR|CR) per #11a convention.
    m = re.match(r"^NBPS\s+IBG\s+DR\s+CA\s+AOBJOM\d+\s+\S+\s+(.+)$", u)
    if m:
        body = m.group(1).strip()
        toks = body.split()
        if toks and re.fullmatch(r"\d+", toks[0]):
            toks.pop(0)
        body = " ".join(toks)
        body = _ab_dedupe_halves(body)
        body = _ab_strip_trailing_refs(body)
        name = _ab_fix_truncation(body)
        if name:
            return name, "pattern"
        return _unnamed_alliance, "pattern"

    return None


# Sprint 7 #10 (V3-A) — Bank Islam (BIMB) tail-entity extractor.
# Person-name entities have no corporate suffix anchor, so _br_extract_entity
# bails to the first token (MOHD / SITI / NUR …). Take leading uppercase
# alpha tokens until a ref-shaped token (XRPP001), pure digit, lowercase
# token (purpose), or (cid:NNN) glyph.
_BIMB_REF_TOKEN_RE = re.compile(r"^[A-Z]+\d+$")
_BIMB_CID_TOKEN_RE = re.compile(r"^\(cid:\d+\)$", re.IGNORECASE)
_BIMB_ALPHA_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9.&\-/]*\.?$")

# Sprint 7 #11 — generic English purpose words that appear in BIMB format1
# `payment_details` for own-party adjustments (CA DR&CR ADVICE: 'REFUND INTO
# CUSTOMER ACCOUNT', 'FUND TRANSFER', 'TRF FUND', 'TRANSFE R FUND'). When
# the entity extractor's first token is one of these, the tail is purpose
# text, not a counterparty. Truncated forms (TRANSFE, TRF) reflect the 20-
# char column-width clipping seen in the actual `payment_details` strings.
_BIMB_PURPOSE_STOP_TOKENS = {
    "REFUND", "FUND", "TRF", "TRANSFE", "TRANSFER", "INTO", "FROM",
    "ADVICE", "INWARD", "OUTWARD", "RETURN", "REVERSAL", "DEBIT",
    "CREDIT", "ACCOUNT", "INTEREST", "PAYMENT", "PYMT",
}


def _bimb_extract_tail_entity(tail: str) -> str:
    out: List[str] = []
    for tok in tail.split():
        if _BIMB_CID_TOKEN_RE.match(tok):
            # Skip glyph noise — don't include but don't terminate (cid:236
            # often appears mid-name from non-Latin character rendering)
            continue
        if _BIMB_REF_TOKEN_RE.match(tok):
            break
        if tok.isdigit():
            break
        clean = tok.rstrip(",.;:")
        if not clean:
            break
        # Require originally-uppercase token — purpose words like 'fees',
        # '1apr' must terminate extraction, not feed into the entity.
        if not _BIMB_ALPHA_TOKEN_RE.match(clean):
            break
        out.append(clean)
    return " ".join(out).strip()


def _extract_counterparty_uob(
    desc: str,
    direction: Optional[str] = None,
    amount: Optional[float] = None,
    own_party: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """BUG-001 (2026-05-05) — UOB-specific counterparty extraction.

    Returns None if no rule matches; caller falls through to the global
    special-bucket / raw fallback.

    Implements 13 family rules from the parser-uob-2026-05-05 handoff. Rules
    apply in priority order, first match wins. Specific entity-bearing
    prefixes run first; the catch-all bank-anchor + pipe rule (UOB-A) runs
    near the end; bank-fee bucket (UOB-F) is the final pattern. The legacy
    Sprint 6 #10 cheque rule (UOB-K) is preserved as a final guard so
    existing UNNAMED UOB TRANSFER (DR) routing for Chq Wdl / bare Cheque
    rows is unchanged.
    """
    if not desc:
        return None
    # AM/PM time-stamp leak cleanup (BUG-003 in handoff): strip standalone
    # AM|PM tokens that appear between two uppercase tokens (the time field
    # is preserved separately on the transaction row, so this is duplicate).
    cleaned = _UOB_AMPM_LEAK_RE.sub(" ", desc).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    u = cleaned.upper()
    direction_norm = direction or "DR"

    # ── UOB-G — XOR outward TT to foreign supplier (typically CN) ──────────
    # 'XOR<digits>C01 [AM|PM] EB <SUPPLIER NAME> <Misc|DuitNow|...>'
    m = re.match(
        r"^XOR\d+C01\s+(?:(?:AM|PM)\s+)?EB\s+(.+?)\s+"
        r"(?:Misc|MISC|DuitNow|DUITNOW|RENTAS|MYISTF|Fund|FUND|"
        r"\d+\s+of\s+\d+|\d{12,})",
        cleaned,
        flags=re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip().rstrip(".,;:")
        if name and len(name) >= 4:
            return name.upper(), "pattern"
    # Unterminated fallback: 'XOR... EB <NAME>' with no trailing keyword.
    m = re.match(
        r"^XOR\d+C01\s+(?:(?:AM|PM)\s+)?EB\s+([A-Z][A-Z &().,'\-]{3,})\s*$",
        cleaned,
    )
    if m:
        return m.group(1).strip().rstrip(".,;:").upper(), "pattern"

    # ── UOB-B — ABMB GOODWILL factoring (canonical regardless of MSF-GW####)
    # The |MSF-GW####| suffix is a per-invoice factoring drawdown ID and
    # MUST NOT disambiguate counterparty. Single biggest-impact fix in the
    # handoff (~75% of CR volume on BORE collapses from 79 fragments to 1).
    if u.startswith("ABMB GOODWILL"):
        return "ABMB GOODWILL EVEREST SDN. BHD.", "pattern"

    # ── UOB-C — LMS sweep (own-party internal) ────────────────────────────
    # SWPSEC = sweep TO LMS, SWPBCK = sweep BACK from LMS, POSINT = LMS
    # interest posting. Treat as own-party so M7 stamping marks these
    # is_related_party=true even when the LMS account is not in the bundle.
    if re.search(r"\b(SWPSEC|SWPBCK|POSINT)\b", u):
        m_acct = re.search(r"\b(2403\d{6,8})\b", u)
        acct = m_acct.group(1) if m_acct else None
        holder_tokens = _own_party_core_tokens(own_party) if own_party else []
        holder_label = " ".join(holder_tokens) if holder_tokens else None
        if holder_label and acct:
            return f"{holder_label} - {acct} (LMS SWEEP)", "pattern"
        if holder_label:
            return f"{holder_label} (LMS SWEEP)", "pattern"
        return "UOB LMS SWEEP", "special"

    # ── UOB-D — Loan drawdown (must come BEFORE UOB-I) ────────────────────
    if re.match(r"^Trf\.\s*Wd\.\s*Loans\b", cleaned, flags=re.IGNORECASE):
        return "UOB LOAN FACILITY (DRAWDOWN)", "pattern"

    # ── UOB-I — Loan service / interest posting ───────────────────────────
    if re.match(r"^Loan\s", cleaned, flags=re.IGNORECASE):
        return "UOB LOAN FACILITY (SERVICE)", "pattern"

    # ── UOB-N — IBG Bulk Payroll (route to existing BULK SALARY bucket) ──
    if re.match(r"^UIEI\d+\s+IBG\s+BULK\s+PAYROLL", cleaned, flags=re.IGNORECASE):
        return "BULK SALARY", "special"

    # ── UOB-M — Trade Bill Transfer (UOB trade finance facility) ─────────
    if re.match(
        r"^\d+(?:BA|IF)\d+\s+(?:AM|PM)\s+Trade\s+Bill\s+Transfer\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return "UOB TRADE BILL FACILITY", "pattern"

    # ── UOB-O — Cheque Deposit (route to existing Unidentified bucket) ──
    if re.match(r"^Cheque\s+Deposit\b", cleaned, flags=re.IGNORECASE):
        return "Unidentified (Cheque)", "special"

    # ── UOB-E — Own-party transfer between holder's accounts ──────────────
    # Two sub-variants:
    #   E1: 'Tt BOREINTERNATIONAL ...' / 'TT BOREINTERNATIONAL ...' — the
    #       Tt prefix is the standard UOB shape.
    #   E2: '<word> <COMPACT_HOLDER> ...' — UOB renders some own-party
    #       transfers with arbitrary leading words ('TT Back', 'Com',
    #       'Fees', 'Boost') followed by the holder's name compacted to
    #       a single all-caps token (e.g. BOREINTERNATIONAL,
    #       UPELLCORPORATION). When own_party is known, build the compact
    #       form and search for it as a whole-word token.
    holder_tokens = _own_party_core_tokens(own_party) if own_party else []
    holder_label = " ".join(holder_tokens) if holder_tokens else None
    if re.match(r"^[Tt][Tt]\s+[A-Z][A-Z]+", cleaned):
        if holder_label:
            return f"{holder_label} (OWN-PARTY)", "pattern"
        return "UOB INTERNAL TRANSFER (OWN-PARTY)", "special"
    if holder_tokens:
        compact = "".join(holder_tokens)
        if len(compact) >= 6 and re.search(rf"\b{re.escape(compact)}\b", u):
            return f"{holder_label} (OWN-PARTY)", "pattern"

    # ── UOB-J — Long-ref fintech (e.g. Boost Bank Berhad via DuitNow) ────
    # '<10+digit-ref> AM|PM <id> <BANK NAME> <purpose>'
    m = re.match(
        r"^\d{10,}\s+(?:AM|PM)\s+\d+\s+"
        r"([A-Z][A-Z &().,'\-]*?(?:BANK|BERHAD|BHD)[A-Z &().,'\-]*?)"
        r"(?:\s+(?:Misc|MISC|DuitNow|DUITNOW|Fund|FUND|RENTAS|ODS)|$)",
        cleaned,
    )
    if m:
        return m.group(1).strip().rstrip(".,;:").upper(), "pattern"

    # ── UOB-H — Dash UOB continuation (multi-line layout artefact) ───────
    # Inward IBG/DuitNow CR rows occasionally render with a leading dash +
    # 'UOB' and a name continuation: '- UOB <NAME> <purpose>'.
    m = re.match(
        r"^-\s*UOB\s+(.+?)"
        r"(?:\s+(?:DuitNow|DUITNOW|Fund|FUND|Misc|MISC|RENTAS|ODS)|$)",
        cleaned,
    )
    if m:
        name = m.group(1).strip().rstrip(".,;:")
        if name and len(name) >= 3:
            return name.upper(), "pattern"

    # ── UOB-A — Pipe + bank-anchor (catch-all, highest volume) ───────────
    # Two sub-variants:
    #   ||  (BORE-style)            — middle is empty, name is in BEFORE
    #   |   (Upell-style, single)   — middle may be ref OR a fuller name
    # Strip MSF-GW#### refs first so they don't pollute middle.
    if "|" in cleaned:
        cleaned_no_msf = _UOB_MSF_REF_RE.sub("||", cleaned)
        if "||" in cleaned_no_msf:
            before = cleaned_no_msf.split("||", 1)[0]
            middle = ""
        else:
            parts = cleaned_no_msf.split("|", 2)
            before = parts[0]
            middle = parts[1] if len(parts) > 1 else ""

        # Try MIDDLE first if it looks like a real entity name.
        # Accept only when (a) middle has a legal-suffix marker (SDN/BHD/
        # BERHAD/etc.) OR (b) middle is all-caps with ≥3 tokens. Reject
        # ref-shaped middles (leading letter+digit, leading digit, leading
        # purpose word) — those would shadow the cleaner BEFORE-anchor name.
        candidate: Optional[str] = None
        if middle:
            mid_clean = middle.strip()
            mid_u = mid_clean.upper()
            looks_like_ref = bool(
                re.match(r"^[A-Z]?\d", mid_clean)
                or re.match(
                    r"^(?:MSF-GW|DR|CR|DUITNOW|PAYMENT|FUND|MISC|CERT|GAJI|"
                    r"SALARY|DIRECT\s+DEBIT|GEAR|PURCHASE)\b",
                    mid_u,
                )
                or len(mid_clean) < 6
            )
            has_legal_suffix = bool(
                re.search(
                    r"\b(SDN\.?\s*BHD\.?|BERHAD|ENTERPRISE|TRADING|MARKETING|"
                    r"RESOURCES|INDUSTRIES|HOLDINGS|CORPORATION|CORP|"
                    r"SUPPLIES|SERVICES|VENTURES|SOLUTIONS)\b",
                    mid_u,
                )
            )
            all_caps = bool(re.match(r"^[A-Z][A-Z0-9 &().,'\-/]*$", mid_clean))
            if not looks_like_ref and (
                has_legal_suffix or (all_caps and len(mid_clean.split()) >= 3)
            ):
                candidate = mid_clean

        # Otherwise extract from BEFORE-segment after the LAST bank anchor.
        if not candidate:
            anchors = list(_UOB_BANK_ANCHOR_RE.finditer(before))
            if anchors:
                last = anchors[-1]
                candidate = before[last.end():].strip(" -|")

        if candidate:
            candidate = _UOB_AMPM_LEAK_RE.sub(" ", candidate).strip(" -|")
            candidate = re.sub(r"\s+", " ", candidate).rstrip(".,;:")
            if candidate and len(candidate) >= 3 and re.search(r"[A-Z]", candidate):
                return candidate.upper(), "pattern"

    # ── UOB-A2 — Bank-anchor + entity, NO pipe (Juta/HSBC trade rows) ────
    # 'Outward ACH MBB EASTERN STEEL SDN. BHD.' / 'PRIVATE TRANSACTION AM
    # MBB UMECH CONSTRUCTION SDN. BHD.' / 'C/MBBIS-... MBB WCT CONSTRUCTION
    # SDN. BHD.'. Anchor sits mid-string; the canonical entity follows it.
    # Tighter than A: require ≥2 uppercase tokens after the anchor and a
    # legal-suffix marker (SDN/BHD/BERHAD/etc.) so we don't pick up purpose
    # phrases like 'MBB transfer fees'.
    if "|" not in cleaned:
        anchors = list(_UOB_BANK_ANCHOR_RE.finditer(cleaned))
        if anchors:
            last = anchors[-1]
            tail = cleaned[last.end():].strip(" -")
            tail_u = tail.upper()
            has_legal_suffix = bool(
                re.search(
                    r"\b(SDN\.?\s*BHD\.?|BERHAD|ENTERPRISE|TRADING|MARKETING|"
                    r"RESOURCES|INDUSTRIES|HOLDINGS|CORPORATION|CORP|"
                    r"SUPPLIES|SERVICES|VENTURES|SOLUTIONS)\b",
                    tail_u,
                )
            )
            tail_caps = bool(
                re.match(r"^[A-Z][A-Z0-9 &().,'\-/]{3,}$", tail.split(" DR")[0].split(" CR")[0].strip())
            )
            tokens_count = len(tail.split())
            if has_legal_suffix and tokens_count >= 2:
                # Trim trailing direction markers and noise tokens
                t = re.sub(r"\s+(?:DR|CR|\d+\s+of\s+\d+).*$", "", tail).strip()
                t = _UOB_AMPM_LEAK_RE.sub(" ", t).rstrip(".,;:")
                if t and len(t) >= 4:
                    return t.upper(), "pattern"

    # ── UOB-F — Bank charges (Service Charge / OD Int / OD Commitment / ──
    # Chq Book Courier / SVC-REQ / Excess Interest). MUST come AFTER UOB-A
    # / UOB-A2 so a real entity isn't mis-routed to BANK FEES.
    if (
        re.search(r"\bSERVICE\s+CHARGE", u)
        or re.search(r"\bOD\s+INT\s+CHARGE", u)
        or re.search(r"\bOD\s+COMMITMENT\s+FEE", u)
        or re.search(r"\bCHQ\s+BOOK\s+COURIER\s+FEE", u)
        or re.search(r"\bSVC[-\s]REQ\s+OF\s+STATEMENT", u)
        or re.search(r"\bEXCESS\s+INTEREST", u)
    ):
        return "BANK FEES", "special"

    # ── UOB-K — Cheque withdrawal / bare Cheque <num> (preserved verbatim
    # from Sprint 6 #10; previously at line ~3155 of this file). DR-only
    # gate kept for cross-bank safety. CR cheque rows stay on existing path.
    if direction_norm == "DR" and (
        re.match(r"^CHQ\s+WDL\b", u) or re.match(r"^CHEQUE\s+\d", u)
    ):
        return "UNNAMED UOB TRANSFER (DR)", "special"

    return None


def _extract_counterparty(
    description: str,
    bank: Optional[str] = None,
    amount: Optional[float] = None,
    direction: Optional[str] = None,
    own_party: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (counterparty_name, extraction_method).

    method ∈ {"special", "pattern", "raw"}.

    `amount` is the absolute transaction value (debit OR credit, whichever is
    non-zero). Optional — passed by build_counterparty_ledger so handlers that
    need amount-aware sanity checks (e.g. routing fee descriptions to BANK
    FEES only when the amount is fee-shaped) can use it. The check mirrors
    the existing C24 v3.2 rule: 'OTHER TRANSFER FEE if amount <= RM1.00 →
    C24, full stop'. We use a slightly looser threshold (RM 100) for RHB
    Reflex SC rows because they're consistently RM 0.50 in the corpus but
    leave headroom for legitimate fee-shaped charges.
    """
    if not description:
        return "UNIDENTIFIED", "raw"
    desc = description.strip()
    u = desc.upper()

    # Alliance Bank: try bank-specific patterns first
    if bank and "ALLIANCE" in bank.upper():
        ab = _extract_counterparty_alliance(desc, direction)
        if ab is not None:
            return ab

    # ── Sprint 7 #10 (V3-A) — Bank Islam (BIMB) entity extraction ──────────
    # Bank Islam descriptions are multi-line in the source PDF. The parser
    # (Sprint 7 #10 fix in bank_islam.py) now appends continuation lines to
    # the description, producing rows like:
    #   '9871 RTP REDIRECT CT CR ROHAZEILLAH ENTERPRISE MYTUTOR ACADEMY SDN.
    #    BHD. 16832 XRPP001'   (Mytutor — RTP entity transfer)
    #   '3110 SA HSE CHQ DEP - CR .50 02011 MOHD RIZAL BIN OSMAN 0179044'
    #    (KMZ — house-cheque deposit from a person)
    #   '0153 eSPICK CHQ PRCSG FEE .50 ...'  (cheque processing fee)
    #   '9124 CDB CS TO IBFTS3 MYTUTOR ACADEMY SDN BHD fees ...' (own-party
    #    cash-to-internal-account transfer)
    #
    # Strategy: bank-gate; route fee/system opcodes to BANK FEES; for
    # entity-bearing opcodes, strip the 4-digit ref + opcode tokens + .50
    # cents fragment + 5-digit branch ref, then use _br_extract_entity on
    # the remainder. _strip_own_party_tokens downstream removes the
    # statement-holder echo (MYTUTOR ACADEMY / KMZ RESTU) so the real
    # counterparty survives. Bare/refs-only tails → UNNAMED BANK ISLAM
    # TRANSFER (CR|DR). Cross-bank safety: bank-gated; opcode tokens
    # (`eSPICK`, `RTP REDIRECT CT`, `SA HSE CHQ DEP`, `CDB CS TO IBFTS3`)
    # are BIMB-specific in corpus.
    if bank and "BANK ISLAM" in bank.upper():
        direction_norm = direction or "CR"

        # Drop leading 4-digit transaction-type code (always present at the
        # start of every BIMB description per format2 capture).
        body = re.sub(r"^\s*\d{4}\s+", "", desc).strip()

        # B1 — fee opcodes
        if re.match(
            r"^(?:e?SPICK\s+CHQ\s+PRCSG\s+FEE"
            r"|CMS\s+SERVICE\s+CHARGE"
            r"|CMS\s+CASH\s+DEP\s+(?:CHRG|FEE)"
            r"|MYC\s+DD\s+CASA(?:\s+-\s+(?:DR|CR))?)\b",
            body, flags=re.IGNORECASE,
        ):
            return "BANK FEES", "special"

        # B2 — profit paid (CR side: profit on hibah / interest analogue)
        if re.match(r"^PROFIT\s+PAID\b", body, flags=re.IGNORECASE):
            return "FD/INTEREST", "special"
        # Profit charged (DR side: financing profit) → bank fees per project convention
        if re.match(r"^PROFIT\s+CHARGED\b", body, flags=re.IGNORECASE):
            return "BANK FEES", "special"

        # B3 — entity-bearing opcodes. Tail extraction strategy:
        #   strip opcode tokens, then strip the per-row own-party echo
        #   (e.g. ' MYTUTOR ACADEMY SDN BHD' continuation echo, present on
        #   nearly every BIMB row), then strip leading .50/cents fragments
        #   and 5+digit branch refs. Pass remainder to _br_extract_entity.
        #   Own-party strip BEFORE _br_extract_entity matters: with the
        #   echo in place, _br_extract_entity's 6-token lookahead reaches
        #   only the echo's 'SDN' (not 'BHD'), and bails to a single-
        #   token result like 'MOHD' — losing the rest of the person's
        #   name.
        bimb_entity_opcodes = re.compile(
            r"^(?:"
            r"(?:SA\s+)?HSE\s+CHQ\s+DEP\s*-\s*CR(?:/DR)?"
            r"|RTP\s+REDIRECT\s+CT\s+(?:CR|DR)"
            r"|RTP\s+IBFT\s+(?:CR|DR)"
            r"|INW\s+DuitNow\s+Transfer"
            r"|IBG\s+TRANSFER\s+TO\s+CA"
            r"|FN\s+AUTO\s+RPY\s*-\s*HC"
            r"|CA\s+DR&CR\s+ADVICE"
            r"|CDB\s+CS\s+TO\s+IBFTS3"
            r"|CDB\s+JOMPAY\s+(?:OFF-US|ON-US)"
            r"|eSPICK\s+INW\s+PT\s+LTD\s+CO"
            r")\s*(.*)$",
            re.IGNORECASE,
        )
        m = bimb_entity_opcodes.match(body)
        if m:
            tail = m.group(1).strip()
            # Strip own-party echo wherever it appears (case-insensitive,
            # tolerant of 'SDN. BHD.' vs 'SDN BHD' spacing). Without this,
            # _br_extract_entity's 6-token lookahead truncates the entity
            # to the first token (MOHD / SITI / NUR …).
            if own_party:
                op_pattern = re.escape(own_party).replace(
                    r"SDN\ BHD", r"SDN\.?\s*BHD\.?"
                )
                tail = re.sub(op_pattern, " ", tail, flags=re.IGNORECASE).strip()
                tail = re.sub(r"\s+", " ", tail)
            # Strip leading cents fragment (.50, 1.50, etc.)
            tail = re.sub(r"^\.?\d+(?:\.\d+)?\s+", "", tail).strip()
            # Strip leading 5+digit branch / system refs
            tail = re.sub(r"^\d{5,}\s+", "", tail).strip()
            # Sprint 7 #12 (V3-A): when own-party strip + ref strip leave an
            # empty tail, the row is a confirmed own-party self-transfer
            # between the statement-holder's accounts (e.g. 'CDB CS TO IBFTS3
            # PRINCIPAL GAS SDN BHD' = own → own internal). Returning the
            # own-party name as the bucket lets C01 (own-party) fire on the
            # row instead of routing to UNNAMED.
            if not tail:
                if own_party:
                    return own_party.upper(), "pattern"
                return f"UNNAMED BANK ISLAM TRANSFER ({direction_norm})", "special"
            entity_norm = _bimb_extract_tail_entity(tail).upper().strip()
            if not entity_norm or not _has_real_word(entity_norm):
                return f"UNNAMED BANK ISLAM TRANSFER ({direction_norm})", "special"
            # Sprint 7 #11 (V3-A): when a BIMB row is an own-party adjustment
            # (e.g. format1 'CA DR&CR ADVICE' with own-party in sndr column +
            # bank-internal purpose in payment_details), the strips leave only
            # purpose words like 'REFUND INTO CUSTOME[R]' or 'FUND TRANSFE[R]'.
            # Same semantic as the empty-tail case above — the row is an own-
            # party adjustment, route to the company-name bucket so C01 fires.
            first_tok = entity_norm.split()[0]
            if first_tok in _BIMB_PURPOSE_STOP_TOKENS:
                if own_party:
                    return own_party.upper(), "pattern"
                return f"UNNAMED BANK ISLAM TRANSFER ({direction_norm})", "special"
            return entity_norm, "pattern"

        # B4 — bare opcode-only system events (no entity expected)
        if re.match(
            r"^(?:BAL\s+B/F|BAYARAN\s+JEMPUTAN|REVERSAL|INWARD\s+RETURN)\b",
            body, flags=re.IGNORECASE,
        ):
            return f"UNNAMED BANK ISLAM TRANSFER ({direction_norm})", "special"

    # ── Sprint 6 #4 — Hong Leong Bank / HLB Islamic entity extraction ──────
    # HLB statements use ~10 distinct prefix formats. ~85% of corpus rows
    # carry an extractable entity name; the remaining ~15% are truly unnamed
    # (Bulk DuitNow aggregate batches, CA IBT internal own-account transfers)
    # and route to UNNAMED HLB TRANSFER (CR|DR) so volume is visible without
    # inventing a fake counterparty. Bank-gated to "Hong Leong" (covers both
    # Hong Leong Bank and Hong Leong Islamic Bank — same description format
    # across both subsidiaries).
    #
    # Audited corpus: 537 rows across MTCE + Detik samples; pre-fix
    # extraction was 0/537 (no HLB handler existed before this commit).
    #
    # Sub-formats handled (named):
    #   A. CIB Instant Transfer at DIO <amount> <inv/ref/purpose> <ENTITY>
    #      <8-digit-date>HLBBMYKL<ref>
    #      Entity = trailing alpha run; PDF column-truncates at ~22 chars.
    #   B. Instant Transfer at KLM <amount> [<balance>] [interbank|ITB TRF
    #      [HLB <bank>]] <ENTITY> <8-digit-date>{ARBK|BMMB|CIBB|MBBE|BIMB}
    #      MYKL<ref>
    #      Entity = trailing alpha run before the bank-id ref. The
    #      `interbank` and `ITB TRF [HLB <bank>]` purpose markers are
    #      stripped before extraction so they don't leak into the bucket.
    #   C. JomPAY Bill Payment at DIO <amount> <bill-ref> C<ref>{Y|N}
    #      <ENTITY> 24IM<ref>
    #      Entity = trailing alpha run after the C{...}{Y|N} confirmation
    #      token. Most billers are utilities (TNB / MAXIS / TM UNIFI) —
    #      Layer 2 AI maps to C24 BILL/UTILITIES via biller-name keyword.
    #      MUST run before the generic JomPAY catch-all at line ~2284
    #      otherwise that handler grabs `JOMPAY BILL` as the biller token.
    #   D. FPX B2B1 <amount> [noise-digits] <ref10> <ENTITY> <ref16>
    #      Entity = name-tokens between the 10-digit and 16-digit refs.
    #      Truncations accepted per Q2 (analyst clue beats UNCATEGORIZED).
    #   E. Fund Transfer at DIO <amount> <invoice/ref/purpose> <ENTITY>
    #      Entity at the very end of the body. No suffix ref code.
    #   F. CIB IBG CA Debit Advice at KLM <amount> <purpose> <ENTITY>
    #      IBGCMP<bank><ref>
    #      Entity = name-tokens between purpose and IBGCMP ref.
    #   G. CA-i Debit Advice - SWIFT <amount> [<balance>] <purpose> OUR
    #      <ENTITY> CPTJ<bank><ref>
    #      Entity = words after `OUR` and before `CPTJ` ref.
    #
    # Truly unnamed (route to UNNAMED HLB TRANSFER — Q3 maps non-cheque
    # nameless transfers to the bank-attributed bucket):
    #   H. Bulk DuitNow <amount> <internal-code> CTHLCF<ref>
    #      Aggregate disbursement batch; description has no entity.
    #   I. CA IBT Debit Advice at SPI <amount>
    #      Inter-account own-bank transfer; no entity present.
    #
    # Cheque (Q3 — cheques stay on existing Unidentified bucket):
    #   J. Inclearing-Cheque <num> <amount>  → Unidentified (Cheque)
    #
    # Bank fees (Q4 — keep current routing intent: fees → BANK FEES). The
    # global BANK FEES regex catches `Cheque Processing Fee` already; the
    # remaining HLB-specific fee shapes need explicit routing here:
    #   - Serv Charge-IBG/TT/Rentas/Misc        (~31 rows)
    #   - Remittance Cable Charge / Commission  (~4 rows)
    #   - Debit Advice - SST                    (~1 row)
    #   - Overdraft/Excess Interest             (~1 row)
    #   - CA Debit Advice <small> CPTJ<ref>     (~5 rows; settlement fees,
    #     paired with parent CA-i SWIFT remittance row)
    #
    # Cross-bank safety: bank-gated to Hong Leong; defence-in-depth via
    # HLB-specific anchor tokens (`at DIO` / `at KLM` / `at SPI` / `IBGCMP`
    # / `CTHLCF` / `HLBBMYKL`) which would not match other banks even
    # without the gate. P7-compliant: bucket name is bank-attributed
    # (UNNAMED HLB TRANSFER), no rail labels in the bucket.
    #
    # Placement note: this block runs BEFORE the global special-bucket /
    # BANK FEES / generic-JomPAY regexes so HLB JomPAY rows route to the
    # actual biller name rather than the catch-all `JOMPAY BILL`. The
    # bank gate ensures non-HLB rows still hit the generic path below.
    if bank and "HONG LEONG" in bank.upper():
        direction_norm = direction or "DR"

        # HLB-specific BANK FEES (Q4). Patterns are HLB-specific in corpus.
        if re.match(
            r"^(?:SERV\s+CHARGE-IBG/TT/RENTAS/MISC"
            r"|REMITTANCE\s+(?:CABLE\s+CHARGE|COMMISSION)"
            r"|DEBIT\s+ADVICE\s*-\s*SST"
            r"|OVERDRAFT/EXCESS\s+INTEREST)\b",
            u,
        ):
            return "BANK FEES", "special"
        # Small CA Debit Advice rows with CPTJ ref are interbank settlement
        # fees paired with parent CA-i SWIFT remittance rows.
        if re.match(r"^CA\s+DEBIT\s+ADVICE\s+\d", u) and "CPTJ" in u:
            return "BANK FEES", "special"
        # CA Debit Advice referencing a cheque number (`#nnnnnn`) is the
        # debit side of an inclearing cheque pair — Q3 cheque routing.
        if re.match(r"^CA\s+DEBIT\s+ADVICE\s+\d", u) and re.search(r"#\d+", u):
            return "Unidentified (Cheque)", "special"
        # 2026-05-08 — CA Debit Advice with explicit 'Fees' marker
        # (e.g. 'CA Debit Advice 30.00 Fees CERTIFIED TRUE COPY').
        if re.match(r"^CA\s+DEBIT\s+ADVICE\s+\d", u) and re.search(r"\bFEES\b", u):
            return "BANK FEES", "special"

        # Sub-format H: Bulk DuitNow (aggregate batch, no entity)
        if re.match(r"^BULK\s+DUITNOW\b", u):
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format I: CA IBT Debit Advice at SPI (own-bank inter-account)
        if re.match(r"^CA\s+IBT\s+DEBIT\s+ADVICE\s+AT\s+SPI\b", u):
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format J: Inclearing-Cheque → Unidentified (Cheque)
        if re.match(r"^INCLEARING-CHEQUE\b", u):
            return "Unidentified (Cheque)", "special"

        # ── 2026-05-08 — Retail/SME (HLConnect) sub-formats ─────────────
        # Sprint 6 #4 was built against the corporate (CIB / HLConnect
        # Bizlink) corpus. Folder 2 (CurrentAcct) is a personal/SME
        # account that emits a different set of opcodes; pre-fix this
        # account ran 86% raw_fallback. Six new sub-formats below cover
        # the observed shapes: HLConnect retail DuitNow, Cr Adv-Interbank
        # GIRO inward, FPX Collection inward, Fund Trf fr CA-Internet,
        # CDM Deposit, Local Cheque RPC, CA Cash Cheque/Withdrawal.

        # Sub-format O: CDM Deposit ... (cash deposit at branch CDM)
        # Global rule catches `CDM <text> CASH DEPOSIT` but bare
        # 'CDM Deposit at <branch>' doesn't match. Route to existing bucket.
        if re.match(r"^CDM\s+DEPOSIT\b", u):
            return "CASH DEPOSIT", "special"

        # Sub-format P: Local Cheque (RPC) <chequeNo> <amount> ...
        # Inward cheque deposit credit; entity unknown by design.
        if re.match(r"^LOCAL\s+CHEQUE\s*\(RPC\)", u):
            return "Unidentified (Cheque)", "special"

        # Sub-format Q: CA Cash Cheque <num> <amount> / CA Cash Withdrawal <amount>
        # / CA IBT Cash Cheque at <branch> <num> <amount>
        # Branch over-the-counter cash withdrawals (corporate + retail variants).
        if re.match(r"^CA\s+(?:IBT\s+)?CASH\s+(?:CHEQUE|WITHDRAWAL)\b", u):
            return "CASH WITHDRAWAL", "special"

        # Sub-format K: HLConnect DuitNow-previously Inst <amount> [<bal>]
        # Fund transfer <purpose> <ENTITY> <date>HLBBMYKL<ref>
        # Retail consumer DuitNow — purpose text is lowercase Malay so we
        # walk back over UPPERCASE tokens only to avoid leaking purpose
        # words into the entity bucket.
        if re.match(r"^HLCONNECT\s+DUITNOW-PREVIOUSLY\s+INST\b", u):
            body = re.sub(
                r"^HLConnect\s+DuitNow-previously\s+Inst\s+[\d,.]+\s*",
                "", desc, flags=re.IGNORECASE,
            )
            body = re.sub(r"\s*\d{8}[A-Z]{4}MYKL\S+.*$", "", body)
            body = re.sub(r"^[\d,.]+\s+", "", body)  # optional balance
            body = re.sub(r"^Fund\s+transfer\s+", "", body, flags=re.IGNORECASE)
            name = _hlb_extract_uppercase_tail(body)
            if name and _has_real_word(name.upper()):
                return name.upper(), "pattern"
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format L: Cr Adv-Interbank GIRO at KLM <amount> [<bal>]
        # [<numeric-ref>] [<JMT-ref>] <ENTITY> [<footer>]
        # Inward interbank GIRO credit (typically government / corporate).
        # Entity is the trailing all-caps run; standard helper works
        # because Cr Adv-Interbank descriptions don't have lowercase
        # purpose text between refs and entity.
        if re.match(r"^CR\s+ADV-INTERBANK\s+GIRO\s+AT\s+KLM\b", u):
            body = re.sub(
                r"^Cr\s+Adv-Interbank\s+GIRO\s+at\s+KLM\s+[\d,.]+\s*",
                "", desc, flags=re.IGNORECASE,
            )
            body = re.sub(r"^[\d,.]+\s+", "", body)  # optional balance
            # Strip leading numeric refs like '202511076802847195' or
            # 'JMT-INV/01-25/600008' — _hlb_extract_entity walks from end
            # so leading refs only matter if they prevent reaching the entity.
            name = _hlb_extract_entity(body)
            if name and _has_real_word(name.upper()):
                return name.upper(), "pattern"
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format M: FPX Collection at KLM <amount> [<bal>] [<biller-id>]
        # <ENTITY> I<10-digit-fpx-ref>
        # Inward FPX collection from a payer (insurance, credit card co, etc).
        if re.match(r"^FPX\s+COLLECTION\s+AT\s+KLM\b", u):
            body = re.sub(
                r"^FPX\s+Collection\s+at\s+KLM\s+[\d,.]+\s*",
                "", desc, flags=re.IGNORECASE,
            )
            # Strip trailing FPX I-ref + footer
            body = re.sub(r"\s*I\d{10,}.*$", "", body)
            name = _hlb_extract_entity(body)
            if name and _has_real_word(name.upper()):
                return name.upper(), "pattern"
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format N: Fund Trf fr CA to CA-Internet <amount> [<bal>]
        # Fund transfer <purpose> <ENTITY>
        # Inward inter-account transfer with named beneficiary on the
        # other side. Purpose text is lowercase Malay → uppercase-tail.
        if re.match(r"^FUND\s+TRF\s+FR\s+CA\s+TO\s+CA-INTERNET\b", u):
            body = re.sub(
                r"^Fund\s+Trf\s+fr\s+CA\s+to\s+CA-Internet\s+[\d,.]+\s*",
                "", desc, flags=re.IGNORECASE,
            )
            body = re.sub(r"^[\d,.]+\s+", "", body)
            body = re.sub(r"^Fund\s+transfer\s+", "", body, flags=re.IGNORECASE)
            name = _hlb_extract_uppercase_tail(body)
            if name and _has_real_word(name.upper()):
                return name.upper(), "pattern"
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format A: CIB Instant Transfer at DIO ...
        if re.match(r"^CIB\s+INSTANT\s+TRANSFER\s+AT\s+DIO\b", u):
            body = re.sub(
                r"^CIB\s+Instant\s+Transfer\s+at\s+DIO\s+[\d,.]+\s*",
                "",
                desc,
                flags=re.IGNORECASE,
            )
            body = re.sub(r"\s*\d{8}HLBBMYKL\S+.*$", "", body)
            name = _hlb_extract_entity(body)
            if name and _has_real_word(name.upper()):
                return name.upper(), "pattern"
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format B: Instant Transfer at KLM ...
        # After stripping the leading "Instant Transfer at KLM <amount>"
        # prefix and the trailing date+bankid ref, an optional balance
        # number and the `interbank` / `ITB TRF [HLB <bank>]` purpose
        # markers may sit between the amount and the entity. Strip them
        # before invoking the trailing-alpha extractor.
        if re.match(r"^INSTANT\s+TRANSFER\s+AT\s+KLM\b", u):
            body = re.sub(
                r"^Instant\s+Transfer\s+at\s+KLM\s+[\d,.]+\s*",
                "",
                desc,
                flags=re.IGNORECASE,
            )
            body = re.sub(r"\s*\d{8}[A-Z]{4}MYKL\S+.*$", "", body)
            # Strip optional leading balance number
            body = re.sub(r"^[\d,.]+\s+", "", body)
            # Strip purpose markers
            body = re.sub(
                r"^(?:interbank|ITB\s+TRF(?:\s+HLB\s+\w+)?)\s+",
                "",
                body,
                flags=re.IGNORECASE,
            )
            name = _hlb_extract_entity(body)
            if name and _has_real_word(name.upper()):
                return name.upper(), "pattern"
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format C: JomPAY Bill Payment at DIO ...
        if re.match(r"^JOMPAY\s+BILL\s+PAYMENT\s+AT\s+DIO\b", u):
            body = re.sub(
                r"^JomPAY\s+Bill\s+Payment\s+at\s+DIO\s+[\d,.]+\s*",
                "",
                desc,
                flags=re.IGNORECASE,
            )
            body = re.sub(r"\s*24IM\d+.*$", "", body)
            # Prefer entity slot AFTER the C{...}{Y|N} confirmation token
            # when present — that's where HLB places the biller name.
            m = re.search(r"\bC[A-Z0-9]{6,16}[YN]\b\s*(.+)$", body)
            tail = m.group(1).strip() if m else body
            name = _hlb_extract_entity(tail)
            if name and _has_real_word(name.upper()):
                return name.upper(), "pattern"
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format D: FPX B2B1 ...
        if re.match(r"^FPX\s+B2B1\b", u):
            body = re.sub(r"^FPX\s+B2B1\s+", "", desc, flags=re.IGNORECASE)
            body = re.sub(r"\s*\d{16,}.*$", "", body)
            name = _hlb_extract_entity(body)
            if name and _has_real_word(name.upper()):
                return name.upper(), "pattern"
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format E: Fund Transfer at DIO ...
        if re.match(r"^FUND\s+TRANSFER\s+AT\s+DIO\b", u):
            body = re.sub(
                r"^Fund\s+Transfer\s+at\s+DIO\s+[\d,.]+\s*",
                "",
                desc,
                flags=re.IGNORECASE,
            )
            name = _hlb_extract_entity(body)
            if name and _has_real_word(name.upper()):
                return name.upper(), "pattern"
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format F: CIB IBG CA Debit Advice at KLM ...
        if re.match(r"^CIB\s+IBG\s+CA\s+DEBIT\s+ADVICE\s+AT\s+KLM\b", u):
            body = re.sub(
                r"^CIB\s+IBG\s+CA\s+Debit\s+Advice\s+at\s+KLM\s+[\d,.]+\s*",
                "",
                desc,
                flags=re.IGNORECASE,
            )
            body = re.sub(r"\s*IBGCMP\S+.*$", "", body)
            name = _hlb_extract_entity(body)
            if name and _has_real_word(name.upper()):
                return name.upper(), "pattern"
            return f"UNNAMED HLB TRANSFER ({direction_norm})", "special"

        # Sub-format G: CA-i Debit Advice - SWIFT ... OUR <ENTITY> CPTJ<ref>
        if re.match(r"^CA-?I\s+DEBIT\s+ADVICE\s*-\s*SWIFT\b", u):
            m = re.search(
                r"\bOUR\s+(.+?)\s+CPTJ\S+",
                desc,
                flags=re.IGNORECASE,
            )
            if m:
                name = _hlb_extract_entity(m.group(1))
                if name and _has_real_word(name.upper()):
                    return name.upper(), "pattern"
            # No OUR marker — fall through to generic raw bucket

    # ── Sprint 6 #9 — Bank Rakyat entity extraction ────────────────────────
    # Felcra-style Bank Rakyat statements emit each transaction as a date-
    # line plus 1–N continuation lines. After bank_rakyat.py was rewritten
    # to capture continuations, descriptions arrive as:
    #   <5-digit-code> <CONCATENATED-OPCODE> <amount> <continuation tokens>
    # where opcode is one of DUITNOWTRANSFER / DUITNOWFEE / TRANSFERFROMSA /
    # IBGCREDIT / CIBDRADVICE / CIBDRADVICE(JomPA…) / CIBDRADVICE(IBG) /
    # CIBSMSFEE / CIBDRCHARGES / CIBCRADVICE / CIBCOMMISSION(IBG) /
    # IBGINWARDRETURN / REVERSALCR / CASHDEPOSIT / CDMCASHDEPOSIT /
    # CREDITPROFIT/HIBAH / REMITTANCECR(-RENT). Entity (when present) is
    # the first non-noise continuation token. Bank-gated to "Bank Rakyat".
    #
    # Routing (per audit + user Q5):
    #   - Bank fees  : CIBSMSFEE / CIBDRCHARGES / DUITNOWFEE / CIBCOMMISSION
    #                  → BANK FEES (global bucket; cross-bank consistency)
    #   - Cash deposits : CASHDEPOSIT / CDMCASHDEPOSIT → CASH DEPOSIT
    #   - Profit/hibah : CREDITPROFIT/HIBAH → FD/INTEREST
    #   - Inward returns / system reversals : IBGINWARDRETURN / REVERSALCR
    #                  → UNNAMED BANK RAKYAT TRANSFER (CR) (Q5 fallback)
    #   - JomPAY billers : CIBDRADVICE(JomPA → biller-name bucket
    #                  (HLB convention; Layer 2 AI classifies via biller)
    #   - Entity-bearing : DUITNOWTRANSFER / CIBDRADVICE / CIBDRADVICE(IBG)
    #                  / CIBCRADVICE / IBGCREDIT / REMITTANCECR / TRANSFERFROMSA
    #                  → first non-noise continuation token via _br_extract_entity
    #   - Truly nameless : UNNAMED BANK RAKYAT TRANSFER (CR|DR)
    #
    # Placement: BEFORE the global special-buckets block so the bank-fee /
    # cash-deposit / FD-interest opcodes are caught here (the global regex
    # doesn't match the concatenated forms like CIBSMSFEE / CIBDRCHARGES).
    # Bank-gating ensures non-Bank-Rakyat rows still hit the generic path.
    if bank and "BANK RAKYAT" in bank.upper():
        direction_norm = direction or "DR"

        # Strip leading Bank Rakyat transaction code. Felcra emits 5-digit
        # codes (94351, 56431); MTCEC emits 2-3 digit codes (65, 516, 691).
        body_after_code = re.sub(r"^\d{1,6}\s+", "", desc)

        # MTCEC-style PDFs emit opcodes with spaces ("DUITNOW TRANSFER");
        # Felcra-style concatenates ("DUITNOWTRANSFER"). Collapse to the
        # concatenated form so the regexes below match either layout.
        body_after_code = _br_normalize_opcode(body_after_code)

        # Bank fees (concatenated forms — global regex doesn't catch these).
        # PROFITCHARGED = OD profit/interest charge on a Bank Rakyat Cashline-i
        # account — Islamic-banking term for interest expense. Routed to BANK
        # FEES as the closest existing bucket (no dedicated finance-cost bucket
        # in the schema; analyst can disambiguate from the flag layer).
        if re.match(
            r"^(?:CIBSMSFEE|CIBDRCHARGES|DUITNOWFEE|CIBCOMMISSION|PROFITCHARGED)\b",
            body_after_code,
            re.IGNORECASE,
        ):
            return "BANK FEES", "special"

        # Cash deposits
        if re.match(
            r"^(?:CDMCASHDEPOSIT|CASHDEPOSIT)\b",
            body_after_code,
            re.IGNORECASE,
        ):
            return "CASH DEPOSIT", "special"

        # Profit / Hibah → FD/INTEREST. The global regex catches
        # `\bHIBAH\b` but the concatenated form `CREDITPROFIT/HIBAH` may
        # match — defensively route here for either form.
        if re.match(
            r"^CREDITPROFIT(?:/HIBAH)?\b",
            body_after_code,
            re.IGNORECASE,
        ):
            return "FD/INTEREST", "special"

        # Inward returns / system reversals / cheque returns → fallback
        # unnamed (Q5). LOCALCHQRTN is the cheque-return shape; route here
        # rather than RETURNED CHEQUE bucket since the global RETURNED
        # CHEQUE keyword regex doesn't match this concatenated form.
        if re.match(
            r"^(?:IBGINWARDRETURN|REVERSALCR|LOCALCHQRTN"
            r"|BILLPAYMENTTOFIN|TRFRSHAREMEMBER|ATMTRANSFERCR)\b",
            body_after_code,
            re.IGNORECASE,
        ):
            return f"UNNAMED BANK RAKYAT TRANSFER ({direction_norm})", "special"

        # Cheque deposits (2-day local cheque)
        if re.match(r"^2DLOCALCHQ\b", body_after_code, re.IGNORECASE):
            return "Unidentified (Cheque)", "special"

        # Cash withdrawal opcode
        if re.match(r"^CASHWITHDRAWAL\b", body_after_code, re.IGNORECASE):
            return "CASH WITHDRAWAL", "special"

        # JomPAY: CIBDRADVICE(JomPA … <amount> <BILLER> …
        if re.match(r"^CIBDRADVICE\(JOMPA", body_after_code, re.IGNORECASE):
            body = re.sub(
                r"^CIBDRADVICE\(JOMPA\S*\s*[\d,.]+\s*",
                "",
                body_after_code,
                flags=re.IGNORECASE,
            )
            name = _br_extract_entity(body)
            if name and _has_real_word(name):
                return name.upper(), "pattern"
            return f"UNNAMED BANK RAKYAT TRANSFER ({direction_norm})", "special"

        # Entity-bearing opcodes. Note: trailing `(?=\s|$)` rather than \b
        # because some opcodes end with `)` (e.g. CIBDRADVICE(IBG)) and \b
        # doesn't match between two non-word characters.
        entity_opcode_re = (
            r"^(?:"
            r"DUITNOWTRANSFER"
            r"|CIBDRADVICE(?:\([A-Z]+\))?"
            r"|CIBCRADVICE|CREDITADV"
            r"|IBGCREDIT"
            r"|REMITTANCECR(?:-[A-Z]+)?"
            r"|TRANSFERFROMSA|TRFROMSA|TRTOSAVINGS"
            r"|ATMMEPSIBFTCR"
            r")(?=\s|$)"
        )
        if re.match(entity_opcode_re, body_after_code, re.IGNORECASE):
            body = re.sub(
                entity_opcode_re + r"\s*[\d,.]+\s*",
                "",
                body_after_code,
                flags=re.IGNORECASE,
            )
            name = _br_extract_entity(body)
            if name and _has_real_word(name):
                return name.upper(), "pattern"
            return f"UNNAMED BANK RAKYAT TRANSFER ({direction_norm})", "special"

    # BUG-001 (2026-05-05) — UOB counterparty extractor (13 family rules).
    # Bank-gated; returns None when no rule matches so the global special-
    # bucket fallback below still has a chance. Subsumes the prior Sprint 6
    # #10 inline block (Chq Wdl / bare Cheque) — preserved as UOB-K inside
    # _extract_counterparty_uob for behaviour-equivalence.
    if bank and "UOB" in bank.upper():
        uob_hit = _extract_counterparty_uob(
            desc, direction=direction, amount=amount, own_party=own_party
        )
        if uob_hit is not None:
            return uob_hit

    # ── Special buckets (hardcoded — deterministic) ────────────────────────
    if re.search(r"\bCDM\b.*CASH DEPOSIT|\bCASH DEPOSIT\b", u):
        return "CASH DEPOSIT", "special"
    if re.search(r"HSE CHQ DEPOSIT|2D LOCAL CHQ|CHQ DEPOSIT|CHEQUE DEPOSIT", u):
        return "Unidentified (Cheque)", "special"  # CP10
    if re.search(r"HOUSE CHQ DR|CLRG CHQ DR|INWARD CLEARING CHQ DEBIT", u):
        return "Unidentified (Cheque)", "special"  # CP10
    if re.search(r"CASH CHQ DR", u):
        return "CASH WITHDRAWAL", "special"
    if re.search(r"RETURN(?:ED)? CHQ|CHQ RETURN|DISHONOUR", u):
        return "RETURNED CHEQUE", "special"
    if re.search(r"IBG INWARD RETURN|GIRO INWARD RETURN", u):
        return "INWARD RETURN", "special"
    if re.search(r"\bREVERSAL\b|\bREVERSED\b|REV CR|CREDIT REVERSAL", u):
        return "REVERSAL", "special"
    if re.search(r"FD MATUR|FIXED DEPOSIT|FD UPLIFT|INTEREST CREDIT|PROFIT CREDIT|SWEEP IN|\bHIBAH\b|DIVIDEND PAID", u):
        return "FD/INTEREST", "special"
    if re.search(r"AUTOPAY CHARGES|OTHER TRANSFER FEE|CH(?:Q|EQUE)\s+PROCESS(?:ING)?\s+FEE|SERVICE TAX|ACCOUNT STATUS CONFIRM|3RD PARTY CHEQUE|STAMP DUTY|CABLE CHARGE|NOSTRO CHARGE|MAS SERVICE CHARGE|AGENT CHARGES|CMS - DR CORP (?:S/)?CHG|\bHANDLING\s+CHRG\b|\bCHEQ(?:UE)?\s+STAMP\s+FEE\b", u):
        return "BANK FEES", "special"

    # AUTOPAY DR U#### = bulk salary batch (CIMB)
    if re.search(r"\bAUTOPAY\s+DR\s+U\d{3,}", u):
        return "BULK SALARY", "special"

    # AUTOPAY CR U\d+ ... RTB\d+ .TXT = inter-account salary batch (no entity).
    # Strict: must be followed by .TXT marker OR nothing but refs — otherwise an
    # entity name follows and should be extracted by the normal AUTOPAY CR branch.
    if re.match(r"AUTOPAY\s+CR\s+U\d+.*RTB\d+.*\.TXT\b", u):
        return "BULK SALARY", "special"
    if re.fullmatch(r"AUTOPAY\s+CR\s+U\d+(?:\s+(?:RTB\d+|U\d+|\d+))*\s*", u):
        return "BULK SALARY", "special"

    # Maybank APS bulk payroll — PMT SLRY / SLRY / PAYROLL
    if re.search(r"\bPMT\s+SLRY\b|\bSLRY\b|\bPAYROLL\b", u):
        return "BULK SALARY", "special"

    # Loan disbursement / trade finance markers (C10)
    if re.search(r"SCF TRADE|LOAN DISBURS|\bFACTORING\b|FINANCING DISBURS|TRADE FINANCE|INVOICE FIN|INVOICE DISCOUNT|BILL PURCHAS|BILL DISCOUNT|BANKERS ACCEPTANCE|FACILITY DRAWDOWN", u):
        return "LOAN DISBURSEMENT", "special"

    # Loan repayment / instalment markers (C11). Parser labels ALL matches;
    # AI layer (Layer 2) refines C11 vs C04 using entity context
    # (company's own loan → C11+C02, related-party personal loan → C04 only).
    if re.search(
        r"\bTERM LOAN\b|\bLOAN REPAY|\bFINANCING REPAY|\bMONTHLY INSTALMENT\b|"
        r"\bIB2G\s+DR\s+CA\s+CR\s+LN\b|\bTRANSFER TO LOAN\b|\bDD CASA PYMT\b|"
        r"\bFINPAL ISSUER REPAYM",
        u,
    ):
        return "LOAN REPAYMENT", "special"

    # Statutory — LHDN / KWSP / SOCSO / HRDF (general path, not just Alliance)
    if re.search(r"LEMBAGA HASIL", u):
        return "LHDN", "special"
    if re.search(r"KUMPULAN WANG SIMPAN", u):
        return "KWSP", "special"
    if re.search(r"PERTUBUHAN KESELAMAT", u):
        return "SOCSO", "special"
    if re.search(r"PEMBANGUNAN SUMBER M", u):
        return "HRDF", "special"

    # CP5 — REMITTANCE CR JANM
    if "JANM" in u:
        return "JANM", "pattern"

    # ── JOMPAY [biller_code] [ref] — group by biller code ──────────────────
    m = re.match(r"JOMPAY\s+([\w:]+)", u)
    if m:
        biller = m.group(1)
        return (f"JOMPAY {biller}"), "special"

    # ── CP2 — IBG CREDIT F ADVANCE [entity] (factoring) ────────────────────
    # CIMB sometimes duplicates the "F ADVANCE" marker:
    #   "IBG CREDIT F ADVANCE F ADVANCE PLANWORTH GLOBAL FAC"
    # Peel any repeated "F ADVANCE" prefix, then strip trailing FAC/FACTORING
    # category marker to isolate the entity name.
    m = re.match(r"IBG\s+CREDIT\s+F\s+ADVANCE\s+(.+)", u)
    if m:
        body = m.group(1).strip()
        body = re.sub(r"^(?:F\s+ADVANCE\s+)+", "", body, flags=re.IGNORECASE).strip()
        body = re.sub(r"\s+FAC(?:TORING)?\b.*$", "", body, flags=re.IGNORECASE).strip()
        name = _strip_trailing_refs(body)
        return (name or "UNIDENTIFIED"), "pattern"

    # ── CP4 — AUTOPAY CR [ref(s)] [entity] ─────────────────────────────────
    # Sub-patterns:
    #   A: AUTOPAY CR <ref> <ref_dup> <ENTITY>
    #   B: AUTOPAY CR <ref1> <ref2> <ref1_dup> <trailer> <ENTITY>
    #   C: AUTOPAY CR <ENTITY>  (no refs — first token purely alphabetic)
    m = re.match(r"AUTOPAY\s+CR\s+(.+)", u)
    if m:
        rest = m.group(1).strip()
        tokens = rest.split()
        # Sub-pattern C: entire remainder is alphabetic entity
        if tokens and re.fullmatch(r"[A-Z&()'.-]+", tokens[0]):
            name = _strip_trailing_refs(rest)
            return (name or "UNIDENTIFIED"), "pattern"
        # Sub-pattern A/B: find duplicated block end, entity = what's after
        end = _find_duplicated_block_end(tokens)
        if end > 0 and end < len(tokens):
            tail = _strip_purpose_prefix_tokens(tokens[end:])
            name = " ".join(tail).strip() or _tail_alpha_run(tokens[end:])
            name = _strip_trailing_refs(name)
            if name:
                return name, "pattern"
        # Fallback: strip all leading ref/noise tokens, use trailing alpha run
        tail = _strip_purpose_prefix_tokens(tokens)
        name = " ".join(tail).strip() or _tail_alpha_run(tokens)
        name = _strip_trailing_refs(name)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # ── CP1 — IBG CREDIT [ref] [ref_dup] [ENTITY] ──────────────────────────
    m = re.match(r"IBG\s+CREDIT\s+(.+)", u)
    if m:
        rest = m.group(1).strip()
        tokens = rest.split()
        end = _find_duplicated_block_end(tokens)
        if end > 0 and end < len(tokens):
            tail = tokens[end:]
            name = _tail_alpha_run(tail, min_len=2) or " ".join(tail).strip()
            name = _strip_trailing_refs(name)
            if name:
                return name, "pattern"
        # No duplication detected — strip leading ref tokens
        skip = 0
        for t in tokens:
            if re.fullmatch(r"[A-Z0-9/.-]{4,}", t) and any(c.isdigit() for c in t):
                skip += 1
            else:
                break
        name = _strip_trailing_refs(" ".join(tokens[skip:]))
        return (name or rest or "UNIDENTIFIED"), "pattern"

    # ── I-PAYMENT FPXPAY [entity] [reference-code] ─────────────────────────
    m = re.match(r"I-?PAYMENT\s+FPXPAY\s+(.+)", u)
    if m:
        rest = m.group(1).strip()
        # Cut at first reference-code token (alphanumeric with digits, 10+ chars, or trailing digits)
        tokens = rest.split()
        kept = []
        for t in tokens:
            if re.fullmatch(r"[A-Z0-9]{10,}", t) and any(c.isdigit() for c in t):
                break
            if re.fullmatch(r"\d{6,}", t):
                break
            kept.append(t)
        name = " ".join(kept).strip()
        # Strip truncated legal suffixes MAL / BER (from MALAYSIA / BERHAD)
        name = re.sub(r"\s+(?:MAL|BER|SDN|BHD)\b.*$", "", name).strip()
        name = _strip_trailing_refs(name)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # ── CP3 Ambank — DuitNow TRF /MISC (CREDIT|DEBIT), <ENTITY>, <purpose>,..
    # Ambank uses a comma-delimited format with entity in the 2nd field.
    # Example: "DuitNow TRF /MISC CREDIT, V2 AUTO SDN BHD, Sent from AmOnline, WVS188"
    m = re.match(r"DUITNOW\s+(?:TRF|TRANSFER)\s+/MISC\s+(?:CREDIT|DEBIT)\s*,\s*([^,]+)", u)
    if m:
        name = m.group(1).strip().rstrip(".,;").strip()
        return (name or "UNIDENTIFIED"), "pattern"

    # Sprint 6 #6 — Ambank Fund Transfer + JomPAY /DEBIT TRANSFER comma-delim.
    # Same shape as the DuitNow /MISC handler above (entity in 2nd comma field)
    # but with different rail prefixes that previously fell through to raw.
    # 131 Fund Transfer + 92 JomPAY rows in corpus. Strip after first comma per
    # the DuitNow precedent. Pattern is specific enough not to fire on other
    # banks (CIMB/OCBC/RHB JomPAY uses biller-code format, caught by the
    # generic JOMPAY handler above).
    m = re.match(r"FUND\s+TRANSFER\s+/DEBIT\s+TRANSFER\s*,\s*([^,]+)", u)
    if m:
        name = m.group(1).strip().rstrip(".,;").strip()
        return (name or "UNIDENTIFIED"), "pattern"
    m = re.match(r"JOMPAY\s+/DEBIT\s+TRANSFER\s*,\s*([^,]+)", u)
    if m:
        name = m.group(1).strip().rstrip(".,;").strip()
        return (name or "UNIDENTIFIED"), "pattern"

    # ── CWS 2026-06 — Ambank "ACCOUNT STATEMENT & TAX INVOICE" rails ───────
    # Same comma-delimited family as CP3 ("<rail>, <entity>, <memo/ref>, …")
    # but with rail prefixes the handlers above don't cover: INWARD IBG /
    # RENTAS / SWIFT, DuitNow CR/DR TRF, Fund Transfer /CREDIT TRANSFER,
    # AUTO DEBIT, Inst Trf CASA OFI, Skim Accum, Cashiers Order. Without
    # this, each row falls through to *raw* and every distinct memo tail
    # ("PAYMENT JAN 2026" vs "PAYMENT OF INVOICE") mints a separate fake
    # counterparty for the same entity (TSM Maintenance bug). Bank-gated:
    # other banks' INWARD IBG shapes (RHB Waja/Kay-R) are not comma-delim.
    # AUTO DEBIT rows carry an empty entity field with the bank itself as
    # beneficiary (", , AmBank, Auto Debit") — facility-instalment flows the
    # analyst must see, so the field scan tolerates one empty/ref field.
    if bank and "AMBANK" in bank.upper():
        m = re.match(
            r"^(?:INWARD\s+(?:IBG|RENTAS|SWIFT)(?:\s*/\s*MISC\s+(?:CREDIT|DEBIT))?"
            r"|DUITNOW\s+(?:CR|DR)?\s*(?:TRF|TRANSFER)\s*/\s*MISC\s+(?:CREDIT|DEBIT)"
            r"|FUND\s+TRANSFER\s*/\s*(?:CREDIT|DEBIT)\s+TRANSFER"
            r"|AUTO\s+DEBIT\s*/\s*DEBIT\s+TRANSFER"
            r"|INST\s+TRF\s+CASA\s+OFI\s*/\s*(?:CREDIT|DEBIT)\s+TRANSFER"
            r"|SKIM\s+ACCUM\s+FOR\s+TRX\s*/\s*DEBIT\s+TRANSFER"
            r"|CASHIERS\s+ORDER\s*/\s*DEBIT\s+TRANSFER"
            r"|CREDIT\s+TRANSFER"
            r")\s*,\s*(.+)$",
            u,
        )
        if m:
            # Older Ambank layouts glue the running balance into the text
            # ("PETROLIAM NASIONAL 448,985.44DR B…"); the comma inside the
            # amount would split the entity field and mint fragment entities
            # ("PETROLIAM NASIONAL 448"). Strip money tokens BEFORE splitting.
            tail = re.sub(r"\s*\d{1,3}(?:,\d{3})*\.\d{2}\s*(?:DR|CR)?\b", " ", m.group(1))
            fields = [f.strip(" .,;").strip() for f in tail.split(",")]

            def _clean_entity_field(f: str) -> str:
                # RENTAS entity fields carry "<acct-no> <NAME> <address>":
                # "001826189021 TANJUNG OFFSHORE SERVICES SDN. BHD. C-16-01+LVL
                # 16+KL TRILLION CORP" / "109070000012 AFFIN BANK CSH-LOC 9TH
                # FLOOR MENARA AFFIN ...". Strip the leading account number,
                # then cut at the legal suffix or the first address token.
                f = re.sub(r"^\d{6,}\s+", "", f).strip()
                sfx = re.search(r"^(.*?\bSDN\.?\s*BHD\.?|.*?\bBERHAD\b|.*?\bBHD\.?(?=\s|$))", f)
                if sfx:
                    return re.sub(r"\s{2,}", " ", sfx.group(1).strip(" .,"))
                addr = re.search(
                    r"\s+(\d+(?:ST|ND|RD|TH)\s+FLOOR|LEVEL\s+\d+|LVL\s+\d+|TINGKAT\s+"
                    r"|MENARA\s|WISMA\s|JALAN\s|PLAZA\s|TAMAN\s|BLOK\s|C-\d+)",
                    f,
                )
                if addr:
                    f = f[: addr.start()].strip(" .,")
                # drop orphan 1–2 letter tail left behind by money-token strip
                f = re.sub(r"\s+[A-Z]{1,2}$", "", f.strip())
                return re.sub(r"\s{2,}", " ", f)

            def _ref_like(f: str) -> bool:
                return bool(re.search(r"\d{4,}", f)) or not _has_real_word(f)

            name = ""
            for f in fields[:2]:  # entity is field 1; field 2 only as fallback
                f = _clean_entity_field(f)
                if f and not _ref_like(f):
                    name = f
                    break
            if name:
                return name, "pattern"
            direction_label = direction or "DR"
            return f"UNNAMED AMBANK TRANSFER ({direction_label})", "special"

    # ── Sprint 6 #12 — OCBC DUITNOW(INST TRF) /IB <ENTITY> handler ─────────
    # OCBC's DuitNow Instant Transfer prints as:
    #   DUITNOW(INST TRF) (DR|CR) /IB <ENTITY truncated ~22ch> DESC: <purpose> REF: <purpose>
    # Entity slice is between `/IB ` and ` DESC:`. OCBC truncates the entity
    # at ~22 chars, so legal-suffix stubs ` SDN`, ` SDN. BH`/` SDN BH`
    # (truncated BHD), and bare ` S` (first letter of SDN) frequently strand
    # at the end. We restore them to `SDN BHD` so the bare-`S` form is
    # filtered through `_has_real_word` and bare initials fall through to
    # UNNAMED. Downstream `_normalise_counterparty` strips legal suffixes
    # entirely, so consolidation also works without restoration — the
    # restoration's primary role is gating bare `MR S`-style inputs.
    # Bare or unparseable rows bucket to UNNAMED OCBC TRANSFER (CR|DR) so
    # volume is visible without inventing a fake counterparty. Bank-scoped
    # to OCBC because `DUITNOW(INST TRF)` (no whitespace after DUITNOW) is
    # OCBC-only. Sprint 6 #11b renamed from rail-named UNNAMED DUITNOW to
    # comply with P7 (rail labels never appear in counterparty buckets).
    if bank and "OCBC" in bank.upper():
        m = re.match(r"^DUITNOW\(INST\s+TRF\)\s+(DR|CR)\s+/IB\s+(.+)$", desc, flags=re.IGNORECASE)
        if m:
            direction = m.group(1).upper()
            tail = m.group(2)
            # Entity is everything up to ` DESC:` (case-sensitive marker OCBC
            # emits). 501/501 corpus rows have ` DESC:`. If absent, bucket bare.
            ent_m = re.match(r"^(.*?)\s+DESC:\s*(.*)$", tail, flags=re.IGNORECASE)
            if not ent_m:
                return f"UNNAMED OCBC TRANSFER ({direction})", "special"
            entity = ent_m.group(1).strip().rstrip(".,;:")
            if not entity:
                return f"UNNAMED OCBC TRANSFER ({direction})", "special"
            ent_u = entity.upper()
            # Restore OCBC-truncated legal suffix:
            #   "<NAME> SDN BH" / "<NAME> SDN. BH"  → "<NAME> SDN BHD"  (truncated BHD)
            #   "<NAME> SDN"                        → "<NAME> SDN BHD"
            #   "<NAME> S"                          → "<NAME> SDN BHD"  (3+ tokens
            #     so we don't restore on genuine single-letter surnames/initials)
            if re.search(r"\bSDN\.?\s+BH\.?\s*$", ent_u):
                entity = re.sub(r"\bSDN\.?\s+BH\.?\s*$", "SDN BHD", entity, flags=re.IGNORECASE)
            elif re.search(r"\bSDN\.?\s*$", ent_u):
                entity = re.sub(r"\bSDN\.?\s*$", "SDN BHD", entity, flags=re.IGNORECASE)
            elif re.search(r"\bS\s*$", ent_u) and len(entity.split()) >= 3:
                entity = re.sub(r"\s+S\s*$", " SDN BHD", entity, flags=re.IGNORECASE)
            entity_norm = entity.upper().strip()
            if not entity_norm or not _has_real_word(entity_norm):
                return f"UNNAMED OCBC TRANSFER ({direction})", "special"
            return entity_norm, "pattern"

    # Sprint 6 #7 — OCBC CA MYDEBIT + CA BANKCARD card-POS rows.
    # Card POS purchases currently fall through to raw with merchant name
    # buried in the description tail. Q3=B chose generic CARD POS buckets
    # over per-merchant entity extraction: card-POS flows are payment-rail,
    # not business relationships, and per-merchant ledgering fragments across
    # ~80-100 unique merchants. Both prefixes are OCBC-exclusive in corpus.
    if bank and "OCBC" in bank.upper():
        if re.match(r"^CA\s+MYDEBIT\b", u):
            return "CARD POS (MYDEBIT)", "special"
        if re.match(r"^CA\s+BANKCARD\b", u):
            return "CARD POS (BANKCARD)", "special"

    # Sprint 6 #10 (UOB Chq Wdl / bare Cheque DR) was relocated to
    # _extract_counterparty_uob (rule UOB-K) by BUG-001 (2026-05-05) so it
    # runs in priority order with the other UOB rules. Behaviour preserved.

    # ── Sprint 6 #15 — RHB Reflex (RFLX) handler + RHB miscellaneous fees ──
    # RHB Reflex is RHB's online corporate banking platform. Four RFLX
    # sub-formats appear in the corpus:
    #   A. RHB Bank (Waja): 'RFLX <ENTITY> / -' — body between leading
    #      'RFLX ' and trailing ' / -'. Many bodies are common single-token
    #      first names truncated by the PDF column (MOHAMMAD ×265, WAN ×59,
    #      ARKAS, ATASHA, ASHRUL…); these are flagged ambiguous because the
    #      bank-truncated text gives no way to disambiguate distinct people
    #      sharing the name. Multi-token bodies (ONG JIA BIN, JATI WAJA,
    #      C.N.T. AUTO) are treated as real entities — specific enough.
    #   B. RHB Bank: 'RFLX / CM112/ -' — RHB Reflex Instant Transfer
    #      Service Charge (RM 0.50 each). CM112 is RHB's internal product
    #      code. Routes to BANK FEES with amount-safety check.
    #   C. RHB Islamic (Kay R): 'RFLX INSTANT TRF (DR|CR) <ref10> <ref12>
    #      <ENTITY> <purpose>' — real-transfer rows. Entity = leading
    #      uppercase token-run; stop at first lower/mixed-case token.
    #   D. RHB Islamic: 'RFLX INSTANT TRF SC (DR|CR) <ref10> [CM112]' —
    #      same Instant Transfer SC as B, fuller print. BANK FEES with
    #      amount-safety check.
    #
    # Amount-safety check (B + D): only auto-route to BANK FEES if amount
    # is fee-shaped (≤ RM 100). Mirrors existing C24 v3.2 rule. One corpus
    # row (2025-08-15, RM 138,791.36) carries the SC label but a non-fee
    # amount — likely a bank/PDF data quality issue. With the safety check
    # it falls through to raw, surfaced for separate Sprint 6 #16
    # investigation rather than silently mis-bucketed as a fee.
    #
    # Bank-scoped to RHB as defence-in-depth.
    if bank and "RHB" in bank.upper():
        # General bank-fee cap (CHQ SVC, statement copy, BANKERS REFER, SST
        # remittance, etc.) — those have legitimate amounts up to ~RM 15-20,
        # so RM 100 is the right ceiling. Used by the misc-fees branch ~50
        # lines below.
        fee_amount_ok = (amount is None) or (amount <= 100.0)
        # BUG-002 (2026-05-02): RFLX service-charge rows are ALWAYS RM 0.50
        # (or 0) — much tighter than the general bank-fee cap. Tightening
        # from <= 100 to <= 1 catches the Kay R Aug-15 anomaly where a
        # multi-line PDF layout bound RM 138,791.36 to the SC label instead
        # of the paired transfer row. Route anomalies to a stable "RFLX SC
        # ANOMALY (NEEDS REVIEW)" bucket (added to the protected-labels
        # set) so the analyst sees them grouped and clearly flagged.
        SC_FEE_MAX = 1.00
        sc_fee_ok = (amount is None) or (amount <= SC_FEE_MAX)
        # Sub-format B: bank-internal Reflex Instant Transfer service fee
        if re.match(r"^RFLX\s*/\s*CM\d+\s*/", desc, flags=re.IGNORECASE):
            if sc_fee_ok:
                return "BANK FEES", "special"
            return "RFLX SC ANOMALY (NEEDS REVIEW)", "special"
        # Sub-format D: 'RFLX INSTANT TRF SC ...' — service charge
        if re.match(r"^RFLX\s+INSTANT\s+TRF\s+SC\s+(?:DR|CR)\b", desc, flags=re.IGNORECASE):
            if sc_fee_ok:
                return "BANK FEES", "special"
            return "RFLX SC ANOMALY (NEEDS REVIEW)", "special"
        # Sub-format C: 'RFLX INSTANT TRF (DR|CR) <ref10> <ref12> <ENTITY> ...'
        m = re.match(
            r"^RFLX\s+INSTANT\s+TRF\s+(DR|CR)\s+\d{10}\s+\d{12}\s+(.+)$",
            desc, flags=re.IGNORECASE,
        )
        if m:
            direction = m.group(1).upper()
            tail = m.group(2).strip()
            if not tail:
                return f"UNNAMED RHB TRANSFER ({direction})", "special"
            # Take leading uppercase / parenthesised tokens (entity); stop at
            # first lower/mixed-case token (purpose).
            ent_toks: List[str] = []
            for t in tail.split():
                if re.fullmatch(r"[A-Z][A-Z0-9.&()\-]*\.?", t) or t == "(M)":
                    ent_toks.append(t)
                else:
                    break
            entity = " ".join(ent_toks).strip().rstrip(".,;:")
            ent_u = entity.upper()
            if re.search(r"\bSDN\.?\s+B\.?\s*$", ent_u):
                entity = re.sub(r"\bSDN\.?\s+B\.?\s*$", "SDN BHD", entity, flags=re.IGNORECASE)
            elif re.search(r"\bSDN\.?\s*$", ent_u):
                entity = re.sub(r"\bSDN\.?\s*$", "SDN BHD", entity, flags=re.IGNORECASE)
            entity_norm = entity.upper().strip()
            if not entity_norm or not _has_real_word(entity_norm):
                return f"UNNAMED RHB TRANSFER ({direction})", "special"
            return entity_norm, "pattern"
        # Sub-format A: 'RFLX <ENTITY> / -' — also handles 'RFLX <NAME> / / -'
        # (extra slash variant). Single-token bodies (common first names)
        # flagged ambiguous; multi-token bodies kept as real entities.
        m = re.match(r"^RFLX\s+(.+?)\s*/\s*(?:/\s*)?-\s*$", desc, flags=re.IGNORECASE)
        if m:
            body = m.group(1).strip().rstrip(".,;:/")
            body_norm = body.upper().strip()
            if not body_norm or not _has_real_word(body_norm):
                return "UNCATEGORIZED", "special"
            tokens = body_norm.split()
            if len(tokens) == 1:
                # Single-token body: PDF-truncated common first name
                # (MOHAMMAD, WAN, ARKAS…). Bucket per name with ambiguity
                # marker so volume is visible without false consolidation.
                # Sprint 6 #16 retro-fit: ambiguity tag is rail-agnostic so
                # the same name from RFLX / REFLEX- / RPP rails consolidates
                # under one bucket. Rail labels are extraction-mechanism
                # noise irrelevant to credit underwriting.
                return f"{body_norm} (possibly multiple parties)", "special"
            return body_norm, "pattern"

        # ── RHB miscellaneous bank fees ────────────────────────────────────
        # All bank-charged service fees route to BANK FEES per the project
        # principle (any service fee → C24, except loan repayment which has
        # its own category). Same amount-safety check applies.
        #   - 'CHQ SVC / -' / 'CHQ SVC / / -'              cheque service (RM 0.50)
        #   - 'SERVICE CASH CHQ / -'                       cash-cheque service (RM 2)
        #   - 'REFLEX- / CM<NNN>/ -'                       other Reflex product fee
        #   - 'SERVICE CHARGES-OTHERS … CTC STATEMENT'     statement-copy fee (RM 15)
        #   - 'BANKERS REFER CHARGES'                      banker's reference (RM 15)
        #   - 'ST - DR <ref10> SST Remittances …'          SST remittance fee (RM 2)
        if fee_amount_ok and re.match(
            r"^(?:CHQ\s+SVC\b|SERVICE\s+CASH\s+CHQ\b|REFLEX-\s*/\s*CM\d+\s*/|"
            r"SERVICE\s+CHARGES-OTHERS\b|BANKERS\s+REFER\s+CHARGES\b|"
            r"ST\s*-\s*DR\s+\d{10}\s+SST\s+Remittances\b)",
            desc, flags=re.IGNORECASE,
        ):
            return "BANK FEES", "special"

        # ── Sprint 6 #16 — RHB Reflex/RPP/IBG/Rentas/cheque entity extraction ─
        # Companion to #15. Covers the remaining raw-method shapes seen in
        # Waja RHB (RHB Bank) and Kay R (RHB Islamic) corpus files. Two
        # families exist:
        #
        #   Family A — RHB Bank / Waja-style (date-line entity):
        #     'RPP <ENTITY> / <purpose> / <ref> -'
        #     'REFLEX- <ENTITY> [<purpose>] / [/ ]-'        (RFLX cousin via Reflex web)
        #     'INWARD IBG <ENTITY-UPPERCASE> [<refs>] [I- -]'
        #     'RENTAS <ENTITY> [/ROC/...] [I-] -'
        #     'MB FUND <ENTITY> / <purpose>/ -'              (Mobile Banking Fund Tfr)
        #     'FPX B2B <ENTITY> [<ref>] -'                   (FPX bulk inward)
        #     'FPX DD <ENTITY> [<ref>/ -]'                   (FPX direct-debit)
        #     'CASH / / -' / 'CASH CASH CHQ / -' / 'CDT CASH / / -' /
        #       'CHEQUE / / -' / 'CLEARING / -'              (cash/cheque markers)
        #
        #   Family B — RHB Islamic / Kay-R-style (entity-at-end):
        #     'INWARD IBG <ref10> [refs] <ENTITY-RUN-AT-END>'
        #     'RPP INWARD INST TRF (DR|CR) <ref10> <ENTITY-RUN> <purpose>'
        #     'REFLEX-FUNDS TFR (DR|CR) <ref10> <ENTITY-RUN> <purpose>'
        #     'RENTAS CREDIT <ref10> <ref> <code> <ENTITY-RUN-AT-END>'
        #     'FPX DD SELLER (DR|CR) <ref10> <ref> <code> <ENTITY-trunc> <ENTITY-full>'
        #     'LOANS/FIN PAYMENT <ref10> <ENTITY> AUTODEBIT'  → LOAN REPAYMENT
        #     'LOCAL CHQ DEP <ref> - -'                       → Unidentified (Cheque)
        #
        # Bucket-naming policy (HARD CONSTRAINT):
        #   - Rail labels (RFLX/REFLEX-/RPP/IBG/RENTAS/FPX) NEVER appear in
        #     bucket names. They are extraction-mechanism noise irrelevant
        #     to credit underwriting.
        #   - Multi-token entities (≥2 tokens) → bare entity, no suffix.
        #     Auto-consolidates across rails.
        #   - Single-token first names → '<NAME> (possibly multiple parties)'.
        #     Rail-agnostic ambiguity tag — same tag as #15 retro-fit so
        #     RFLX/REFLEX-/RPP rows of the same name fold into one bucket.
        #   - Bare-prefix or unparseable bodies → 'UNNAMED RHB TRANSFER (CR|DR)'.
        #
        # Routing rules:
        #   - LOAN keyword in purpose slot (RPP <entity> / Loan / -, etc.) →
        #     LOAN_DISBURSEMENT (CR) / LOAN_REPAYMENT (DR) per side.
        #     CR=loan received from counterparty (analyst codes as
        #     disbursement-in); DR=we are repaying.
        #   - Cash/cheque markers route to CASH DEPOSIT / CASH WITHDRAWAL /
        #     Unidentified (Cheque) per existing buckets.
        #   - LOCAL CHQ DEP (no payer name in PDF) → Unidentified (Cheque).
        #   - LOANS/FIN PAYMENT AUTODEBIT → LOAN REPAYMENT.
        #
        # Cross-bank safety: bank-scoped to RHB. Patterns like 'RPP <NAME> /'
        # and 'INWARD IBG' are not unique to RHB but the bank gate makes the
        # handler defence-in-depth.
        #
        # LOAN keyword policy: keep the entity name. The AI classifier reads
        # the full description ('RPP RN BINA / Loan / -') and assigns C10/C11
        # via the 'Loan' keyword. Routing to a generic LOAN bucket here would
        # discard the funder/lender identity that credit underwriting needs.

        # Helper: is the body just a bare prefix or junk?
        def _rhb16_clean_body(body: str) -> str:
            b = body.strip().rstrip("/-,. ").strip()
            # Drop trailing single-letter / hyphen residues like 'I-', '- -'
            b = re.sub(r"\s+[Ii]-+\s*-*\s*$", "", b)
            b = re.sub(r"\s+-\s*-\s*$", "", b)
            # Drop trailing alphanum ref tokens (B02975432, INV6585) and bare digits
            b = re.sub(r"\s+[A-Za-z]+\d+/?\s*$", "", b)
            b = re.sub(r"\s+\d+/?\s*$", "", b)
            # Drop residual trailing single letter / single-letter+digit tokens
            b = re.sub(r"\s+[A-Za-z]\d*/?\s*$", "", b)
            return b.strip().rstrip("/-,. ").strip()

        def _rhb16_finalize(body: str) -> Tuple[str, str]:
            """Apply rail-agnostic naming rules."""
            body_norm = body.upper().strip()
            if not body_norm or not _has_real_word(body_norm):
                return ("", "")  # caller picks UNNAMED
            tokens = body_norm.split()
            if len(tokens) == 1:
                return (f"{body_norm} (possibly multiple parties)", "special")
            return (body_norm, "pattern")

        # Family-A — Waja-style date-line entity rails (RHB Bank only;
        # RHB Islamic uses entity-at-end). All use trailing ` / -` or ` -`.

        # RPP <ENTITY> / <purpose> / <ref> [-]
        # LOAN keyword in tail (e.g. 'RPP RN BINA / Loan / -') keeps the entity
        # — AI classifier picks up C10/C11 from the description's 'Loan' word.
        m = re.match(r"^RPP\s+(.+?)\s*/\s*(.*?)\s*$", desc, flags=re.IGNORECASE)
        if m and not re.match(r"^RPP\s+INWARD\b", desc, flags=re.IGNORECASE):
            body_raw = m.group(1).strip()
            body = _rhb16_clean_body(body_raw)
            name, method = _rhb16_finalize(body)
            if name:
                return name, method
            return "UNNAMED RHB TRANSFER (CR)", "special"

        # REFLEX- <ENTITY> [<purpose>] / [/ ]-     OR     REFLEX- <ENTITY> I- -
        # Bare 'REFLEX- / CM<n>/ -' already handled above; bare 'REFLEX-/<bare>'
        # without entity → UNNAMED.
        # LOAN keyword in body (e.g. 'REFLEX- ASHRUL LOAN') keeps the entity —
        # AI classifier picks up C10/C11 from the description's 'Loan' word.
        m = re.match(
            r"^REFLEX-\s*(.+?)\s*(?:/\s*(?:/\s*)?-|[Ii]-+\s*-+)\s*$",
            desc, flags=re.IGNORECASE,
        )
        if m:
            body_raw = m.group(1).strip()
            body = _rhb16_clean_body(body_raw)
            # Strip trailing one-token purpose (ALLOWANCE, SAVING, COMPYNY, etc.)
            # iff that leaves at least one alphabetic token AND the original
            # body has 2+ tokens. Otherwise keep as-is.
            tokens = body.split()
            # Drop a trailing month abbrev / single-purpose token to avoid
            # 'MOHAMAD NOV' / 'MOHAMAD Dec' splitting one person 5 ways.
            MONTH_OR_PURPOSE = {
                "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                "SEP", "SEPT", "OCT", "NOV", "DEC",
                "ALLOWANCE", "SAVING", "COMPYNY", "COMPNY", "LOAN",
                "ADVANCE", "ALIGNMENT", "MINYAK", "OPEN",
            }
            while len(tokens) >= 2 and tokens[-1].upper() in MONTH_OR_PURPOSE:
                tokens.pop()
            body = " ".join(tokens).rstrip("/-,. ").strip()
            name, method = _rhb16_finalize(body)
            if name:
                return name, method
            return "UNNAMED RHB TRANSFER (DR)", "special"

        # MB FUND <ENTITY> / <purpose>/ -    (Mobile Banking Fund Tfr inward)
        m = re.match(r"^MB\s+FUND\s+(.+?)\s*/\s*.*$", desc, flags=re.IGNORECASE)
        if m:
            body = _rhb16_clean_body(m.group(1))
            name, method = _rhb16_finalize(body)
            if name:
                return name, method
            return "UNNAMED RHB TRANSFER (CR)", "special"

        # INWARD IBG <ENTITY> [<refs/I- -/...>]   (Waja-style — date-line entity)
        # Distinguished from Kay-R-style 'INWARD IBG <ref10> ...' by leading
        # token after IBG: if it starts with 10-digit ref, fall through to
        # Family-B handler below.
        if not re.match(r"^INWARD\s+IBG\s+\d{10}\b", desc, flags=re.IGNORECASE):
            m2 = re.match(r"^INWARD\s+IBG\s+(.+?)\s*$", desc, flags=re.IGNORECASE)
            if m2:
                body_raw = m2.group(1).strip()
                # Strip trailing tail markers in priority order. Patterns
                # are ordered narrowest-first since each regex is anchored
                # to end-of-string and re.sub returns the original on miss.
                body = body_raw
                # First strip trailing tail markers (' I- -', ' - -', ' -', '/ -')
                # Without this, the more specific ref regexes below fail because
                # they're $-anchored.
                body = re.sub(r"\s*/\s*-\s*$", "", body)
                body = re.sub(r"\s+[Ii]-+\s*-+\s*$", "", body)
                body = re.sub(r"\s+-\s*-\s*$", "", body)
                body = re.sub(r"\s+-\s*$", "", body)
                # ' B12345/ I- -' / ' P01887164/ BAY.UTK -' (account-id refs)
                body = re.sub(r"\s+[BPbp]\d+/\s*.*$", "", body)
                # 'JATI/25-' / 'JATI/25' style account-period markers
                body = re.sub(r"\s+[A-Z]+/\d+\S*$", "", body)
                # Trailing ' <digits>, <digits>/' style noise (TNB HQ rows)
                body = re.sub(r"\s+\d[\d, /]+/?\s*$", "", body)
                # Repeat tail-marker stripping in case earlier subs exposed them
                body = re.sub(r"\s+[Ii]-+\s*-*\s*$", "", body)
                body = re.sub(r"\s+-+\s*$", "", body)
                # Final clean
                body = body.strip().rstrip("/-,. ").strip()
                # 'AKAUN SUB I-' → after stripping → 'AKAUN SUB' (multi-token, kept)
                # 'JABATAN I- -' → 'JABATAN' (single-token, ambiguous)
                name, method = _rhb16_finalize(body)
                if name:
                    return name, method
                return "UNNAMED RHB TRANSFER (CR)", "special"

        # RENTAS <ENTITY> [/ROC/...] [I-] -
        m = re.match(r"^RENTAS\s+(?!CREDIT\b)(.+?)\s*$", desc, flags=re.IGNORECASE)
        if m:
            body_raw = m.group(1).strip()
            body = re.sub(r"/ROC/\d+\s*", "", body_raw)
            body = re.sub(r"\s+[Ii]-+\s*-+\s*$", "", body)
            body = re.sub(r"\s+-\s*-\s*$", "", body)
            body = re.sub(r"\s+-\s*$", "", body)
            body = re.sub(r"/\s*-\s*$", "", body)
            body = body.strip().rstrip("/-,. ").strip()
            name, method = _rhb16_finalize(body)
            if name:
                return name, method
            return "UNNAMED RHB TRANSFER (CR)", "special"

        # FPX B2B <ENTITY> [<ref>] -    (FPX bulk inward — entity is body)
        m = re.match(r"^FPX\s+B2B\s+(.+?)\s*-\s*$", desc, flags=re.IGNORECASE)
        if m:
            body = m.group(1).strip()
            # Strip trailing ref like 'CP_271125_0'
            body = re.sub(r"\s+CP_\d+_\d+\s*$", "", body, flags=re.IGNORECASE)
            body = body.strip().rstrip("/-,. ").strip()
            name, method = _rhb16_finalize(body)
            if name:
                return name, method
            return "UNNAMED RHB TRANSFER (DR)", "special"

        # FPX DD <ENTITY> [<ref>] -    (FPX direct-debit — entity is first token)
        m = re.match(r"^FPX\s+DD\s+(?!SELLER\b)(.+?)\s*-?\s*$", desc, flags=re.IGNORECASE)
        if m:
            body_raw = m.group(1).strip()
            # Trim trailing ref/digits (e.g. 'BOOST 62410210/')
            body = re.sub(r"\s+\d[\d/]*\s*/?\s*$", "", body_raw)
            body = body.strip().rstrip("/-,. ").strip()
            name, method = _rhb16_finalize(body)
            if name:
                return name, method
            return "UNNAMED RHB TRANSFER (DR)", "special"

        # CASH / cheque markers (Waja own-account / cash handling)
        if re.match(r"^CASH\s+CASH\s+CHQ\b", desc, flags=re.IGNORECASE):
            return "CASH WITHDRAWAL", "special"
        if re.match(r"^CDT\s+CASH\b", desc, flags=re.IGNORECASE):
            return "CASH DEPOSIT", "special"
        if re.match(r"^CASH\s*/\s*", desc, flags=re.IGNORECASE):
            return "CASH DEPOSIT", "special"
        if re.match(r"^CHEQUE\s*/\s*", desc, flags=re.IGNORECASE):
            return "Unidentified (Cheque)", "special"
        if re.match(r"^CLEARING\s*/\s*-", desc, flags=re.IGNORECASE):
            return "Unidentified (Cheque)", "special"

        # Family-B — Kay-R-style entity-at-end (RHB Islamic).

        # LOCAL CHQ DEP <ref> - -    (cheque deposit, no payer name)
        if re.match(r"^LOCAL\s+CHQ\s+DEP\b", desc, flags=re.IGNORECASE):
            return "Unidentified (Cheque)", "special"

        # Helpers for Family-B (Kay-R-style entity extraction):
        # An "entity token" is an uppercase / ALL-CAPS token, optionally
        # parenthesised ((M), (MALAYSIA)), or a legal-suffix punctuation
        # form (SDN., BHD., BHD.). Stops at: lowercase token, mixed-case,
        # alphanumeric ref (INVKR..., MYCN..., JNQ6211, FIN1212259), 'INVOICE',
        # date-month words. Trailing punctuation is normalized.
        _ENT_TOKEN = re.compile(r"^(?:\(?[A-Z][A-Z\-]*\)?\.?|[A-Z]\.?|&)$")
        _STOP_WORDS_KAYR = {
            "INVOICE", "DUITNOW", "TRANSFER", "INVKR", "MYCN", "INV", "NO",
            "PAYMENT", "PYMT", "AUTODEBIT", "JANUARY", "FEBRUARY", "MARCH",
            "APRIL", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER",
            "NOVEMBER", "DECEMBER", "JUN", "JUL", "AUG", "SEP", "SEPT", "OCT",
            "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY", "BAYARAN",
            "PINDAHAN", "LOAN", "RENTAL", "SETTLEMENT", "INTERBANK", "GIRO",
            "IBG", "FAMILY", "YARD", "ASSET", "TRANSPORTATION", "FEE",
            "FOR", "FP", "PYMT", "AP", "DD", "AC", "SC", "CR", "DR",
            "ADV", "MULTI", "IVS", "INVS", "PV", "CSR", "PROGRAM", "SUBCONT",
            "SUBC", "SHORT", "PREPAY", "INS", "DO", "PO", "SO",
        }
        def _take_entity_leading(tail: str) -> str:
            """Take leading entity tokens. Stops at non-entity / stop-word /
            alphanumeric-ref tokens."""
            toks = tail.split()
            out = []
            for t in toks:
                # Reject alphanumeric ref tokens (INVKR..., FIN1212259, JNQ6211)
                if re.search(r"\d", t):
                    break
                tu = t.upper()
                # Stop on purpose / stop words
                if tu.rstrip(".,;:") in _STOP_WORDS_KAYR:
                    break
                if not _ENT_TOKEN.match(t):
                    break
                out.append(t)
            return " ".join(out).strip().rstrip(".,;:")
        def _take_entity_trailing(tail: str) -> str:
            """Take trailing entity tokens — entity is at end of string. Stops
            on hitting lowercase / digit-bearing / stop-word token."""
            toks = tail.split()
            out = []
            for t in reversed(toks):
                if re.search(r"\d", t):
                    break
                tu = t.upper()
                if tu.rstrip(".,;:") in _STOP_WORDS_KAYR:
                    break
                if not _ENT_TOKEN.match(t):
                    break
                out.insert(0, t)
            return " ".join(out).strip().rstrip(".,;:")
        def _canonicalize_legal_suffix(entity: str) -> str:
            """Normalize SDN/BHD truncations and trailing dots."""
            # 'SDN. BHD.' / 'SDN BHD.' → 'SDN BHD'
            ent = re.sub(r"\bSDN\.?\s+BHD\.?\s*$", "SDN BHD", entity, flags=re.IGNORECASE)
            # 'SDN B' (truncated) → 'SDN BHD'
            ent = re.sub(r"\bSDN\.?\s+B\.?\s*$", "SDN BHD", ent, flags=re.IGNORECASE)
            # bare 'SDN' (truncated entity) → 'SDN BHD'
            ent = re.sub(r"\bSDN\.?\s*$", "SDN BHD", ent, flags=re.IGNORECASE)
            # '(MALAY' (truncated) → '(MALAYSIA)'
            ent = re.sub(r"\(MALAY\)?\s*$", "(MALAYSIA)", ent, flags=re.IGNORECASE)
            return ent.strip()

        def _dedupe_entity_halves(entity: str) -> str:
            """Dedupe entity printed twice ('X Y X Y' → 'X Y'). Searches for
            the longest non-trivial repeated block anywhere in the entity
            (not just at the start) and takes from the SECOND occurrence on.
            Used for FPX DD SELLER and INWARD IBG/RENTAS shapes where the
            entity prints as 22-char trunc followed by full name, often with
            an unrelated prefix (purpose words, account designation)."""
            toks = entity.split()
            n = len(toks)
            if n < 4:
                return entity
            best_a = best_b = 0
            best_k = 0
            # Scan for repeated block of length k starting at positions a, b
            for k in range(min(n // 2, 6), 1, -1):
                for a in range(0, n - 2 * k + 1):
                    block = toks[a:a + k]
                    # Skip blocks that are just legal suffixes (SDN, BHD)
                    if all(t.upper().rstrip(".,") in {"SDN", "BHD", "BH", "B", "(M)"} for t in block):
                        continue
                    for b in range(a + k, n - k + 1):
                        if toks[b:b + k] == block:
                            if k > best_k:
                                best_a, best_b, best_k = a, b, k
                            break
                if best_k:
                    break
            if best_k:
                # Take from the second occurrence onward (drops the trunc copy
                # and any unrelated prefix words like 'YAYASAN LTAT' or 'JULY 25').
                return " ".join(toks[best_b:])
            return entity

        # LOANS/FIN PAYMENT <ref10> <ENTITY> AUTODEBIT
        # Keep the entity name (the lender) — AI classifier picks up C11 from
        # the LOANS/FIN keyword in description; counterparty preserves who
        # the borrower is repaying.
        m = re.match(
            r"^LOANS?/FIN\s+PAYMENT\s+\d{10}\s+(.+?)\s+AUTODEBIT\s*$",
            desc, flags=re.IGNORECASE,
        )
        if m:
            entity = _canonicalize_legal_suffix(m.group(1).strip())
            entity_norm = entity.upper().strip()
            if entity_norm and _has_real_word(entity_norm):
                return entity_norm, "pattern"
            return "UNNAMED RHB TRANSFER (DR)", "special"

        # RPP INWARD INST TRF (DR|CR) <ref10> <ENTITY-RUN> <purpose>
        m = re.match(
            r"^RPP\s+INWARD\s+INST\s+TRF\s+(DR|CR)\s+\d{10}\s*(.*)$",
            desc, flags=re.IGNORECASE,
        )
        if m:
            direction = m.group(1).upper()
            tail = m.group(2).strip()
            if not tail:
                return f"UNNAMED RHB TRANSFER ({direction})", "special"
            entity = _canonicalize_legal_suffix(_take_entity_leading(tail))
            entity_norm = entity.upper().strip()
            if not entity_norm or not _has_real_word(entity_norm):
                return f"UNNAMED RHB TRANSFER ({direction})", "special"
            return entity_norm, "pattern"

        # REFLEX-FUNDS TFR (DR|CR) <ref10> <ENTITY-RUN> <purpose>
        m = re.match(
            r"^REFLEX-FUNDS\s+TFR\s+(DR|CR)\s+\d{10}\s*(.*)$",
            desc, flags=re.IGNORECASE,
        )
        if m:
            direction = m.group(1).upper()
            tail = m.group(2).strip()
            if not tail:
                return f"UNNAMED RHB TRANSFER ({direction})", "special"
            entity = _canonicalize_legal_suffix(_take_entity_leading(tail))
            entity_norm = entity.upper().strip()
            if not entity_norm or not _has_real_word(entity_norm):
                return f"UNNAMED RHB TRANSFER ({direction})", "special"
            return entity_norm, "pattern"

        # FPX DD SELLER (DR|CR) <ref10> <ref> <code> <ENTITY-trunc> <ENTITY-full>
        # Entity printed twice. Both copies start with same prefix; the second
        # is "fuller" though may itself be truncated. Strategy: take trailing
        # uppercase run, then dedupe duplicated prefix so 'X Y X Y' → 'X Y'.
        m = re.match(
            r"^FPX\s+DD\s+SELLER\s+(DR|CR)\s+\d{10}\s+\d+\s+\d+\s+(.+)$",
            desc, flags=re.IGNORECASE,
        )
        if m:
            direction = m.group(1).upper()
            tail = m.group(2).strip()
            entity = _canonicalize_legal_suffix(_take_entity_trailing(tail))
            entity = _canonicalize_legal_suffix(_dedupe_entity_halves(entity))
            entity_norm = entity.upper().strip()
            if not entity_norm or not _has_real_word(entity_norm):
                return f"UNNAMED RHB TRANSFER ({direction})", "special"
            return entity_norm, "pattern"

        # RENTAS CREDIT <ref10> <ref> <code> <ENTITY-RUN-AT-END>  (Kay-R-style)
        m = re.match(
            r"^RENTAS\s+CREDIT\s+\d{10}\s+(.+)$",
            desc, flags=re.IGNORECASE,
        )
        if m:
            tail = m.group(1).strip()
            entity = _canonicalize_legal_suffix(_take_entity_trailing(tail))
            entity = _canonicalize_legal_suffix(_dedupe_entity_halves(entity))
            entity_norm = entity.upper().strip()
            if not entity_norm or not _has_real_word(entity_norm):
                return "UNNAMED RHB TRANSFER (CR)", "special"
            # Statutory check on the extracted entity (e.g. AKAUNTAN NEGARA
            # MALAYSIA / JABATAN AKAUNTAN NEGARA → JANM existing bucket).
            if "AKAUNTAN NEGARA" in entity_norm or "JABATAN AKAUNTAN" in entity_norm:
                return "JANM", "pattern"
            return entity_norm, "pattern"

        # INWARD IBG <ref10> [refs] <ENTITY-RUN-AT-END>   (Kay-R-style)
        m = re.match(r"^INWARD\s+IBG\s+\d{10}\s*(.*)$", desc, flags=re.IGNORECASE)
        if m:
            tail = m.group(1).strip()
            if not tail:
                return "UNNAMED RHB TRANSFER (CR)", "special"
            entity = _canonicalize_legal_suffix(_take_entity_trailing(tail))
            entity = _canonicalize_legal_suffix(_dedupe_entity_halves(entity))
            entity_norm = entity.upper().strip()
            if not entity_norm or not _has_real_word(entity_norm):
                return "UNNAMED RHB TRANSFER (CR)", "special"
            return entity_norm, "pattern"

    # ── Sprint 7 #8 (V3-A) — Public Bank DUITNOW TRSF / RMT entity extraction ─
    # PBB descriptions are MTCEC-style (space-separated opcode); when a
    # counterparty is present it follows the 6-digit reference. Two patterns:
    #   A. 'DUITNOW TRSF (DR|CR) <6-digit-ref> [<ENTITY>] [<amount> <balance>]'
    #      Folder 3 (Mazaa) CR rows carry an own-party echo (`MAZAA SDN BHD`)
    #      stripped downstream by _strip_own_party_tokens. DR rows almost
    #      never carry an entity tail — bucket to UNNAMED so volume stays
    #      visible without inventing a fake counterparty.
    #   B. 'RMT (DR|CR) [<ref>] [AT CPC] [<ENTITY>]' — wire remittance.
    #      `RMT CHRG DR` is a remittance fee → BANK FEES.
    #
    # Without this branch the generic DUITNOW handler below grabs `TRSF CR`/
    # `TRSF DR` as the counterparty (P7 violation — rail-label leak into the
    # bucket) and the RMT rows fall through to raw. Bank-gated to PBB; cross-
    # bank safety also via the `DUITNOW TRSF` token (vs the more common
    # `DUITNOW TRANSFER` / `DUITNOW TO ACCOUNT` shapes other banks use).
    if bank and "PUBLIC BANK" in bank.upper():
        # Sprint 7 Phase 2A — DEP-ECP / DR-ECP : Electronic Cheque
        # Presentment. PBB-specific opcodes (cross-bank-keyword-unsafe).
        # DEP-ECP = inbound cheque clearing → CHEQUE DEPOSIT bucket → C19.
        # DR-ECP  = outbound cheque clearing → CHEQUE ISSUE bucket   → C20.
        # 371 + 20 rows in Mazaa corpus alone (75% of statement volume).
        if re.match(r"^DEP-ECP\b", u):
            return "CHEQUE DEPOSIT", "special"
        if re.match(r"^DR-ECP\b", u):
            return "CHEQUE ISSUE", "special"

        # B1 — RMT charge: remittance service fee → BANK FEES
        if re.match(r"^RMT\s+CHRG\s+(?:DR|CR)\b", u):
            return "BANK FEES", "special"

        # Helper: strip parser-leaked trailing amount/balance tuples and the
        # leading 6-digit reference so _br_extract_entity sees a clean tail.
        def _pbb_clean_tail(tail: str) -> str:
            tail = re.sub(
                r"(?:\s+\d{1,3}(?:,\d{3})*(?:\.\d+)?){1,3}\s*$", "", tail
            ).strip()
            tail = re.sub(r"^\d{6,}\b\s*", "", tail).strip()
            return tail

        # A — DUITNOW TRSF
        m = re.match(
            r"^DUITNOW\s+TRSF\s+(DR|CR)(?:\s+(.*))?$",
            desc, flags=re.IGNORECASE,
        )
        if m:
            side = m.group(1).upper()
            tail = _pbb_clean_tail((m.group(2) or "").strip())
            if not tail:
                return f"UNNAMED PUBLIC BANK TRANSFER ({side})", "special"
            entity_norm = _br_extract_entity(tail).upper().strip()
            if not entity_norm or not _has_real_word(entity_norm):
                return f"UNNAMED PUBLIC BANK TRANSFER ({side})", "special"
            return entity_norm, "pattern"

        # B2 — RMT (DR|CR) [tail with optional entity]
        m = re.match(r"^RMT\s+(DR|CR)(?:\s+(.*))?$", u)
        if m:
            side = m.group(1).upper()
            tail = (m.group(2) or "").strip()
            tail = _pbb_clean_tail(tail)
            tail = re.sub(r"^AT\s+CPC\b\s*", "", tail, flags=re.IGNORECASE).strip()
            if not tail:
                return f"UNNAMED PUBLIC BANK TRANSFER ({side})", "special"
            entity_norm = _br_extract_entity(tail).upper().strip()
            if not entity_norm or not _has_real_word(entity_norm):
                return f"UNNAMED PUBLIC BANK TRANSFER ({side})", "special"
            return entity_norm, "pattern"

    # ── CP3 — DUITNOW TO ACCOUNT / CR / TRANSFER ────────────────────────────
    # Sub-formats:
    #   A: [purpose_or_ref] [purpose_or_ref_dup] [ENTITY] [SDN] [BANK]
    #   B: [MYCN\d+] DuitNow (Transfer) [MYCN\d+] [ENTITY]
    #   C: malformed B where purpose runs into "DuitNow (Transfer)"
    m = re.match(r"DUITNOW(?:\s+(?:TO\s+ACCOUNT|CR|TRANSFER))?\s+(.+)", u)
    if m:
        raw = m.group(1).strip()
        # Strip trailing bank suffix (e.g. " MBB", " CIMB", " HLBB", " AMFB")
        raw = re.sub(rf"\s+{_CP_BANK_SUFFIX}\s*$", "", raw).strip()
        # Strip trailing SDN/BHD/SD/BH/PLT noise
        raw = re.sub(r"[\s.,]+(?:SDN\.?|BHD\.?|SD|BH|PLT)(?:\s+(?:SDN\.?|BHD\.?|SD|BH|PLT))*\s*$", "", raw).strip()

        # Sub-format B/C: MYCN reference pattern
        mycn = list(re.finditer(r"MYCN\d+", raw))
        if len(mycn) >= 2:
            tail = raw[mycn[-1].end():].strip()
            tail = _strip_trailing_refs(tail)
            tail = _strip_stop_tokens(tail)
            if tail:
                return tail, "pattern"

        tokens = raw.split()
        # Sub-format A: find duplicated block, entity = tail
        end = _find_duplicated_block_end(tokens)
        if end > 0 and end < len(tokens):
            tail_tokens = _strip_purpose_prefix_tokens(tokens[end:])
            tail = " ".join(tail_tokens).strip()
            tail = _clip_at_stop_keyword(tail)
            tail = _strip_trailing_refs(tail)
            tail = _strip_stop_tokens(tail)
            if tail:
                return tail, "pattern"

        # No duplication — collapse prefix, clip, strip noise, fall back to alpha tail
        rest = _dedupe_duplicated_prefix(raw)
        rest = _clip_at_stop_keyword(rest)
        rest = _strip_trailing_refs(rest)
        rest = _strip_stop_tokens(rest)
        if not rest or re.fullmatch(r"[A-Z0-9/.-]+", rest.split()[0] if rest else ""):
            tail_alpha = _tail_alpha_run(raw.split(), min_len=2)
            if tail_alpha:
                rest = tail_alpha
        return (rest or "UNIDENTIFIED"), "pattern"

    # ── I-FUNDS TR FROM SA [purpose] [PERSON NAME] ─────────────────────────
    m = re.match(r"I-?FUNDS\s+TR\s+FROM\s+SA\s+(.+)", u)
    if m:
        rest = m.group(1).strip()
        tokens = rest.split()
        # Prefer: find BIN/BINTI/ANAK anchor, take 1 token before + anchor + 1-2 after.
        anchor_idx = -1
        for i, t in enumerate(tokens):
            if t.upper() in _CP_NAME_ANCHORS or t.upper().replace("/", "") in ("AL", "AP"):
                anchor_idx = i
                break
        if anchor_idx > 0:
            start = max(0, anchor_idx - 1)
            # Take firstname(s) before anchor — walk back through alpha tokens
            while start > 0 and tokens[start - 1].isalpha() and not _CP_PURPOSE_WORDS.match(tokens[start - 1].upper()):
                start -= 1
            end_i = min(len(tokens), anchor_idx + 3)
            # Trim trailing non-alpha
            while end_i > anchor_idx + 1 and not tokens[end_i - 1].strip(".,()").isalpha():
                end_i -= 1
            name = " ".join(tokens[start:end_i]).strip()
        else:
            # No anchor — take trailing alpha run after stripping purpose prefix
            tail = _strip_purpose_prefix_tokens(tokens)
            name = _tail_alpha_run(tail, min_len=2) or " ".join(tail).strip()
        name = _strip_trailing_refs(name)
        return (name or "UNIDENTIFIED"), ("pattern" if name else "raw")

    # ── I-PYMT TO CCARD [FIRSTNAME SURNAME] [CC_DETAILS] ───────────────────
    m = re.match(r"I-?PYMT\s+TO\s+CCARD\s+(.+)", u)
    if m:
        rest = m.group(1).strip()
        # Stop at first CC / bank-name / CREDIT CARD / BOS
        stop = re.search(
            rf"\b(?:CC|CREDIT\s+CARD|{_CP_BANK_SUFFIX}|BOS)\b",
            rest, flags=re.IGNORECASE,
        )
        if stop and stop.start() > 0:
            rest = rest[:stop.start()].strip()
        rest = _strip_trailing_refs(rest)
        return (rest or "UNIDENTIFIED"), "pattern"

    # ── CP6 — CIMB TR IBG / TR TO C/A / TR TO SAVINGS ──────────────────────
    m = re.match(r"(?:TR\s+IBG|TR\s+TO\s+C/A|TR\s+TO\s+SAVINGS|TR\s+TO)\s+(.+)", u)
    if m:
        rest = m.group(1).strip()
        stop = re.search(rf"\b{_CP_STOP_KEYWORDS}\b", rest)
        if stop and stop.start() > 0:
            rest = rest[:stop.start()].strip()
        # Peel leading purpose / date prefix tokens (e.g. "JAN AND FEB 2026 ")
        # so the entity name doesn't get consumed by _strip_trailing_refs'
        # greedy year-match (e.g. " 2026 <name>" incorrectly treated as a ref).
        rest = " ".join(_strip_purpose_prefix_tokens(rest.split()))
        rest = _strip_trailing_refs(rest)
        rest = _strip_stop_tokens(rest)
        # Preserve SDN/BHD boundary — if name ends at SDN (no BHD), keep "… SDN BHD"
        if re.search(r"\bSDN\s*$", rest):
            rest = rest + " BHD"
        return (rest or "UNIDENTIFIED"), "pattern"

    # ── Sprint 6 #11a — Maybank CMS bulk-disbursement / direct-debit ───────
    # Maybank's Cash Management Service emits three sub-formats:
    #   CMS - CR PYMT MARS <ENTITY> <REF> [Book Transfer Third]
    #   CMS - DR DIRECT DEBIT <ENTITY> <REF> [purpose]
    #   CMS - DR PYMT MARS <ENTITY_or_PURPOSE> <REF>
    # Entity follows the 4-token prefix. Body is cut at the first reference
    # token (e.g. IT2507170347332, L2507233912092, MA458130272994, KL2500072,
    # SF_2025-06-10_, SB130220030082, A7A1220, E-2025…), the "Book Transfer
    # Third" purpose marker, or "PayNet". Bare advice rows and intercompany /
    # claim / hotel-booking rows (Shahnaz CMS-DR PYMT MARS variants) bucket to
    # UNNAMED MAYBANK TRANSFER so analysts see the volume without inventing fake
    # entities. Sprint 6 #11c — was UNNAMED CMS until 2026-04-27; renamed because
    # CMS is a rail/product name (Maybank Cash Management Service), not a
    # counterparty attribute. Audit confirmed handler is Maybank-only in corpus
    # (regex specificity is sufficient — no static bank guard needed).
    m = re.match(r"^CMS\s*-\s*(CR|DR)\s+(PYMT\s+MARS|DIRECT\s+DEBIT)(?:\s+(.+))?$", u)
    if m:
        direction = m.group(1).upper()
        body = (m.group(3) or "").strip()
        if not body:
            return f"UNNAMED MAYBANK TRANSFER ({direction})", "special"
        # Intercompany / own-account / generic-purpose markers — no third-party.
        if re.match(
            r"^(?:INTERCO\b|INTERCOMPANIES\b|TRANSFER\s+FUNDS\b|"
            r"PYMT\s+(?:KK|MAKAN|HOTEL)|PAYMENT\s+|HOTEL\s+BOOKING|"
            r"CLAIM(?:S|_)?\b|CLAIMS_\w|\d+\s+INTERCO\b|001\s+001\b)",
            body, flags=re.IGNORECASE,
        ):
            return f"UNNAMED MAYBANK TRANSFER ({direction})", "special"
        tokens = body.split()
        # Reference-token = uppercase letter prefix + digits, or pure digits 6+,
        # or alnum like SF_, E-, A7A1220 (mixed case codes).
        ref_re = re.compile(
            r"^(?:[A-Z]{1,6}[-_]?\d{3,}|\d{6,}|SF_\d|E-\d|[A-Z]\d[A-Z]\d{4,})",
            re.IGNORECASE,
        )
        cut = len(tokens)
        for i, t in enumerate(tokens):
            if ref_re.match(t):
                cut = i
                break
            tl = t.lower()
            if tl == "book" and i + 2 < len(tokens) and tokens[i + 1].lower() == "transfer":
                cut = i
                break
            if tl == "paynet":
                cut = i
                break
        name = " ".join(tokens[:cut]).strip()
        if not name:
            return f"UNNAMED MAYBANK TRANSFER ({direction})", "special"
        # Strip dangling truncated "(M" / "(MALAY" / trailing punctuation.
        name = re.sub(r"\s*\((?:M|MALAY)\.?\s*$", "", name).strip()
        name = name.rstrip(".,;:")
        # Preserve "… SDN" → "… SDN BHD" (Maybank truncates legal suffix).
        if re.search(r"\bSDN\.?\s*$", name, flags=re.IGNORECASE):
            name = re.sub(r"\bSDN\.?\s*$", "SDN BHD", name, flags=re.IGNORECASE)
        name = _strip_trailing_refs(name)
        if not _has_real_word(name):
            return f"UNNAMED MAYBANK TRANSFER ({direction})", "special"
        return name, "pattern"

    # ── Sprint 6 #11b — Maybank PAYMENT DEBIT - APS /OTHERS MAS PAYMENT ────
    # Maybank APS bulk-payment format. Body after the `*` separator is almost
    # always an internal payroll / statutory / petty-cash purpose tagged with
    # the statement holder's own short name (e.g. NAARA SALARY OCT 2025,
    # NAARA EPF SOCSO EIS HRDF, NAARA PETTY CASH SEPT). No third-party entity
    # is ever present. Route to existing salary/statutory buckets where the
    # purpose is clear; otherwise bucket as UNNAMED INTERNAL PAYROLL (DR) so
    # the analyst can see the volume. Sprint 6 #11d — was UNNAMED MAS PAYMENT
    # (DR) until 2026-04-27; renamed to a neutral operational label because
    # MAS PAYMENT is a rail/product name (Maybank's APS rail), and the audit
    # confirmed every row in this bucket is internal payroll / statutory /
    # petty cash, never a third-party transfer.
    m = re.match(r"^PAYMENT\s+DEBIT\s*-\s*APS\s*/?\s*OTHERS(?:\s+MAS\s+PAYMENT\s*\*?)?\s*(.*)$", u)
    if m:
        body = m.group(1).strip()
        if not body:
            return "UNNAMED INTERNAL PAYROLL (DR)", "special"
        bu = body.upper()
        # KWSP / SOCSO / HRDF first — combined-batch rows like
        # "NAARA EPF SOCSO EIS HRDF" should bucket consistently. Pick the
        # first statutory keyword present (KWSP → SOCSO → HRDF priority).
        if re.search(r"\b(?:EPF|KWSP)\b", bu):
            return "KWSP", "special"
        if re.search(r"\b(?:SOCSO|PERKESO|EIS)\b", bu):
            return "SOCSO", "special"
        if re.search(r"\b(?:HRDF|HRDC)\b", bu):
            return "HRDF", "special"
        if re.search(r"\b(?:SALARY|GAJI|PAYROLL|WAGES|BONUS|ELAUN)\b", bu):
            return "BULK SALARY", "special"
        return "UNNAMED INTERNAL PAYROLL (DR)", "special"

    # ── CP9 — Maybank PAYMENT FR A/C [entity]*[entity_or_ref] ──────────────
    # The real counterparty can sit on either side of `*`: usually AFTER (e.g.
    # "PAYMENT FR A/C <ref> * <ENTITY>"), but for Maybank VISA card repayments
    # and many biller auto-debits the entity is BEFORE `*` and the text AFTER
    # is a masked card number or ref code ("MAYBANK VISA CARD * XXXX-XXXX-..").
    # Prefer whichever side has a real alphabetic word (not just X's or digits).
    # Ghost-verb guard: bare "PAYMENT FR A/C" with nothing useful goes to a
    # consolidated UNNAMED PAYMENT bucket, NOT UNIDENTIFIED — so the analyst
    # can see these as a single data-quality line in top_parties.
    m = re.match(r"PAYMENT\s+FR\s+A/C(?:\s+(.+))?$", u)
    if m:
        rest = (m.group(1) or "").strip()
        if not rest:
            return "UNNAMED PAYMENT (DR)", "special"
        if "*" in rest:
            before, after = [p.strip() for p in rest.split("*", 1)]
            if _has_real_word(after):
                name = after
            elif _has_real_word(before):
                name = before
            else:
                return "UNNAMED PAYMENT (DR)", "special"
        else:
            name = rest
        name = _strip_trailing_refs(name)
        if not _has_real_word(name):
            return "UNNAMED PAYMENT (DR)", "special"
        return name, "pattern"

    # ── Maybank INTER-BANK PAYMENT INTO/FROM A/C [entity] [/ROC/ref] [purpose] ──
    # Example: "INTER-BANK PAYMENT INTO A/C KHAN MOHAMMED ABU B /ROC/106493872511/// EDUCATION"
    # Entity is the alpha tokens between "A/C" and the first ref/slash marker.
    m = re.match(r"INTER[- ]?BANK\s+PAYMENT\s+(INTO|FROM)\s+A/C(?:\s+(.+))?$", u)
    if m:
        direction = "CR" if m.group(1) == "INTO" else "DR"
        rest = (m.group(2) or "").strip()
        if not rest:
            return f"UNNAMED INTER-BANK ({direction})", "special"
        # Take tokens until we hit a ref marker: starts with '/', or is digit-heavy
        name_toks = []
        for tok in rest.split():
            if tok.startswith("/"):
                break
            if re.fullmatch(r"\d+", tok):
                break
            if len(tok) >= 4 and any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
                # Mixed alphanumeric with digits — likely a ref code
                break
            name_toks.append(tok)
        name = _strip_trailing_refs(" ".join(name_toks))
        if not _has_real_word(name):
            return f"UNNAMED INTER-BANK ({direction})", "special"
        return name, "pattern"

    # ── CP7/CP8 — Maybank TRANSFER TO/FR A/C [name]*[purpose] ──────────────
    # Ghost-verb guard: many Maybank Islamic statements emit bare "TRANSFER FR A/C"
    # or "TRANSFER TO A/C" with no counterparty at all, or with only a ref number
    # after `*`. These were previously silently dumped into UNIDENTIFIED; now
    # they consolidate into UNNAMED TRANSFER (CR/DR) so the analyst sees them
    # as one clear line in top_parties instead of being blind-spot flows.
    m = re.match(r"(TRANSFER\s+(TO|FR)\s+A/C|TRF\s+(TO|FR))(?:\s+(.+))?$", u)
    if m:
        direction = "CR" if (m.group(2) or m.group(3)) == "FR" else "DR"
        rest = (m.group(4) or "").strip()
        if not rest:
            return f"UNNAMED TRANSFER ({direction})", "special"
        if "*" in rest:
            before, after = [p.strip() for p in rest.split("*", 1)]
            # For TRANSFER patterns, name is typically BEFORE the asterisk
            # (purpose text follows *). Fall back to AFTER only if BEFORE is garbage.
            if _has_real_word(before):
                name = before
            elif _has_real_word(after):
                name = after
            else:
                return f"UNNAMED TRANSFER ({direction})", "special"
        else:
            stop = re.search(rf"\s+{_CP_STOP_KEYWORDS}\b", rest)
            name = rest[:stop.start()].strip() if stop else rest
        name = _strip_trailing_refs(name)
        if not _has_real_word(name):
            return f"UNNAMED TRANSFER ({direction})", "special"
        return name, "pattern"

    # ── IBG DEBIT / IBG DR (outward) ───────────────────────────────────────
    m = re.match(r"IBG\s+(?:DEBIT|DR)\s+(.+)", u)
    if m:
        rest = m.group(1).strip()
        tokens = rest.split()
        skip = 0
        for t in tokens:
            if re.fullmatch(r"[A-Z0-9]{4,}", t) and any(c.isdigit() for c in t):
                skip += 1
            else:
                break
        name = _strip_trailing_refs(" ".join(tokens[skip:]))
        return (name or rest or "UNIDENTIFIED"), "pattern"

    # Fallback — raw description
    return desc.upper().strip(), "raw"


_CP_NOISE_NAMES = {
    "LT", "MAN", "ACC NO", "ACC", "ALLEN", "EVENT", "PENYERAHAN", "PERMIT CNU",
    "PERMIT", "KRMK", "KRMK 2025", "JAN AND FEB", "OKTOBER", "CNU",
    "JANUARI", "FEBRUARI", "MAC", "APRIL", "MEI", "JUN", "JULAI", "OGOS",
    "SEPTEMBER", "NOVEMBER", "DISEMBER",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
}


_OWN_PARTY_BOILER_SUFFIX = {"SDN", "BHD", "BERHAD", "PTY", "LTD", "(M)", "&", "CO"}
_OWN_PARTY_PROTECTED_LABELS = {
    "UNIDENTIFIED", "UNCATEGORIZED", "CASH DEPOSIT", "CASH WITHDRAWAL",
    "BANK FEES", "BULK SALARY", "FD/INTEREST", "LOAN REPAYMENT",
    "LOAN DISBURSEMENT", "KWSP", "SOCSO", "LHDN", "HRDF", "REVERSAL",
    "RETURNED CHEQUE", "INWARD RETURN", "JANM", "APAYLATER",
    # Sprint 6 #11b — OCBC DUITNOW(INST TRF) defensive bucket renamed from
    # rail-named UNNAMED DUITNOW to comply with P7. Bank-scoped — handler
    # at _extract_counterparty only reaches this branch for OCBC rows.
    "UNNAMED OCBC TRANSFER (CR)", "UNNAMED OCBC TRANSFER (DR)",
    # Sprint 6 #16 — RHB unnamed transfer buckets are rail-agnostic; the
    # rail name (RFLX / REFLEX- / RPP / IBG / RENTAS) is extraction-
    # mechanism detail irrelevant to credit underwriting.
    "UNNAMED RHB TRANSFER (CR)", "UNNAMED RHB TRANSFER (DR)",
    # BUG-002 (2026-05-02) — RFLX SC rows whose amount is not fee-shaped
    # (> RM 1.00). Stable label so all anomalies bucket together and the
    # analyst sees one clearly-flagged group instead of N raw-description
    # buckets.
    "RFLX SC ANOMALY (NEEDS REVIEW)",
    # Sprint 6 #11a — Alliance unnamed transfer is rail-agnostic. Replaces
    # 7 rail-named buckets (INSTANT TRANSFER / CR ADVICE / IB2G FUND
    # TRANSFER / IB2G DEBIT / FPX PAYMENT / RENTAS CREDIT / DD CASA).
    "UNNAMED ALLIANCE TRANSFER (CR)", "UNNAMED ALLIANCE TRANSFER (DR)",
    # Sprint 6 #11c — Maybank CMS bulk-disbursement / direct-debit unnamed
    # bucket renamed from rail-named UNNAMED CMS. Audit confirmed handler is
    # Maybank-only in corpus.
    "UNNAMED MAYBANK TRANSFER (CR)", "UNNAMED MAYBANK TRANSFER (DR)",
    # Sprint 6 #11d — Maybank APS bulk-payment unnamed bucket renamed from
    # rail-named UNNAMED MAS PAYMENT (DR) to a neutral operational label.
    # Audit confirmed every row in this format is internal payroll /
    # statutory / petty-cash (never a third-party transfer), so the bucket
    # encodes operational meaning rather than bank attribution.
    "UNNAMED INTERNAL PAYROLL (DR)",
    # Sprint 6 #10 — UOB Chq Wdl + bare Cheque <num> DR rows. Bank-gated
    # handler at _extract_counterparty only reaches this branch for UOB rows.
    "UNNAMED UOB TRANSFER (DR)",
    # Sprint 6 #7 — OCBC card-POS generic buckets. CA MYDEBIT (debit-card
    # POS) and CA BANKCARD (credit-card POS) consolidate card payment-rail
    # flows; merchant identity is intentionally not extracted (Q3=B).
    "CARD POS (MYDEBIT)", "CARD POS (BANKCARD)",
    # Sprint 6 #4 — Hong Leong Bank / HLB Islamic unnamed bucket for truly
    # nameless rows (Bulk DuitNow aggregate batches, CA IBT internal own-
    # account transfers) per Q3. Bank-gated handler at _extract_counterparty
    # only reaches this branch for Hong Leong rows.
    "UNNAMED HLB TRANSFER (CR)", "UNNAMED HLB TRANSFER (DR)",
    # Sprint 6 #9 — Bank Rakyat fallback bucket for IBGINWARDRETURN /
    # REVERSALCR (system events, no entity) and entity-bearing opcodes
    # whose continuation tokens are all noise. Bank-gated handler at
    # _extract_counterparty only reaches this branch for Bank Rakyat rows.
    "UNNAMED BANK RAKYAT TRANSFER (CR)", "UNNAMED BANK RAKYAT TRANSFER (DR)",
    # Sprint 7 #8 — Public Bank DUITNOW TRSF / RMT bucket. PBB DR rows almost
    # never carry an entity tail; CR rows often carry only an own-party echo.
    "UNNAMED PUBLIC BANK TRANSFER (CR)", "UNNAMED PUBLIC BANK TRANSFER (DR)",
    # Sprint 7 #10 — Bank Islam (BIMB) bare-opcode / system-event bucket. Used
    # when entity-bearing opcodes find no entity in the tail (refs-only).
    "UNNAMED BANK ISLAM TRANSFER (CR)", "UNNAMED BANK ISLAM TRANSFER (DR)",
    "Unidentified (Cheque)",
}


def _strip_own_party_tokens(name: str, own_party: str) -> str:
    """Sprint 6 #10 — strip own-party (statement holder) name tokens from the
    extracted counterparty name. Handles prefix, suffix, and bracketing forms
    plus column-width truncation (e.g. 'NEWTON' emitted as 'NEW' when the PDF
    column cuts the last 3 chars).

    Conservative — requires at least 2 non-boilerplate core tokens of the
    own-party to match, and does not strip anything when the remainder would
    be < 3 characters.
    """
    if not name or not own_party:
        return name
    name_up = name.upper().split()
    if not name_up:
        return name
    own_core = [
        t for t in own_party.upper().split()
        if t not in _OWN_PARTY_BOILER_SUFFIX
    ]
    if len(own_core) < 2:
        return name

    # Sprint 6 #14 — exact-match case: counterparty (after legal-suffix
    # stripping) is literally identical to the own-party core tokens. The
    # prefix/suffix/mid-span matchers below all require seq to be STRICTLY
    # longer than the own-party tokens, so without this fast-path the row
    # leaks as a real counterparty. Relabel with (OWN-PARTY) so the rows
    # consolidate under one bucket; the suffix survives _normalise_counterparty
    # and is excluded from re-stripping by the "(OWN-PARTY)" not in norm_name
    # guard in build_counterparty_ledger. Cross-bank-safe — verified via
    # corpus survey that no Alliance / Maybank / etc. counterparty currently
    # equals the holder name token-for-token.
    if name_up == own_core:
        return f"{name} (OWN-PARTY)"

    def _prefix_matches(seq: list, prefix: list) -> bool:
        if len(seq) <= len(prefix):
            return False
        if seq[: len(prefix)] == prefix:
            return True
        # Truncation tolerance: last prefix token may be truncated in seq.
        if len(prefix) >= 2 and seq[: len(prefix) - 1] == prefix[:-1]:
            full = prefix[-1]
            trunc = seq[len(prefix) - 1]
            if len(trunc) >= 3 and full.startswith(trunc):
                return True
        return False

    def _suffix_matches(seq: list, suffix: list) -> bool:
        if len(seq) <= len(suffix):
            return False
        if seq[-len(suffix) :] == suffix:
            return True
        # Truncation tolerance on last token.
        if len(suffix) >= 2 and seq[-len(suffix) : -1] == suffix[:-1]:
            full = suffix[-1]
            trunc = seq[-1]
            if len(trunc) >= 3 and full.startswith(trunc):
                return True
        return False

    def _mid_match_span(seq: list, own: list):
        """Return (start, end) of the longest own-prefix that appears as a
        contiguous sub-sequence of seq, not at the boundaries. None otherwise.
        """
        for k in range(len(own), 1, -1):
            group = own[:k]
            # Scan interior positions only (not position 0, not position len-k).
            for i in range(1, len(seq) - k):
                if seq[i : i + k] == group:
                    return (i, i + k, k)
                # Truncation tolerance on last token.
                if k >= 2 and seq[i : i + k - 1] == group[:-1]:
                    full = group[-1]
                    trunc = seq[i + k - 1]
                    if len(trunc) >= 3 and full.startswith(trunc):
                        return (i, i + k, k)
        return None

    changed = True
    guard = 0
    while changed and guard < 5:
        guard += 1
        changed = False
        # Prefix.
        for k in range(len(own_core), 1, -1):
            if _prefix_matches(name_up, own_core[:k]):
                name_up = name_up[k:]
                changed = True
                break
        if changed:
            continue
        # Suffix.
        for k in range(len(own_core), 1, -1):
            if _suffix_matches(name_up, own_core[:k]):
                name_up = name_up[: -k]
                changed = True
                break
        if changed:
            continue
        # Mid-string span (own-party between two entity fragments, e.g.
        # "PREMIER INTEGRATED L KLINIK DRS YOUNG NEWTON PREMIER INTEGRATED LABS").
        # Prefer the longer surrounding fragment as the true entity.
        span = _mid_match_span(name_up, own_core)
        if span:
            left = name_up[: span[0]]
            right = name_up[span[1] :]
            # Keep the LONGER side by token count; ties go to the right (usually fuller name).
            if len(" ".join(right)) >= len(" ".join(left)):
                name_up = right
            else:
                name_up = left
            changed = True

    cleaned = " ".join(name_up).strip()
    if len(cleaned) < 3:
        return name  # would erase the real counterparty; keep original
    return cleaned


# 2026-05-02 BUG-004 — description-based own-party fallback.
# The existing _strip_own_party_tokens fast-path (line ~4164) only stamps
# (OWN-PARTY) when the EXTRACTED counterparty name equals the holder's core
# tokens exactly. Two failure modes seen on Kay R RHB:
#   (a) Parser fell through to a fallback bucket (UNNAMED RHB TRANSFER (CR))
#       so the holder's name never reached the strip pass — but the raw
#       description ("INWARD IBG ... KAY R RESOURCES SETTLEMENT") clearly
#       names the holder.
#   (b) Parser extracted only one token ("RESOURCES" from a row whose
#       description was "KAY R RESOURCES (M) SDN. BHD. KAY R RESOURCES
#       PINDAHAN") — too short for the prefix/suffix matchers.
# Solution: scan the description directly for ≥2 distinctive holder tokens
# covering ≥50% of the holder's core. When the test passes, override the
# extracted name with "<holder> (OWN-PARTY)" so all three cases bucket
# together. Conservative: requires ≥2 tokens AND ≥50% coverage, so a stray
# "RESOURCES" token alone in a third-party description does NOT trigger.
_OWN_PARTY_DESC_TOKEN_RE = re.compile(r"[A-Z0-9]+")


def _own_party_core_tokens(own_party: str) -> List[str]:
    """Return holder's distinctive (non-boilerplate) tokens, uppercased."""
    if not own_party:
        return []
    return [
        t for t in str(own_party).upper().split()
        if t and t not in _OWN_PARTY_BOILER_SUFFIX
    ]


def _description_implies_own_party(desc: str, own_party: str) -> bool:
    """Return True iff at least 2 holder core tokens (and ≥50% of them)
    appear in the description text. Used to detect own-party rows that the
    parser missed because of fallback-bucket routing or partial extraction.
    """
    own_core = _own_party_core_tokens(own_party)
    if len(own_core) < 2:
        return False
    desc_tokens = set(_OWN_PARTY_DESC_TOKEN_RE.findall(str(desc).upper()))
    if not desc_tokens:
        return False
    matched = sum(1 for t in own_core if t in desc_tokens)
    return matched >= 2 and matched / len(own_core) >= 0.5


def _normalise_counterparty(name: str) -> str:
    """CP11 normalisation: uppercase, strip legal suffixes, collapse spaces, merge known variants.

    Conservative — does NOT do fragment/prefix merging (spec CP11: wrong
    normalisation is worse than duplicates; RP5/AI handles truncation).
    """
    if not name:
        return "UNIDENTIFIED"
    n = name.upper().strip()
    n = re.sub(r"[.,;:]", " ", n)
    # Strip leading purpose prefixes (PAYM/PAYMENT/SI = "sila isi" reference marker)
    n = re.sub(r"^(?:PAYM|PAYMENT|SI)\s+", "", n).strip()
    # PRESERVE legal entity suffixes (analyst directive 2026-06-07): never
    # delete SDN BHD / BHD / LTD. The suffix distinguishes a registered
    # company from an individual and is load-bearing for the C26/C27
    # trade-income/expense dispatch — wiping it reclassifies real trade
    # flows as unclassified and lets one-way foreign vendors be mistaken
    # for related parties. Truncated / abbreviated forms (SB, SDN, SDN B)
    # are EXPANDED to the full canonical "SDN BHD", not stripped.
    n = normalize_company_suffix(n)              # SB / SDN / SDN B / SDN BH → SDN BHD
    n = re.sub(r"\bBERHAD\b", "BHD", n)          # public BERHAD → BHD
    n = re.sub(r"\bBER\b\.?(?=\s|$)", "BHD", n)  # truncated BERHAD → BHD
    # Non-entity location noise can still be normalised away.
    n = re.sub(r"\b(?:MAL|\(M\)|& CO)\b\.?", " ", n)
    # Strip parenthesised location/type suffixes (truncated or complete).
    # \b after the alternation prevents `(M` from matching inside `(MAYBANK`
    # (e.g. "BA SETTLEMENT (MAYBANK ISLAMIC)") — the single-letter M token only
    # matches when followed by a non-word boundary (`)`, whitespace, EOS).
    n = re.sub(r"\((?:SARAWAK|SABAH|MALAYSI[A]?|SAR|L|M)\b\)?", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    # Strip bare trailing truncated-suffix tokens (BH, SD, B, M, & as LAST token only)
    while True:
        m2 = re.match(r"^(.*?)(?<=\s)(?:BH|SD|B|M|&|MALA|MALAY)\s*$", n)
        if not m2:
            break
        n = m2.group(1).strip()
        if not n:
            break

    # Hardcoded variant merges (per spec)
    if "PLANWORTH" in n:
        return "PLANWORTH GLOBAL"
    if n == "JANM" or n.startswith("JANM ") or " JANM" in f" {n} " or "JANM CAWANGAN" in n:
        return "JANM"

    # Noise / purpose-only names → UNCATEGORIZED bucket
    if n in _CP_NOISE_NAMES or len(n) < 3:
        return "UNCATEGORIZED"

    return n or "UNIDENTIFIED"


def _cp_merge_key(name: str) -> str:
    """Aggressive key for M1 exact-normalised match: strip all non-alnum, collapse BIN/BINTI noise."""
    if not name:
        return ""
    k = name.upper()
    k = re.sub(r"\b(?:SDN|BHD|PTY|LTD|MAL|BER|\(M\)|& CO)\b\.?", " ", k)
    k = re.sub(r"[^A-Z0-9 ]+", " ", k)
    k = re.sub(r"\s+", " ", k).strip()
    return k


def _cp_tokens(name: str) -> List[str]:
    return [t for t in re.split(r"\s+", _cp_merge_key(name)) if t]


def _cp_score(group: dict) -> Tuple[int, float]:
    """Prefer the group with more transactions then higher turnover as survivor."""
    txns = group.get("credit_count", 0) + group.get("debit_count", 0)
    turnover = group.get("total_credits", 0.0) + group.get("total_debits", 0.0)
    return (txns, turnover)


def _cp_absorb(dst: dict, src: dict) -> None:
    dst["total_credits"] += src["total_credits"]
    dst["total_debits"] += src["total_debits"]
    dst["credit_count"] += src["credit_count"]
    dst["debit_count"] += src["debit_count"]
    dst["transactions"].extend(src["transactions"])


def _merge_counterparty_groups(groups: Dict[str, dict]) -> Dict[str, dict]:
    """Iterative M1–M5 merge. Runs until a pass produces no merges.

    Preserves total credit/debit invariants — every transaction stays assigned
    to exactly one counterparty; only the key changes.
    """
    # Never touch these synthetic buckets.
    PROTECTED = {
        "UNIDENTIFIED", "UNIDENTIFIED (CHEQUE)", "CASH DEPOSIT", "CASH WITHDRAWAL",
        "RETURNED CHEQUE", "INWARD RETURN", "REVERSAL", "FD/INTEREST", "BANK FEES",
        "BULK SALARY", "JANM", "PLANWORTH GLOBAL", "UNCATEGORIZED",
    }

    def _is_protected(name: str) -> bool:
        return name.upper() in PROTECTED

    for _iteration in range(5):
        merged_any = False
        names = list(groups.keys())

        # M1 — exact normalised-key match
        by_key: Dict[str, List[str]] = {}
        for n in names:
            if _is_protected(n):
                continue
            k = _cp_merge_key(n)
            if not k:
                continue
            by_key.setdefault(k, []).append(n)
        for k, variants in by_key.items():
            if len(variants) < 2:
                continue
            variants.sort(key=lambda nm: _cp_score(groups[nm]), reverse=True)
            survivor = variants[0]
            for loser in variants[1:]:
                if loser in groups and survivor in groups and loser != survivor:
                    _cp_absorb(groups[survivor], groups[loser])
                    del groups[loser]
                    merged_any = True

        # Rebuild working name list.
        names = [n for n in groups.keys() if not _is_protected(n)]
        keys = {n: _cp_merge_key(n) for n in names}
        toks = {n: _cp_tokens(n) for n in names}

        # Index for fast lookups
        names_sorted = sorted(names, key=lambda nm: len(keys[nm]), reverse=True)

        # M2 — prefix match (shorter is ≥10 chars AND full prefix of longer's key)
        for short in list(names):
            if short not in groups:
                continue
            sk = keys.get(short, "")
            if len(sk) < 10:
                continue
            for long in names_sorted:
                if long == short or long not in groups or short not in groups:
                    continue
                lk = keys.get(long, "")
                if len(lk) <= len(sk):
                    continue
                # Prefix must end on a word boundary in the longer key
                if lk.startswith(sk) and (len(lk) == len(sk) or lk[len(sk)] == " "):
                    survivor, loser = (long, short) if _cp_score(groups[long]) >= _cp_score(groups[short]) else (short, long)
                    _cp_absorb(groups[survivor], groups[loser])
                    del groups[loser]
                    merged_any = True
                    break

        # Refresh
        names = [n for n in groups.keys() if not _is_protected(n)]
        toks = {n: _cp_tokens(n) for n in names}

        # M3 — BIN/BINTI truncation: first name matches, and truncated surname (3-4 chars)
        # is a prefix of the fuller surname at the same token position.
        def _bin_split(tok_list: List[str]) -> Optional[Tuple[List[str], str, List[str]]]:
            for i, t in enumerate(tok_list):
                if t in ("BIN", "BINTI", "BT", "BTE"):
                    return tok_list[:i], t, tok_list[i + 1:]
            return None

        name_list = list(names)
        for i, a in enumerate(name_list):
            if a not in groups:
                continue
            pa = _bin_split(toks.get(a, []))
            if not pa:
                continue
            for b in name_list[i + 1:]:
                if b not in groups or a not in groups:
                    continue
                pb = _bin_split(toks.get(b, []))
                if not pb:
                    continue
                fa, _, sa = pa
                fb, _, sb = pb
                if fa != fb or not sa or not sb:
                    continue
                # First surname token prefix check (3-4 char truncation)
                x, y = sa[0], sb[0]
                if x == y:
                    continue
                short_tok, long_tok = (x, y) if len(x) < len(y) else (y, x)
                if 3 <= len(short_tok) <= 4 and long_tok.startswith(short_tok):
                    survivor, loser = (a, b) if _cp_score(groups[a]) >= _cp_score(groups[b]) else (b, a)
                    _cp_absorb(groups[survivor], groups[loser])
                    del groups[loser]
                    merged_any = True

        # Refresh
        names = [n for n in groups.keys() if not _is_protected(n)]
        toks = {n: _cp_tokens(n) for n in names}

        # M4 — CCARD FIRSTNAME SURNAME → BIN/BINTI form with same firstname+surname-start
        for a in list(names):
            if a not in groups:
                continue
            ta = toks.get(a, [])
            if len(ta) != 2:
                continue  # FIRSTNAME SURNAME form
            first_a, surn_a = ta
            for b in list(names):
                if b == a or b not in groups or a not in groups:
                    continue
                pb = _bin_split(toks.get(b, []))
                if not pb:
                    continue
                fb, _, sb = pb
                if not fb or fb[0] != first_a or not sb:
                    continue
                # Either surname-start matches or is a prefix of the other (truncation)
                sa_full = surn_a
                sb_full = sb[0]
                if sa_full == sb_full or sa_full.startswith(sb_full) or sb_full.startswith(sa_full):
                    if 3 <= min(len(sa_full), len(sb_full)):
                        survivor, loser = (a, b) if _cp_score(groups[a]) >= _cp_score(groups[b]) else (b, a)
                        _cp_absorb(groups[survivor], groups[loser])
                        del groups[loser]
                        merged_any = True

        # Refresh
        names = [n for n in groups.keys() if not _is_protected(n)]
        keys = {n: _cp_merge_key(n) for n in names}

        # M5 — company fragment: shorter complete entity (≥2 tokens, ≥2 txns) whose
        # full key appears as a contiguous token run inside a longer name's key.
        for short in list(names):
            if short not in groups:
                continue
            sk = keys.get(short, "")
            stoks = sk.split()
            if len(stoks) < 2:
                continue
            g_short = groups[short]
            if g_short["credit_count"] + g_short["debit_count"] < 2:
                continue
            for long in list(names):
                if long == short or long not in groups or short not in groups:
                    continue
                lk = keys.get(long, "")
                ltoks = lk.split()
                if len(ltoks) <= len(stoks):
                    continue
                # Contiguous token-run match
                found = False
                for i in range(0, len(ltoks) - len(stoks) + 1):
                    if ltoks[i:i + len(stoks)] == stoks:
                        found = True
                        break
                if not found:
                    continue
                survivor, loser = (long, short) if _cp_score(groups[long]) >= _cp_score(groups[short]) else (short, long)
                _cp_absorb(groups[survivor], groups[loser])
                del groups[loser]
                merged_any = True

        # Refresh
        names = [n for n in groups.keys() if not _is_protected(n)]
        toks = {n: _cp_tokens(n) for n in names}

        # M6 — trailing-token truncation: same number of tokens, identical prefix
        # tokens 0..n-2, and last tokens are prefix-related.
        # Catches: FATHIN SYAIRAH NAJL ↔ NAJLA, SUPREME LANDMOB ↔ LANDMOBILE,
        # PDIGITALS COMMU ↔ COMMUNICAT, DAMINA SECURITY D ↔ DE.
        name_list = list(names)
        for i, a in enumerate(name_list):
            if a not in groups:
                continue
            ta = toks.get(a, [])
            if len(ta) < 2:
                continue
            for b in name_list[i + 1:]:
                if b not in groups or a not in groups:
                    continue
                tb = toks.get(b, [])
                if len(tb) != len(ta):
                    continue
                if ta[:-1] != tb[:-1]:
                    continue
                xa, xb = ta[-1], tb[-1]
                if xa == xb:
                    continue
                short_tok, long_tok = (xa, xb) if len(xa) < len(xb) else (xb, xa)
                if not long_tok.startswith(short_tok):
                    continue
                # Short must be ≥3 chars, OR ≥1 char when we have additional context tokens (n≥3)
                if len(short_tok) < 3 and len(ta) < 3:
                    continue
                survivor, loser = (a, b) if _cp_score(groups[a]) >= _cp_score(groups[b]) else (b, a)
                _cp_absorb(groups[survivor], groups[loser])
                del groups[loser]
                merged_any = True

        if not merged_any:
            break

    return groups


def build_counterparty_ledger(transactions: List[dict]) -> dict:
    """Group extracted transactions by counterparty.

    Every transaction with a non-zero credit or debit lands in exactly one
    counterparty group. Balance-only rows are skipped. Returns the
    `counterparty_ledger` section per CHANGE 6 spec.
    """
    empty = {
        "version": "1.0",
        "total_counterparties": 0,
        "extraction_stats": {
            "pattern_matched": 0,
            "special_bucket": 0,
            "raw_fallback": 0,
            "total_transactions": 0,
        },
        "counterparties": [],
    }
    if not transactions:
        return empty

    groups: Dict[str, dict] = {}
    stats = {"pattern": 0, "special": 0, "raw": 0}

    for t in transactions:
        desc = str(t.get("description") or "").strip()
        debit = safe_float(t.get("debit")) or 0.0
        credit = safe_float(t.get("credit")) or 0.0

        if debit == 0 and credit == 0:
            continue

        raw_name, method = _extract_counterparty(
            desc, t.get("bank"),
            amount=max(debit, credit),
            direction=("DR" if debit > 0 else "CR"),
            own_party=t.get("own_party_name"),
        )
        # BUG-003 (2026-05-02): restore truncated "SDN BHD" tails (SB / SDN B /
        # SDN BH / SDN BHD.) before bucketing, so column-clipped variants merge
        # with their full-suffix counterparts.
        raw_name = normalize_company_suffix(raw_name)
        norm_name = _normalise_counterparty(raw_name)
        # Sprint 6 #6: when the extractor fell back to raw (no pattern/special bucket
        # matched), purge purpose-word-only and month-only names into UNCATEGORIZED
        # so top_parties isn't polluted by ghost-verb residuals (DEBIT ADVICE,
        # INSTANT TRANSFER, etc.). Special/pattern buckets are NOT touched because
        # their labels (CASH DEPOSIT, BULK SALARY, LOAN REPAYMENT) are intentionally
        # composed of stop-word tokens.
        if method == "raw" and should_drop_as_counterparty(norm_name):
            norm_name = "UNCATEGORIZED"
        # Sprint 6 #10: strip statement-holder's own company name from the
        # extracted counterparty (Alliance leaks "KLINIK DRS YOUNG NEW" /
        # "BESTLITE ELECTRICAL" suffixes into real counterparties because the
        # PDF column-width truncation smuggles the own name into continuation
        # lines). Parser stamps own_party_name per-row; protected synthetic
        # labels (UNIDENTIFIED / BULK SALARY / ...) are NOT stripped.
        op = t.get("own_party_name")
        if (
            op
            and norm_name not in _OWN_PARTY_PROTECTED_LABELS
            and not norm_name.startswith("UNIDENTIFIED")
            and not norm_name.startswith("UNNAMED")
            and "(OWN-PARTY)" not in norm_name
        ):
            stripped = _strip_own_party_tokens(norm_name, str(op))
            if stripped and stripped != norm_name:
                norm_name = _normalise_counterparty(stripped)
        # BUG-004 (2026-05-02): description-based own-party fallback.
        # Catches the cases the prefix/suffix strip pass misses — most often
        # rows that fell through to the UNNAMED <bank> TRANSFER fallback
        # bucket OR rows where the parser extracted a single tail token
        # ("RESOURCES") from a description that nonetheless names the holder
        # in full.
        #
        # Facility-memo guard (2026-06-10): a third-party loan / facility
        # payment routed through a financier or trustee commonly echoes the
        # borrower's own name in the memo (e.g. "TRANSFER FR A/C MALAYSIAN
        # TRUSTEES * Funding Societes Zaim Express Sdn Bhd"): the holder tokens
        # are present, but the real counterparty (MALAYSIAN TRUSTEES) was
        # correctly extracted and must NOT be clobbered to OWN-PARTY — that hid
        # the loan repayment as an own-account transfer. When the description
        # carries an explicit loan / facility keyword, skip the description-
        # based own-party override and keep the extracted counterparty so the
        # row reaches the C10/C11 rung. Narrow on purpose: genuine own-account
        # transfers (incl. multi-word fallback labels like "INTER ACC TXN OWN
        # ACC TXN") never carry these keywords, so they stay stamped.
        _facility_memo = bool(
            _LOAN_DISBURSEMENT_RE.search(desc) or _LOAN_REPAYMENT_RE.search(desc)
        )
        if (
            op
            and "(OWN-PARTY)" not in norm_name
            and not _facility_memo
            and _description_implies_own_party(desc, str(op))
        ):
            holder_label = " ".join(_own_party_core_tokens(op))
            if holder_label:
                norm_name = f"{holder_label} (OWN-PARTY)"
        stats[method] = stats.get(method, 0) + 1

        g = groups.setdefault(
            norm_name,
            {
                "counterparty_name": norm_name,
                "total_credits": 0.0,
                "total_debits": 0.0,
                "credit_count": 0,
                "debit_count": 0,
                "transactions": [],
            },
        )

        if credit > 0:
            g["total_credits"] += credit
            g["credit_count"] += 1
            amount, ttype = credit, "CREDIT"
        else:
            g["total_debits"] += debit
            g["debit_count"] += 1
            amount, ttype = debit, "DEBIT"

        g["transactions"].append(
            {
                "date": t.get("date"),
                "description": desc,
                "amount": round(float(amount), 2),
                "type": ttype,
                "balance": safe_float(t.get("balance")),
                "bank": t.get("bank"),
                "account_no": t.get("account_no"),
                "source_file": t.get("source_file"),
                "extraction_method": method,
            }
        )

    # ── M1–M5 merge pass ────────────────────────────────────────────────────
    groups = _merge_counterparty_groups(groups)

    counterparties = []
    for g in groups.values():
        g["total_credits"] = round(g["total_credits"], 2)
        g["total_debits"] = round(g["total_debits"], 2)
        g["net_position"] = round(g["total_credits"] - g["total_debits"], 2)
        g["transaction_count"] = g["credit_count"] + g["debit_count"]
        g["transactions"].sort(key=lambda x: str(x.get("date") or ""))
        counterparties.append(g)

    counterparties.sort(
        key=lambda x: x["total_credits"] + x["total_debits"], reverse=True
    )

    return {
        "version": "1.0",
        "total_counterparties": len(counterparties),
        "extraction_stats": {
            "pattern_matched": stats["pattern"],
            "special_bucket": stats["special"],
            "raw_fallback": stats["raw"],
            "total_transactions": stats["pattern"] + stats["special"] + stats["raw"],
        },
        "counterparties": counterparties,
    }


# ---------------------------------------------------
# DISPLAY
# ---------------------------------------------------
if st.session_state.results or (bank_choice == "Affin Bank" and st.session_state.affin_statement_totals) or (
    bank_choice == "Ambank" and st.session_state.ambank_statement_totals
) or (bank_choice == "CIMB Bank" and st.session_state.cimb_statement_totals) or (
    bank_choice == "RHB Bank" and st.session_state.rhb_statement_totals
):
    # ✅ PDF Integrity Report (expandable per file)
    if st.session_state.pdf_integrity_results:
        st.subheader("🛡️ PDF Integrity Report")

        # Overall summary
        total_files = len(st.session_state.pdf_integrity_results)
        high_risk_files = [
            fname for fname, r in st.session_state.pdf_integrity_results.items()
            if r.get("overall_risk") == "HIGH"
        ]
        med_risk_files = [
            fname for fname, r in st.session_state.pdf_integrity_results.items()
            if r.get("overall_risk") == "MEDIUM"
        ]
        clean_files = total_files - len(high_risk_files) - len(med_risk_files)

        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Files Scanned", total_files)
        with summary_cols[1]:
            st.metric("🟢 Clean", clean_files)
        with summary_cols[2]:
            st.metric("🟡 Medium Risk", len(med_risk_files))
        with summary_cols[3]:
            st.metric("🔴 High Risk", len(high_risk_files))

        for fname, result in st.session_state.pdf_integrity_results.items():
            risk = result.get("overall_risk", "LOW")
            risk_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk, "⚪")
            high_c = result.get("high_count", 0)
            med_c = result.get("medium_count", 0)
            low_c = result.get("low_count", 0)

            with st.expander(f"{risk_icon} {fname} — Risk: **{risk}** ({high_c}H / {med_c}M / {low_c}L)"):
                for layer_name, layer_findings in result.get("layer_results", {}).items():
                    if not layer_findings:
                        continue
                    # Skip pure-informational entries (render hashes with LOW severity only)
                    actionable = [f for f in layer_findings if f.get("severity") != "LOW" or "disagreement" in f.get("message", "").lower()]
                    display_findings = actionable if actionable else layer_findings

                    layer_label = layer_name.replace("_", " ").title()
                    st.markdown(f"**Layer: {layer_label}**")
                    for f in display_findings:
                        sev = f.get("severity", "LOW")
                        sev_badge = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "ℹ️"}.get(sev, "")
                        st.markdown(f"- {sev_badge} **{sev}**: {f.get('message', '')}")

                        detail = f.get("detail")
                        if detail and isinstance(detail, dict):
                            # Show anomalous amounts detail if present
                            if "anomalous_amounts" in detail:
                                st.markdown("  Suspicious amounts:")
                                for amt in detail["anomalous_amounts"]:
                                    st.markdown(
                                        f"  - Page {amt['page']}: `{amt['text']}` "
                                        f"(font: {amt['font']}, size: {amt['size']})"
                                    )

        st.markdown("---")

    df = pd.DataFrame(st.session_state.results) if st.session_state.results else pd.DataFrame()

    # Pre-compute counterparty ledger once (reused by display + downloads)
    _ledger_txns_pre = df.to_dict(orient="records") if not df.empty else []
    counterparty_ledger = build_counterparty_ledger(_ledger_txns_pre)

    # ── Top-level KPI metrics ───────────────────────────────────────────────
    if not df.empty:
        _total_credit = float(pd.to_numeric(df.get("credit", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        _total_debit = float(pd.to_numeric(df.get("debit", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        _date_min = df["date"].min() if "date" in df.columns else None
        _date_max = df["date"].max() if "date" in df.columns else None

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Transactions", f"{len(df):,}")
        k2.metric("Total Credits", f"{_total_credit:,.2f}")
        k3.metric("Total Debits", f"{_total_debit:,.2f}")
        k4.metric("Net", f"{(_total_credit - _total_debit):,.2f}")
        k5.metric("Counterparties", counterparty_ledger.get("total_counterparties", 0))
        k6.metric("Date Range", f"{_date_min} → {_date_max}" if _date_min and _date_max else "—")

    st.subheader("📊 Extracted Transactions")

    if not df.empty:
        display_cols = [
            "date",
            "description",
            "debit",
            "credit",
            "balance",
            "company_name",
            "account_no",
            "page",
            "seq",
            "bank",
            "source_file",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        _num_fmt_txn = {c: "{:,.2f}" for c in ("debit", "credit", "balance") if c in display_cols}
        st.dataframe(df[display_cols].style.format(_num_fmt_txn, na_rep=""), use_container_width=True)
    else:
        st.info("No line-item transactions extracted.")

    monthly_summary_raw = calculate_monthly_summary(st.session_state.results)
    monthly_summary = present_monthly_summary_standard(monthly_summary_raw)

    if monthly_summary:
        st.subheader("📅 Monthly Summary (Standardized)")
        summary_df = pd.DataFrame(monthly_summary)
        desired_cols = [
            "month",
            "company_name",
            "account_no",
            "opening_balance",
            "total_debit",
            "total_credit",
            "highest_balance",
            "lowest_balance",
            "swing",
            "ending_balance",
            "source_files",
        ]
        summary_df = summary_df[[c for c in desired_cols if c in summary_df.columns]]
        _num_fmt_ms = {
            c: "{:,.2f}"
            for c in ("opening_balance", "total_debit", "total_credit", "highest_balance", "lowest_balance", "swing", "ending_balance")
            if c in summary_df.columns
        }
        st.dataframe(summary_df.style.format(_num_fmt_ms, na_rep=""), use_container_width=True)

    # ── Counterparty Ledger (on-screen summary) ────────────────────────────
    _cps = counterparty_ledger.get("counterparties", []) if counterparty_ledger else []
    if _cps:
        st.subheader("💼 Counterparty Ledger")
        _stats = counterparty_ledger.get("extraction_stats", {})
        cs1, cs2, cs3, cs4 = st.columns(4)
        cs1.metric("Total Counterparties", f"{counterparty_ledger.get('total_counterparties', 0):,}")
        cs2.metric("Pattern matched", f"{_stats.get('pattern_matched', 0):,}")
        cs3.metric("Special bucket", f"{_stats.get('special_bucket', 0):,}")
        cs4.metric("Raw fallback", f"{_stats.get('raw_fallback', 0):,}")

        _credits_cps = sorted(
            [c for c in _cps if c.get("total_credits", 0) > 0],
            key=lambda c: c["total_credits"],
            reverse=True,
        )
        _debits_cps = sorted(
            [c for c in _cps if c.get("total_debits", 0) > 0],
            key=lambda c: c["total_debits"],
            reverse=True,
        )

        def _ledger_df(cps, amount_key, count_key):
            rows = [
                {
                    "counterparty": c["counterparty_name"],
                    "amount": c[amount_key],
                    "txn_count": c[count_key],
                    "net_position": c["net_position"],
                }
                for c in cps
            ]
            return pd.DataFrame(rows)

        def _render_tx_expander(cps, header, amount_key):
            if not cps:
                return
            with st.expander(f"🔍 {header} — drill-down (top {min(25, len(cps))})"):
                for c in cps[:25]:
                    st.markdown(
                        f"**{c['counterparty_name']}** — "
                        f"Cr {c['total_credits']:,.2f} / Dr {c['total_debits']:,.2f} "
                        f"(net {c['net_position']:,.2f}, {c['transaction_count']:,} txns)"
                    )
                    tx_df = pd.DataFrame(c.get("transactions", []))
                    if not tx_df.empty:
                        show_cols = [col for col in ["date", "description", "type", "amount", "balance", "bank", "source_file"] if col in tx_df.columns]
                        _tx_fmt = {k: "{:,.2f}" for k in ("amount", "balance") if k in show_cols}
                        st.dataframe(
                            tx_df[show_cols].style.format(_tx_fmt, na_rep=""),
                            use_container_width=True,
                            height=min(240, 40 + 32 * len(tx_df)),
                        )
                    st.markdown("---")

        _amt_fmt = {"amount": "{:,.2f}", "net_position": "{:,.2f}"}

        # ── Credits first ──
        st.markdown("### 💰 Credits (money received)")
        if _credits_cps:
            cr_df = _ledger_df(_credits_cps, "total_credits", "credit_count")
            st.dataframe(cr_df.style.format(_amt_fmt, na_rep=""), use_container_width=True, height=400)

            top_cr = cr_df.head(10)
            st.markdown("**🏆 Top 10 Credit Counterparties**")
            st.bar_chart(top_cr.set_index("counterparty")["amount"])
            _render_tx_expander(_credits_cps, "Credits", "total_credits")
        else:
            st.info("No credit transactions.")

        st.markdown("---")

        # ── Debits second ──
        st.markdown("### 💸 Debits (money paid out)")
        if _debits_cps:
            dr_df = _ledger_df(_debits_cps, "total_debits", "debit_count")
            st.dataframe(dr_df.style.format(_amt_fmt, na_rep=""), use_container_width=True, height=400)

            top_dr = dr_df.head(10)
            st.markdown("**🏆 Top 10 Debit Counterparties**")
            st.bar_chart(top_dr.set_index("counterparty")["amount"])
            _render_tx_expander(_debits_cps, "Debits", "total_debits")
        else:
            st.info("No debit transactions.")

    st.subheader("⬇️ Download Options")
    col1, col2, col3 = st.columns(3)

    df_download = df.copy() if not df.empty else pd.DataFrame([])

    def _sanitize_records(records):
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and math.isnan(v):
                    rec[k] = None
        return records

    with col1:
        st.download_button(
            "📄 Download Transactions (JSON)",
            json.dumps(_sanitize_records(df_download.to_dict(orient="records")), indent=4),
            "transactions.json",
            "application/json",
        )

    with col2:
        date_min = df_download["date"].min() if "date" in df_download.columns and not df_download.empty else None
        date_max = df_download["date"].max() if "date" in df_download.columns and not df_download.empty else None

        total_files_processed = None
        if "source_file" in df_download.columns and not df_download.empty:
            total_files_processed = int(df_download["source_file"].nunique())
        else:
            if bank_choice == "Affin Bank":
                total_files_processed = len(st.session_state.affin_statement_totals)
            elif bank_choice == "Ambank":
                total_files_processed = len(st.session_state.ambank_statement_totals)
            elif bank_choice == "CIMB Bank":
                total_files_processed = len(st.session_state.cimb_statement_totals)
            elif bank_choice == "RHB Bank":
                total_files_processed = len(st.session_state.rhb_statement_totals)

        company_names = sorted(
            {x for x in df_download.get("company_name", pd.Series([], dtype=object)).dropna().astype(str).tolist() if x.strip()}
        )

        account_nos = sorted(
            {x for x in df_download.get("account_no", pd.Series([], dtype=object)).dropna().astype(str).tolist() if x.strip()}
        )

        # ✅ Build pdf_integrity section for the report
        _integrity_for_report = {}
        if st.session_state.pdf_integrity_results:
            for _fname, _res in st.session_state.pdf_integrity_results.items():
                _integrity_for_report[_fname] = {
                    "overall_risk": _res.get("overall_risk"),
                    "finding_count": _res.get("finding_count"),
                    "high_count": _res.get("high_count"),
                    "medium_count": _res.get("medium_count"),
                    "low_count": _res.get("low_count"),
                    "findings": [
                        {
                            "layer": f.get("layer"),
                            "severity": f.get("severity"),
                            "message": f.get("message"),
                            "detail": f.get("detail"),
                        }
                        for f in _res.get("all_findings", [])
                    ],
                }

        full_report = {
            "summary": {
                "total_transactions": int(len(df_download)),
                "date_range": f"{date_min} to {date_max}" if date_min and date_max else None,
                "total_files_processed": total_files_processed,
                "company_names": company_names,
                "account_nos": account_nos,
                "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "pdf_integrity": _integrity_for_report,
            "account_type_determinations": list(st.session_state.account_type_determinations),
            "monthly_summary": monthly_summary,
            "counterparty_ledger": counterparty_ledger,
            "transactions": _sanitize_records(df_download.to_dict(orient="records")),
        }

        st.download_button(
            "📊 Download Full Report (JSON)",
            json.dumps(full_report, indent=4),
            "full_report.json",
            "application/json",
        )

    with col3:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_download.to_excel(writer, sheet_name="Transactions", index=False)
            if monthly_summary:
                pd.DataFrame(monthly_summary).to_excel(writer, sheet_name="Monthly Summary", index=False)

        st.download_button(
            "📊 Download Full Report (XLSX)",
            output.getvalue(),
            "full_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if _USE_TRACK_2:
        st.markdown("---")
        st.caption(
            "🔬 Track 2 deterministic classifier (USE_TRACK_2=1). "
            "Produces the v6.3.5-schema engine output the claude.ai "
            "pre-analysis template consumes."
        )
        try:
            t2_account_meta = _track2_account_meta(
                st.session_state.account_type_determinations
            )
            t2_result = _build_track2_result(
                transactions=_sanitize_records(
                    df_download.to_dict(orient="records")
                ),
                counterparty_ledger=counterparty_ledger,
                pdf_integrity=_integrity_for_report,
                company_names=company_names,
                related_parties=[],
                factoring_entities=[],
                account_meta=t2_account_meta,
            )
            t2_ok, t2_errors = _validate_track2_result(t2_result)
            if not t2_ok:
                st.warning(
                    "Track 2 schema validation failed — first errors: "
                    f"{t2_errors[:3]}"
                )
            st.download_button(
                "🔬 Download Track 2 Analysis (JSON)",
                json.dumps(t2_result, indent=4),
                "track2_analysis.json",
                "application/json",
            )
            # === TRACK 2 RENDERER INTEGRATION (added — remove block to revert) ===
            # Two-path render of the v6.3.5 result into the analyst deliverable.
            # JSON is intentionally NOT re-offered here — the button above is the
            # single source for the engine JSON (it doubles as the claude.ai
            # input), so we only add HTML + Excel to avoid duplicate outputs.
            if _t2_render is not None:
                _t2_company = (
                    (t2_result.get("report_info") or {}).get("company_name")
                    or "report"
                )
                _t2_safe = (
                    _t2_company.replace(" ", "_").replace("/", "_").replace(".", "")
                )[:60] or "report"
                _tab_direct, _tab_ai = st.tabs(
                    ["📑 Direct → Report", "🤖 AI Analysis → Report"]
                )
                with _tab_direct:
                    st.caption(
                        "Render the engine result above straight to the analyst "
                        "deliverable — no AI round-trip. Observations are "
                        "engine-authored."
                    )
                    try:
                        # Render on a normalised COPY so the engine JSON download
                        # above stays canonical. normalize_claude_v635 rescales the
                        # 0-1 decimal overall_success_rate to a percent for display
                        # (same step the standalone renderer applies on upload).
                        _t2_view = _t2_render.normalize_claude_v635(
                            json.loads(json.dumps(t2_result))
                        )
                        _html = _t2_render.generate_interactive_html(_t2_view)
                        st.download_button(
                            "🌐 Analyst Report (HTML)",
                            _html.encode("utf-8")
                            if isinstance(_html, str)
                            else _html,
                            f"{_t2_safe}_v6_report.html",
                            "text/html; charset=utf-8",
                            key="t2_direct_html_dl",
                        )
                        _xls = _t2_render.generate_excel(_t2_view)
                        if _xls is not None:
                            st.download_button(
                                "📊 Analyst Report (Excel)",
                                _xls.getvalue()
                                if hasattr(_xls, "getvalue")
                                else _xls,
                                f"{_t2_safe}_v6_analysis.xlsx",
                                "application/vnd.openxmlformats-officedocument"
                                ".spreadsheetml.sheet",
                                key="t2_direct_xls_dl",
                            )
                        else:
                            st.info("Install `openpyxl` for the Excel deliverable.")
                    except Exception as _direct_exc:
                        st.error(f"Direct render failed: {_direct_exc}")
                with _tab_ai:
                    st.caption(
                        "Already have the claude.ai-enriched analysis JSON? Upload "
                        "it to render the same HTML/Excel deliverable (adds the AI "
                        "narrative + related-party/classification layer)."
                    )
                    _ai_json = st.file_uploader(
                        "Upload AI analysis JSON (v6.3.x)",
                        type=["json"],
                        key="t2_ai_json_upload",
                    )
                    if _ai_json is not None:
                        try:
                            _ai_data = json.load(_ai_json)
                            if (
                                isinstance(_ai_data, dict)
                                and "monthly_analysis" in _ai_data
                                and (
                                    "consolidated" not in _ai_data
                                    or "top_parties" not in _ai_data
                                )
                            ):
                                _ai_data = _t2_render.normalize_claude_v633(_ai_data)
                            if isinstance(_ai_data, dict) and (
                                _ai_data.get("report_info") or {}
                            ).get("schema_version", "") in ("6.3.4", "6.3.5"):
                                _ai_data = _t2_render.normalize_claude_v635(_ai_data)
                            _ai_company = (
                                (_ai_data.get("report_info") or {}).get(
                                    "company_name"
                                )
                                or "report"
                            )
                            _ai_safe = (
                                _ai_company.replace(" ", "_")
                                .replace("/", "_")
                                .replace(".", "")
                            )[:60] or "report"
                            _ai_html = _t2_render.generate_interactive_html(_ai_data)
                            st.download_button(
                                "🌐 Analyst Report (HTML)",
                                _ai_html.encode("utf-8")
                                if isinstance(_ai_html, str)
                                else _ai_html,
                                f"{_ai_safe}_v6_report.html",
                                "text/html; charset=utf-8",
                                key="t2_ai_html_dl",
                            )
                            _ai_xls = _t2_render.generate_excel(_ai_data)
                            if _ai_xls is not None:
                                st.download_button(
                                    "📊 Analyst Report (Excel)",
                                    _ai_xls.getvalue()
                                    if hasattr(_ai_xls, "getvalue")
                                    else _ai_xls,
                                    f"{_ai_safe}_v6_analysis.xlsx",
                                    "application/vnd.openxmlformats-officedocument"
                                    ".spreadsheetml.sheet",
                                    key="t2_ai_xls_dl",
                                )
                        except Exception as _ai_exc:
                            st.error(f"AI-JSON render failed: {_ai_exc}")
            # === END TRACK 2 RENDERER INTEGRATION ===
        except Exception as t2_exc:
            st.error(f"Track 2 engine failed: {t2_exc}")
            st.exception(t2_exc)

else:
    if uploaded_files:
        st.warning("⚠️ No transactions found — click **Start Processing**.")


# === MULTI-BANK COMBINE (CWS 2026-06) ====================================
# A customer with accounts at several banks needs ONE report. Each app run
# parses one bank format and exports a full_report.json; this section merges
# 2+ of those exports into one combined report, runs the Track 2 engine once
# over all accounts, and renders the single analyst HTML/Excel deliverable.
# Mirrors scripts/merge_full_reports.py (same merge functions).
st.markdown("---")
st.subheader("🔗 Combine Multi-Bank Reports")
st.caption(
    "One customer, several banks? Run the parser once per bank format and "
    "download each **Full Report (JSON)**. Then upload those exports here to "
    "merge them into ONE combined report — single Track 2 engine JSON (for "
    "the claude.ai template) plus the combined analyst HTML / Excel."
)
_merge_uploads = st.file_uploader(
    "Upload 2 or more full_report.json exports (one per bank run)",
    type=["json"],
    accept_multiple_files=True,
    key="multibank_merge_upload",
)
if _merge_uploads and len(_merge_uploads) < 2:
    st.info("Upload at least two full_report.json files to combine.")
elif _merge_uploads:
    try:
        from scripts.merge_full_reports import (
            merge_reports as _mb_merge,
            _dominant_bank as _mb_bank_label,
        )

        _mb_reports = []
        _mb_bad = False
        for _up in _merge_uploads:
            _rep = json.load(_up)
            if not isinstance(_rep, dict) or "transactions" not in _rep:
                st.error(
                    f"**{_up.name}** is not a full_report.json export "
                    "(no 'transactions' key) — did you upload the Track 2 "
                    "analysis JSON by mistake?"
                )
                _mb_bad = True
            else:
                _mb_reports.append(_rep)
        if not _mb_bad:
            _mb_labels = [_mb_bank_label(r) for r in _mb_reports]
            _mb_seen: Dict[str, int] = {}
            for _i, _lab in enumerate(_mb_labels):
                _mb_seen[_lab] = _mb_seen.get(_lab, 0) + 1
                if _mb_seen[_lab] > 1:
                    _mb_labels[_i] = f"{_lab} #{_mb_seen[_lab]}"

            _mb_merged = _mb_merge(_mb_reports, _mb_labels)
            _mb_accounts = {
                str(t.get("account_no") or t.get("account_number"))
                for t in _mb_merged["transactions"]
            }
            st.success(
                f"Merged {len(_mb_reports)} reports ({', '.join(_mb_labels)}) — "
                f"{_mb_merged['summary']['total_transactions']} transactions "
                f"across {len(_mb_accounts)} accounts."
            )

            _mb_col1, _mb_col2 = st.columns(2)
            with _mb_col1:
                st.download_button(
                    "📊 Merged Full Report (JSON)",
                    json.dumps(_mb_merged, indent=2, default=str),
                    "merged_full_report.json",
                    "application/json",
                    key="mb_merged_dl",
                )

            from kredit_lab_classify_track2 import (
                account_meta_from_determinations as _mb_meta_fn,
                build_track2_result as _mb_build,
                validate_track2_result as _mb_validate,
            )

            _mb_result = _mb_build(
                transactions=_mb_merged["transactions"],
                counterparty_ledger=_mb_merged["counterparty_ledger"],
                pdf_integrity=_mb_merged["pdf_integrity"],
                company_names=_mb_merged["summary"]["company_names"],
                related_parties=[],
                factoring_entities=[],
                account_meta=_mb_meta_fn(
                    _mb_merged["account_type_determinations"]
                ),
            )
            _mb_ok, _mb_errors = _mb_validate(_mb_result)
            if not _mb_ok:
                st.warning(
                    "Track 2 schema validation failed — first errors: "
                    f"{_mb_errors[:3]}"
                )
            with _mb_col2:
                st.download_button(
                    "🔬 Combined Track 2 Analysis (JSON)",
                    json.dumps(_mb_result, indent=4),
                    "track2_analysis.json",
                    "application/json",
                    key="mb_t2_dl",
                )

            try:
                import renderer_core as _mb_render
            except Exception:
                _mb_render = None
            if _mb_render is not None:
                _mb_company = (
                    (_mb_result.get("report_info") or {}).get("company_name")
                    or "report"
                )
                _mb_safe = (
                    _mb_company.replace(" ", "_")
                    .replace("/", "_")
                    .replace(".", "")
                )[:60] or "report"
                _mb_view = _mb_render.normalize_claude_v635(
                    json.loads(json.dumps(_mb_result))
                )
                _mb_html = _mb_render.generate_interactive_html(_mb_view)
                st.download_button(
                    "🌐 Combined Analyst Report (HTML)",
                    _mb_html.encode("utf-8")
                    if isinstance(_mb_html, str)
                    else _mb_html,
                    f"{_mb_safe}_v6_report.html",
                    "text/html; charset=utf-8",
                    key="mb_html_dl",
                )
                _mb_xls = _mb_render.generate_excel(_mb_view)
                if _mb_xls is not None:
                    st.download_button(
                        "📊 Combined Analyst Report (Excel)",
                        _mb_xls.getvalue()
                        if hasattr(_mb_xls, "getvalue")
                        else _mb_xls,
                        f"{_mb_safe}_v6_analysis.xlsx",
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet",
                        key="mb_xls_dl",
                    )
            else:
                st.info(
                    "renderer_core unavailable — HTML/Excel render skipped "
                    "(JSON downloads above still work)."
                )
    except Exception as _mb_exc:
        st.error(f"Merge failed: {_mb_exc}")
        st.exception(_mb_exc)
# === END MULTI-BANK COMBINE ==============================================
