# alliance.py
# Alliance Bank Malaysia Berhad statement parser
#
# Interface matches project convention:
#   parse_transactions_alliance(pdf, filename) -> List[dict]
# where pdf is a pdfplumber.PDF instance (from bytes_to_pdfplumber)

import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from core_utils import finalize_parser_output


_TX_START_RE = re.compile(r"^(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{2})\s+(?P<rest>.+)$")
_MONEY_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*\.\d{2}|\-?\d+\.\d{2}")

_HEADER_SUBSTRS = (
    "STATEMENT OF ACCOUNT",
    "PENYATA AKAUN",
    "PAGE ",
    "HALAMAN ",
    "CURRENT A/C",
    "ACCOUNT NO",
    "NO. AKAUN",
    "CURRENCY",
    "MATAWANG",
    "PROTECTED BY PIDM",
    "DILINDUNGI",
    "CIF NO",
)

# Sprint 6 #10: match an all-caps company-name line in the page-1 header.
# Statement header lines identifying the account holder follow shapes like
# "KLINIK DRS YOUNG NEWTON SDN BHD" or "BESTLITE ELECTRICAL SDN BHD".
# Restrict to UPPERCASE to avoid matching address/CIF/disclaimer wording.
_OWN_PARTY_SUFFIX_RE = re.compile(
    r"\b(?:SDN\s+BHD|SDN\s*BHD\.?|BERHAD|BHD|ENTERPRISE|TRADING|\(M\)\s+SDN\s+BHD)\b",
    re.IGNORECASE,
)
_OWN_PARTY_SKIP_PREFIXES = (
    "STATEMENT", "PENYATA", "CIF", "PAGE", "HALAMAN", "CURRENT", "SAVINGS",
    "ACCOUNT", "NO.", "CURRENCY", "MATAWANG", "PROTECTED", "DILINDUNGI",
    "LEVEL", "NO ", "JLN", "JALAN", "LOT ", "LOT\t", "TAMAN",
    "KUALA", "KL ", "SELANGOR", "JOHOR", "PERAK", "PENANG", "SABAH", "SARAWAK",
    "CAPITAL SQUARE",
)

_STOP_MARKERS = (
    "THE ITEMS AND BALANCES SHOWN ABOVE",
    "SEGALA BUTIRAN DAN BAKI AKAUN",
    "PINJAMAN/PEMBIAYAAN DAN AKTIFKAN SEBARANG AKAUN DORMAN",
    "ALLIANCE BANK MALAYSIA BERHAD",
)

# Defensive post-strip: even if the stop marker is missed at line level
# (e.g. footer wraps onto the same line as a transaction), strip it from
# the joined description.
_FOOTER_STRIP_RE = re.compile(
    r"\s*(?:The items and balances shown above|Segala butiran dan baki akaun"
    r"|PINJAMAN/PEMBIAYAAN DAN AKTIFKAN SEBARANG AKAUN DORMAN).*$",
    re.IGNORECASE | re.DOTALL,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\x00", " ")).strip()


def _is_noise(line: str) -> bool:
    up = _norm(line).upper()
    if not up:
        return True
    if any(k in up for k in _HEADER_SUBSTRS):
        return True
    # common table header
    if (
        "TRANSACTION DETAILS" in up
        and "CHEQUE" in up
        and "DEBIT" in up
        and "CREDIT" in up
        and "BALANCE" in up
    ):
        return True
    if up.startswith("DATE TRANSACTION DETAILS") or up.startswith("TARIKH KETERANGAN"):
        return True
    return False


def _is_stop(line: str) -> bool:
    up = _norm(line).upper()
    return any(m in up for m in _STOP_MARKERS)


def _parse_money_tokens(text: str) -> List[float]:
    out: List[float] = []
    for m in _MONEY_RE.finditer(text):
        try:
            out.append(float(m.group().replace(",", "")))
        except Exception:
            continue
    return out


# Match "<number> CR|DR" as the balance-with-side suffix at the end of a line.
_BAL_SIGN_RE = re.compile(
    r"(-?\d{1,3}(?:,\d{3})*\.\d{2}|\-?\d+\.\d{2})\s+(CR|DR)\s*$",
    re.IGNORECASE,
)


def _balance_side_from_line(line: str) -> Optional[str]:
    """Return 'CR' or 'DR' if the line ends with a balance+side suffix, else None."""
    m = _BAL_SIGN_RE.search(line)
    if m:
        return m.group(2).upper()
    return None


def _iso_from_ddmmyy(dd: str, mm: str, yy: str) -> Optional[str]:
    y = 2000 + int(yy)
    # Sprint 6 #8: defence-in-depth date-clamp. The Alliance row detector keys
    # off a 6-digit DDMMYY prefix; OCR occasionally emits ref codes (e.g.
    # "272666 PV25-154309-A") whose first 6 digits happen to form a parseable
    # datetime (d=27, m=26 was caught by ValueError, but d=27, m=6, y=66 would
    # slip through as 2066-06-27). The BUG-001 ghost-row filter catches most of
    # these via (debit=0 ∧ credit=0 ∧ balance=None), but rejecting implausible
    # years here closes the hole before a leak reaches the ghost-row check.
    if y < 2010 or y > datetime.now().year + 1:
        return None
    try:
        dt = datetime(y, int(mm), int(dd))
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d")


def _extract_own_party_name(header_text: str) -> Optional[str]:
    """Sprint 6 #10 — pull the account-holder company name from the page-1
    header text. Used downstream by the counterparty extractor to strip
    own-name tokens from extracted counterparty names (e.g. Bestlite's
    'PYM FPT ENGINEERING BESTLITE ELECTRICAL' -> 'PYM FPT ENGINEERING').

    Conservative: returns the FIRST uppercase header line that (a) is not a
    known header-metadata prefix, (b) contains a company suffix (SDN BHD /
    BERHAD / ENTERPRISE / TRADING). Returns None if no candidate matches.
    """
    if not header_text:
        return None
    for raw_line in header_text.splitlines():
        line = _norm(raw_line)
        if not line or len(line) < 6:
            continue
        up = line.upper()
        if any(up.startswith(pref) for pref in _OWN_PARTY_SKIP_PREFIXES):
            continue
        if any(k in up for k in _HEADER_SUBSTRS):
            continue
        # Must look like a company name: all-caps letters + spaces + & / ( ) + digits minimal.
        if line != up:
            continue
        # Must contain a company suffix token
        if not _OWN_PARTY_SUFFIX_RE.search(line):
            continue
        # Filter out address-like lines that happen to contain "BHD" by accident (rare; defensive).
        if re.search(r"\b\d{5}\b", line):  # postcode
            continue
        return line
    return None


def _strip_trailing_amounts(s: str) -> str:
    t = _norm(s)
    t = re.sub(r"\s+\b(CR|DR)\b\s*$", "", t, flags=re.I)
    t = re.sub(r"\s+-?\d[\d,]*\.\d{2}\s+-?\d[\d,]*\.\d{2}\s*$", "", t)
    t = re.sub(r"\s+-?\d[\d,]*\.\d{2}\s*$", "", t)
    return _norm(t)


def parse_transactions_alliance(pdf, filename: str) -> List[Dict[str, Any]]:
    """
    Parse Alliance Bank statement into transaction dicts.
    Uses per-row signed-balance delta (CR = positive, DR = negative) so that
    CR accounts, DR accounts (overdraft / revolving), and CR↔DR transitions
    within a single account (e.g. a current account that goes overdrawn
    mid-month) all produce correct debit/credit assignments.
    """
    raw_rows: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    header_text_parts: List[str] = []

    for page_no, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        if page_no == 1:
            # Header extraction: pre-transaction portion of page 1. Alliance
            # boilerplate at the bottom of page 1 generically mentions
            # "overdraft/cashline accounts" in disclaimers — including that
            # would falsely promote a plain OD statement to CASH_LINE. Cut at
            # the table header ("Date Transaction Details / Tarikh Keterangan")
            # to capture only the Portfolio Summary / Account Type / facility-
            # name region.
            cut = text
            for marker in (
                "Date Transaction Details",
                "Tarikh Keterangan Urusniaga",
                "DATE TRANSACTION DETAILS",
            ):
                idx = cut.find(marker)
                if idx != -1:
                    cut = cut[:idx]
                    break
            header_text_parts.append(cut)
        for raw_line in text.splitlines():
            line = _norm(raw_line)
            if _is_noise(line):
                continue

            if _is_stop(line):
                if current:
                    raw_rows.append(current)
                    current = None
                continue

            m = _TX_START_RE.match(line)
            if m:
                date_iso = _iso_from_ddmmyy(m.group("d"), m.group("m"), m.group("y"))
                if not date_iso:
                    # 6-digit prefix that isn't a valid date (e.g. a cheque
                    # number like "403836 19/11/25" on a continuation line).
                    # Treat as continuation — do NOT flush/replace current,
                    # which previously caused a phantom duplicate row.
                    if current is not None:
                        current["description_parts"].append(line)
                        if current.get("balance") is None:
                            vals = _parse_money_tokens(line)
                            if len(vals) >= 2:
                                current["amount"] = vals[-2]
                                current["balance"] = vals[-1]
                                current["balance_sign"] = _balance_side_from_line(line)
                            elif len(vals) == 1:
                                current["balance"] = vals[-1]
                                current["balance_sign"] = _balance_side_from_line(line)
                    continue

                rest = m.group("rest")
                vals = _parse_money_tokens(line)

                if current:
                    raw_rows.append(current)

                current = {
                    "date": date_iso,
                    "description_parts": [_strip_trailing_amounts(rest)],
                    "amount": None,
                    "balance": None,
                    "balance_sign": None,
                    "page": page_no,
                }

                # typical: ... <amount> <balance>
                if len(vals) >= 2:
                    current["amount"] = vals[-2]
                    current["balance"] = vals[-1]
                    current["balance_sign"] = _balance_side_from_line(line)
                elif len(vals) == 1:
                    current["balance"] = vals[-1]
                    current["balance_sign"] = _balance_side_from_line(line)
                continue

            # continuation line
            if not current:
                continue

            up = line.upper()
            if "DATE" in up and "TRANSACTION" in up and "DETAILS" in up:
                continue

            current["description_parts"].append(line)

            # sometimes numeric tokens appear on continuation line
            if current.get("balance") is None:
                vals = _parse_money_tokens(line)
                if len(vals) >= 2:
                    current["amount"] = vals[-2]
                    current["balance"] = vals[-1]
                    current["balance_sign"] = _balance_side_from_line(line)
                elif len(vals) == 1:
                    current["balance"] = vals[-1]
                    current["balance_sign"] = _balance_side_from_line(line)

    if current:
        raw_rows.append(current)

    out: List[Dict[str, Any]] = []
    # Track balance as a SIGNED value (CR = positive, DR = negative) so that
    # delta math works uniformly across CR/DR transitions within a single
    # account (e.g. a current account that goes overdrawn mid-month and its
    # later balances switch from CR to DR suffix).
    prev_signed_balance: Optional[float] = None
    seq = 0

    def _signed(bal_val: Optional[float], bal_sign: Optional[str]) -> Optional[float]:
        if not isinstance(bal_val, (int, float)):
            return None
        if (bal_sign or "").upper() == "DR":
            return -float(bal_val)
        return float(bal_val)

    for r in raw_rows:
        seq += 1
        desc = _norm(" ".join(r.get("description_parts") or []))
        # Defensive footer strip: sometimes the disclaimer runs onto the
        # same line as the last transaction with no clean newline.
        desc = _norm(_FOOTER_STRIP_RE.sub("", desc))
        desc_up = desc.upper()
        bal = r.get("balance")
        bal_sign = r.get("balance_sign")
        signed_bal = _signed(bal, bal_sign)
        row_is_dr = (bal_sign or "").upper() == "DR"

        # BEGINNING BALANCE row
        if "BEGINNING BALANCE" in desc_up and isinstance(bal, (int, float)):
            prev_signed_balance = signed_bal
            out.append(
                {
                    "date": r["date"],
                    "description": "BEGINNING BALANCE",
                    "debit": 0.0,
                    "credit": 0.0,
                    "balance": float(bal),
                    "balance_sign": bal_sign,
                    "page": int(r.get("page") or 0),
                    "seq": seq,
                    "bank": "Alliance Bank",
                    "source_file": filename,
                }
            )
            continue

        debit = 0.0
        credit = 0.0

        if isinstance(prev_signed_balance, (int, float)) and isinstance(signed_bal, (int, float)):
            delta = round(signed_bal - prev_signed_balance, 2)
            # Signed-balance convention: positive delta = money IN = credit,
            # negative delta = money OUT = debit. Works for CR-only, DR-only,
            # and mid-statement CR↔DR transitions.
            if delta >= 0:
                credit = abs(delta)
            else:
                debit = abs(delta)
            prev_signed_balance = signed_bal
        else:
            # fallback if beginning balance wasn't parsed
            amt = r.get("amount")
            if amt is None:
                amt = 0.0
            debit = float(amt)

        if "ENDING BALANCE" in desc_up and isinstance(bal, (int, float)):
            desc = "ENDING BALANCE"

        row = {
            "date": r["date"],
            "description": desc,
            "debit": float(debit),
            "credit": float(credit),
            "balance": float(bal) if isinstance(bal, (int, float)) else None,
            "balance_sign": bal_sign,
            "page": int(r.get("page") or 0),
            "seq": seq,
            "bank": "Alliance Bank",
            "source_file": filename,
        }

        # Flag suspicious zero-amount rows that aren't balance markers.
        is_balance_row = (
            "BEGINNING BALANCE" in desc_up or "ENDING BALANCE" in desc_up
        )
        row["needs_review"] = (not is_balance_row and debit == 0.0 and credit == 0.0)

        out.append(row)

    # BUG-001 fix: drop ghost rows where debit=0, credit=0, balance is None.
    # These are voucher codes / footer fragments that the row detector mistook
    # for transactions (e.g. the KDYN 2094-10-31 / "PV25-184472-A" case).
    # Real transactions have money movement; balance markers always carry a balance.
    cleaned = [
        row for row in out
        if not (row["debit"] == 0.0 and row["credit"] == 0.0 and row["balance"] is None)
    ]

    # Sprint 4.5: per-PDF account_type determination + statutory stamping.
    # Alliance stores OD as a positive debt magnitude with a DR suffix on balance;
    # the signed-balance pass above has already normalised balance to negative for
    # DR rows, so row-math downstream in determine_account_type matches CR formula
    # for OD accounts with sustained-negative balance — which is what we want.
    # Opening balance comes from the BEGINNING BALANCE row if present; closing
    # from the last row with a balance.
    opening = next(
        (r["balance"] for r in cleaned
         if r.get("description") == "BEGINNING BALANCE" and r.get("balance") is not None),
        None,
    )
    closing = next(
        (r["balance"] for r in reversed(cleaned) if r.get("balance") is not None),
        None,
    )
    header_text = "\n".join(header_text_parts) if header_text_parts else None
    # Sprint 6 #10: stamp own-party name on every row so the counterparty
    # extractor downstream can strip own-name tokens from extracted names.
    own_party = _extract_own_party_name(header_text) if header_text else None
    if own_party:
        for row in cleaned:
            row["own_party_name"] = own_party
    return finalize_parser_output(
        cleaned,
        header_text=header_text,
        opening_balance=opening,
        closing_balance=closing,
    )
