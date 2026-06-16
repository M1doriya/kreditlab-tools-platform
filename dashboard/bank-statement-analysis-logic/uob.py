"""uob.py

UOB Malaysia - "Account Activities" PDF export parser.

Observed layout (from provided samples):
  - Header contains:
      Company
      Account <account_no> <company_name> MYR <account_no>
      Statement Date <dd/mm/yyyy> - <dd/mm/yyyy>
  - Transaction table columns:
      Statement Date | Transaction Date | Description | Deposit(MYR) | Withdrawal(MYR) | Ledger Balance(MYR)

Notes:
  - Descriptions are frequently multi-line.
  - Amount columns are consistently present as numeric tokens (e.g. 0.00, 1,090.00, -644,255.96).
  - This parser is resilient to line wraps by "row stitching": we detect a new row when a line
    starts with TWO dates (dd/mm/yyyy dd/mm/yyyy).

Output:
  A list of dicts with canonical keys: date, description, debit, credit, balance, page, bank, source_file.
  Extra metadata (company_name/account_no) may also be included and will be preserved by core_utils.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from core_utils import normalize_date, normalize_text, safe_float, finalize_parser_output


BANK_NAME = "UOB Bank"


_STMT_TIME_RE = re.compile(r"^(?P<stmt>\d{2}/\d{2}/\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})(?:\s+(?P<rest>.*))?$")
_TRX_LINE_RE = re.compile(r"^(?P<trx>\d{2}/\d{2}/\d{4})\s+(?P<body>.*)$")
_MONEY_RE = re.compile(r"^-?(\d{1,3}(?:,\d{3})*|\d+)\.\d{2}$")


def _extract_header_meta(first_page_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
    """Return (company_name, account_no, statement_end_iso, ledger_balance_header)."""
    if not first_page_text:
        return None, None, None, None

    company_name: Optional[str] = None
    account_no: Optional[str] = None
    statement_end_iso: Optional[str] = None
    ledger_balance_header: Optional[float] = None

    # Company name: UOB export sometimes shows "Company Available Balance" followed by the name.
    m = re.search(
        r"Company\s+Available\s+Balance\s*\n\s*([A-Z0-9 &().,'\/-]{3,})\b",
        first_page_text,
        re.IGNORECASE,
    )
    if m:
        company_name = normalize_text(m.group(1))
        # sometimes the export appends currency + balance after the company name
        company_name = re.split(r"\bMYR\b", company_name, maxsplit=1, flags=re.IGNORECASE)[0].strip() or company_name

    # Fallback: the line after a standalone "Company" label.
    if not company_name:
        m = re.search(r"\bCompany\b\s*\n\s*([A-Z0-9 &().,'\/-]{3,})\s*(?:\n|$)", first_page_text, re.IGNORECASE)
        if m:
            cand = normalize_text(m.group(1))
            if cand and cand.upper() not in {"ACCOUNT", "COMPANY / ACCOUNT"}:
                company_name = cand

    # Account number: commonly shown beneath "Account Ledger Balance".
    m = re.search(r"Account\s+Ledger\s+Balance\s*\n\s*(\d{6,20})\b", first_page_text, re.IGNORECASE)
    if m:
        account_no = m.group(1)

    # Fallback: first long digit group after "Account" label.
    if not account_no:
        m = re.search(r"\bAccount\b\s*(?:\n\s*)?(\d{6,20})\b", first_page_text, re.IGNORECASE)
        if m:
            account_no = m.group(1)

    # Statement period end date.
    m = re.search(
        r"Statement\s+Date\s*(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})",
        first_page_text,
        re.IGNORECASE,
    )
    if m:
        statement_end_iso = normalize_date(m.group(2))

    # Ledger balance header (used for no-transaction fallback)
    # Example: "... MYR -644,255.96" on the same line as the account number.
    m = re.search(r"Account\s+Ledger\s+Balance.*?\bMYR\b\s*([-()\d,]+\.\d{2})", first_page_text, re.IGNORECASE | re.DOTALL)
    if m:
        ledger_balance_header = safe_float(m.group(1))

    return company_name, account_no, statement_end_iso, ledger_balance_header


def _split_amounts_from_tail(body: str) -> Optional[Tuple[str, float, float, float]]:
    """Return (desc, deposit, withdrawal, balance) by taking last 3 money tokens."""
    tokens = (body or "").split()
    money_idx = [i for i, t in enumerate(tokens) if _MONEY_RE.match(t)]
    if len(money_idx) < 3:
        return None

    dep = safe_float(tokens[money_idx[-3]])
    wd = safe_float(tokens[money_idx[-2]])
    bal = safe_float(tokens[money_idx[-1]])

    desc = normalize_text(" ".join(tokens[: money_idx[-3]]))
    return desc, dep, wd, bal


def parse_transactions_uob(pdf: pdfplumber.PDF, source_file: str = "") -> List[Dict[str, Any]]:
    transactions: List[Dict[str, Any]] = []

    company_name: Optional[str] = None
    account_no: Optional[str] = None
    statement_end_iso: Optional[str] = None
    ledger_balance_header: Optional[float] = None

    # Sprint 4.5: capture page-1 header text (pre-transaction-table region) for
    # determine_account_type. UOB exposes "Overdraft Facility <amount>" in the
    # Account Details block — this is what flags UOB Upell as OD. Truncate at
    # the transaction-table marker.
    header_text: Optional[str] = None
    if pdf.pages:
        page1_full = pdf.pages[0].extract_text() or ""
        cut = page1_full
        for marker in (
            "Statement Date Transaction Date Description",
            "Account Transactions",
            "Statement Date Transaction Date",
        ):
            idx = cut.find(marker)
            if idx != -1:
                cut = cut[:idx]
                break
        header_text = cut or None

    prev_tx: Optional[Dict[str, Any]] = None

    pending_stmt_date: Optional[str] = None
    pending_time: Optional[str] = None
    pending_ampm: Optional[str] = None
    pending_desc_head: str = ""

    for page_idx, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        if not text:
            continue

        if page_idx == 1:
            company_name, account_no, statement_end_iso, ledger_balance_header = _extract_header_meta(text)

        lines = [normalize_text(ln) for ln in (text.splitlines() or []) if normalize_text(ln)]

        for line in lines:
            up = line.upper().strip()

            # stop at footer/summary sections
            if up.startswith("TOTAL DEPOSITS") or up.startswith("NOTE"):
                break

            if "DATE OF EXPORT" in up:
                continue
            if up in {"ACCOUNT ACTIVITIES"}:
                continue
            if up.startswith("STATEMENT DATE TRANSACTION DATE"):
                continue
            if up.startswith("STATEMENT DATE") and "TRANSACTION" in up and "DESCRIPTION" in up:
                continue

            # AM/PM line wrap (belongs to the pending statement time)
            if up in {"AM", "PM"} and pending_stmt_date and pending_time and not pending_ampm:
                pending_ampm = up
                continue

            # 1) Statement date + time line (sometimes also contains the start of description)
            m1 = _STMT_TIME_RE.match(line)
            if m1:
                pending_stmt_date = normalize_date(m1.group("stmt")) or m1.group("stmt")
                pending_time = m1.group("time")
                pending_ampm = None
                pending_desc_head = normalize_text(m1.group("rest") or "")
                continue

            # 2) Transaction date line (contains amounts/balance)
            m2 = _TRX_LINE_RE.match(line)
            if m2:
                trx_date = m2.group("trx")
                body = m2.group("body") or ""

                split = _split_amounts_from_tail(body)
                if not split:
                    # Some rows have transaction date + no amounts on that line; treat as description continuation
                    if prev_tx is not None and line and not _MONEY_RE.match(line):
                        prev_tx["description"] = normalize_text(prev_tx.get("description", "") + " " + line)
                    continue

                desc_body, dep, wd, bal = split
                desc = normalize_text(" ".join([pending_desc_head, desc_body]).strip()) if pending_desc_head or desc_body else ""

                # If still empty, keep a placeholder
                if not desc:
                    desc = "(NO DESCRIPTION)"

                credit = abs(dep) if dep else 0.0
                debit = abs(wd) if wd else 0.0

                trx_iso = normalize_date(trx_date) or trx_date
                post_iso = pending_stmt_date

                tx = {
                    # IMPORTANT:
                    # "date" is the *transaction date* (value date) so monthly summaries align to the period users expect.
                    # Posting/statement date is preserved separately as "posting_date".
                    "date": trx_iso,
                    "posting_date": post_iso,
                    "transaction_date": trx_iso,
                    "time": (
                        f"{pending_time} {pending_ampm}".strip() if (pending_time and pending_ampm) else pending_time
                    ),
                    "description": desc,
                    "debit": round(float(debit), 2),
                    "credit": round(float(credit), 2),
                    "balance": round(float(bal), 2),
                    "page": page_idx,
                    "bank": BANK_NAME,
                    "source_file": source_file,
                    "company_name": company_name,
                    "account_no": account_no,
                }
                transactions.append(tx)
                prev_tx = tx

                # reset pending for next row
                pending_stmt_date = None
                pending_time = None
                pending_ampm = None
                pending_desc_head = ""
                continue

            # 3) Continuation lines for description
            if prev_tx is not None:
                # If AM/PM wraps after the transaction line, attach it to the time field (not description)
                if up in {"AM", "PM"} and prev_tx.get("time") and prev_tx.get("time") not in {"AM", "PM"}:
                    prev_tx["time"] = normalize_text(f"{prev_tx.get('time')} {up}")
                    continue
                # Skip obvious noise like "AM Total 1 Cheque(s)"
                if re.match(r"^(AM|PM)\s+TOTAL\b", up, flags=re.IGNORECASE):
                    continue
                if up.startswith("ACCOUNT ACTIVITIES") or up.startswith("RECORD"):
                    continue
                prev_tx["description"] = normalize_text(prev_tx.get("description", "") + " " + line)

    # Fallback: no transactions but we still want month to appear.
    if not transactions and ledger_balance_header is not None:
        transactions.append(
            {
                "date": statement_end_iso or "2000-01-01",
                "description": "NO TRANSACTIONS (LEDGER BALANCE)",
                "debit": 0.0,
                "credit": 0.0,
                "balance": round(float(ledger_balance_header), 2),
                "page": None,
                "bank": BANK_NAME,
                "source_file": source_file,
                "company_name": company_name,
                "account_no": account_no,
                "is_statement_balance": True,
            }
        )

    for t in transactions:
        t.setdefault("is_statement_balance", False)

    # Sprint 4.5: per-PDF account_type determination + statutory stamping.
    # UOB corpus (Upell) uses sustained-negative Ledger Balance to signal OD;
    # the "Overdraft Facility <amount>" header line is the primary lock.
    closing_balance_value = None
    try:
        closing_balance_value = (
            float(ledger_balance_header) if ledger_balance_header is not None else None
        )
    except Exception:
        closing_balance_value = None
    return finalize_parser_output(
        transactions,
        header_text=header_text,
        opening_balance=None,
        closing_balance=closing_balance_value,
    )
