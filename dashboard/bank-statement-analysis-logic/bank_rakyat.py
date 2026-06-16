
# bank_rakyat.py
# Bank Rakyat – Balance-driven, summary-aware parser (FINAL)

import re
from datetime import datetime

from core_utils import finalize_parser_output


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_amount(val):
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return None


def parse_date(raw):
    try:
        return datetime.strptime(raw, "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        return None


# ---------------------------------------------------------
# Extract summary section (source of truth)
# ---------------------------------------------------------

def extract_summary(full_text):
    """
    Extracts opening, total debit, total credit, closing.
    Works even if values appear BELOW labels.
    """

    nums = [clean_amount(x) for x in re.findall(r"[-]?\d[\d,]*\.\d{2}", full_text)]
    nums = [n for n in nums if n is not None]

    summary = {
        "opening": None,
        "total_debit": None,
        "total_credit": None,
        "closing": None,
    }

    # Explicit patterns (preferred)
    m = re.search(r"(Opening Balance|Baki Permulaan)[^\d\-]*([-]?\d[\d,]*\.\d{2})", full_text, re.I | re.S)
    if m:
        summary["opening"] = clean_amount(m.group(2))

    m = re.search(r"(Closing Balance|Baki Penutup)[^\d\-]*([-]?\d[\d,]*\.\d{2})", full_text, re.I | re.S)
    if m:
        summary["closing"] = clean_amount(m.group(2))

    # Fallback: Bank Rakyat summary row ALWAYS has 4 numbers
    # [opening, total debit, total credit, closing]
    if len(nums) >= 4:
        summary["opening"] = summary["opening"] or nums[-4]
        summary["total_debit"] = nums[-3]
        summary["total_credit"] = nums[-2]
        summary["closing"] = summary["closing"] or nums[-1]

    return summary


# ---------------------------------------------------------
# Extract raw transaction rows (order-independent)
# ---------------------------------------------------------

# Stop markers for the multi-line continuation walker. Bank Rakyat
# (Felcra-style) statements emit each transaction as a date-line followed
# by 1–N continuation lines (entity name, sub-account tag, refs, purpose).
# The walker collects those continuations until it hits the next date-line,
# the summary footer, or a page header re-emitted on continuation pages.
_BR_CONTINUATION_STOP_RE = re.compile(
    r"^\s*(?:"
    r"Baki\s*Permulaan|Opening\s*Balance|"
    r"Closing\s*Balance|Baki\s*Penutup|"
    r"Tarikh\s*Kod|Date\s*Transaction|"
    r"Mukasurat|Page\s*\d+\s*of|"
    r"This\s+is\s+a\s+computer\s+generated"
    r")",
    re.IGNORECASE,
)


def extract_transactions(pdf):
    """Capture multi-line transaction blocks.

    Felcra-style Bank Rakyat PDFs (CASA_DATAPOS_*) put the entity name and
    purpose on continuation lines below each date-line. The previous parser
    only kept the date-line, so descriptions were just `<code> <opcode>
    <amount>` with no entity. The walker now appends continuation tokens
    space-joined onto the description so the bank-gated counterparty
    extractor in app.py can recover the entity.
    """
    rows = []

    for page_no, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            date_match = re.search(r"\d{2}/\d{2}/\d{4}", line)
            if not date_match:
                i += 1
                continue

            amounts = re.findall(r"[-]?\d[\d,]*\.\d{2}", line)
            if not amounts:
                i += 1
                continue

            balance = clean_amount(amounts[-1])
            if balance is None:
                i += 1
                continue

            iso_date = parse_date(date_match.group())
            if not iso_date:
                i += 1
                continue

            desc = line.replace(date_match.group(), "")
            desc = desc.replace(amounts[-1], "", 1)
            desc = " ".join(desc.split())

            # Walk forward collecting continuation lines until we hit the
            # next date-line, the summary footer, or a re-emitted page
            # header. Empty / whitespace-only lines are skipped.
            j = i + 1
            continuation = []
            while j < len(lines):
                nxt = lines[j]
                if re.match(r"^\s*\d{2}/\d{2}/\d{4}", nxt):
                    break
                if _BR_CONTINUATION_STOP_RE.match(nxt):
                    break
                stripped = nxt.strip()
                if stripped:
                    continuation.append(stripped)
                j += 1

            if continuation:
                desc = (desc + " " + " ".join(continuation)).strip()

            rows.append({
                "date": iso_date,
                "description": desc,
                "balance": balance,
                "page": page_no,
            })

            i = j  # resume at the next date-line / stop marker

    return rows


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------

def parse_bank_rakyat(pdf, source_filename=""):
    # Sprint 4.5: capture page-1 header text (pre-transaction-table region) for
    # determine_account_type. Bank Rakyat Cashline-i statements disclose the
    # "Cashline-i Limit : RM ..." here — required for OD lock.
    header_text = None
    if pdf.pages:
        page1 = pdf.pages[0].extract_text() or ""
        cut = page1
        for marker in (
            "Tarikh Kod Transaksi",
            "Date Transaction Description",
            "Tarikh Kod",
            "Date Transaction",
        ):
            idx = cut.find(marker)
            if idx != -1:
                cut = cut[:idx]
                break
        header_text = cut or None

    # Read entire document text
    full_text = ""
    for p in pdf.pages:
        full_text += (p.extract_text() or "") + "\n"

    summary = extract_summary(full_text)
    raw_rows = extract_transactions(pdf)

    if not raw_rows:
        return finalize_parser_output(
            [],
            header_text=header_text,
            opening_balance=summary.get("opening"),
            closing_balance=summary.get("closing"),
        )

    # Sort chronologically
    raw_rows.sort(key=lambda x: (x["date"], x["page"]))

    # Determine opening balance (BEST METHOD)
    opening = summary["opening"]

    if opening is None and summary["closing"] is not None:
        opening = (
            summary["closing"]
            - (summary["total_credit"] or 0)
            + (summary["total_debit"] or 0)
        )

    results = []
    prev_balance = opening

    for row in raw_rows:
        debit = credit = 0.0

        if prev_balance is not None:
            delta = round(row["balance"] - prev_balance, 2)
            if delta > 0:
                credit = delta
            elif delta < 0:
                debit = abs(delta)

        prev_balance = row["balance"]

        results.append({
            "date": row["date"],
            "description": row["description"],
            "debit": debit,
            "credit": credit,
            "balance": row["balance"],
            "page": row["page"],
            "bank": "Bank Rakyat",
            "source_file": source_filename,
        })

    # Sprint 4.5: per-PDF account_type determination + statutory stamping.
    # Bank Rakyat OD statements (Cashline-i) use negative-balance convention;
    # the header's "Cashline-i Limit : RM ..." disclosure triggers CASH_LINE.
    return finalize_parser_output(
        results,
        header_text=header_text,
        opening_balance=summary.get("opening"),
        closing_balance=summary.get("closing"),
    )
