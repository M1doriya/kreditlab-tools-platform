# public_bank.py - Standalone Public Bank Parser
import re

from core_utils import advance_year_on_rollover, finalize_parser_output

# ---------------------------------------------------------
# YEAR EXTRACTION
# ---------------------------------------------------------

def extract_year_from_text(text):
    """
    Extract year from Public Bank / Public Islamic Bank statement text.
    Safely handles:
    - Statement Date 31 Jul 2024
    - STATEMENT DATE : 30/09/24
    - Statement Date: DD/MM/YYYY
    - FOR THE PERIOD : DD/MM/YYYY
    """

    # -------------------------------------------------
    # Pattern 1: Statement Date 31 Jul 2024 (MOST COMMON)
    # -------------------------------------------------
    match = re.search(
        r'(?:Statement Date|Tarikh Penyata)\s*[:\s]+\d{1,2}\s+[A-Za-z]{3,}\s+(\d{4})',
        text,
        re.IGNORECASE
    )
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2100:
            return str(year)

    # -------------------------------------------------
    # Pattern 2: STATEMENT DATE : 30/09/24 or 30/09/2024
    # -------------------------------------------------
    match = re.search(
        r'(?:STATEMENT DATE|TARIKH PENYATA)\s*[:\s]+\d{1,2}/\d{1,2}/(\d{2,4})',
        text,
        re.IGNORECASE
    )
    if match:
        year_str = match.group(1)
        if len(year_str) == 4:
            year = int(year_str)
        else:
            year = 2000 + int(year_str)

        if 2000 <= year <= 2100:
            return str(year)

    # -------------------------------------------------
    # Pattern 3: Statement Date: DD/MM/YYYY
    # -------------------------------------------------
    match = re.search(
        r'Statement\s+(?:Date|Period)\s*[:\s]+\d{1,2}/\d{1,2}/(\d{4})',
        text,
        re.IGNORECASE
    )
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2100:
            return str(year)

    # -------------------------------------------------
    # Pattern 4: FOR THE PERIOD : DD/MM/YYYY
    # -------------------------------------------------
    match = re.search(
        r'FOR\s+THE\s+PERIOD\s*[:\s]+\d{1,2}/\d{1,2}/(\d{4})',
        text,
        re.IGNORECASE
    )
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2100:
            return str(year)

    # -------------------------------------------------
    # No safe year found
    # -------------------------------------------------
    return None



# ---------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------

# Matches date at start of line: "05/06 ..."
DATE_LINE = re.compile(r"^(?P<date>\d{2}/\d{2})\s+(?P<rest>.*)$")

# Matches amount + balance at end of line: "1,200.00 45,000.00"
AMOUNT_BAL = re.compile(r"(?P<amount>\d{1,3}(?:,\d{3})*\.\d{2})\s+(?P<balance>\d{1,3}(?:,\d{3})*\.\d{2})$")

# Matches "Balance B/F" lines
BAL_ONLY = re.compile(r"^(?P<date>\d{2}/\d{2})\s+(Balance.*)\s+(?P<balance>\d{1,3}(?:,\d{3})*\.\d{2})$", re.IGNORECASE)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TX_KEYWORDS = [
    "TSFR", "DUITNOW", "GIRO", "JOMPAY", "RMT", "DR-ECP",
    "HANDLING", "FEE", "DEP", "RTN", "PROFIT", "AUTOMATED",
    "CHARGES", "DEBIT", "CREDIT", "TRANSFER", "PAYMENT"
]

IGNORE_PREFIXES = [
    "CLEAR WATER", "/ROC", "PVCWS", "IMEPS",
    "PUBLIC BANK", "PAGE", "TEL:", "MUKA SURAT", "TARIKH",
    "DATE", "NO.", "URUS NIAGA", "STATEMENT", "ACCOUNT"
]

# While capturing post-amount detail lines (counterparty / reference / rail —
# see parse_transactions_pbb), only true page furniture may be skipped.
# Notably NOT skipped here: own-company beneficiary lines ("CLEAR WATER ..."),
# payment-voucher refs ("PVCWS-26-..."), IMEPS refs — those are real
# transaction detail that statutory / facility / own-party detection needs.
HARD_IGNORE_PREFIXES = [
    "PUBLIC BANK", "PAGE", "TEL:", "MUKA SURAT", "TARIKH",
    "DATE", "URUS NIAGA", "STATEMENT", "PROTECTED BY PIDM", "DILINDUNGI",
]

# Markers that end the transaction table region of a page: once one of these
# is seen, nothing below it on the page may be glued onto a transaction.
TABLE_END_MARKERS = [
    "BALANCE C/F", "BAKI PENUTUP", "CLOSING BALANCE",
    "TOTAL DEBIT", "TOTAL CREDIT", "JUMLAH DEBIT", "JUMLAH KREDIT",
    "TEGASAN", "HIGHLIGHTS", "RINGKASAN", "PERINGATAN", "NOTIS",
]

# Max detail lines glued onto one transaction (observed blocks are 2-4 lines;
# the cap is a guard against a missing table-end marker swallowing a footer).
MAX_TRAILING_LINES = 6


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def is_ignored(line):
    """Check if line should be ignored"""
    line_upper = line.upper()
    return any(line_upper.startswith(p) for p in IGNORE_PREFIXES)

def is_hard_ignored(line):
    """Page furniture only — used while capturing post-amount detail lines."""
    line_upper = line.upper()
    return any(line_upper.startswith(p) for p in HARD_IGNORE_PREFIXES)

def is_table_end(line):
    """Detect the end of the transaction table region on a page."""
    line_upper = line.upper()
    return any(m in line_upper for m in TABLE_END_MARKERS)

def is_tx_start(line):
    """Check if line starts a new transaction"""
    return any(line.upper().startswith(k) for k in TX_KEYWORDS)


# ---------------------------------------------------------
# Main Parser
# ---------------------------------------------------------

def parse_transactions_pbb(pdf, source_filename=""):
    """
    Main parser for Public Bank statements.
    Automatically extracts year and parses all transactions.
    
    Args:
        pdf: pdfplumber PDF object
        source_filename: Name of the source file
    
    Returns:
        List of transaction dictionaries
    """
    all_transactions = []
    detected_year = None

    # Sprint 4.5: capture page-1 header text (pre-transaction-table region) for
    # determine_account_type. Public Bank's table begins at bilingual markers
    # "TARIKH URUS NIAGA" / "DATE TRANSACTION".
    header_text = None
    if pdf.pages:
        page1 = pdf.pages[0].extract_text() or ""
        cut = page1
        for marker in (
            "TARIKH URUS NIAGA",
            "DATE TRANSACTION",
        ):
            idx = cut.find(marker)
            if idx != -1:
                cut = cut[:idx]
                break
        header_text = cut or None

    # Extract year from first few pages
    for page in pdf.pages[:3]:
        text = page.extract_text() or ""
        detected_year = extract_year_from_text(text)
        if detected_year:
            break
    
    # Fallback to current year
    if not detected_year:
        from datetime import datetime
        detected_year = str(datetime.now().year)
    
    # Process all pages
    prev_date_iso = None
    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""

        tx = []
        current_date = None
        prev_balance = None
        desc_accum = ""
        waiting_for_amount = False

        # Post-amount detail capture: Public Bank prints the counterparty /
        # reference / rail lines BELOW the amount-bearing transaction line
        # (e.g. "DR-ECP ... 24,457.20 854,766.99" followed by "LEMBAGA HASIL
        # DALAM NEGERI MAL"). Without capturing these, statutory bodies (EPF /
        # SOCSO / LHDN), facility instalments (SME Bank / ORIX) and all
        # counterparty names are invisible downstream. Detail blocks can spill
        # across a page break (orphan lines arrive right after the next page's
        # "Balance B/F"), so the target survives page boundaries but capture
        # only re-arms once the table region resumes.
        trailing_target = all_transactions[-1] if all_transactions else None
        trailing_active = False
        trailing_count = 0

        lines = text.splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if is_table_end(line):
                trailing_active = False
                trailing_target = None
                continue
            if trailing_active and trailing_target is not None:
                if is_hard_ignored(line):
                    continue
            elif is_ignored(line):
                continue

            # Check for amounts FIRST
            amount_match = AMOUNT_BAL.search(line)
            has_amount = bool(amount_match)

            # Check for start of new transaction
            date_match = DATE_LINE.match(line)
            keyword_match = is_tx_start(line)
            is_new_start = date_match or keyword_match

            # Handle Balance B/F
            bal_match = BAL_ONLY.match(line)
            if bal_match:
                current_date = bal_match.group("date")
                prev_balance = float(bal_match.group("balance").replace(",", ""))
                desc_accum = ""
                waiting_for_amount = False
                # Table region (re)started: orphan detail lines of the previous
                # page's last transaction may follow directly below this line.
                trailing_active = True
                trailing_count = 0
                continue
            
            # CASE A: Line has amounts
            if has_amount:
                amount = float(amount_match.group("amount").replace(",", ""))
                balance = float(amount_match.group("balance").replace(",", ""))
                
                # Determine description
                if is_new_start:
                    if date_match:
                        current_date = date_match.group("date")
                        line_desc = date_match.group("rest")
                    else:
                        line_desc = line.replace(amount_match.group(0), "").strip()
                    final_desc = line_desc
                else:
                    final_desc = desc_accum + " " + line.replace(amount_match.group(0), "").strip()
                
                # Determine debit vs credit
                debit = 0.0
                credit = 0.0
                
                if prev_balance is not None:
                    if balance < prev_balance:
                        debit = amount
                    elif balance > prev_balance:
                        credit = amount
                else:
                    # Fallback based on keywords
                    upper_desc = final_desc.upper()
                    if "CR" in upper_desc or "DEP" in upper_desc or "CREDIT" in upper_desc:
                        credit = amount
                    else:
                        debit = amount
                
                # Format date
                if current_date:
                    dd, mm = current_date.split("/")
                    iso_date = f"{detected_year}-{mm}-{dd}"
                else:
                    iso_date = f"{detected_year}-01-01"
                iso_date = advance_year_on_rollover(iso_date, prev_date_iso)
                prev_date_iso = iso_date
                
                # Append transaction
                tx.append({
                    "date": iso_date,
                    "description": final_desc.strip(),
                    "debit": debit,
                    "credit": credit,
                    "balance": balance,
                    "page": page_num,
                    "source_file": source_filename,
                    "bank": "Public Bank"
                })

                # Reset state
                prev_balance = balance
                desc_accum = ""
                waiting_for_amount = False
                # Detail lines (counterparty/ref/rail) follow below this line.
                trailing_target = tx[-1]
                trailing_active = True
                trailing_count = 0

            # CASE B: No amounts, but starts new transaction
            elif is_new_start:
                if date_match:
                    current_date = date_match.group("date")
                    desc_accum = date_match.group("rest")
                else:
                    desc_accum = line
                waiting_for_amount = True
                # A new transaction has begun: stop gluing onto the previous one.
                trailing_active = False

            # CASE C: Continuation text
            elif waiting_for_amount:
                desc_accum += " " + line

            # CASE D: post-amount detail lines belonging to the last
            # completed transaction (counterparty name, reference, FPX rail)
            elif trailing_active and trailing_target is not None:
                if trailing_count < MAX_TRAILING_LINES:
                    trailing_target["description"] = (
                        trailing_target["description"] + " " + line
                    ).strip()
                    trailing_count += 1
                else:
                    trailing_active = False
        
        all_transactions.extend(tx)

    # Sprint 4.5: per-PDF account_type determination + statutory stamping.
    # Public Bank corpus is 17/17 CR; no pre-extracted opening/closing, pass None.
    return finalize_parser_output(
        all_transactions,
        header_text=header_text,
        opening_balance=None,
        closing_balance=None,
    )
