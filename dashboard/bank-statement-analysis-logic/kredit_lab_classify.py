"""kredit_lab_classify.py — Stage 2 deterministic classifier (Streamlit UI).

Pipeline:
    full_report.json  ->  this module  ->  analysis.json + narrative_brief.json + parser_quality.json

V1 / V1.1 / V2 categories wired:
    C01/C02 (own-party)         from counterparty_name == company_name
    C03/C04 (related-party)     analyst-confirmed names; substring match
    C05     (salary)            from BULK SALARY bucket + STAFF OVERTIME/INCENTIVE/BONUS/ADVANCE
    C06-C09 (statutory)         from KWSP/SOCSO/LHDN/HRDF buckets + statutory_bucket_for fallback
    C10     (loan disbursement) from LOAN DISBURSEMENT bucket / analyst factoring entities
    C11     (loan repayment)    from LOAN REPAYMENT bucket
    C12     (FD/interest)       from FD/INTEREST bucket
    C13     (reversal)          from REVERSAL keyword + INWARD RETURN bucket (CR side only)
    C14/C15 (returned cheques)  cross-bank keyword fallback only (RETURNED CHEQUE / DISHONOURED).
                                Bank-specific shorthand (Maybank RTD, etc.) intentionally NOT added —
                                cross-bank purity wins; parser layer should bucket these.
    C16     (IBG inward return) from INWARD RETURN bucket
    C17/C18 (cash dep/wdl)      from CASH DEPOSIT / CASH WITHDRAWAL buckets
    C19/C20 (cheque dep/issue)  keyword fallback (CHQ DEPOSIT / CHQ ISSUE / CLRG CHQ DR)
    C24     (bank fees)         from BANK FEES bucket
    C25     (balance row)       from is_statement_balance / is_opening_balance flags
    C26/C27 (trade in/out)      V2: corporate-marker counterparty (SDN BHD / BHD / BERHAD / ENTERPRISE
                                / TRADING / CORPORATION / CORP / GROUP / HOLDINGS / INDUSTRIES) AFTER
                                all other rules fail. Natural-person counterparties (BIN/BINTI/A/L/A/P)
                                explicitly do NOT fire — per CLASSIFICATION_RULES_v3_5 LOCKED note.

V1 deferred / intentional flag-layer only:
    C21-C23 monitoring          round-figure / high-value detected at flag stage only —
                                schema does not reserve C21-C23 fields per-row; flags.indicators[]
                                already covers AML monitoring (Round Figure, High Value, Cash Deposits).

All classifications are cross-bank-safe — no bank-specific code paths.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
import streamlit as st

from core_utils import (
    determine_account_type,
    statutory_bucket_for,
    should_drop_as_counterparty,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
PROMPT_DIR = REPO_ROOT / "validation runs - json" / "claude ai prompt file"
SCHEMA_PATH = PROMPT_DIR / "BANK_ANALYSIS_SCHEMA_v6_3_5.json"
RULES_PATH = PROMPT_DIR / "CLASSIFICATION_RULES_v3_5.json"
TARGET_SCHEMA_VERSION = "6.3.5"
TARGET_RULES_VERSION = "3.5"
TARGET_PROMPT_VERSION = "v3.5.6"
CLASSIFIER_VERSION = "1.0.0"

ROUND_FIGURE_MIN = 10_000.0
HIGH_VALUE_THRESHOLD = 100_000.0
LOW_BALANCE_THRESHOLD = 1_000.0

# Canonical 16-flag list (must match BANK_ANALYSIS_SCHEMA_v6_3_5 enum exactly)
FLAG_DEFINITIONS: list[tuple[int, str]] = [
    (1, "Returned Cheques (Inward)"),
    (2, "Returned Cheques (Outward)"),
    (3, "Round Figure Credits (AML)"),
    (4, "High Value Credits (>3x EOD)"),
    (5, "Cash Deposits (AML)"),
    (6, "EPF Compliance"),
    (7, "SOCSO Compliance"),
    (8, "LHDN Tax Payments"),
    (9, "Large Credits (>=RM100K)"),
    (10, "Own Party Transactions"),
    (11, "Related Party Transactions"),
    (12, "Loan Activity"),
    (13, "Data Quality"),
    (14, "FX Transactions"),
    (15, "Low Closing Balance"),
    (16, "HRDF Payments"),
]

# Counterparty bucket -> primary category code
# Codes are anchored to CLASSIFICATION_RULES_v3_5.json:
#   C13 = Reversal Credit (generic CR-side reversal)
#   C14/C15 = Returned Cheques Inward(DR)/Outward(CR)
#   C16 = IBG/GIRO Inward Return (CR)
#   C17/C18 = Cash Deposit/Withdrawal
#   C19/C20 = Cheque Deposit/Issue
BUCKET_TO_CATEGORY: dict[str, str] = {
    "BULK SALARY": "C05",
    "BANK FEES": "C24",
    "KWSP": "C06",
    "SOCSO": "C07",
    "LHDN": "C08",
    "HRDF": "C09",
    "LOAN REPAYMENT": "C11",
    "LOAN DISBURSEMENT": "C10",
    "FD/INTEREST": "C12",
    "CASH DEPOSIT": "C17",
    "CASH WITHDRAWAL": "C18",
    "CHEQUE DEPOSIT": "C19",
    "CHEQUE ISSUE": "C20",
    "INWARD RETURN": "C16",
}

# Description-keyword fallback patterns. Run AFTER bucket-direct + own-party,
# BEFORE statutory_bucket_for. Each entry: (compiled regex, side requirement, code).
# Cross-bank phrases only — bank-specific tokens belong in parser layer.
_KEYWORD_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bRETURN(ED)?\s*CHE?Q\b|\bCHE?Q\s*RETURN\b|\bDISHONOURED\b"), "DR", "C14"),
    (re.compile(r"\bRETURN(ED)?\s*CHE?Q\b|\bCHE?Q\s*RETURN\b|\bDISHONOURED\b"), "CR", "C15"),
    (re.compile(r"\bIBG\s*INWARD\s*RETURN\b|\bGIRO\s*INWARD\s*RETURN\b"), "CR", "C16"),
    (re.compile(r"\bCDM\s*CASH\s*DEPOSIT\b|\bCASH\s*DEPOSIT\b"), "CR", "C17"),
    (re.compile(r"\bCASH\s*CHE?Q\s*DR\b|\bCASH\s*WITHDRAW(AL)?\b|\bATM\s*WITHDRAW\b"), "DR", "C18"),
    (re.compile(r"\bHSE\s*CHE?Q\s*DEPOSIT\b|\bCHE?Q(UE)?\s*DEPOSIT\b|\b2D\s*LOCAL\s*CHE?Q\b"), "CR", "C19"),
    (re.compile(r"\bHOUSE\s*CHE?Q\s*DR\b|\bCLRG\s*CHE?Q\s*DR\b|\bINWARD\s*CLEARING\s*CHE?Q\s*DEBIT\b|\bCHE?Q\s*ISSUE\b"), "DR", "C20"),
    (re.compile(r"\bREVERSAL\b|\bREVERSED\b|\bREV\s*CR\b|\bCREDIT\s*REVERSAL\b"), "CR", "C13"),
    (re.compile(
        r"\bBULK\s*SALARY\b|\bAUTO\s*SALARY\b|\bAUTOPAY\s*DR\b|\bPAYROLL\b|\bGAJI\b|"
        r"\bSALARY\b|\bSTAFF\s*OVERTIME\b|\bSTAFF\s*INCENTIVE\b|\bSTAFF\s*BONUS\b|"
        r"\bSTAFF\s*ADVANCE\b"
    ), "DR", "C05"),
]

# Per-category expected side. Bucket-direct firings must agree with this — guards
# against mis-bucketed parser rows (e.g. CR row stamped as BULK SALARY).
_CATEGORY_SIDES: dict[str, str] = {
    "C01": "CR", "C02": "DR", "C03": "CR", "C04": "DR",
    "C05": "DR", "C06": "DR", "C07": "DR", "C08": "DR", "C09": "DR",
    "C10": "CR", "C11": "DR", "C12": "CR", "C13": "CR",
    "C14": "DR", "C15": "CR", "C16": "CR",
    "C17": "CR", "C18": "DR", "C19": "CR", "C20": "DR",
    "C24": "ANY",  # bank fees can be either side
    "C25": "ANY",  # balance rows
    "C26": "CR", "C27": "DR",
}

PERSONAL_KEYWORDS_RP4 = [
    "ADV FI", "FI ", "CLAIM", "DIVIDEND", "LOAN", "PETTY",
    "HOUSING", "CREDIT CARD", "BONUS", "MEDICAL", "REIMBURSE",
]

# C26/C27 trade detection — corporate-entity markers (LOCKED in CLASSIFICATION_RULES_v3_5).
# Anchored at word boundary; case-insensitive at use site.
_CORPORATE_ENTITY_MARKERS = re.compile(
    r"\b(SDN\s*\.?\s*BHD\.?|BHD\.?|BERHAD|ENTERPRISE|TRADING|"
    r"CORPORATION|CORP|GROUP|HOLDINGS|INDUSTRIES)\b",
    re.IGNORECASE,
)

# Natural-person markers — explicitly DO NOT fire C26/C27 per rulebook note.
# Stick to unambiguous patrilineage tokens; bare "AL"/"AP" cause false-exclusion on
# legitimate names like "AL-IKHWAN HOLDINGS".
_NATURAL_PERSON_MARKERS = re.compile(
    r"\bBIN\b|\bBINTI\b|\bA/L\b|\bA/P\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Analyst decisions
# ---------------------------------------------------------------------------


@dataclass
class AnalystDecisions:
    related_parties: list[str] = field(default_factory=list)
    factoring_entities: list[str] = field(default_factory=list)
    commission_treatment: str = "regular_expense"
    rp4_director_decisions: dict[str, str] = field(default_factory=dict)
    od_limit: float | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Stage A — load + pre-analysis gate
# ---------------------------------------------------------------------------


def load_parser_output(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_rulebook(path: Path = RULES_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def detect_account_type(data: dict[str, Any], od_limit: float | None = None) -> dict[str, Any]:
    """Wrap core_utils.determine_account_type. Trust parser stamping if present.
    od_limit is accepted for forward-compat but not yet plumbed into core_utils."""
    txs = data.get("transactions", [])
    stamped = next(
        (t.get("account_type_determination") for t in txs if t.get("account_type_determination")),
        None,
    )
    monthly = data.get("monthly_summary", [])

    def _safe_float(v: Any) -> float | None:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    opening = _safe_float(monthly[0].get("opening_balance")) if monthly else None
    closing = _safe_float(monthly[-1].get("ending_balance")) if monthly else None
    determination = stamped or determine_account_type(
        txs, opening_balance=opening, closing_balance=closing, header_text=""
    )
    locked = determination.get("locked_type", "CR")
    is_od = locked == "OD"
    return {
        "type": "OD" if is_od else "Current",
        "convention": "OD" if is_od else "CR",
        "is_od": is_od,
        "determination": determination,
    }


def reconcile_balance_trail(data: dict[str, Any], convention: str) -> dict[str, Any]:
    """Per (account_no, month) reconciliation. Tolerance RM 1.00.

    CR convention (default): expected_closing = opening + credit - debit.
    OD convention (Alliance-style positive-magnitude debt): expected = opening + debit - credit.
    Ambank OD is parser-pre-negated so it falls through as CR convention here — correct.
    """
    is_od = convention == "OD"
    deltas = []
    all_pass = True
    for m in data.get("monthly_summary", []):
        opening = float(m.get("opening_balance") or 0.0)
        debit = float(m.get("total_debit") or 0.0)
        credit = float(m.get("total_credit") or 0.0)
        actual = float(m.get("ending_balance") or 0.0)
        expected = opening + debit - credit if is_od else opening + credit - debit
        delta = round(expected - actual, 2)
        passed = abs(delta) <= 1.0
        if not passed:
            all_pass = False
        deltas.append({
            "month": m.get("month") or "",
            "account_number": m.get("account_no") or "",
            "bank_name": _bank_for_month(data, m),
            "opening_balance": round(opening, 2),
            "closing_balance": round(actual, 2),
            "gross_credits": round(credit, 2),
            "gross_debits": round(debit, 2),
            "expected_closing": round(expected, 2),
            "reconciliation_delta": delta,
            "passed": passed,
            "transactions_extracted": _count_txs_for_month(data, m),
        })
    return {"pass": all_pass, "deltas": deltas}


def _bank_for_month(data: dict[str, Any], month_summary: dict[str, Any]) -> str:
    month = month_summary.get("month")
    acct = month_summary.get("account_no")
    for tx in data.get("transactions", []):
        if tx.get("account_no") == acct and (tx.get("date") or "").startswith(month or ""):
            return tx.get("bank") or "Unknown"
    return "Unknown"


def _count_txs_for_month(data: dict[str, Any], month_summary: dict[str, Any]) -> int:
    month = month_summary.get("month")
    acct = month_summary.get("account_no")
    return sum(
        1 for tx in data.get("transactions", [])
        if tx.get("account_no") == acct and (tx.get("date") or "").startswith(month or "")
    )


# V3-A auto-RP Step 1 — deterministic behavioral signals on counterparty_ledger.
# Each signal carries a weight; total score maps to LOW/MEDIUM/HIGH confidence.
# HIGH-confidence candidates auto-flow into AnalystDecisions.related_parties so
# C03/C04 fires without manual checkbox confirmation. MEDIUM/LOW still surface
# for analyst review.
_RP_CONCENTRATION_DR_THRESHOLD = 0.05    # cp gross DR / total gross DR
_RP_RECURRENCE_MIN_MONTHS = 3            # distinct calendar months with DRs
_RP_BIDIRECTIONAL_MIN_SIDE_COUNT = 2     # min(cr_count, dr_count)
_RP_ROUND_AMOUNT_FLOOR = 1000.0
_RP_ROUND_AMOUNT_MULTIPLE = 100.0
_RP_ROUND_HITS_MIN = 2
# Sustained-round director-draw pattern: ≥5 round DRs spread across
# ≥4 calendar months upgrades round-amount signal from weight 1 → 2.
# Captures revolving director advances (the textbook RP shape) while
# leaving the weak "≥2 round DRs" tier untouched for one-off vendor refunds.
_RP_ROUND_SUSTAINED_HITS_MIN = 5
_RP_ROUND_SUSTAINED_MONTHS_MIN = 4

# Synthetic / pattern-fallback labels that must never be treated as personal names.
_RP_EXCLUDE_PREFIXES = ("UNIDENTIFIED", "UNNAMED")
_RP_EXCLUDE_NAMES = {
    "UNCATEGORIZED", "REVERSAL", "RETURNED CHEQUE", "JANM", "APAYLATER",
    "BULK SALARY", "BANK FEES", "FD/INTEREST", "LOAN REPAYMENT",
    "LOAN DISBURSEMENT", "CASH DEPOSIT", "CASH WITHDRAWAL", "INWARD RETURN",
    "KWSP", "SOCSO", "LHDN", "HRDF",
}

# Score → confidence tier. Strong signals (weight 2) are concentration,
# personal-keyword sweep, and bidirectional flow — each captures one of the
# two director patterns the user called out: "many payments going into
# personal account" or "receive/give advances or loan". Two strong or
# strong+weak land HIGH; a single strong is MEDIUM; a single weak is LOW.
_RP_HIGH_SCORE = 3
_RP_MEDIUM_SCORE = 2


def _compute_rp_signals(cp: dict[str, Any], gross_dr: float) -> dict[str, Any] | None:
    """Score one counterparty against the five RP signals. Return None if no
    signal fires. Each entry in `signals` is a (name, weight, evidence) tuple."""
    debit_count = int(cp.get("debit_count", 0) or 0)
    credit_count = int(cp.get("credit_count", 0) or 0)
    total_dr = float(cp.get("total_debits", 0.0) or 0.0)
    total_cr = float(cp.get("total_credits", 0.0) or 0.0)
    txs = cp.get("transactions", []) or []
    cp_name = cp.get("counterparty_name") or ""

    # Parser-stamped suffix (app.py _rhb16_finalize / RFLX single-token branch)
    # indicating the bucket consolidates multiple distinct people sharing a
    # common first name (MOHAMMAD, WAN, ARKAS…). Cannot auto-confirm as RP
    # without analyst disambiguation — force LOW regardless of signal score.
    is_ambiguous_multi_party = "(possibly multiple parties)" in cp_name.lower()

    signals: list[tuple[str, int, str]] = []

    # 1. Personal-keyword sweep (existing baseline; weight 2).
    if debit_count >= 3:
        personal_hits = sum(
            1 for tx in txs
            if any(kw in (tx.get("description") or "").upper() for kw in PERSONAL_KEYWORDS_RP4)
        )
        if personal_hits >= 2:
            signals.append(("personal_keyword_sweep", 2, f"{personal_hits} personal-kw rows"))

    # 2. Concentration: cp gross DR ≥ 5% of total gross DR (weight 2).
    if gross_dr > 0 and total_dr / gross_dr >= _RP_CONCENTRATION_DR_THRESHOLD:
        signals.append((
            "concentration_dr",
            2,
            f"DR {100 * total_dr / gross_dr:.1f}% of gross",
        ))

    # 3. Monthly recurrence: DR rows span ≥ 3 distinct calendar months (weight 1).
    dr_months = {
        (tx.get("date") or "")[:7]
        for tx in txs if tx.get("type") == "DEBIT"
    }
    dr_months.discard("")
    if len(dr_months) >= _RP_RECURRENCE_MIN_MONTHS:
        signals.append(("monthly_recurrence", 1, f"DR over {len(dr_months)} months"))

    # 4. Bidirectional flow (director loan-account pattern; weight 2 — strong).
    # Require min(cr_count, dr_count) ≥ 2 to filter out one-off vendor refunds.
    if min(credit_count, debit_count) >= _RP_BIDIRECTIONAL_MIN_SIDE_COUNT:
        signals.append((
            "bidirectional_flow",
            2,
            f"{credit_count}CR / {debit_count}DR",
        ))

    # 5. Round-number advances. Two tiers:
    #   - Weak (weight 1): ≥ 2 round DRs (multiples of 100 above 1000) —
    #     could be one-off vendor refund or genuine draw.
    #   - Sustained (weight 2): ≥ 5 round DRs across ≥ 4 calendar months —
    #     revolving director-draw cadence; very unlikely to be coincidence.
    round_dr_months: set[str] = set()
    round_hits = 0
    for tx in txs:
        if tx.get("type") != "DEBIT":
            continue
        amt = float(tx.get("amount") or 0.0)
        if amt < _RP_ROUND_AMOUNT_FLOOR or amt % _RP_ROUND_AMOUNT_MULTIPLE != 0:
            continue
        round_hits += 1
        m = (tx.get("date") or "")[:7]
        if m:
            round_dr_months.add(m)
    if (
        round_hits >= _RP_ROUND_SUSTAINED_HITS_MIN
        and len(round_dr_months) >= _RP_ROUND_SUSTAINED_MONTHS_MIN
    ):
        signals.append((
            "round_amount_sustained", 2,
            f"{round_hits} round DRs over {len(round_dr_months)} months",
        ))
    elif round_hits >= _RP_ROUND_HITS_MIN:
        signals.append(("round_amount_advance", 1, f"{round_hits} round DRs"))

    if not signals:
        return None

    score = sum(weight for _, weight, _ in signals)

    # Ambiguous + one-way flow → analyst must disambiguate (force LOW).
    # Ambiguous + bidirectional → director-loan-account pattern (e.g. RHB
    # Waja ASHRUL: 57 DR / 22 CR with 'Bayar balik IBK' memos). The
    # back-and-forth itself is the disambiguating signal — it would be
    # extraordinarily unlikely for N different ATASHAs / ASHRULs to all
    # exhibit revolving flow with the same descriptor template — so we
    # keep the computed score there.
    has_bidirectional = any(sig == "bidirectional_flow" for sig, _, _ in signals)
    force_low_ambiguous = is_ambiguous_multi_party and not has_bidirectional

    if force_low_ambiguous:
        confidence = "LOW"
    elif score >= _RP_HIGH_SCORE:
        confidence = "HIGH"
    elif score >= _RP_MEDIUM_SCORE:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "signals": [sig for sig, _, _ in signals],
        "score": score,
        "confidence": confidence,
        "ambiguous_multi_party": is_ambiguous_multi_party,
        "evidence": " · ".join(ev for _, _, ev in signals),
        "total_dr": round(total_dr, 2),
        "total_cr": round(total_cr, 2),
        "debit_count": debit_count,
        "credit_count": credit_count,
    }


def scan_related_party_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    """RP4 director-like sweep — surface for analyst confirmation.

    V3-A auto-RP Step 1: candidates are scored across five deterministic
    behavioral signals (personal keywords, DR concentration, monthly recurrence,
    bidirectional flow, round-amount advances). HIGH-confidence candidates are
    auto-confirmed downstream; MEDIUM/LOW surface in the analyst form.
    """
    cps = data.get("counterparty_ledger", {}).get("counterparties", [])
    gross_dr = sum(float(cp.get("total_debits", 0.0) or 0.0) for cp in cps)

    candidates: list[dict[str, Any]] = []
    for cp in cps:
        name = cp.get("counterparty_name") or ""
        if not name:
            continue
        upper = name.upper()
        if _looks_like_company(name) or upper in BUCKET_TO_CATEGORY:
            continue
        if upper in _RP_EXCLUDE_NAMES:
            continue
        if any(upper.startswith(p) for p in _RP_EXCLUDE_PREFIXES):
            continue

        scored = _compute_rp_signals(cp, gross_dr)
        if scored is None:
            continue

        candidates.append({
            "name": name,
            "method": scored["signals"][0],  # backwards-compat: primary trigger
            **scored,
        })

    confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    candidates.sort(key=lambda c: (confidence_order.get(c["confidence"], 9), -c["total_dr"]))
    return candidates


def auto_confirmed_related_parties(candidates: list[dict[str, Any]]) -> list[str]:
    """Names from RP scan whose deterministic score reached HIGH confidence.
    These are merged into AnalystDecisions.related_parties without analyst
    checkbox — Step 1 auto-RP behavior."""
    return [c["name"] for c in candidates if c.get("confidence") == "HIGH"]


def _looks_like_company(name: str) -> bool:
    upper = name.upper()
    company_markers = ["SDN", "BHD", "ENTERPRISE", "BANK", "TRADING", "HOLDINGS",
                       "SERVICES", "CORPORATION", "GROUP", "PRIVATE"]
    return any(m in upper for m in company_markers)


def scan_purpose_clusters(data: dict[str, Any]) -> dict[str, Any]:
    """Histogram DR purpose keywords. Flag cluster >20% of gross DR."""
    keyword_buckets = {
        "commission": ["COMMISSION", "KOMISEN", "HABUAN"],
        "salary": ["SALARY", "GAJI", "PAYROLL", "BULK SALARY"],
        "fee_income": ["FEE", "FI ", "ADV FI"],
    }
    totals = {k: 0.0 for k in keyword_buckets}
    gross_dr = 0.0
    for tx in data.get("transactions", []):
        amt = float(tx.get("debit") or 0.0)
        if amt <= 0:
            continue
        gross_dr += amt
        desc = (tx.get("description") or "").upper()
        for cluster, kws in keyword_buckets.items():
            if any(kw in desc for kw in kws):
                totals[cluster] += amt
                break
    clusters = {k: {"amount": round(v, 2), "pct": round(100 * v / gross_dr, 2) if gross_dr else 0}
                for k, v in totals.items()}
    dominant = max(clusters.items(), key=lambda kv: kv[1]["pct"], default=(None, {"pct": 0}))
    needs_confirmation = dominant[1]["pct"] >= 20.0 if dominant[0] else False
    return {"clusters": clusters, "dominant": dominant[0], "needs_confirmation": needs_confirmation}


# ---------------------------------------------------------------------------
# Stage B — classification
# ---------------------------------------------------------------------------


def _build_counterparty_lookup(data: dict[str, Any]) -> dict[tuple, str]:
    """Map (date, description, amount, type) -> counterparty_name from ledger."""
    lookup: dict[tuple, str] = {}
    for cp in data.get("counterparty_ledger", {}).get("counterparties", []):
        name = cp.get("counterparty_name") or ""
        for tx in cp.get("transactions", []):
            key = (tx.get("date"), tx.get("description"), tx.get("amount"), tx.get("type"))
            lookup[key] = name
    return lookup


def _tx_side_amount(tx: dict[str, Any]) -> tuple[str | None, float]:
    debit = float(tx.get("debit") or 0.0)
    credit = float(tx.get("credit") or 0.0)
    if credit > 0:
        return "CR", credit
    if debit > 0:
        return "DR", debit
    return None, 0.0


_COMPANY_SUFFIXES = re.compile(
    r"\b(SDN\.?\s*BHD\.?|BERHAD|ENTERPRISE|HOLDINGS|TRADING|SERVICES|"
    r"CORPORATION|GROUP|PRIVATE|LIMITED|LTD|PLT|CORP|INC|BHD)\b",
    re.IGNORECASE,
)

# V2.3 — parenthetical disambiguators that don't change identity
# (e.g. "ALPHA (M)", "BETA (SARAWAK)" should canonicalize without the parens).
_PAREN_DISAMBIGUATOR = re.compile(r"\s*\([^)]*\)\s*")

# Trailing single-letter tokens are likely truncation artifacts ("PLANWORTH GLOBAL S"
# is the parser's clipped form of "...SDN BHD"). Strip them after suffix removal.
_TRAILING_SINGLE_TOKEN = re.compile(r"\s+[A-Z]\s*$")


def _company_root(name: str) -> str:
    if not name:
        return ""
    # Strip parenthetical disambiguators BEFORE corporate suffixes so concat-
    # form holders like `KOPERASIKAKITANGANFELCRA(M)BERHAD` reduce to a clean
    # `KOPERASIKAKITANGANFELCRA` root that substring-matches the descriptive
    # bucket `KOPERASIKAKITANGANFELCRA BERHAD`. Mirrors the order used by
    # `_canonicalize_for_merge` (line 529) so the two helpers stay aligned.
    upper = _PAREN_DISAMBIGUATOR.sub(" ", name.upper())
    cleaned = _COMPANY_SUFFIXES.sub("", upper)
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _canonicalize_for_merge(name: str) -> str:
    """V2.3 M1+M2: strip corporate suffixes and parenthetical disambiguators.
    Returns a normalized key for grouping variants of the same entity.
    Empty string for falsy input — caller should treat empty as ungroupable."""
    if not name:
        return ""
    upper = name.upper().strip()
    upper = _PAREN_DISAMBIGUATOR.sub(" ", upper)
    cleaned = _COMPANY_SUFFIXES.sub("", upper)
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Drop trailing single-letter token (likely truncated suffix initial)
    prev = None
    while cleaned != prev:
        prev = cleaned
        cleaned = _TRAILING_SINGLE_TOKEN.sub("", cleaned).strip()
    return cleaned


_MERGE_M3_TOP_N = 200  # bound M3 substring search to top-N by amount; tail won't reach top_parties anyway


def _merge_counterparty_groups(
    bucket: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """V2.3 entity canonicalization for top_parties.

    M1+M2 — group by _canonicalize_for_merge() output (cheap hash-based).
    M3    — substring/token-subset merge among the top-N canonical keys by total_amount
            (bounded: noisy parser-extraction tails dominate cost without affecting output).
    M4    — Levenshtein fuzzy merge: deferred. Marginal recall gain over M1+M3 vs cubic
            scaling on files with >500 unique counterparties (KYDN: 1742 CPs).

    Returns a new dict keyed by canonical form. Each entry merges totals + monthly
    breakdowns from variants and picks the longest original name as display.
    """
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    canon_amount: dict[str, float] = {}
    for original, entry in bucket.items():
        canon = _canonicalize_for_merge(original) or original.upper().strip()
        grouped.setdefault(canon, []).append((original, entry))
        canon_amount[canon] = canon_amount.get(canon, 0.0) + entry["total_amount"]

    canon_remap: dict[str, str] = {k: k for k in grouped}

    # M3 — substring/token-subset merge, bounded to top-N keys by amount
    top_keys = sorted(canon_amount, key=lambda k: -canon_amount[k])[:_MERGE_M3_TOP_N]
    top_keys_by_len = sorted(top_keys, key=len, reverse=True)
    token_cache = {k: set(k.split()) for k in top_keys}
    for short in top_keys_by_len:
        if not short or len(short) < 6 or canon_remap[short] != short:
            continue
        short_tokens = token_cache[short]
        if not short_tokens:
            continue
        for long in top_keys_by_len:
            if long == short or len(long) <= len(short) or canon_remap[long] != long:
                continue
            if short_tokens.issubset(token_cache[long]):
                canon_remap[short] = long
                break

    def resolve(k: str) -> str:
        seen = set()
        while canon_remap[k] != k and k not in seen:
            seen.add(k)
            k = canon_remap[k]
        return k

    merged: dict[str, dict[str, Any]] = {}
    for canon, items in grouped.items():
        target = resolve(canon)
        slot = merged.setdefault(target, {
            "party_name": "",
            "total_amount": 0.0,
            "transaction_count": 0,
            "is_related_party": False,
            "_monthly": {},
        })
        for original, entry in items:
            # Pick the longest original as display name (strip leading/trailing whitespace)
            if len(original.strip()) > len(slot["party_name"]):
                slot["party_name"] = original.strip()
            slot["total_amount"] += entry["total_amount"]
            slot["transaction_count"] += entry["transaction_count"]
            slot["is_related_party"] = slot["is_related_party"] or entry["is_related_party"]
            for month, mb in entry["_monthly"].items():
                target_mb = slot["_monthly"].setdefault(
                    month, {"month": month, "amount": 0.0, "count": 0}
                )
                target_mb["amount"] += mb["amount"]
                target_mb["count"] += mb["count"]
    return merged


def _own_party_match(cp_upper: str, desc_upper: str, company_roots: list[str]) -> bool:
    """Bidirectional match: company-root vs counterparty-bucket vs description.
    Min length 5 chars on root (lowered from 6 in Sprint 7 Phase 2A to unblock
    short distinctive holder names like `MAZAA`). The cp-side check keeps its
    6-char floor as a guard against generic bucket-name FPs (`UNNAMED`, etc.).
    Audit (2026-04-28): Felcra/Waja/KMZ/Principal Gas/Mytutor all have roots
    ≥9 chars; Mazaa is the only 5-char-root corpus, and its 55 `MAZAA` desc
    rows are 100% legitimate own-party DuitNow self-transfers."""
    for root in company_roots:
        if not root or len(root) < 5:
            continue
        if root in desc_upper:
            return True
        if cp_upper and (root in cp_upper or (len(cp_upper) >= 6 and cp_upper in root)):
            return True
    return False


def _ledger_key(tx: dict[str, Any]) -> tuple:
    side, amt = _tx_side_amount(tx)
    return (
        tx.get("date"),
        tx.get("description"),
        round(amt, 2),
        "CREDIT" if side == "CR" else "DEBIT" if side == "DR" else None,
    )


def classify_transactions(
    data: dict[str, Any],
    rulebook: dict[str, Any],
    decisions: AnalystDecisions,
) -> list[dict[str, Any]]:
    cp_lookup = _build_counterparty_lookup(data)
    company_roots = [_company_root(n) for n in data.get("summary", {}).get("company_names", []) if n]
    company_roots = [r for r in company_roots if r]
    related_upper = [r.upper() for r in decisions.related_parties if r]
    factoring_upper = [f.upper() for f in decisions.factoring_entities if f]

    classified = []
    for tx in data.get("transactions", []):
        side, amount = _tx_side_amount(tx)
        desc_upper = (tx.get("description") or "").upper()
        cp_name = cp_lookup.get(_ledger_key(tx)) or ""
        cp_upper = cp_name.upper()

        primary: str | None = None
        reason: str | None = None

        # C25 — balance/opening rows always first
        if tx.get("is_statement_balance") or tx.get("is_opening_balance"):
            primary = "C25"
            reason = "Statement/opening balance row"
        # C01/C02 — own-party (runs BEFORE bucket-direct so LOAN REPAYMENT etc.
        # don't override own-account transfers; matches AI-reference convention).
        # Per CLASSIFICATION_RULES_v3_5: C01=CR, C02=DR.
        elif side and _own_party_match(cp_upper, desc_upper, company_roots):
            primary = "C01" if side == "CR" else "C02"
            reason = f"Own-party: {cp_name}" if cp_name else "Own-party (description match)"
        # C03/C04 — related-party (analyst-confirmed); also priority over bucket
        elif side and related_upper and any(r in cp_upper or r in desc_upper for r in related_upper):
            primary = "C03" if side == "CR" else "C04"
            reason = "Related-party (analyst-confirmed)"
        # Counterparty-bucket direct mapping (parser already labeled).
        # Side must match the category's canonical side — guards against
        # mis-bucketed CR rows ending up in DR-only buckets like BULK SALARY.
        elif cp_upper in BUCKET_TO_CATEGORY:
            candidate = BUCKET_TO_CATEGORY[cp_upper]
            expected = _CATEGORY_SIDES.get(candidate, "ANY")
            if expected == "ANY" or expected == side:
                primary = candidate
                reason = f"Counterparty bucket: {cp_upper}"
        # C10 — factoring disbursement (analyst-confirmed counterparty)
        elif side == "CR" and any(f in cp_upper or f in desc_upper for f in factoring_upper):
            primary = "C10"
            reason = "Factoring disbursement (analyst-confirmed)"
        # C06–C09 — statutory fallback when bucket missed it
        elif side == "DR" and (stat := statutory_bucket_for(tx.get("description") or "")):
            primary = {"KWSP": "C06", "SOCSO": "C07", "LHDN": "C08", "HRDF": "C09"}[stat]
            reason = f"Statutory keyword: {stat}"
        # Description-keyword fallbacks (cheque/cash/reversal/salary) when
        # parser hasn't routed the row to a known bucket
        else:
            for pattern, required_side, code in _KEYWORD_RULES:
                if side != required_side:
                    continue
                if pattern.search(desc_upper):
                    primary = code
                    reason = f"Keyword fallback: {code}"
                    break

        # C26/C27 — trade in/out final fallback. Per CLASSIFICATION_RULES_v3_5
        # LOCKED rule: counterparty has corporate marker, NOT natural person,
        # NOT already routed by any earlier rule. Bucket-only counterparties
        # (BULK SALARY, KWSP etc.) and parser-dropouts (TRANSFER FR/TO A/C)
        # are excluded by the existing bucket / drop checks.
        if primary is None and side and cp_name and not should_drop_as_counterparty(cp_name):
            if (cp_upper not in BUCKET_TO_CATEGORY
                    and _CORPORATE_ENTITY_MARKERS.search(cp_upper)
                    and not _NATURAL_PERSON_MARKERS.search(cp_upper)):
                primary = "C26" if side == "CR" else "C27"
                reason = f"Trade {'income' if side == 'CR' else 'expense'}: {cp_name}"

        out = dict(tx)
        out["_side"] = side
        out["_amount"] = round(amount, 2)
        out["_counterparty_name"] = cp_name
        out["classification"] = {
            "primary": primary,
            "dual_tags": [],
            "side": side,
            "reason": reason,
            "mode": "FULL_CODE",
        }
        classified.append(out)

    return classified


# ---------------------------------------------------------------------------
# Stage C — aggregation
# ---------------------------------------------------------------------------


def _empty_monthly_row(month: str, account_no: str, bank_name: str) -> dict[str, Any]:
    return {
        "month": month,
        "account_number": account_no,
        "bank_name": bank_name,
        "gross_credits": 0.0, "gross_debits": 0.0,
        "net_credits": 0.0, "net_debits": 0.0,
        "credit_count": 0, "debit_count": 0,
        "own_party_cr": 0.0, "own_party_dr": 0.0,
        "related_party_cr": 0.0, "related_party_dr": 0.0,
        "reversal_cr": 0.0,
        "returned_cheques_inward_count": 0, "returned_cheques_inward_amount": 0.0,
        "returned_cheques_outward_count": 0, "returned_cheques_outward_amount": 0.0,
        "loan_disbursement_cr": 0.0, "fd_interest_cr": 0.0,
        "round_figure_cr": 0.0, "high_value_cr": 0.0,
        "cash_deposits_count": 0, "cash_deposits_amount": 0.0,
        "cash_withdrawals_count": 0, "cash_withdrawals_amount": 0.0,
        "cheque_deposits_count": 0, "cheque_deposits_amount": 0.0,
        "cheque_issues_count": 0, "cheque_issues_amount": 0.0,
        "loan_repayment_dr": 0.0, "salary_paid": 0.0,
        "statutory_epf": 0.0, "statutory_socso": 0.0,
        "statutory_tax": 0.0, "statutory_hrdf": 0.0,
        "eod_lowest": 0.0, "eod_highest": 0.0, "eod_average": 0.0,
        "opening_balance": 0.0, "closing_balance": 0.0,
        "fx_credit_count": 0, "fx_credit_amount": 0.0,
        "fx_debit_count": 0, "fx_debit_amount": 0.0,
        "reconciliation_status": "PASS", "reconciliation_delta": 0.0,
        "extraction_gaps": 0,
        "missing_debit_amount": 0.0, "missing_credit_amount": 0.0,
        "own_party_cr_count": 0, "own_party_dr_count": 0,
        "related_party_cr_count": 0, "related_party_dr_count": 0,
        "loan_repayment_count": 0,
        "inward_return_cr": 0.0,
        "unclassified_cr_count": 0, "unclassified_cr_amount": 0.0,
        "unclassified_dr_count": 0, "unclassified_dr_amount": 0.0,
        "trade_income_count": 0, "trade_income_amount": 0.0,
        "trade_expense_count": 0, "trade_expense_amount": 0.0,
    }


def build_monthly_analysis(
    classified: list[dict[str, Any]],
    data: dict[str, Any],
    reconciliation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Per (account, month) aggregation with all 60 schema fields."""
    rows: dict[tuple, dict[str, Any]] = {}
    eod_balances: dict[tuple, dict[str, float]] = {}  # (acct, month) -> {date: balance}
    recon_by_key = {(d["account_number"], d["month"]): d for d in reconciliation.get("deltas", [])}

    # Seed rows from monthly_summary
    for m in data.get("monthly_summary", []):
        key = (m.get("account_no"), m.get("month"))
        bank = _bank_for_month(data, m)
        row = _empty_monthly_row(m.get("month") or "", m.get("account_no") or "", bank)
        row["opening_balance"] = round(float(m.get("opening_balance") or 0.0), 2)
        row["closing_balance"] = round(float(m.get("ending_balance") or 0.0), 2)
        row["eod_lowest"] = round(float(m.get("lowest_balance") or 0.0), 2)
        row["eod_highest"] = round(float(m.get("highest_balance") or 0.0), 2)
        rd = recon_by_key.get(key)
        if rd:
            row["reconciliation_status"] = "PASS" if rd["passed"] else "FAIL"
            row["reconciliation_delta"] = rd["reconciliation_delta"]
        rows[key] = row
        eod_balances[key] = {}

    # Walk classified rows
    for tx in classified:
        date = tx.get("date") or ""
        month = date[:7] if date else ""
        acct = tx.get("account_no") or ""
        key = (acct, month)
        if key not in rows:
            rows[key] = _empty_monthly_row(month, acct, tx.get("bank") or "Unknown")
            eod_balances[key] = {}

        row = rows[key]
        primary = tx.get("classification", {}).get("primary")
        side = tx.get("_side")
        amt = tx.get("_amount", 0.0)

        # Skip C25 (balance rows) from all aggregation
        if primary == "C25":
            continue

        if side == "CR":
            row["gross_credits"] += amt
            row["credit_count"] += 1
        elif side == "DR":
            row["gross_debits"] += amt
            row["debit_count"] += 1

        # Track EOD balance per date (last balance seen wins)
        bal = tx.get("balance")
        if bal is not None and date:
            eod_balances[key][date] = float(bal)

        # Category-specific buckets
        if primary == "C01":  # own-party DR
            row["own_party_dr"] += amt
            row["own_party_dr_count"] += 1
        elif primary == "C02":  # own-party CR
            row["own_party_cr"] += amt
            row["own_party_cr_count"] += 1
        elif primary == "C03":
            row["related_party_dr"] += amt
            row["related_party_dr_count"] += 1
        elif primary == "C04":
            row["related_party_cr"] += amt
            row["related_party_cr_count"] += 1
        elif primary == "C05":
            row["salary_paid"] += amt
        elif primary == "C06":
            row["statutory_epf"] += amt
        elif primary == "C07":
            row["statutory_socso"] += amt
        elif primary == "C08":
            row["statutory_tax"] += amt
        elif primary == "C09":
            row["statutory_hrdf"] += amt
        elif primary == "C10":
            row["loan_disbursement_cr"] += amt
        elif primary == "C11":
            row["loan_repayment_dr"] += amt
            row["loan_repayment_count"] += 1
        elif primary == "C12":
            row["fd_interest_cr"] += amt
        elif primary == "C13":
            row["reversal_cr"] += amt
        elif primary == "C14":  # Returned Cheques Inward (DR)
            row["returned_cheques_inward_count"] += 1
            row["returned_cheques_inward_amount"] += amt
        elif primary == "C15":  # Returned Cheques Outward (CR)
            row["returned_cheques_outward_count"] += 1
            row["returned_cheques_outward_amount"] += amt
        elif primary == "C16":  # IBG/GIRO Inward Return (CR)
            row["inward_return_cr"] += amt
        elif primary == "C17":  # Cash Deposit (CR)
            row["cash_deposits_count"] += 1
            row["cash_deposits_amount"] += amt
        elif primary == "C18":  # Cash Withdrawal (DR)
            row["cash_withdrawals_count"] += 1
            row["cash_withdrawals_amount"] += amt
        elif primary == "C19":  # Cheque Deposit (CR)
            row["cheque_deposits_count"] += 1
            row["cheque_deposits_amount"] += amt
        elif primary == "C20":  # Cheque Issue (DR)
            row["cheque_issues_count"] += 1
            row["cheque_issues_amount"] += amt
        elif primary == "C26":  # Trade Income (CR)
            row["trade_income_count"] += 1
            row["trade_income_amount"] += amt
        elif primary == "C27":  # Trade Expense (DR)
            row["trade_expense_count"] += 1
            row["trade_expense_amount"] += amt
        elif primary is None:
            if side == "CR":
                row["unclassified_cr_count"] += 1
                row["unclassified_cr_amount"] += amt
            elif side == "DR":
                row["unclassified_dr_count"] += 1
                row["unclassified_dr_amount"] += amt

        # AML / monitoring
        if side == "CR" and amt >= ROUND_FIGURE_MIN and amt == round(amt / 1000) * 1000:
            row["round_figure_cr"] += amt
        if side == "CR" and amt >= HIGH_VALUE_THRESHOLD:
            row["high_value_cr"] += amt

    # Net + EOD average
    for key, row in rows.items():
        row["net_credits"] = round(row["gross_credits"] - row["own_party_cr"]
                                    - row["related_party_cr"] - row["reversal_cr"]
                                    - row["loan_disbursement_cr"], 2)
        row["net_debits"] = round(row["gross_debits"] - row["own_party_dr"]
                                   - row["related_party_dr"] - row["loan_repayment_dr"], 2)
        row["gross_credits"] = round(row["gross_credits"], 2)
        row["gross_debits"] = round(row["gross_debits"], 2)
        row["own_party_cr"] = round(row["own_party_cr"], 2)
        row["own_party_dr"] = round(row["own_party_dr"], 2)
        row["related_party_cr"] = round(row["related_party_cr"], 2)
        row["related_party_dr"] = round(row["related_party_dr"], 2)
        row["reversal_cr"] = round(row["reversal_cr"], 2)
        row["loan_disbursement_cr"] = round(row["loan_disbursement_cr"], 2)
        row["loan_repayment_dr"] = round(row["loan_repayment_dr"], 2)
        row["fd_interest_cr"] = round(row["fd_interest_cr"], 2)
        row["round_figure_cr"] = round(row["round_figure_cr"], 2)
        row["high_value_cr"] = round(row["high_value_cr"], 2)
        row["cash_deposits_amount"] = round(row["cash_deposits_amount"], 2)
        row["cash_withdrawals_amount"] = round(row["cash_withdrawals_amount"], 2)
        row["salary_paid"] = round(row["salary_paid"], 2)
        row["statutory_epf"] = round(row["statutory_epf"], 2)
        row["statutory_socso"] = round(row["statutory_socso"], 2)
        row["statutory_tax"] = round(row["statutory_tax"], 2)
        row["statutory_hrdf"] = round(row["statutory_hrdf"], 2)
        row["inward_return_cr"] = round(row["inward_return_cr"], 2)
        row["unclassified_cr_amount"] = round(row["unclassified_cr_amount"], 2)
        row["unclassified_dr_amount"] = round(row["unclassified_dr_amount"], 2)
        row["trade_income_amount"] = round(row["trade_income_amount"], 2)
        row["trade_expense_amount"] = round(row["trade_expense_amount"], 2)

        bals = list(eod_balances.get(key, {}).values())
        if bals:
            row["eod_average"] = round(sum(bals) / len(bals), 2)

    return sorted(rows.values(), key=lambda r: (r["account_number"], r["month"]))


def build_consolidated(monthly: list[dict[str, Any]]) -> dict[str, Any]:
    if not monthly:
        return _empty_consolidated()

    months_count = len({m["month"] for m in monthly}) or 1
    s = lambda key: round(sum(m.get(key, 0) for m in monthly), 2)
    sc = lambda key: int(sum(m.get(key, 0) for m in monthly))

    consolidated = {
        "gross_credits": s("gross_credits"),
        "gross_debits": s("gross_debits"),
        "net_credits": s("net_credits"),
        "net_debits": s("net_debits"),
        "annualized_net_credits": round(s("net_credits") * 12 / months_count, 2),
        "annualized_net_debits": round(s("net_debits") * 12 / months_count, 2),
        "total_own_party_cr": s("own_party_cr"),
        "total_own_party_dr": s("own_party_dr"),
        "total_related_party_cr": s("related_party_cr"),
        "total_related_party_dr": s("related_party_dr"),
        "total_reversal_cr": s("reversal_cr"),
        "total_returned_cheques_inward": s("returned_cheques_inward_amount"),
        "total_returned_cheques_outward": s("returned_cheques_outward_amount"),
        "total_loan_disbursement_cr": s("loan_disbursement_cr"),
        "total_fd_interest_cr": s("fd_interest_cr"),
        "total_round_figure_cr": s("round_figure_cr"),
        "total_high_value_cr": s("high_value_cr"),
        "total_cash_deposits": s("cash_deposits_amount"),
        "total_cash_withdrawals": s("cash_withdrawals_amount"),
        "total_cheque_deposits": s("cheque_deposits_amount"),
        "total_cheque_issues": s("cheque_issues_amount"),
        "total_loan_repayment_dr": s("loan_repayment_dr"),
        "total_salary_paid": s("salary_paid"),
        "total_statutory_epf": s("statutory_epf"),
        "total_statutory_socso": s("statutory_socso"),
        "total_statutory_tax": s("statutory_tax"),
        "total_statutory_hrdf": s("statutory_hrdf"),
        "eod_lowest": min((m.get("eod_lowest", 0) for m in monthly), default=0.0),
        "eod_highest": max((m.get("eod_highest", 0) for m in monthly), default=0.0),
        "eod_average": round(sum(m.get("eod_average", 0) for m in monthly) / months_count, 2),
        "total_fx_credits": s("fx_credit_amount"),
        "total_fx_debits": s("fx_debit_amount"),
        "data_completeness": "COMPLETE" if all(m["reconciliation_status"] == "PASS" for m in monthly) else "INCOMPLETE",
        "months_with_gaps": sum(1 for m in monthly if m["reconciliation_status"] != "PASS"),
        "total_extraction_gaps": sc("extraction_gaps"),
        "total_missing_debits": s("missing_debit_amount"),
        "total_missing_credits": s("missing_credit_amount"),
        "total_inward_return_cr": s("inward_return_cr"),
        "total_unclassified_cr": s("unclassified_cr_amount"),
        "total_unclassified_dr": s("unclassified_dr_amount"),
        "total_trade_income_cr": s("trade_income_amount"),
        "total_trade_income_count": sc("trade_income_count"),
        "total_trade_expense_dr": s("trade_expense_amount"),
        "total_trade_expense_count": sc("trade_expense_count"),
        "statutory_compliance": _build_statutory_compliance(monthly),
    }
    return consolidated


def _empty_consolidated() -> dict[str, Any]:
    return {
        "gross_credits": 0.0, "gross_debits": 0.0, "net_credits": 0.0, "net_debits": 0.0,
        "annualized_net_credits": 0.0, "annualized_net_debits": 0.0,
        "total_own_party_cr": 0.0, "total_own_party_dr": 0.0,
        "total_related_party_cr": 0.0, "total_related_party_dr": 0.0,
        "total_reversal_cr": 0.0, "total_returned_cheques_inward": 0.0, "total_returned_cheques_outward": 0.0,
        "total_loan_disbursement_cr": 0.0, "total_fd_interest_cr": 0.0,
        "total_round_figure_cr": 0.0, "total_high_value_cr": 0.0,
        "total_cash_deposits": 0.0, "total_cash_withdrawals": 0.0,
        "total_cheque_deposits": 0.0, "total_cheque_issues": 0.0,
        "total_loan_repayment_dr": 0.0, "total_salary_paid": 0.0,
        "total_statutory_epf": 0.0, "total_statutory_socso": 0.0, "total_statutory_tax": 0.0, "total_statutory_hrdf": 0.0,
        "eod_lowest": 0.0, "eod_highest": 0.0, "eod_average": 0.0,
        "total_fx_credits": 0.0, "total_fx_debits": 0.0,
        "data_completeness": "COMPLETE", "months_with_gaps": 0, "total_extraction_gaps": 0,
        "total_missing_debits": 0.0, "total_missing_credits": 0.0, "total_inward_return_cr": 0.0,
        "total_unclassified_cr": 0.0, "total_unclassified_dr": 0.0,
        "total_trade_income_cr": 0.0, "total_trade_income_count": 0,
        "total_trade_expense_dr": 0.0, "total_trade_expense_count": 0,
        "statutory_compliance": _empty_statutory_compliance(),
    }


def _build_statutory_compliance(monthly: list[dict[str, Any]]) -> dict[str, Any]:
    salary_months = [m["month"] for m in monthly if m.get("salary_paid", 0) > 0]
    epf_months = [m["month"] for m in monthly if m.get("statutory_epf", 0) > 0]
    socso_months = [m["month"] for m in monthly if m.get("statutory_socso", 0) > 0]
    lhdn_months = [m["month"] for m in monthly if m.get("statutory_tax", 0) > 0]
    hrdf_months = [m["month"] for m in monthly if m.get("statutory_hrdf", 0) > 0]

    salary_set = set(salary_months)
    epf_paid = sorted(salary_set & set(epf_months))
    socso_paid = sorted(salary_set & set(socso_months))
    epf_missing = sorted(salary_set - set(epf_months))
    socso_missing = sorted(salary_set - set(socso_months))

    epf_pct = round(100 * len(epf_paid) / len(salary_set), 2) if salary_set else 0.0
    socso_pct = round(100 * len(socso_paid) / len(salary_set), 2) if salary_set else 0.0

    epf_ratios = []
    socso_ratios = []
    for m in monthly:
        sal = float(m.get("salary_paid", 0))
        if sal <= 0:
            continue
        epf_amt = float(m.get("statutory_epf", 0))
        epf_ratio = round(100 * epf_amt / sal, 2) if sal else 0.0
        epf_ratios.append({
            "month": m["month"], "epf_amount": round(epf_amt, 2),
            "salary_amount": round(sal, 2), "ratio_pct": epf_ratio,
            "status": _epf_status(epf_ratio, epf_amt > 0),
        })
        socso_amt = float(m.get("statutory_socso", 0))
        socso_ratio = round(100 * socso_amt / sal, 2) if sal else 0.0
        socso_ratios.append({
            "month": m["month"], "socso_amount": round(socso_amt, 2),
            "salary_amount": round(sal, 2), "ratio_pct": socso_ratio,
            "status": _socso_status(socso_ratio, socso_amt > 0),
        })

    if not salary_set:
        overall = "COMPLIANT"
    elif epf_pct < 50 or socso_pct < 50:
        overall = "CRITICAL"
    elif epf_pct < 100 or socso_pct < 100:
        overall = "GAPS_DETECTED"
    else:
        overall = "COMPLIANT"

    return {
        "salary_months_active": len(salary_set),
        "salary_months_list": sorted(salary_set),
        "epf_months_paid": len(epf_paid),
        "epf_months_list": epf_paid,
        "epf_months_missing": epf_missing,
        "epf_coverage_pct": epf_pct,
        "socso_months_paid": len(socso_paid),
        "socso_months_list": socso_paid,
        "socso_months_missing": socso_missing,
        "socso_coverage_pct": socso_pct,
        "lhdn_months_paid": len(lhdn_months),
        "lhdn_detected": bool(lhdn_months),
        "hrdf_months_paid": len(hrdf_months),
        "hrdf_detected": bool(hrdf_months),
        "epf_per_month_ratios": epf_ratios,
        "socso_per_month_ratios": socso_ratios,
        "overall_status": overall,
    }


def _epf_status(ratio_pct: float, paid: bool) -> str:
    if not paid:
        return "WARNING"
    # Dual-band: 11-15% employer-only OR 20-26% combined (per v3.5.3 KDYN rule)
    if 11 <= ratio_pct <= 15 or 20 <= ratio_pct <= 26:
        return "OK"
    if ratio_pct > 26:
        return "CATCH_UP"
    if ratio_pct < 11:
        return "STRUCTURAL"
    return "WARNING"


def _socso_status(ratio_pct: float, paid: bool) -> str:
    if not paid:
        return "WARNING"
    if 0.5 <= ratio_pct <= 2.5:
        return "OK"
    if ratio_pct > 2.5:
        return "CATCH_UP"
    return "STRUCTURAL"


def _empty_statutory_compliance() -> dict[str, Any]:
    return {
        "salary_months_active": 0, "salary_months_list": [],
        "epf_months_paid": 0, "epf_months_list": [], "epf_months_missing": [],
        "epf_coverage_pct": 0.0,
        "socso_months_paid": 0, "socso_months_list": [], "socso_months_missing": [],
        "socso_coverage_pct": 0.0,
        "lhdn_months_paid": 0, "lhdn_detected": False,
        "hrdf_months_paid": 0, "hrdf_detected": False,
        "epf_per_month_ratios": [], "socso_per_month_ratios": [],
        "overall_status": "COMPLIANT",
    }


def build_top_parties(
    classified: list[dict[str, Any]],
    related_parties: list[str],
) -> dict[str, Any]:
    related_upper = [r.upper() for r in related_parties if r]
    payers: dict[str, dict[str, Any]] = {}  # CR
    payees: dict[str, dict[str, Any]] = {}  # DR

    def _ensure(bucket: dict, cp: str) -> dict[str, Any]:
        return bucket.setdefault(cp, {
            "party_name": cp, "total_amount": 0.0, "transaction_count": 0,
            "is_related_party": any(r in cp.upper() for r in related_upper),
            "_monthly": {},  # month -> {amount, count}
        })

    for tx in classified:
        primary = tx.get("classification", {}).get("primary")
        if primary == "C25":
            continue
        cp = tx.get("_counterparty_name") or ""
        if not cp or cp in BUCKET_TO_CATEGORY or should_drop_as_counterparty(cp):
            continue
        side = tx.get("_side")
        amt = tx.get("_amount", 0.0)
        bucket = payers if side == "CR" else payees if side == "DR" else None
        if bucket is None:
            continue
        entry = _ensure(bucket, cp)
        entry["total_amount"] += amt
        entry["transaction_count"] += 1
        month = (tx.get("date") or "")[:7]
        if month:
            mb = entry["_monthly"].setdefault(month, {"month": month, "amount": 0.0, "count": 0})
            mb["amount"] += amt
            mb["count"] += 1

    payers = _merge_counterparty_groups(payers)
    payees = _merge_counterparty_groups(payees)

    def _top(d: dict, k: int = 10) -> list[dict[str, Any]]:
        ranked = sorted(d.values(), key=lambda x: -x["total_amount"])[:k]
        out = []
        for i, e in enumerate(ranked, 1):
            mb = sorted(e["_monthly"].values(), key=lambda x: x["month"])
            for m in mb:
                m["amount"] = round(m["amount"], 2)
            out.append({
                "rank": i,
                "party_name": e["party_name"],
                "total_amount": round(e["total_amount"], 2),
                "transaction_count": e["transaction_count"],
                "is_related_party": e["is_related_party"],
                "monthly_breakdown": mb,
            })
        return out

    return {"top_payers": _top(payers), "top_payees": _top(payees)}


def build_large_credits(classified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tx in classified:
        if tx.get("classification", {}).get("primary") == "C25":
            continue
        if tx.get("_side") == "CR" and tx.get("_amount", 0) >= HIGH_VALUE_THRESHOLD:
            out.append({
                "date": tx.get("date"),
                "description": tx.get("description"),
                "amount": tx.get("_amount"),
                "type": "CREDIT",
                "account_number": tx.get("account_no") or "",
                "counterparty": tx.get("_counterparty_name") or None,
            })
    return out


def build_own_related_transactions(classified: list[dict[str, Any]]) -> dict[str, Any]:
    txs = []
    summary = {"own_party_cr": 0.0, "own_party_dr": 0.0,
               "related_party_cr": 0.0, "related_party_dr": 0.0}
    for tx in classified:
        primary = tx.get("classification", {}).get("primary")
        if primary not in ("C01", "C02", "C03", "C04"):
            continue
        side = tx.get("_side")
        amt = tx.get("_amount", 0.0)
        ptype = "OWN" if primary in ("C01", "C02") else "RELATED"
        txs.append({
            "date": tx.get("date"),
            "description": tx.get("description"),
            "amount": amt,
            "type": "CREDIT" if side == "CR" else "DEBIT",
            "party_type": ptype,
            "account_number": tx.get("account_no") or "",
        })
        if primary == "C01":
            summary["own_party_dr"] += amt
        elif primary == "C02":
            summary["own_party_cr"] += amt
        elif primary == "C03":
            summary["related_party_dr"] += amt
        elif primary == "C04":
            summary["related_party_cr"] += amt
    summary = {k: round(v, 2) for k, v in summary.items()}
    return {"summary": summary, "transactions": txs}


def build_loan_transactions(classified: list[dict[str, Any]]) -> dict[str, Any]:
    disb, repay = [], []
    for tx in classified:
        primary = tx.get("classification", {}).get("primary")
        entry = {
            "date": tx.get("date"),
            "description": tx.get("description"),
            "amount": tx.get("_amount"),
            "type": "CREDIT" if tx.get("_side") == "CR" else "DEBIT",
            "account_number": tx.get("account_no") or "",
            "counterparty": tx.get("_counterparty_name") or None,
        }
        if primary == "C10":
            disb.append(entry)
        elif primary == "C11":
            repay.append(entry)
    return {"disbursements": disb, "repayments": repay}


def build_unclassified(classified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tx in classified:
        if tx.get("classification", {}).get("primary") is not None:
            continue
        side = tx.get("_side")
        if side is None:
            continue
        out.append({
            "date": tx.get("date"),
            "description": tx.get("description"),
            "amount": tx.get("_amount"),
            "type": "CREDIT" if side == "CR" else "DEBIT",
            "account_number": tx.get("account_no") or "",
            "reason": "No matching rule (V1 deferred categories)",
        })
    return out


def build_flags(consolidated: dict[str, Any], monthly: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sc = consolidated.get("statutory_compliance", {})
    flags = []
    for fid, name in FLAG_DEFINITIONS:
        detected, remarks = _evaluate_flag(fid, name, consolidated, monthly, sc)
        flags.append({"id": fid, "name": name, "detected": detected, "remarks": remarks})
    return flags


def _evaluate_flag(
    fid: int, name: str,
    consolidated: dict[str, Any],
    monthly: list[dict[str, Any]],
    sc: dict[str, Any],
) -> tuple[bool, str]:
    if name == "Returned Cheques (Inward)":
        amt = consolidated.get("total_returned_cheques_inward", 0)
        return amt > 0, f"RM {amt:,.2f} across {sum(m['returned_cheques_inward_count'] for m in monthly)} cheques"
    if name == "Returned Cheques (Outward)":
        amt = consolidated.get("total_returned_cheques_outward", 0)
        return amt > 0, f"RM {amt:,.2f} across {sum(m['returned_cheques_outward_count'] for m in monthly)} cheques"
    if name == "Round Figure Credits (AML)":
        amt = consolidated.get("total_round_figure_cr", 0)
        return amt > 0, f"Round-figure credits totalling RM {amt:,.2f}"
    if name == "High Value Credits (>3x EOD)":
        amt = consolidated.get("total_high_value_cr", 0)
        return amt > 0, f"High-value credits totalling RM {amt:,.2f}"
    if name == "Cash Deposits (AML)":
        amt = consolidated.get("total_cash_deposits", 0)
        cnt = sum(m["cash_deposits_count"] for m in monthly)
        return cnt > 0, f"{cnt} cash deposits totalling RM {amt:,.2f}"
    if name == "EPF Compliance":
        pct = sc.get("epf_coverage_pct", 0)
        status = sc.get("overall_status", "COMPLIANT")
        return status != "COMPLIANT" and sc.get("salary_months_active", 0) > 0, \
               f"EPF coverage {pct}% — {status}"
    if name == "SOCSO Compliance":
        pct = sc.get("socso_coverage_pct", 0)
        status = sc.get("overall_status", "COMPLIANT")
        return status != "COMPLIANT" and sc.get("salary_months_active", 0) > 0, \
               f"SOCSO coverage {pct}% — {status}"
    if name == "LHDN Tax Payments":
        n = sc.get("lhdn_months_paid", 0)
        return n > 0, f"LHDN paid in {n} months" if n else "No LHDN payments detected"
    if name == "Large Credits (>=RM100K)":
        amt = consolidated.get("total_high_value_cr", 0)
        return amt > 0, f"Large credits totalling RM {amt:,.2f}"
    if name == "Own Party Transactions":
        cr, dr = consolidated.get("total_own_party_cr", 0), consolidated.get("total_own_party_dr", 0)
        return (cr + dr) > 0, f"CR RM {cr:,.2f} / DR RM {dr:,.2f}"
    if name == "Related Party Transactions":
        cr, dr = consolidated.get("total_related_party_cr", 0), consolidated.get("total_related_party_dr", 0)
        return (cr + dr) > 0, f"CR RM {cr:,.2f} / DR RM {dr:,.2f}"
    if name == "Loan Activity":
        cr = consolidated.get("total_loan_disbursement_cr", 0)
        dr = consolidated.get("total_loan_repayment_dr", 0)
        return (cr + dr) > 0, f"Disbursements RM {cr:,.2f} / Repayments RM {dr:,.2f}"
    if name == "Data Quality":
        gaps = consolidated.get("months_with_gaps", 0)
        return gaps > 0, f"{gaps} months failed reconciliation" if gaps else "All months reconciled"
    if name == "FX Transactions":
        cr, dr = consolidated.get("total_fx_credits", 0), consolidated.get("total_fx_debits", 0)
        return (cr + dr) > 0, f"FX CR RM {cr:,.2f} / FX DR RM {dr:,.2f}"
    if name == "Low Closing Balance":
        low_months = [m for m in monthly if m.get("closing_balance", 0) < LOW_BALANCE_THRESHOLD]
        return bool(low_months), f"{len(low_months)} months below RM {LOW_BALANCE_THRESHOLD:,.0f}"
    if name == "HRDF Payments":
        n = sc.get("hrdf_months_paid", 0)
        return n > 0, f"HRDF paid in {n} months" if n else "No HRDF payments detected"
    return False, ""


def build_observations(consolidated: dict[str, Any], flags: list[dict[str, Any]]) -> dict[str, Any]:
    positive: list[str] = []
    concerns: list[str] = []
    sc = consolidated.get("statutory_compliance", {})
    if consolidated.get("data_completeness") == "COMPLETE":
        positive.append("All months reconciled to bank statements within tolerance.")
    if sc.get("overall_status") == "COMPLIANT" and sc.get("salary_months_active", 0) > 0:
        positive.append(f"EPF and SOCSO fully covered across {sc['salary_months_active']} salary months.")
    if consolidated.get("net_credits", 0) > 0:
        positive.append(f"Net credits of RM {consolidated['net_credits']:,.2f} indicate active turnover.")

    if consolidated.get("data_completeness") != "COMPLETE":
        concerns.append(f"Reconciliation gaps in {consolidated.get('months_with_gaps')} months — investigate before relying on totals.")
    if sc.get("overall_status") == "CRITICAL":
        concerns.append(f"Statutory compliance CRITICAL — EPF {sc.get('epf_coverage_pct')}% / SOCSO {sc.get('socso_coverage_pct')}%.")
    elif sc.get("overall_status") == "GAPS_DETECTED":
        concerns.append(f"Statutory gaps — EPF {sc.get('epf_coverage_pct')}% / SOCSO {sc.get('socso_coverage_pct')}%.")
    if consolidated.get("total_unclassified_cr", 0) + consolidated.get("total_unclassified_dr", 0) > 0:
        concerns.append(
            f"Unclassified rows: CR RM {consolidated['total_unclassified_cr']:,.2f} / "
            f"DR RM {consolidated['total_unclassified_dr']:,.2f} — V1 deferred categories.")

    return {"positive": positive[:8], "concerns": concerns[:8]}


# ---------------------------------------------------------------------------
# Stage D — final assembly + deliverables
# ---------------------------------------------------------------------------


def build_accounts(data: dict[str, Any], account_meta: dict[str, Any], monthly: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accounts = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for m in monthly:
        grouped.setdefault(m["account_number"], []).append(m)

    for acct_no, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: r["month"])
        bank = rows_sorted[0]["bank_name"]
        company_names = data.get("summary", {}).get("company_names") or [""]
        determination = account_meta.get("determination", {})
        accounts.append({
            "bank_name": bank,
            "account_number": acct_no,
            "account_holder": company_names[0],
            "account_type": account_meta["type"],
            "is_od": account_meta["is_od"],
            "period_start": rows_sorted[0]["month"] + "-01",
            "period_end": _last_day_of_month(rows_sorted[-1]["month"]),
            "opening_balance": rows_sorted[0]["opening_balance"],
            "closing_balance": rows_sorted[-1]["closing_balance"],
            "total_credits": round(sum(r["gross_credits"] for r in rows_sorted), 2),
            "total_debits": round(sum(r["gross_debits"] for r in rows_sorted), 2),
            "transaction_count": sum(r["credit_count"] + r["debit_count"] for r in rows_sorted),
            "account_type_determination": {
                "tested_formulas": determination.get("tested_formulas", ["CR"]),
                "row_level_test": determination.get("row_level_test"),
                "cr_trail": determination.get("cr_trail"),
                "od_trail": determination.get("od_trail"),
                "header_signal": determination.get("header_signal"),
                "locked_type": determination.get("locked_type", "CR"),
                "confidence": determination.get("confidence", "HIGH"),
                "locked_rationale": determination.get("locked_rationale", "Per-row formula match"),
            },
        })
    return accounts


def _last_day_of_month(yyyy_mm: str) -> str:
    if not yyyy_mm or "-" not in yyyy_mm:
        return ""
    y, m = yyyy_mm.split("-")[:2]
    last = {"01": "31", "02": "28", "03": "31", "04": "30", "05": "31", "06": "30",
            "07": "31", "08": "31", "09": "30", "10": "31", "11": "30", "12": "31"}.get(m, "28")
    if m == "02" and int(y) % 4 == 0 and (int(y) % 100 != 0 or int(y) % 400 == 0):
        last = "29"
    return f"{y}-{m}-{last}"


def build_parsing_metadata(data: dict[str, Any], reconciliation: dict[str, Any]) -> dict[str, Any]:
    deltas = reconciliation.get("deltas", [])
    passed = sum(1 for d in deltas if d["passed"])
    total = len(deltas) or 1
    total_tx = sum(d["transactions_extracted"] for d in deltas)
    return {
        "overall_success_rate": round(100 * passed / total, 2),
        "total_transactions_extracted": total_tx,
        "total_balance_checks": len(deltas),
        "total_balance_checks_passed": passed,
        "account_month_checks": deltas,
    }


def assemble_analysis_json(
    *,
    data: dict[str, Any],
    classified: list[dict[str, Any]],
    monthly: list[dict[str, Any]],
    consolidated: dict[str, Any],
    top_parties: dict[str, Any],
    large_credits: list[dict[str, Any]],
    own_related: dict[str, Any],
    loans: dict[str, Any],
    flags: list[dict[str, Any]],
    observations: dict[str, Any],
    unclassified: list[dict[str, Any]],
    parsing_metadata: dict[str, Any],
    account_meta: dict[str, Any],
) -> dict[str, Any]:
    summary = data.get("summary", {})
    period_range = (summary.get("date_range") or "").split(" to ")
    period_start = period_range[0] if len(period_range) > 0 else ""
    period_end = period_range[1] if len(period_range) > 1 else period_start

    return {
        "report_info": {
            "schema_version": TARGET_SCHEMA_VERSION,
            "company_name": (summary.get("company_names") or [""])[0],
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "period_start": period_start,
            "period_end": period_end,
            "total_accounts": len(summary.get("account_nos") or []),
            "total_months": len({m["month"] for m in monthly}),
            "related_parties": [],
        },
        "accounts": build_accounts(data, account_meta, monthly),
        "monthly_analysis": monthly,
        "consolidated": consolidated,
        "top_parties": top_parties,
        "large_credits": large_credits,
        "own_related_transactions": own_related,
        "loan_transactions": loans,
        "flags": {"indicators": flags},
        "observations": observations,
        "parsing_metadata": parsing_metadata,
        "unclassified_transactions": unclassified,
        "pdf_integrity": _sanitize_pdf_integrity(data.get("pdf_integrity", {})),
        "counterparty_ledger": data.get("counterparty_ledger", {}),
        "classification_config": {
            "execution_mode": "FULL_CODE",
            "classifier_version": CLASSIFIER_VERSION,
            "rules_version": TARGET_RULES_VERSION,
        },
    }


# Schema only allows 5 layer enums; the parser's 8-layer detector emits
# extras ('text_layers', 'metadata', 'cross_validation'). Map to closest schema-valid value.
_PDF_LAYER_MAP = {
    "text_layers": "structural",
    "text_layer": "structural",
    "metadata": "structural",
    "cross_validation": "structural",
    "cross-validation": "structural",
}
_PDF_LAYER_VALID = {"fonts", "visual", "bank_profile", "structural", "arithmetic"}


def _sanitize_pdf_integrity(pdf_integrity: dict[str, Any]) -> dict[str, Any]:
    """Coerce parser pdf_integrity findings into the schema's layer enum."""
    out: dict[str, Any] = {}
    for filename, payload in (pdf_integrity or {}).items():
        if not isinstance(payload, dict):
            continue
        sanitized = dict(payload)
        findings = []
        for f in payload.get("findings", []) or []:
            if not isinstance(f, dict):
                continue
            layer = f.get("layer")
            if layer not in _PDF_LAYER_VALID:
                layer = _PDF_LAYER_MAP.get(layer, "structural")
            cleaned = {**f, "layer": layer}
            # Schema requires detail to be an object when present; drop null/non-object
            if "detail" in cleaned and not isinstance(cleaned["detail"], dict):
                cleaned.pop("detail")
            findings.append(cleaned)
        sanitized["findings"] = findings
        out[filename] = sanitized
    return out


def validate_against_schema(analysis: dict[str, Any], schema: dict[str, Any]) -> None:
    jsonschema.validate(instance=analysis, schema=schema)


def build_parser_quality_report(
    data: dict[str, Any],
    monthly: list[dict[str, Any]],
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    """Lean Deliverable-2 grade A-F based on extraction stats + balance trail."""
    extraction = data.get("counterparty_ledger", {}).get("extraction_stats", {})
    pattern_matched = extraction.get("pattern_matched", 0)
    special_bucket = extraction.get("special_bucket", 0)
    raw_fallback = extraction.get("raw_fallback", 0)
    total_tx = extraction.get("total_transactions", 1) or 1
    effective_match_rate = round(100 * (pattern_matched + special_bucket) / total_tx, 2)

    deltas = reconciliation.get("deltas", [])
    passed = sum(1 for d in deltas if d["passed"])
    balance_pass_rate = round(100 * passed / max(len(deltas), 1), 2)

    if effective_match_rate >= 95 and balance_pass_rate == 100:
        grade = "A"
    elif effective_match_rate >= 90 and balance_pass_rate >= 95:
        grade = "B"
    elif effective_match_rate >= 80 and balance_pass_rate >= 90:
        grade = "C"
    elif effective_match_rate >= 70:
        grade = "D"
    else:
        grade = "F"

    return {
        "grade": grade,
        "effective_match_rate_pct": effective_match_rate,
        "balance_trail_pass_rate_pct": balance_pass_rate,
        "extraction_stats": {
            "pattern_matched": pattern_matched,
            "special_bucket": special_bucket,
            "raw_fallback": raw_fallback,
            "total_transactions": total_tx,
        },
        "balance_checks": {
            "total": len(deltas),
            "passed": passed,
            "failed_months": [d["month"] for d in deltas if not d["passed"]],
        },
        "company_name": (data.get("summary", {}).get("company_names") or [""])[0],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": TARGET_SCHEMA_VERSION,
    }


def build_narrative_brief(
    *,
    data: dict[str, Any],
    consolidated: dict[str, Any],
    flags: list[dict[str, Any]],
    related_parties: list[str],
    purpose_clusters: dict[str, Any],
    rp4_candidates: list[dict[str, Any]],
    missing_months: list[str],
) -> dict[str, Any]:
    summary = data.get("summary", {})
    detected_flags = [f["name"] for f in flags if f["detected"]]
    sc = consolidated.get("statutory_compliance", {})
    return {
        "target_prompt_version": TARGET_PROMPT_VERSION,
        "schema_version": TARGET_SCHEMA_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "company_name": (summary.get("company_names") or [""])[0],
        "period": summary.get("date_range"),
        "headline_numbers": {
            "gross_credits": consolidated.get("gross_credits"),
            "gross_debits": consolidated.get("gross_debits"),
            "net_credits": consolidated.get("net_credits"),
            "net_debits": consolidated.get("net_debits"),
            "annualized_net_credits": consolidated.get("annualized_net_credits"),
            "eod_average": consolidated.get("eod_average"),
            "salary_paid": consolidated.get("total_salary_paid"),
            "epf_coverage_pct": sc.get("epf_coverage_pct"),
            "socso_coverage_pct": sc.get("socso_coverage_pct"),
            "trade_income_cr": consolidated.get("total_trade_income_cr"),
            "trade_expense_dr": consolidated.get("total_trade_expense_dr"),
        },
        "detected_patterns": {
            "purpose_clusters": purpose_clusters,
            "flags_detected": detected_flags,
        },
        "missing_months": missing_months,
        "rp4_candidates": rp4_candidates,
        "analyst_inputs": {
            "related_parties": related_parties,
        },
        "analyst_asks": [
            "Confirm RP4 candidates are directors/family vs employees vs contractors.",
            "Confirm dominant purpose-cluster treatment (commission/salary).",
            "Provide narrative observations on top 3 concerns and overall risk posture.",
        ],
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def _gate_basic_auth() -> None:
    user = os.environ.get("BASIC_AUTH_USER")
    pwd = os.environ.get("BASIC_AUTH_PASS")
    if not user and not pwd:
        return
    if not user or not pwd:
        st.error("BASIC_AUTH_USER and BASIC_AUTH_PASS must both be set.")
        st.stop()
    if "_authed" not in st.session_state:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in"):
                if u == user and p == pwd:
                    st.session_state["_authed"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            st.stop()


def _render_decisions_form(rp4_candidates: list[dict[str, Any]]) -> AnalystDecisions:
    """Two-pass UI: RP4 checkboxes seeded from the pre-analysis gate appear above
    the existing free-text fields. Confirmed RP4 names are appended to related_parties
    so the classifier re-runs with C03/C04 firing on those rows.

    V3-A auto-RP Step 1: HIGH-confidence candidates are pre-checked by default;
    analyst can untick to override. MEDIUM/LOW unchecked by default."""
    confirmed_rp4: list[str] = []
    if rp4_candidates:
        n_high = sum(1 for c in rp4_candidates if c.get("confidence") == "HIGH")
        with st.expander(
            f"RP4 candidates ({len(rp4_candidates)}; {n_high} auto-confirmed HIGH) — review directors / family",
            expanded=True,
        ):
            st.caption(
                "Deterministic behavioral sweep over the counterparty ledger "
                "(personal keywords, DR concentration, monthly recurrence, "
                "bidirectional flow, round-amount advances). HIGH-confidence "
                "names are pre-ticked — untick to exclude. MEDIUM/LOW are "
                "unchecked; tick any director, shareholder, or close family."
            )
            for cand in rp4_candidates:
                name = cand["name"]
                conf = cand.get("confidence", "LOW")
                label = (
                    f"[{conf}] {name} — {cand['evidence']} · "
                    f"DR RM {cand.get('total_dr', 0):,.2f}"
                )
                default = conf == "HIGH"
                if st.checkbox(label, key=f"rp4_{name}", value=default):
                    confirmed_rp4.append(name)

    with st.expander("Analyst decisions", expanded=True):
        rp_text = st.text_area(
            "Related parties (one per line)",
            help="Names matched as C03/C04. Case-insensitive substring vs counterparty + description. "
                 "Confirmed RP4 candidates above are appended automatically.",
        )
        fact_text = st.text_area(
            "Factoring entities (one per line)",
            help="C10 disbursement counterparties (e.g. PLANWORTH GLOBAL).",
        )
        commission = st.radio(
            "Commission cluster treatment (if dominant)",
            options=["regular_expense", "salary_c05_with_statutory"],
            index=0,
        )
        od_limit = st.number_input("OD limit (RM, optional)", min_value=0.0, step=10000.0, value=0.0)
        notes = st.text_area("Notes (free-form)", height=80)

    free_text_rp = [s.strip() for s in rp_text.splitlines() if s.strip()]
    # Dedupe while preserving order; confirmed RP4 first so they survive substring collisions
    seen: set[str] = set()
    related_parties: list[str] = []
    for name in confirmed_rp4 + free_text_rp:
        if name.upper() not in seen:
            seen.add(name.upper())
            related_parties.append(name)

    return AnalystDecisions(
        related_parties=related_parties,
        factoring_entities=[s.strip() for s in fact_text.splitlines() if s.strip()],
        commission_treatment=commission,
        od_limit=od_limit or None,
        notes=notes,
    )


def _run_pre_analysis_gate(data: dict[str, Any]) -> dict[str, Any]:
    """Pass 1 of the two-pass flow. Cheap; runs on upload to drive the decisions form."""
    account_meta = detect_account_type(data)
    recon = reconcile_balance_trail(data, account_meta["convention"])
    rp_candidates = scan_related_party_candidates(data)
    clusters = scan_purpose_clusters(data)
    return {
        "account_meta": account_meta,
        "recon": recon,
        "rp4_candidates": rp_candidates,
        "purpose_clusters": clusters,
    }


def streamlit_main() -> None:
    st.set_page_config(page_title="Kredit Lab Classifier", layout="wide")
    _gate_basic_auth()
    st.title("Kredit Lab — Stage 2 Classifier")
    st.caption(
        f"Schema v{TARGET_SCHEMA_VERSION} · Rules v{TARGET_RULES_VERSION} · "
        f"Classifier v{CLASSIFIER_VERSION}"
    )

    upload = st.file_uploader("Drop full_report.json", type=["json"])
    if upload is None:
        st.info("Upload a parser-produced full_report.json to begin.")
        st.session_state.pop("upload_key", None)
        return

    raw = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    upload_key = (upload.name, len(raw))
    if st.session_state.get("upload_key") != upload_key:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return
        try:
            gate = _run_pre_analysis_gate(data)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            return
        st.session_state["upload_key"] = upload_key
        st.session_state["data"] = data
        st.session_state["gate"] = gate

    data = st.session_state["data"]
    gate = st.session_state["gate"]

    # Pass 1 summary — what the analyst should see before confirming
    with st.expander("Pre-analysis gate", expanded=True):
        meta = gate["account_meta"]
        recon = gate["recon"]
        passed = sum(1 for d in recon["deltas"] if d["passed"])
        total = len(recon["deltas"])
        cols = st.columns(3)
        cols[0].metric("Account type", meta["type"])
        cols[1].metric("Convention", meta["convention"])
        cols[2].metric("Balance trail", f"{passed}/{total} months")
        if passed != total:
            failing = [d["month"] for d in recon["deltas"] if not d["passed"]]
            st.warning(f"Reconciliation gaps: {', '.join(failing)}")

    decisions = _render_decisions_form(gate["rp4_candidates"])

    if not st.button("Classify", type="primary"):
        return

    try:
        # Re-run pre-analysis gate ONLY if od_limit was provided (account_meta depends on it).
        if decisions.od_limit is not None:
            account_meta = detect_account_type(data, od_limit=decisions.od_limit)
            recon = reconcile_balance_trail(data, account_meta["convention"])
        else:
            account_meta = gate["account_meta"]
            recon = gate["recon"]
        rp_candidates = gate["rp4_candidates"]
        clusters = gate["purpose_clusters"]

        rulebook = load_rulebook()
        schema = load_schema()

        with st.status("Classifying...", expanded=False):
            classified = classify_transactions(data, rulebook, decisions)

        with st.status("Aggregating...", expanded=False):
            monthly = build_monthly_analysis(classified, data, recon)
            consolidated = build_consolidated(monthly)
            top_parties = build_top_parties(classified, decisions.related_parties)
            large_credits = build_large_credits(classified)
            own_related = build_own_related_transactions(classified)
            loans = build_loan_transactions(classified)
            unclassified = build_unclassified(classified)
            flags = build_flags(consolidated, monthly)
            observations = build_observations(consolidated, flags)
            parsing_metadata = build_parsing_metadata(data, recon)

        analysis = assemble_analysis_json(
            data=data, classified=classified, monthly=monthly,
            consolidated=consolidated, top_parties=top_parties,
            large_credits=large_credits, own_related=own_related,
            loans=loans, flags=flags, observations=observations,
            unclassified=unclassified, parsing_metadata=parsing_metadata,
            account_meta=account_meta,
        )

        with st.status("Schema-validating...", expanded=False):
            validate_against_schema(analysis, schema)

        narrative_brief = build_narrative_brief(
            data=data, consolidated=consolidated, flags=flags,
            related_parties=decisions.related_parties,
            purpose_clusters=clusters, rp4_candidates=rp_candidates,
            missing_months=[d["month"] for d in recon["deltas"] if not d["passed"]],
        )
        parser_quality = build_parser_quality_report(data, monthly, recon)

        st.success("Classification complete.")
        st.metric("Net credits", f"RM {consolidated['net_credits']:,.2f}")
        st.metric("EPF coverage", f"{consolidated['statutory_compliance']['epf_coverage_pct']}%")
        st.metric("Overall status", consolidated["statutory_compliance"]["overall_status"])
        if rp_candidates:
            st.warning(f"{len(rp_candidates)} RP4 candidate(s) surfaced — see narrative_brief.json.")

        col1, col2, col3 = st.columns(3)
        col1.download_button("analysis.json",
            data=json.dumps(analysis, indent=2),
            file_name="analysis.json", mime="application/json")
        col2.download_button("narrative_brief.json",
            data=json.dumps(narrative_brief, indent=2),
            file_name="narrative_brief.json", mime="application/json")
        col3.download_button("parser_quality.json",
            data=json.dumps(parser_quality, indent=2),
            file_name="parser_quality.json", mime="application/json")

    except jsonschema.ValidationError as exc:
        st.error(f"Schema validation failed: {exc.message}")
        st.code(" / ".join(str(p) for p in exc.absolute_path), language="text")
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)


if __name__ == "__main__":
    streamlit_main()
