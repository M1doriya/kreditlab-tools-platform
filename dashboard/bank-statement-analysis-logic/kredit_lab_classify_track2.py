"""kredit_lab_classify_track2.py — Track 2 deterministic classifier.

Engine-heavy migration of v3.5.6 AI-prompt classification logic into Python.
Companion to ``kredit_lab_classify.py`` (Track 1, frozen). Track 1 must not
import from this module, and this module must not import from Track 1; the
two are kept fully separated by design — see the architecture handoff under
``prompts/NEXT_CHAT_PROMPT.md`` and the project memory entry
``project_track_2_architecture.md``.

Session 1: ``compute_monthly_eod`` — port of the EOD (End-of-Day balance)
algorithm specified in ``SYSTEM_PROMPT_v3_5_6.md`` lines 558-583.

Session 2: ``compute_risk_flags`` — port of the canonical 16-flag reducer
specified in ``SYSTEM_PROMPT_v3_5_6.md`` lines 644-667 and the
``flags.indicators[]`` schema (``BANK_ANALYSIS_SCHEMA_v6_3_5.json`` line
1119+). The reducer is the *terminal* step of risk-signal computation —
it consumes already-computed summary fields and emits the 16-record
output array. Computing those summary fields (round-figure detection,
high-value detection, statutory monthly ratios, large-credits selection,
etc.) is a separate set of Tier-2 migrations scheduled for later sessions.

Session 3: ``compute_statutory_compliance`` — port of the EPF/SOCSO/LHDN/
HRDF coverage computation specified in ``SYSTEM_PROMPT_v3_5_6.md`` lines
488-521, producing the ``statutory_compliance`` sub-object whose schema
is at ``BANK_ANALYSIS_SCHEMA_v6_3_5.json`` line 777. Output drops
directly into ``compute_risk_flags(..., statutory_compliance=...)``
populating Flags 6/7/8/16 with their schema-mandatory month-by-month
remarks.

Session 4: ``compute_round_figure_credits`` (C21), ``compute_large_credits``
(C23), and ``compute_high_value_credits`` (C22) — port of the three
locked monitoring rules in ``CLASSIFICATION_RULES_v3_5.json`` lines
1036-1083. C21 = exact-multiple-of-RM10K credits (structuring/AML
indicator). C23 = credits at or above the configurable absolute
threshold (default RM100K). C22 = credits exceeding ``multiplier``
times that month's EOD average (composes with session-1
``compute_monthly_eod``); skipped entirely when ``eod_unreliable=True``
per the prompt's "no proxy values" rule. All three feed
``compute_risk_flags`` populating Flags 3/4/9.

Session 5: ``compute_returned_cheques`` (C14/C15), ``compute_data_completeness``
(consolidated reducer), and ``compute_fx_totals`` (Flag 14 inputs).
Returned-cheques uses the LOCKED regex from ``CLASSIFICATION_RULES_v3_5.json``
line 907 and routes by row side (DR -> inward C14, CR -> outward C15).
Data-completeness aggregates per-month reconciliation results into the
consolidated ``data_completeness`` enum + gap counts. FX detection is
negative-list-first per the schema's $comment_fx_classification: RENTAS/
JANM, IBG with internal voucher codes (GBPV/USDP/EURK), DuitNow/FPX/
JomPAY, and MYR-to-MYR are NEVER FX; positive triggers are explicit
keywords (FOREX, FX CONV, FOREIGN EXCHANGE) and SWIFT/TT-with-foreign-
currency markers. Default-to-NOT-FX when ambiguous.

Session 6: ``compute_monthly_aggregates`` — bank-agnostic, Streamlit-free
per-month aggregation derived purely from canonical-schema transaction
rows. Outputs ``month``, totals (``gross_credits`` / ``gross_debits`` /
``net_change``), counts, balance metrics (``opening_balance`` /
``closing_balance`` / ``lowest_balance`` / ``highest_balance`` /
``swing``), and EOD metrics composed from ``compute_monthly_eod``.
Opening-balance derivation is account-type-aware: first-month opening
back-computed from the first row's balance via the CR formula
(``balance - credit + debit``) or OD formula (``balance + credit - debit``);
subsequent months' opening = previous month's closing. Track 1's
``app.py:calculate_monthly_summary`` is bank-branched and Streamlit-coupled;
this is the clean equivalent that Track 2 owns.

Subsequent sessions add further Tier-2 migrations (counterparty
aggregation, RP detection, classification, etc.) per the migration plan.
"""

from __future__ import annotations

import json
import re
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Canonical 16 flags — fixed IDs, names, and order.
# Source of truth: BANK_ANALYSIS_SCHEMA_v6_3_5.json `flags.indicators` enum.
# Never reorder, rename, or change without updating the schema, the v3.5.6
# prompt, and the classification rules JSON in lockstep.
# ---------------------------------------------------------------------------
CANONICAL_FLAGS: tuple[tuple[int, str], ...] = (
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
)


LOW_CLOSING_THRESHOLD_CR = 1_000.00
OD_HIGH_UTILISATION_RATIO = 0.90  # flag 15 OD branch: >=90% of od_limit drawn down

# v6.3.4 dual-band thresholds for statutory_compliance per-month ratios.
# Source: BANK_ANALYSIS_SCHEMA_v6_3_5.json line 915 (EPF) and 955 (SOCSO).
EPF_BAND_EMPLOYER_ONLY = (11.0, 15.0)   # employer-only contribution band
EPF_BAND_COMBINED = (20.0, 26.0)        # employer + employee combined band
SOCSO_BAND = (1.0, 5.0)
STRUCTURAL_RUN_LENGTH = 4               # >=4 consecutive above-band months -> STRUCTURAL

# Sub-threshold employer cut-off — extension of the v3.3.1 commission_policy
# "no payroll obligation" spirit to sole-prop / director-only / very small
# payrolls where 0% EPF / SOCSO coverage may be correct (no employer
# obligation). Surfaced by the s12 statutory-chain calibration: RE Concept
# (RM 11.8K total) and Calvin Skin (RM 15.25K total) tripped CRITICAL on
# Flag 6 but appear to be director-only / sub-threshold employers. Total
# salary across the statement window <= this threshold flags the account
# as sub-threshold so the s3 reducer downgrades from CRITICAL to a softer
# SUB_THRESHOLD status that downstream consumers can interpret as
# "EPF / SOCSO obligation may not apply — verify employer status".
SUBTHRESHOLD_TOTAL_SALARY_RM = 30_000.0

# Channel-blind employer thresholds — heuristic for accounts where
# statutory contributions are likely paid via cheque or off-account, so
# 0% EPF / SOCSO detected by keyword is uninformative rather than a real
# compliance gap. Surfaced by the s12 statutory-chain calibration: Juta
# Kenangan (RM 24.2M cheque-DR, 90.7% of gross) and Hou Tian (RM 1.66M /
# 24.9%) both detect real bulk payroll but zero EPF / SOCSO / LHDN
# substring matches across the entire window. Cheque-DR magnitude alone
# can come from supplier payments; the *ratio* check ensures cheques are
# a significant share of total outflows (not just absolute large spend).
#
# Both gates must trip for the account to be classified as channel-blind:
#   * cheque_dr_amount >= ``CHANNEL_BLIND_CHEQUE_DR_MIN_RM`` (magnitude)
#   * cheque_dr_amount / gross_dr_amount >= ``CHANNEL_BLIND_CHEQUE_DR_MIN_RATIO`` (significance)
#
# The reducer treats CHANNEL_BLIND as a softer verdict than CRITICAL but
# weaker than SUB_THRESHOLD: a sub-threshold employer with no obligation
# stays SUB_THRESHOLD even if it also happens to be cheque-heavy.
CHANNEL_BLIND_CHEQUE_DR_MIN_RM = 500_000.0
CHANNEL_BLIND_CHEQUE_DR_MIN_RATIO = 0.10
# Loose cheque-DR regex for the channel-blind heuristic ONLY. Broader than
# the strict ``CHEQUE_ISSUE_RE`` (C20) because the calibration uncovered
# bank-specific cheque shapes (UOB ``Chq Wdl``, ``Cheque NNNNNNN``) that
# C20 deliberately doesn't match. Soft-signal use: over-matching small
# cheque-fee rows is fine because the absolute-RM threshold filters them.
CHEQUE_DR_HEURISTIC_RE = re.compile(r"\bCHEQUE\b|\bCHQ\b", re.IGNORECASE)


def compute_monthly_eod(
    transactions: list[dict[str, Any]],
    year_month: str,
) -> dict[str, float | int | None]:
    """Compute end-of-day balance metrics for a single account-month.

    Walks transactions in statement order, groups by date, takes the LAST
    row's balance for each date as that date's EOD, then reduces to
    min/max/average across the month. Only dates with transactions count;
    missing days are not back-filled.

    Args:
        transactions: rows in canonical schema, already in statement order
            (the order they appear in the source PDF). Must contain only
            rows for ONE account; the caller is responsible for per-account
            grouping. Each row needs at minimum ``date`` (ISO ``YYYY-MM-DD``)
            and ``balance`` (float | None). Rows with ``balance is None``
            are skipped.
        year_month: target month in ``YYYY-MM`` format. Only rows whose
            ``date`` starts with this prefix are included — the caller is
            responsible for excluding previous-month opening-balance rows
            and including/excluding same-day opening rows per the prompt
            spec (this function trusts its input).

    Returns:
        dict with keys ``eod_lowest``, ``eod_highest``, ``eod_average``
        (all float, or ``None`` when the month has no qualifying rows) and
        ``eod_dates_count`` (int — number of distinct dates with at least
        one transaction whose balance is not None).
    """
    daily_eod: dict[str, float] = {}
    for row in transactions:
        date = row.get("date")
        if not isinstance(date, str) or not date.startswith(year_month):
            continue
        balance = row.get("balance")
        if balance is None:
            continue
        daily_eod[date] = float(balance)

    if not daily_eod:
        return {
            "eod_lowest": None,
            "eod_highest": None,
            "eod_average": None,
            "eod_dates_count": 0,
        }

    values = list(daily_eod.values())
    return {
        "eod_lowest": min(values),
        "eod_highest": max(values),
        "eod_average": sum(values) / len(values),
        "eod_dates_count": len(values),
    }


# ---------------------------------------------------------------------------
# Risk-signal flag reducer — the canonical 16 flags
# ---------------------------------------------------------------------------


def _rm(amount: float | int | None) -> str:
    """Format a ringgit amount as ``RM 1,234.56``. ``None`` → ``RM 0.00``."""
    return f"RM {float(amount or 0):,.2f}"


def _pct(part: float | int | None, total: float | int | None) -> str:
    """Format ``part/total`` as a percent string. ``0/0`` → ``0.0%``."""
    p = float(part or 0)
    t = float(total or 0)
    if t == 0:
        return "0.0%"
    return f"{(p / t) * 100:.1f}%"


def _missing_months(monthly: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return entries whose ``ratio`` is below 1.0 (i.e. coverage < 100%).

    Entries without a numeric ``ratio`` field are skipped silently — the
    caller is responsible for shape correctness.
    """
    if not monthly:
        return []
    out: list[dict[str, Any]] = []
    for row in monthly:
        ratio = row.get("ratio")
        if isinstance(ratio, (int, float)) and ratio < 1.0:
            out.append(row)
    return out


def _coverage_remarks(
    label: str, coverage_pct: float, monthly: list[dict[str, Any]] | None
) -> str:
    """Build remarks for EPF/SOCSO flags listing missing months + ratios."""
    missing = _missing_months(monthly)
    if not missing:
        return f"{label} coverage {coverage_pct:.1f}% — review monthly contributions."
    parts = [f"{m.get('month', '?')} ratio {float(m.get('ratio', 0)):.2f}" for m in missing]
    return (
        f"{label} coverage {coverage_pct:.1f}%; missing/partial months: "
        + ", ".join(parts)
    )


def _flag(idx: int, detected: bool, remarks: str) -> dict[str, Any]:
    fid, fname = CANONICAL_FLAGS[idx]
    return {"id": fid, "name": fname, "detected": detected, "remarks": remarks}


def compute_risk_flags(
    summary: dict[str, Any],
    *,
    monthly_analysis: list[dict[str, Any]] | None = None,
    statutory_compliance: dict[str, Any] | None = None,
    account_type: str = "CR",
    od_limit: float | None = None,
) -> list[dict[str, Any]]:
    """Build the canonical 16-flag ``flags.indicators[]`` array.

    Pure reducer — consumes already-computed summary fields and emits the
    final flag records. Does NOT compute any of the upstream metrics
    (round-figure detection, statutory ratios, large-credit selection,
    etc.); those land in their own Tier-2 migrations and the caller is
    responsible for populating the summary dict before calling this.

    Args:
        summary: pre-computed metric dict. Keys read (all optional —
            missing keys treated as zero / empty / False):
                returned_cheques_inward_count, returned_cheques_inward_amount,
                returned_cheques_outward_count, returned_cheques_outward_amount,
                round_figure_cr, round_figure_count,
                high_value_cr, high_value_count, eod_unreliable,
                cash_deposits_count, cash_deposits_amount, gross_credits, gross_debits,
                large_credits (list[dict] each with at least 'amount'),
                own_party_cr, own_party_dr,
                related_party_cr, related_party_dr, related_party_names (list[str]),
                loan_disbursement_cr, loan_repayment_dr,
                data_completeness ('COMPLETE' | 'INCOMPLETE'), data_gaps (str),
                total_fx_credits, total_fx_debits,
                salary_months_active.
        monthly_analysis: per-month rows used by flag 15 — each row must have
            ``month`` (e.g. ``'2026-04'``) and ``closing_balance`` (float).
        statutory_compliance: sub-object used by flags 6/7/8/16. Keys read:
                epf_coverage_pct, epf_monthly (list[{month, ratio}]),
                socso_coverage_pct, socso_monthly (list[{month, ratio}]),
                lhdn_detected (bool), lhdn_count, lhdn_total,
                hrdf_detected (bool), hrdf_count, hrdf_total.
        account_type: ``'CR'`` (default) or ``'OD'``. Flag 15 inverts on OD.
        od_limit: facility limit for OD accounts. Required for flag 15 to
            fire on OD utilisation; if missing, flag 15 stays clean and
            remarks note that ``od_limit`` was not supplied.

    Returns:
        list of exactly 16 dicts, each ``{id, name, detected, remarks}``,
        in canonical order. Names match the schema enum exactly so the
        downstream HTML renderer can match by string.
    """
    summary = summary or {}
    statutory_compliance = statutory_compliance or {}
    monthly_analysis = monthly_analysis or []

    flags: list[dict[str, Any]] = []

    # --- Flag 1 — Returned Cheques (Inward) ---------------------------------
    rc_in_count = int(summary.get("returned_cheques_inward_count") or 0)
    rc_in_amount = float(summary.get("returned_cheques_inward_amount") or 0)
    flags.append(
        _flag(
            0,
            rc_in_count > 0,
            (
                f"{rc_in_count} inward returned cheques totalling {_rm(rc_in_amount)}."
                if rc_in_count > 0
                else "No inward returned cheques in the period."
            ),
        )
    )

    # --- Flag 2 — Returned Cheques (Outward) --------------------------------
    rc_out_count = int(summary.get("returned_cheques_outward_count") or 0)
    rc_out_amount = float(summary.get("returned_cheques_outward_amount") or 0)
    flags.append(
        _flag(
            1,
            rc_out_count > 0,
            (
                f"{rc_out_count} outward returned cheques totalling {_rm(rc_out_amount)}."
                if rc_out_count > 0
                else "No outward returned cheques in the period."
            ),
        )
    )

    # --- Flag 3 — Round Figure Credits (AML) --------------------------------
    rf_amount = float(summary.get("round_figure_cr") or 0)
    rf_count = int(summary.get("round_figure_count") or 0)
    flags.append(
        _flag(
            2,
            rf_amount > 0,
            (
                f"{rf_count} round-figure credits totalling {_rm(rf_amount)}."
                if rf_amount > 0
                else "No round-figure credits flagged."
            ),
        )
    )

    # --- Flag 4 — High Value Credits (>3x EOD) ------------------------------
    hv_amount = float(summary.get("high_value_cr") or 0)
    hv_count = int(summary.get("high_value_count") or 0)
    eod_unreliable = bool(summary.get("eod_unreliable", False))
    if eod_unreliable:
        flags.append(
            _flag(3, False, "Skipped — EOD computation unreliable for this account.")
        )
    else:
        flags.append(
            _flag(
                3,
                hv_amount > 0,
                (
                    f"{hv_count} credits exceed 3x daily EOD totalling {_rm(hv_amount)}."
                    if hv_amount > 0
                    else "No credits exceeded 3x daily EOD."
                ),
            )
        )

    # --- Flag 5 — Cash Deposits (AML) ---------------------------------------
    cd_count = int(summary.get("cash_deposits_count") or 0)
    cd_amount = float(summary.get("cash_deposits_amount") or 0)
    gross_credits = float(summary.get("gross_credits") or 0)
    flags.append(
        _flag(
            4,
            cd_count > 0,
            (
                f"{cd_count} cash deposits totalling {_rm(cd_amount)} "
                f"({_pct(cd_amount, gross_credits)} of gross credits)."
                if cd_count > 0
                else "No cash deposits in the period."
            ),
        )
    )

    # --- Flag 6 — EPF Compliance --------------------------------------------
    # Sub-threshold / channel-blind employer downgrades: when
    # statutory_compliance flags the account as sub-threshold or
    # channel-blind, the EPF / SOCSO gap remark is contextualised so
    # downstream consumers don't read 0% coverage as a hard CRITICAL.
    # Priority order (matches the s3 reducer): sub-threshold context
    # wins over channel-blind ("no obligation" beats "can't verify").
    #
    # Coverage fallback: a missing ``epf_coverage_pct`` defaults to 100
    # (no detection); a present 0.0 must remain 0.0 (do NOT use
    # ``... or 100``, which treats 0.0 as falsy and silently silences
    # the most common CRITICAL case).
    subthreshold_info = statutory_compliance.get("subthreshold_employer") or {}
    is_subthreshold = bool(subthreshold_info.get("is_subthreshold"))
    subthreshold_reason = str(subthreshold_info.get("reason") or "")
    channel_blind_info = statutory_compliance.get("channel_blind_employer") or {}
    is_channel_blind = bool(channel_blind_info.get("is_channel_blind"))
    channel_blind_reason = str(channel_blind_info.get("reason") or "")
    if is_subthreshold and subthreshold_reason:
        context_remark = subthreshold_reason
    elif is_channel_blind and channel_blind_reason:
        context_remark = channel_blind_reason
    else:
        context_remark = ""

    epf_pct_raw = statutory_compliance.get("epf_coverage_pct")
    epf_pct = float(epf_pct_raw) if epf_pct_raw is not None else 100.0
    epf_monthly = statutory_compliance.get("epf_monthly")
    if epf_pct < 100:
        epf_remark = _coverage_remarks("EPF", epf_pct, epf_monthly)
        if context_remark:
            epf_remark = f"{epf_remark} {context_remark}"
    else:
        epf_remark = "EPF coverage 100% across salary months."
    flags.append(_flag(5, epf_pct < 100, epf_remark))

    # --- Flag 7 — SOCSO Compliance ------------------------------------------
    socso_pct_raw = statutory_compliance.get("socso_coverage_pct")
    socso_pct = float(socso_pct_raw) if socso_pct_raw is not None else 100.0
    socso_monthly = statutory_compliance.get("socso_monthly")
    if socso_pct < 100:
        socso_remark = _coverage_remarks("SOCSO", socso_pct, socso_monthly)
        if context_remark:
            socso_remark = f"{socso_remark} {context_remark}"
    else:
        socso_remark = "SOCSO coverage 100% across salary months."
    flags.append(_flag(6, socso_pct < 100, socso_remark))

    # --- Flag 8 — LHDN Tax Payments (informational) -------------------------
    lhdn_detected = bool(statutory_compliance.get("lhdn_detected", False))
    lhdn_count = int(statutory_compliance.get("lhdn_count") or 0)
    lhdn_total = float(statutory_compliance.get("lhdn_total") or 0)
    salary_months_active = int(summary.get("salary_months_active") or 0)
    if lhdn_detected:
        flags.append(
            _flag(
                7,
                False,
                f"LHDN payments detected: {lhdn_count} tx totalling {_rm(lhdn_total)} "
                "(PCB/CP204/SST — schedules differ; informational only).",
            )
        )
    else:
        detected_8 = salary_months_active > 0
        flags.append(
            _flag(
                7,
                detected_8,
                (
                    "No LHDN tax payments detected despite active salary months — "
                    "possible PCB non-remittance (soft concern, not hard failure)."
                    if detected_8
                    else "No LHDN tax payments detected; no salary activity to compare against."
                ),
            )
        )

    # --- Flag 9 — Large Credits (>=RM100K) ----------------------------------
    large_credits = summary.get("large_credits") or []
    lc_count = len(large_credits)
    lc_total = sum(float(c.get("amount") or 0) for c in large_credits)
    flags.append(
        _flag(
            8,
            lc_count > 0,
            (
                f"{lc_count} large credits (>=RM100K) totalling {_rm(lc_total)}."
                if lc_count > 0
                else "No credits at or above RM100,000."
            ),
        )
    )

    # --- Flag 10 — Own Party Transactions -----------------------------------
    op_cr = float(summary.get("own_party_cr") or 0)
    op_dr = float(summary.get("own_party_dr") or 0)
    gross_debits = float(summary.get("gross_debits") or 0)
    flags.append(
        _flag(
            9,
            op_cr > 0 or op_dr > 0,
            (
                f"Own-party CR {_rm(op_cr)} ({_pct(op_cr, gross_credits)} of gross credits); "
                f"DR {_rm(op_dr)} ({_pct(op_dr, gross_debits)} of gross debits)."
                if (op_cr > 0 or op_dr > 0)
                else "No own-party transactions detected."
            ),
        )
    )

    # --- Flag 11 — Related Party Transactions -------------------------------
    rp_cr = float(summary.get("related_party_cr") or 0)
    rp_dr = float(summary.get("related_party_dr") or 0)
    rp_names = summary.get("related_party_names") or []
    if rp_cr > 0 or rp_dr > 0:
        names_str = ", ".join(rp_names) if rp_names else "(no canonical names provided)"
        remarks = (
            f"Related-party CR {_rm(rp_cr)} ({_pct(rp_cr, gross_credits)} of gross credits); "
            f"DR {_rm(rp_dr)} ({_pct(rp_dr, gross_debits)} of gross debits). Parties: {names_str}."
        )
        flags.append(_flag(10, True, remarks))
    else:
        flags.append(_flag(10, False, "No related-party transactions detected."))

    # --- Flag 12 — Loan Activity --------------------------------------------
    ld_cr = float(summary.get("loan_disbursement_cr") or 0)
    lr_dr = float(summary.get("loan_repayment_dr") or 0)
    # Loan-review net: rows that look like facility activity but were booked as
    # neither facility nor related-party (see ``_build_loan_review_track2``).
    # Appended to the remarks so a non-zero count is never silent — the failure
    # mode this guards against is "0 / No loan activity detected" printed over a
    # statement that plainly carries monthly facility repayments.
    review_count = int(summary.get("loan_review_count") or 0)
    review_note = (
        f" {review_count} loan-shaped row(s) unclassified — review."
        if review_count > 0
        else ""
    )
    flags.append(
        _flag(
            11,
            ld_cr > 0 or lr_dr > 0 or review_count > 0,
            (
                f"Loan disbursements {_rm(ld_cr)}; loan repayments {_rm(lr_dr)}."
                + review_note
                if (ld_cr > 0 or lr_dr > 0)
                else (
                    "No loan disbursements or repayments classified."
                    + review_note
                    if review_count > 0
                    else "No loan disbursements or repayments detected."
                )
            ),
        )
    )

    # --- Flag 13 — Data Quality ---------------------------------------------
    data_completeness = str(summary.get("data_completeness") or "COMPLETE").upper()
    data_gaps = str(summary.get("data_gaps") or "")
    flags.append(
        _flag(
            12,
            data_completeness == "INCOMPLETE",
            (
                f"Statement data INCOMPLETE: {data_gaps}"
                if data_completeness == "INCOMPLETE"
                else "Statement data complete across the period."
            ),
        )
    )

    # --- Flag 14 — FX Transactions ------------------------------------------
    fx_cr = float(summary.get("total_fx_credits") or 0)
    fx_dr = float(summary.get("total_fx_debits") or 0)
    flags.append(
        _flag(
            13,
            fx_cr > 0 or fx_dr > 0,
            (
                f"FX credits {_rm(fx_cr)}; FX debits {_rm(fx_dr)}."
                if (fx_cr > 0 or fx_dr > 0)
                else "No FX (foreign-currency) activity detected."
            ),
        )
    )

    # --- Flag 15 — Low Closing Balance (CR) / High Utilisation (OD) ---------
    is_od = account_type.upper() == "OD"
    if is_od:
        if od_limit is None or od_limit <= 0:
            flags.append(
                _flag(
                    14,
                    False,
                    "OD account — od_limit not provided; cannot evaluate utilisation.",
                )
            )
        else:
            high_util_threshold = float(od_limit) * OD_HIGH_UTILISATION_RATIO
            # Compare on magnitude so both OD sign conventions work:
            # signed-negative (modern parsers, drawdown stored as -X) and
            # positive-magnitude (legacy Alliance, drawdown stored as +X).
            offenders = [
                m
                for m in monthly_analysis
                if isinstance(m.get("closing_balance"), (int, float))
                and abs(float(m["closing_balance"])) >= high_util_threshold
            ]
            if offenders:
                months_str = ", ".join(
                    f"{m.get('month', '?')} ({_rm(m.get('closing_balance'))})"
                    for m in offenders
                )
                flags.append(
                    _flag(
                        14,
                        True,
                        f"OD utilisation high (>=90% of {_rm(od_limit)} limit): {months_str}.",
                    )
                )
            else:
                flags.append(
                    _flag(
                        14,
                        False,
                        f"OD utilisation healthy across all months (limit {_rm(od_limit)}).",
                    )
                )
    else:
        offenders = [
            m
            for m in monthly_analysis
            if isinstance(m.get("closing_balance"), (int, float))
            and float(m["closing_balance"]) < LOW_CLOSING_THRESHOLD_CR
        ]
        if offenders:
            months_str = ", ".join(
                f"{m.get('month', '?')} ({_rm(m.get('closing_balance'))})"
                for m in offenders
            )
            flags.append(
                _flag(
                    14,
                    True,
                    f"Closing balance below {_rm(LOW_CLOSING_THRESHOLD_CR)} in: {months_str}.",
                )
            )
        else:
            flags.append(
                _flag(
                    14,
                    False,
                    f"Closing balance stayed at or above {_rm(LOW_CLOSING_THRESHOLD_CR)} every month.",
                )
            )

    # --- Flag 16 — HRDF Payments (informational) ----------------------------
    hrdf_detected = bool(statutory_compliance.get("hrdf_detected", False))
    hrdf_count = int(statutory_compliance.get("hrdf_count") or 0)
    hrdf_total = float(statutory_compliance.get("hrdf_total") or 0)
    if hrdf_detected:
        flags.append(
            _flag(
                15,
                False,
                f"HRDF payments detected: {hrdf_count} tx totalling {_rm(hrdf_total)} "
                "(informational; no coverage ratio computed).",
            )
        )
    else:
        detected_16 = salary_months_active > 0
        flags.append(
            _flag(
                15,
                detected_16,
                (
                    "No HRDF payments detected despite active salary months — "
                    "soft concern (PSMB exempts <10-employee firms in covered industries)."
                    if detected_16
                    else "No HRDF payments detected; no salary activity to compare against."
                ),
            )
        )

    return flags


# ---------------------------------------------------------------------------
# Statutory compliance computation — feeds Flags 6/7/8/16
# ---------------------------------------------------------------------------


def _epf_status(ratio_pct: float) -> str:
    """Categorize a single-month EPF ratio against the v6.3.4 dual band.

    OK when within EITHER the employer-only band [11, 15] OR the combined
    band [20, 26]. Below 11 -> WARNING. Above 26 (or in the gap 15-20) ->
    above-band; the caller decides CATCH_UP vs STRUCTURAL by run length.
    """
    if EPF_BAND_EMPLOYER_ONLY[0] <= ratio_pct <= EPF_BAND_EMPLOYER_ONLY[1]:
        return "OK"
    if EPF_BAND_COMBINED[0] <= ratio_pct <= EPF_BAND_COMBINED[1]:
        return "OK"
    if ratio_pct < EPF_BAND_EMPLOYER_ONLY[0]:
        return "WARNING"
    return "ABOVE"  # transient label resolved to CATCH_UP / STRUCTURAL below


def _socso_status(ratio_pct: float) -> str:
    if SOCSO_BAND[0] <= ratio_pct <= SOCSO_BAND[1]:
        return "OK"
    if ratio_pct < SOCSO_BAND[0]:
        return "WARNING"
    return "ABOVE"


def _resolve_above_runs(statuses: list[str]) -> list[str]:
    """Replace each ``ABOVE`` with ``CATCH_UP`` or ``STRUCTURAL``.

    A run of ``>= STRUCTURAL_RUN_LENGTH`` consecutive ``ABOVE`` entries
    becomes ``STRUCTURAL`` for every month in the run. Shorter runs (or
    isolated months) become ``CATCH_UP``.
    """
    resolved = list(statuses)
    n = len(resolved)
    i = 0
    while i < n:
        if resolved[i] != "ABOVE":
            i += 1
            continue
        j = i
        while j < n and resolved[j] == "ABOVE":
            j += 1
        run_len = j - i
        label = "STRUCTURAL" if run_len >= STRUCTURAL_RUN_LENGTH else "CATCH_UP"
        for k in range(i, j):
            resolved[k] = label
        i = j
    return resolved


def compute_channel_blind_indicator(
    transactions: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Heuristic for cheque-channel-heavy accounts where statutory
    contributions are likely paid via cheque or off-account.

    Walks the raw transaction list and computes:
      * ``cheque_dr_amount`` — sum of DR-side rows whose description
        matches ``CHEQUE_DR_HEURISTIC_RE`` (broader than the strict C20
        ``CHEQUE_ISSUE_RE``; tolerates UOB ``Chq Wdl`` / ``Cheque NNNN``
        bank-specific shapes that C20 doesn't match).
      * ``gross_dr_amount`` — sum of all DR-side rows.

    Channel-blind iff BOTH gates trip — magnitude (cheque_dr_amount >=
    ``CHANNEL_BLIND_CHEQUE_DR_MIN_RM``) AND significance
    (cheque_dr_amount / gross_dr_amount >=
    ``CHANNEL_BLIND_CHEQUE_DR_MIN_RATIO``). The ratio gate prevents a
    large absolute number from supplier-cheque-heavy accounts being
    misclassified when cheques are a small share of total outflows.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``. ``None`` / empty -> non-blind indicator.

    Returns:
        dict with:
            is_channel_blind (bool): True iff both gates trip.
            cheque_dr_amount (float): summed cheque-DR amount.
            gross_dr_amount (float): summed gross DR.
            cheque_dr_ratio (float): ratio in 0..1 (0 when gross_dr=0).
            threshold_amount (float): ``CHANNEL_BLIND_CHEQUE_DR_MIN_RM``.
            threshold_ratio (float): ``CHANNEL_BLIND_CHEQUE_DR_MIN_RATIO``.
            reason (str): one-line explanation for downstream remark.
    """
    cheque_dr = 0.0
    gross_dr = 0.0
    for row in transactions or ():
        if not isinstance(row, dict):
            continue
        debit = float(row.get("debit") or 0)
        if debit <= 0:
            continue
        gross_dr += debit
        description = str(row.get("description") or "")
        if CHEQUE_DR_HEURISTIC_RE.search(description):
            cheque_dr += debit
    ratio = (cheque_dr / gross_dr) if gross_dr > 0 else 0.0
    is_blind = (
        cheque_dr >= CHANNEL_BLIND_CHEQUE_DR_MIN_RM
        and ratio >= CHANNEL_BLIND_CHEQUE_DR_MIN_RATIO
    )
    if is_blind:
        reason = (
            f"Cheque-DR RM {cheque_dr:,.2f} ({ratio * 100:.1f}% of gross DR "
            f"RM {gross_dr:,.2f}) exceeds the channel-blind thresholds "
            f"(>= RM {CHANNEL_BLIND_CHEQUE_DR_MIN_RM:,.0f} and "
            f">= {CHANNEL_BLIND_CHEQUE_DR_MIN_RATIO * 100:.0f}% of gross DR) "
            "— statutory contributions may be paid via cheque or off-account "
            "and not detectable by keyword."
        )
    elif cheque_dr == 0.0:
        reason = "No cheque-DR activity; channel-blind check N/A."
    else:
        reason = (
            f"Cheque-DR RM {cheque_dr:,.2f} ({ratio * 100:.1f}% of gross DR) "
            "below channel-blind thresholds."
        )
    return {
        "is_channel_blind": is_blind,
        "cheque_dr_amount": round(cheque_dr, 2),
        "gross_dr_amount": round(gross_dr, 2),
        "cheque_dr_ratio": round(ratio, 4),
        "threshold_amount": CHANNEL_BLIND_CHEQUE_DR_MIN_RM,
        "threshold_ratio": CHANNEL_BLIND_CHEQUE_DR_MIN_RATIO,
        "reason": reason,
    }


def is_subthreshold_employer(
    monthly_amounts: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    """Heuristic for sole-prop / director-only / sub-threshold employers
    where 0% EPF coverage may be correct (no employer obligation).

    Reads ``salary_paid`` totals from ``monthly_amounts`` and applies the
    v3.3.1 commission_policy spirit ("no payroll obligation -> don't fire
    CRITICAL") to small-payroll accounts that the literal v3.3.1 list
    doesn't cover. Rule: total salary across the statement window <=
    ``SUBTHRESHOLD_TOTAL_SALARY_RM`` is treated as sub-threshold.

    Rationale (s12 calibration): RM 30K total over a 6-12 month window
    averages RM 2.5-5K / month, which for a single payee is either a
    director's basic / EA-sized "salary" (typically below EPF voluntary-
    contribution thresholds in practice) or a sole-prop owner draw, not
    a true employer-of-record relationship. Both s12 false-positive
    CRITICALs (RE Concept RM 11.8K, Calvin Skin RM 15.25K) fit this
    shape; all five true-employer accounts (Juta Kenangan, Hou Tian,
    HLB MTCE post-own-account-guard, plus the two non-CRITICAL controls)
    were well above the threshold.

    Args:
        monthly_amounts: same per-month aggregate dict that
            ``compute_statutory_compliance`` consumes. Reads only
            ``salary_paid`` per month. ``None`` / empty -> non-subthreshold.

    Returns:
        dict with:
            is_subthreshold (bool): True when 0 < total <= threshold.
            total_salary_amount (float): summed ``salary_paid``.
            threshold_amount (float): ``SUBTHRESHOLD_TOTAL_SALARY_RM``.
            reason (str): one-line explanation for downstream remark.
    """
    total = 0.0
    for month_dict in (monthly_amounts or {}).values():
        if not isinstance(month_dict, dict):
            continue
        total += float(month_dict.get("salary_paid") or 0)
    is_sub = 0.0 < total <= SUBTHRESHOLD_TOTAL_SALARY_RM
    return {
        "is_subthreshold": is_sub,
        "total_salary_amount": round(total, 2),
        "threshold_amount": SUBTHRESHOLD_TOTAL_SALARY_RM,
        "reason": (
            f"Total salary RM {total:,.2f} <= sub-threshold "
            f"RM {SUBTHRESHOLD_TOTAL_SALARY_RM:,.0f} — likely "
            "sole-prop / director-only / sub-threshold employer; EPF / "
            "SOCSO obligation may not apply."
            if is_sub
            else (
                "No salary detected; sub-threshold check N/A."
                if total == 0.0
                else f"Total salary RM {total:,.2f} above sub-threshold."
            )
        ),
    }


def compute_statutory_compliance(
    monthly_amounts: dict[str, dict[str, float]],
    *,
    transactions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``statutory_compliance`` sub-object from per-month aggregates.

    Implements the algorithm in ``SYSTEM_PROMPT_v3_5_6.md`` lines 488-521:
    coverage via SET INTERSECTION (capped at 100), per-month ratios for
    EPF/SOCSO with v6.3.4 dual-band status, LHDN/HRDF presence-only (no
    coverage ratio), overall_status that does not degrade on STRUCTURAL.

    Sub-threshold employer downgrade (s12): when
    ``is_subthreshold_employer`` flags the account as sub-threshold and
    the verdict would otherwise be CRITICAL (0% EPF or SOCSO coverage),
    ``overall_status`` becomes ``SUB_THRESHOLD`` instead. The
    ``subthreshold_employer`` indicator is returned in the output so
    downstream consumers (risk-flag remarks) can surface the context.

    Channel-blind employer downgrade (s12): when ``transactions`` are
    provided and ``compute_channel_blind_indicator`` flags the account
    as cheque-channel-heavy, an otherwise-CRITICAL verdict becomes
    ``CHANNEL_BLIND``. Priority: ``SUB_THRESHOLD`` > ``CHANNEL_BLIND`` >
    ``CRITICAL`` (no-obligation beats can't-verify beats real-gap). The
    ``channel_blind_employer`` indicator is always returned (with
    ``is_channel_blind=False`` when transactions are not provided), so
    callers can branch on it unconditionally.

    Args:
        monthly_amounts: mapping ``YYYY-MM`` -> dict of monthly totals.
            Each month dict reads (missing keys treated as 0.0):
                ``salary_paid``, ``statutory_epf``, ``statutory_socso``,
                ``statutory_tax`` (LHDN bucket: PCB+CP204+SST+stamp+RPGT),
                ``statutory_hrdf`` (PSMB).
            Months with all zeros are still iterated (months without any
            payroll activity simply do not enter ``salary_months_list``).

    Returns:
        dict matching the ``statutory_compliance`` schema at
        ``BANK_ANALYSIS_SCHEMA_v6_3_5.json`` line 777, with all 17
        required keys populated. Output is suitable to pass directly as
        ``compute_risk_flags(..., statutory_compliance=...)``.
    """
    monthly_amounts = monthly_amounts or {}

    def _amt(month: str, key: str) -> float:
        return float((monthly_amounts.get(month) or {}).get(key) or 0)

    months_sorted = sorted(monthly_amounts.keys())

    salary_months_list = [m for m in months_sorted if _amt(m, "salary_paid") > 0]
    epf_months_list = [m for m in months_sorted if _amt(m, "statutory_epf") > 0]
    socso_months_list = [m for m in months_sorted if _amt(m, "statutory_socso") > 0]
    lhdn_months = [m for m in months_sorted if _amt(m, "statutory_tax") > 0]
    hrdf_months = [m for m in months_sorted if _amt(m, "statutory_hrdf") > 0]

    salary_set = set(salary_months_list)
    epf_set = set(epf_months_list)
    socso_set = set(socso_months_list)

    # Coverage via SET INTERSECTION, capped at 100. When salary_set is empty,
    # the entity has no employer obligation to compare against; emit 100 so
    # Flag 6/7 do not fire on non-employer accounts.
    if salary_set:
        epf_coverage_pct = min(
            100.0, (len(epf_set & salary_set) / len(salary_set)) * 100.0
        )
        socso_coverage_pct = min(
            100.0, (len(socso_set & salary_set) / len(salary_set)) * 100.0
        )
    else:
        epf_coverage_pct = 100.0
        socso_coverage_pct = 100.0

    epf_months_missing = sorted(salary_set - epf_set)
    socso_months_missing = sorted(salary_set - socso_set)

    # Per-month ratios (EPF/SOCSO only) for months where BOTH amounts exist.
    epf_overlap = sorted(epf_set & salary_set)
    epf_raw_statuses: list[str] = []
    epf_per_month_ratios_pre: list[dict[str, Any]] = []
    for m in epf_overlap:
        salary_amount = _amt(m, "salary_paid")
        epf_amount = _amt(m, "statutory_epf")
        ratio_pct = (epf_amount / salary_amount) * 100.0 if salary_amount else 0.0
        epf_raw_statuses.append(_epf_status(ratio_pct))
        epf_per_month_ratios_pre.append(
            {
                "month": m,
                "epf_amount": round(epf_amount, 2),
                "salary_amount": round(salary_amount, 2),
                "ratio_pct": round(ratio_pct, 2),
            }
        )
    epf_resolved = _resolve_above_runs(epf_raw_statuses)
    epf_per_month_ratios = [
        {**row, "status": status}
        for row, status in zip(epf_per_month_ratios_pre, epf_resolved)
    ]

    socso_overlap = sorted(socso_set & salary_set)
    socso_raw_statuses: list[str] = []
    socso_per_month_ratios_pre: list[dict[str, Any]] = []
    for m in socso_overlap:
        salary_amount = _amt(m, "salary_paid")
        socso_amount = _amt(m, "statutory_socso")
        ratio_pct = (socso_amount / salary_amount) * 100.0 if salary_amount else 0.0
        socso_raw_statuses.append(_socso_status(ratio_pct))
        socso_per_month_ratios_pre.append(
            {
                "month": m,
                "socso_amount": round(socso_amount, 2),
                "salary_amount": round(salary_amount, 2),
                "ratio_pct": round(ratio_pct, 2),
            }
        )
    socso_resolved = _resolve_above_runs(socso_raw_statuses)
    socso_per_month_ratios = [
        {**row, "status": status}
        for row, status in zip(socso_per_month_ratios_pre, socso_resolved)
    ]

    # Sub-threshold indicator — extends the v3.3.1 commission_policy
    # "no payroll obligation" spirit to small-payroll / director-only
    # accounts. Surfaced by the s12 calibration (see is_subthreshold_employer).
    subthreshold = is_subthreshold_employer(monthly_amounts)

    # Channel-blind indicator — cheque-DR-heavy accounts where statutory
    # may be paid via cheque or off-account. Requires raw transactions
    # (cheque-DR shape isn't carried in monthly_amounts); skipped with a
    # neutral indicator when ``transactions`` is None.
    if transactions is not None:
        channel_blind = compute_channel_blind_indicator(transactions)
    else:
        channel_blind = {
            "is_channel_blind": False,
            "cheque_dr_amount": 0.0,
            "gross_dr_amount": 0.0,
            "cheque_dr_ratio": 0.0,
            "threshold_amount": CHANNEL_BLIND_CHEQUE_DR_MIN_RM,
            "threshold_ratio": CHANNEL_BLIND_CHEQUE_DR_MIN_RATIO,
            "reason": (
                "transactions not provided to compute_statutory_compliance; "
                "channel-blind check skipped."
            ),
        }

    # overall_status — coverage-only (STRUCTURAL alone does not degrade).
    # Sub-threshold and channel-blind downgrade priority:
    # SUB_THRESHOLD > CHANNEL_BLIND > CRITICAL. Sub-threshold wins because
    # "no employer obligation" is a stronger statement than "obligation
    # may exist but not visible by keyword".
    salary_active = bool(salary_set)
    if salary_active and (epf_coverage_pct == 0.0 or socso_coverage_pct == 0.0):
        if subthreshold["is_subthreshold"]:
            overall_status = "SUB_THRESHOLD"
        elif channel_blind["is_channel_blind"]:
            overall_status = "CHANNEL_BLIND"
        else:
            overall_status = "CRITICAL"
    elif epf_coverage_pct >= 100.0 and socso_coverage_pct >= 100.0:
        overall_status = "COMPLIANT"
    else:
        overall_status = "GAPS_DETECTED"

    return {
        "salary_months_active": len(salary_set),
        "salary_months_list": sorted(salary_set),
        "epf_months_paid": len(epf_set),
        "epf_months_list": sorted(epf_set),
        "epf_months_missing": epf_months_missing,
        "epf_coverage_pct": round(epf_coverage_pct, 2),
        "socso_months_paid": len(socso_set),
        "socso_months_list": sorted(socso_set),
        "socso_months_missing": socso_months_missing,
        "socso_coverage_pct": round(socso_coverage_pct, 2),
        "lhdn_months_paid": len(lhdn_months),
        "lhdn_detected": bool(lhdn_months),
        "hrdf_months_paid": len(hrdf_months),
        "hrdf_detected": bool(hrdf_months),
        "epf_per_month_ratios": epf_per_month_ratios,
        "socso_per_month_ratios": socso_per_month_ratios,
        "subthreshold_employer": subthreshold,
        "channel_blind_employer": channel_blind,
        "overall_status": overall_status,
    }


# ---------------------------------------------------------------------------
# C21 / C22 / C23 — row-level monitoring computations
# Source of truth: CLASSIFICATION_RULES_v3_5.json lines 1036-1083 (LOCKED).
# These three are CR-side monitoring flags applied AFTER classification —
# they do not require any classifier output, only the canonical row schema.
# ---------------------------------------------------------------------------

ROUND_FIGURE_DEFAULT_MULTIPLE = 10_000.0
ROUND_FIGURE_DEFAULT_MIN_AMOUNT = 10_000.0
LARGE_CREDIT_DEFAULT_THRESHOLD = 100_000.0
HIGH_VALUE_DEFAULT_MULTIPLIER = 3.0


def _credit_amount(row: dict[str, Any]) -> float:
    """Return the row's credit amount as a non-negative float, or 0.0."""
    value = row.get("credit")
    if value is None:
        return 0.0
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return 0.0
    return amount if amount > 0 else 0.0


def _to_entry(row: dict[str, Any], amount: float) -> dict[str, Any]:
    """Build a ``transaction_entry`` (date, description, amount[, balance])."""
    entry: dict[str, Any] = {
        "date": row.get("date"),
        "description": row.get("description") or "",
        "amount": round(amount, 2),
    }
    balance = row.get("balance")
    if isinstance(balance, (int, float)):
        entry["balance"] = float(balance)
    return entry


def compute_round_figure_credits(
    transactions: list[dict[str, Any]],
    *,
    multiple: float = ROUND_FIGURE_DEFAULT_MULTIPLE,
    min_amount: float = ROUND_FIGURE_DEFAULT_MIN_AMOUNT,
) -> dict[str, Any]:
    """C21 — credits whose amount is an exact multiple of ``multiple`` and
    at or above ``min_amount``.

    Rule (verbatim from CLASSIFICATION_RULES_v3_5.json line 1047):
        ``amount >= 10000 AND amount % 10000 == 0``

    Pure math — no keyword matching. Applied to credits only; debits and
    rows without a credit amount are silently skipped.

    Args:
        transactions: rows in canonical schema. Reads ``credit``, ``date``,
            ``description``, ``balance``.
        multiple: structuring step (default RM10,000). The ``%`` test uses
            this value.
        min_amount: lower bound (default RM10,000). Rows below this are
            skipped even if divisible.

    Returns:
        dict with keys ``round_figure_cr`` (float, sum), ``round_figure_count``
        (int), and ``round_figure_entries`` (list[transaction_entry]).
    """
    if multiple <= 0:
        return {
            "round_figure_cr": 0.0,
            "round_figure_count": 0,
            "round_figure_entries": [],
        }

    entries: list[dict[str, Any]] = []
    for row in transactions:
        amount = _credit_amount(row)
        if amount < min_amount:
            continue
        # Float modulo is fragile; use integer-cents math.
        cents = round(amount * 100)
        step_cents = round(multiple * 100)
        if step_cents <= 0 or cents % step_cents != 0:
            continue
        entries.append(_to_entry(row, amount))

    total = round(sum(e["amount"] for e in entries), 2)
    return {
        "round_figure_cr": total,
        "round_figure_count": len(entries),
        "round_figure_entries": entries,
    }


def compute_large_credits(
    transactions: list[dict[str, Any]],
    *,
    threshold: float = LARGE_CREDIT_DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """C23 — credits at or above an absolute threshold (default RM100,000).

    Rule (verbatim from CLASSIFICATION_RULES_v3_5.json line 1078):
        ``credit >= large_credit_threshold``

    Args:
        transactions: rows in canonical schema. Reads ``credit``, ``date``,
            ``description``, ``balance``.
        threshold: absolute floor in ringgit. Configurable per the LOCKED
            rule (``configurable: true`` in the rulebook).

    Returns:
        dict with key ``large_credits`` (list[transaction_entry]) suitable
        to drop straight into the analysis JSON's top-level ``large_credits``
        array AND the 16-flag reducer's ``summary['large_credits']`` input.
    """
    entries: list[dict[str, Any]] = []
    for row in transactions:
        amount = _credit_amount(row)
        if amount < threshold:
            continue
        entries.append(_to_entry(row, amount))

    return {"large_credits": entries}


def compute_high_value_credits(
    transactions: list[dict[str, Any]],
    *,
    multiplier: float = HIGH_VALUE_DEFAULT_MULTIPLIER,
    eod_unreliable: bool = False,
) -> dict[str, Any]:
    """C22 — credits exceeding ``multiplier`` times that month's EOD average.

    Rule (verbatim from CLASSIFICATION_RULES_v3_5.json line 1062):
        ``credit > 3 * monthly_eod_avg``

    Per-month threshold — composes with session-1 ``compute_monthly_eod``
    over each ``YYYY-MM`` present in ``transactions``. When
    ``eod_unreliable=True`` (caller's reconciliation harness flagged the
    balance trail as unreliable), the rule is SKIPPED ENTIRELY per the
    prompt's "no proxy values" fallback — output is all-zeros with the
    ``eod_unreliable`` flag echoed so ``compute_risk_flags`` Flag 4 honours
    the skip.

    Months whose EOD average is ``None`` (no balances or no transactions
    that month) contribute zero entries — the function does not abort
    when individual months lack EOD data, only when the caller globally
    declares EOD unreliable.

    Args:
        transactions: rows in canonical schema. Reads ``credit``, ``date``,
            ``balance``, ``description``. Should contain only rows for ONE
            account; the caller is responsible for per-account grouping.
        multiplier: threshold multiplier (default 3.0).
        eod_unreliable: when True, skip the rule entirely.

    Returns:
        dict with keys ``high_value_cr`` (float, sum), ``high_value_count``
        (int), ``high_value_entries`` (list[transaction_entry]), and
        ``eod_unreliable`` (bool, echoed input).
    """
    if eod_unreliable:
        return {
            "high_value_cr": 0.0,
            "high_value_count": 0,
            "high_value_entries": [],
            "eod_unreliable": True,
        }

    months: set[str] = set()
    for row in transactions:
        date = row.get("date")
        if isinstance(date, str) and len(date) >= 7:
            months.add(date[:7])

    monthly_threshold: dict[str, float] = {}
    for ym in months:
        eod = compute_monthly_eod(transactions, ym)
        avg = eod.get("eod_average")
        if avg is None or avg <= 0:
            continue
        monthly_threshold[ym] = multiplier * float(avg)

    entries: list[dict[str, Any]] = []
    for row in transactions:
        date = row.get("date")
        if not isinstance(date, str) or len(date) < 7:
            continue
        threshold = monthly_threshold.get(date[:7])
        if threshold is None:
            continue
        amount = _credit_amount(row)
        if amount <= threshold:
            continue
        entries.append(_to_entry(row, amount))

    total = round(sum(e["amount"] for e in entries), 2)
    return {
        "high_value_cr": total,
        "high_value_count": len(entries),
        "high_value_entries": entries,
        "eod_unreliable": False,
    }


# ---------------------------------------------------------------------------
# C14 / C15 — Returned cheques (inward DR / outward CR)
# Source of truth: CLASSIFICATION_RULES_v3_5.json line 907.
# ---------------------------------------------------------------------------

RETURNED_CHEQUE_RE = re.compile(
    r"(?:RETURN(?:ED)?\s+CHQ|CHQ\s+RETURN|DISHONOUR)",
    re.IGNORECASE,
)


def _row_side(row: dict[str, Any]) -> str | None:
    """Classify row as ``CR`` / ``DR`` / ``None`` based on credit/debit values."""
    credit = row.get("credit")
    debit = row.get("debit")
    try:
        c = float(credit or 0)
    except (TypeError, ValueError):
        c = 0.0
    try:
        d = float(debit or 0)
    except (TypeError, ValueError):
        d = 0.0
    if c > 0 and d <= 0:
        return "CR"
    if d > 0 and c <= 0:
        return "DR"
    return None


def compute_returned_cheques(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect returned cheques on both sides via the LOCKED v3.5 regex.

    Rule: a row matching the ``RETURNED_CHEQUE_RE`` pattern in its
    description is a returned cheque. The side discriminates inward vs
    outward:

      * DR side (debit) -> C14 inward (cheque the company DEPOSITED bounced).
      * CR side (credit) -> C15 outward (a cheque the company ISSUED bounced
        and the bank credited the amount back).

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict with C14/C15 schema fields populated for the 16-flag reducer:
            returned_cheques_inward_count, returned_cheques_inward_amount,
            returned_cheques_outward_count, returned_cheques_outward_amount,
            inward_entries, outward_entries (transaction_entry list).
    """
    inward_entries: list[dict[str, Any]] = []
    outward_entries: list[dict[str, Any]] = []
    inward_total = 0.0
    outward_total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not RETURNED_CHEQUE_RE.search(str(description)):
            continue
        side = _row_side(row)
        if side == "DR":
            amount = float(row.get("debit") or 0)
            inward_entries.append(_to_entry(row, amount))
            inward_total += amount
        elif side == "CR":
            amount = float(row.get("credit") or 0)
            outward_entries.append(_to_entry(row, amount))
            outward_total += amount

    return {
        "returned_cheques_inward_count": len(inward_entries),
        "returned_cheques_inward_amount": round(inward_total, 2),
        "returned_cheques_outward_count": len(outward_entries),
        "returned_cheques_outward_amount": round(outward_total, 2),
        "inward_entries": inward_entries,
        "outward_entries": outward_entries,
    }


# ---------------------------------------------------------------------------
# C06 / C07 / C08 / C09 — Statutory keyword detectors (DR side).
# Source of truth: CLASSIFICATION_RULES_v3_5.json categories C06-C09 (LOCKED).
#
# All four feed the per-month ``monthly_amounts`` dict consumed by
# ``compute_statutory_compliance`` (session 3) via a monthly aggregator that
# is NOT YET WIRED. Aggregator wiring is a follow-on session.
#
# These are pure keyword detectors. The v3.3.1 MANDATORY side-gate from
# C06 (``side == 'DR' AND NOT match_own_party``) is split here:
#
#   * The DR-side half IS enforced (rows on the credit side are skipped) —
#     C06-C09 are employer-paying-out flows by definition.
#   * The own-party half is NOT enforced here. ``match_own_party`` requires
#     ``normalize_company_suffix`` (Track-1 BUG-003) which is not yet on
#     the Track-2 side; per the s9 handoff this lives at the dispatcher
#     hook, not in the per-row detector. Callers that need the own-party
#     gate must apply it before consuming these entries.
#
# Likewise, the ``JomPAY without entity name visible`` exclusion shared by
# all four rules is dispatcher-level — Track 2 has ``is_jompay_biller_code_only``
# (s8) for that short-circuit; do NOT bake it into these detectors.
# ---------------------------------------------------------------------------

EPF_PAYMENT_RE = re.compile(
    r"(?:KUMPULAN WANG SIMPAN(?:AN PEKERJA)?|EPF DPE|\bEPF\b|\bKWSP(?=\s|[/\-]|$))",
    re.IGNORECASE,
)


def compute_epf_payments(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect EPF / KWSP employer contributions via the LOCKED v3.5 regex.

    Rule: a DR-side row whose description matches ``EPF_PAYMENT_RE`` is an
    EPF payment (C06) — the employer paying its monthly contribution to
    KWSP. CR-side rows are NEVER C06 by definition (an inbound credit
    referencing EPF/KWSP is a refund or claim disbursement, not an
    employer contribution) so they are filtered out here.

    Regex shape preserved verbatim from v3.5 LOCKED rules:
      * ``KUMPULAN WANG SIMPAN(AN PEKERJA)?`` — full Malay name with the
        truncated trailing-token variant some banks emit.
      * ``EPF DPE`` — bank-specific abbreviation (Direct Payment EPF).
      * ``\\bEPF\\b`` — standalone abbreviation with word boundaries.
      * ``\\bKWSP(?=\\s|[/\\-]|$)`` — KWSP followed only by whitespace,
        slash, hyphen, or end-of-string. Lookahead is what stops
        reference-number patterns like ``KWSP0559246`` (KWSP Account 2
        medical-claim IDs) from matching as employer contributions; the
        rules explicitly call this exclusion out.

    Own-party suppression and JomPAY biller-code suppression are NOT
    applied here — see the section-level comment above for the contract.

    Output is suitable for downstream monthly aggregation into the
    ``statutory_epf`` field that ``compute_statutory_compliance`` (s3)
    consumes; the aggregator itself is not yet wired.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict:
            epf_payments_count, epf_payments_amount, epf_payments_entries.
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not EPF_PAYMENT_RE.search(str(description)):
            continue
        if _row_side(row) != "DR":
            continue
        amount = float(row.get("debit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "epf_payments_count": len(entries),
        "epf_payments_amount": round(total, 2),
        "epf_payments_entries": entries,
    }


SOCSO_PAYMENT_RE = re.compile(
    r"(?:\b(?:SOCSO|PERKESO|PERTUBUHAN\s+KESELAM(?:AT(?:AN)?|A)(?:\s+SOSIAL)?|EIS)\b"
    r"|\bFPX\s*B2B\s+PERTUBUH(?=\s|$|/|-)"
    r"|\bPERTUBUH(?:AN)?\s+CP(?=[_\s]|$))",
    re.IGNORECASE,
)


def compute_socso_payments(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect SOCSO / PERKESO employer contributions via the LOCKED v3.5 regex.

    Rule: a DR-side row whose description matches ``SOCSO_PAYMENT_RE`` is
    a SOCSO payment (C07). CR-side rows are NEVER C07 — inbound credits
    referencing SOCSO/PERKESO are refunds, claim disbursements or welfare
    fund payments (e.g. TABUNG PRIHATIN PROTEK PERKESO), not contributions.

    Four shape families per v3.5 keywords + the truncation_note:
      * ``SOCSO`` / ``PERKESO`` / ``EIS`` — short forms with word
        boundaries. ``EIS`` is the Employment Insurance System paid into
        the same PERKESO collection channel.
      * ``PERTUBUHAN KESELAM(AT(AN)? | A)(\\s+SOSIAL)?`` — full Malay name
        plus the Maybank / further-truncation variants
        (``PERTUBUHAN KESELAMAT``, ``PERTUBUHAN KESELAMA``).
      * ``FPX B2B PERTUBUH`` — RHB FPX B2B one-token cap form.
      * ``PERTUBUH(AN)? CP`` — column-truncated form where the FPX prefix
        was dropped. Lookahead ``(?=[_\\s]|$)`` keeps it from matching
        legitimate ``PERTUBUHAN PELADANG / PELANCONGAN / NELAYAN``
        organisations (which use different trailing tokens).

    Own-party suppression and JomPAY biller-code suppression are NOT
    applied here.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict:
            socso_payments_count, socso_payments_amount,
            socso_payments_entries.
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not SOCSO_PAYMENT_RE.search(str(description)):
            continue
        if _row_side(row) != "DR":
            continue
        amount = float(row.get("debit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "socso_payments_count": len(entries),
        "socso_payments_amount": round(total, 2),
        "socso_payments_entries": entries,
    }


LHDN_TAX_PAYMENT_RE = re.compile(
    r"LEMBAGA HASIL(?:\s+DALAM NEGERI)?|\bLHDN\b",
    re.IGNORECASE,
)


def compute_lhdn_tax_payments(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect LHDN income-tax payments via the LOCKED v3.5 regex.

    Rule: a DR-side row whose description matches ``LHDN_TAX_PAYMENT_RE``
    is an income-tax payment to LHDN (C08). Two shape families:
      * ``LEMBAGA HASIL`` (optionally followed by ``DALAM NEGERI``) —
        the full Malay name.
      * ``\\bLHDN\\b`` — the abbreviation, word-bounded.

    The matching note in v3.5 is critical: do NOT match bare ``HASIL``.
    Person names like ``HASILA BINTI HASHIM`` contain ``HASIL`` as a
    substring but are not LHDN; the regex requires ``LEMBAGA HASIL``
    as the prefix to avoid that misfire.

    Bank ``SERVICE TAX 8% SST`` does NOT match here (no ``LEMBAGA HASIL``
    / ``LHDN`` token) and instead routes to C24 via ``BANK_FEES_RE`` —
    matching the v3.5 critical_exclusion.

    C08 is informational only — there is no salary-coverage check (some
    companies pay tax from a different account legitimately).

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict:
            lhdn_tax_payments_count, lhdn_tax_payments_amount,
            lhdn_tax_payments_entries.
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not LHDN_TAX_PAYMENT_RE.search(str(description)):
            continue
        if _row_side(row) != "DR":
            continue
        amount = float(row.get("debit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "lhdn_tax_payments_count": len(entries),
        "lhdn_tax_payments_amount": round(total, 2),
        "lhdn_tax_payments_entries": entries,
    }


HRDF_PAYMENT_RE = re.compile(
    r"(?:PEMBANGUNAN SUMBER M(?:ANUSIA)?|\bPSMB\b|\bHRDF\b)",
    re.IGNORECASE,
)


def compute_hrdf_payments(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect HRDF / PSMB payments via the LOCKED v3.5 regex.

    Rule: a DR-side row whose description matches ``HRDF_PAYMENT_RE`` is
    an HRDF levy payment (C09). Three shape families:
      * ``PEMBANGUNAN SUMBER M(ANUSIA)?`` — full Malay name with the
        truncated trailing-token variant some banks emit
        (``PEMBANGUNAN SUMBER M``).
      * ``\\bPSMB\\b`` — the agency abbreviation (Pembangunan Sumber
        Manusia Berhad).
      * ``\\bHRDF\\b`` — the levy abbreviation (Human Resources
        Development Fund).

    HRDF is often legitimately absent for small companies — v3.5 fallback
    is "not flagged as error". Coverage is presence-only.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict:
            hrdf_payments_count, hrdf_payments_amount, hrdf_payments_entries.
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not HRDF_PAYMENT_RE.search(str(description)):
            continue
        if _row_side(row) != "DR":
            continue
        amount = float(row.get("debit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "hrdf_payments_count": len(entries),
        "hrdf_payments_amount": round(total, 2),
        "hrdf_payments_entries": entries,
    }


# ---------------------------------------------------------------------------
# C05 — Salary / Payroll (DR side, keyword match + commission v3.3.1 block).
# Source of truth: CLASSIFICATION_RULES_v3_5.json category C05 (UPDATED v3.3.1).
#
# Two regexes drive C05:
#   * SALARY_KEYWORD_RE — the v3.5 salary_keywords list, with the
#     ``salary_regex_note`` ``\bGAJI\b`` word-boundary requirement that
#     avoids ``MENGAJI`` / ``NGAJI`` (Malay: "teaching") substring
#     collisions on tuition / education business statements.
#   * COMMISSION_BLOCK_RE — the v3.3.1 commission_policy keyword list.
#     A row matching this regex is NOT C05 even when a salary keyword
#     is also present. Commission agents are typically independent
#     contractors, not employees; the policy was driven by the MYTUTOR
#     ACADEMY validation run where 1,852 "Comm" tutor payments
#     distorted the EPF ratio to 46.91%.
#
# The two CIMB AUTOPAY DR ``U\d{4}`` (bulk_salary FULL_CODE) and Maybank
# ``TRANSFER FR A/C [name]* [purpose]`` (FULL_CODE) bank patterns are
# matched naturally via the salary_keywords list — ``AUTOPAY DR`` is
# in the keyword list (catches the CIMB bulk form regardless of the
# U-account suffix), and any salary keyword anywhere in the description
# catches the Maybank pattern's "salary keyword in purpose" requirement.
# The CIMB AI_ASSIST individual branch (``TR TO SAVINGS + [name] +
# salary_keyword``) is dispatcher-level — when the row has no salary
# keyword in the bank-emitted description, only AI scoring against the
# extracted counterparty name can decide. That branch is deliberately
# NOT enforced here.
#
# Per v3.5 cross_ref: "Salary takes priority over C04 if salary keyword
# present." Priority resolution between this detector and C04 is the
# dispatcher's responsibility; this function just returns the salary
# evidence. Per v3.5 exclusions: ``AUTOPAY CHARGES`` routes to C24 — the
# salary regex requires ``AUTOPAY DR`` (not the bare ``AUTOPAY``) so
# ``AUTOPAY CHARGES`` does not match here.
# ---------------------------------------------------------------------------

# Custom token boundary: like ``\b`` but only rejects alphabetic-letter
# adjacency. Plain ``\b`` does NOT match between an underscore and a letter
# (Python treats ``_`` as a word character), so ``\bNET\s+PAY\b`` silently
# misses the HLB CIB shape ``Sep 2025_Net Pay <NAME>`` — and the same flaw
# silently misses underscore- or digit-adjacent variants of every other
# salary keyword. The custom anchors permit ``_``, digits, punctuation, and
# whitespace on either side while still blocking embedded substrings like
# ``INTERNET PAYMENT`` (T-letter before NET) or ``MENGAJI`` (G-letter
# before AJI). See test_underscore_bounded_net_pay_matches +
# test_internet_payment_does_not_match_net_pay in test_track2_salary.py.
_TOK_LB = r"(?<![A-Za-z])"
_TOK_RB = r"(?![A-Za-z])"

SALARY_KEYWORD_RE = re.compile(
    _TOK_LB + r"SALAR(?:Y|IES)" + _TOK_RB
    # Full Malay name shapes with explicit GAJI right-boundary via the
    # trailing token alternation. Month-suffix variants cover the v3.5
    # keyword list's GAJI JAN .. GAJI DIS (English + BM forms).
    + r"|" + _TOK_LB + r"GAJI\s+(?:BULANAN|BLN|JAN|FEB|MAR|MAC|APR|MAY|MEI|JUN|JUL|AUG"
                       r"|OGOS|SEP|SEPT|OCT|OKT|NOV|DEC|DIS)" + _TOK_RB
    + r"|" + _TOK_LB + r"(?:BAYARAN|PEMBAYARAN)\s+GAJI" + _TOK_RB
    + r"|" + _TOK_LB + r"STAFF\s+(?:SALARY|INCENTIVE|OVERTIME|BONUS|ADVANCE)" + _TOK_RB
    + r"|" + _TOK_LB + r"EXTRA\s+SALARY" + _TOK_RB
    + r"|" + _TOK_LB + r"GUARD\s+SALARY" + _TOK_RB
    + r"|" + _TOK_LB + r"PMT\s+SLRY" + _TOK_RB
    + r"|" + _TOK_LB + r"SLRY" + _TOK_RB
    + r"|" + _TOK_LB + r"PAYROLL" + _TOK_RB
    + r"|" + _TOK_LB + r"NET\s+PAY" + _TOK_RB
    + r"|" + _TOK_LB + r"AUTOPAY\s+DR" + _TOK_RB,
    re.IGNORECASE,
)


# Concat-form salary keywords for Bank Rakyat DATAPOS exports. BR emits
# parsed descriptions like ``94040 DUITNOWTRANSFER 5555.29 NORMALABTYAAKUB
# StaffID005 GAJINOVEMBER2023`` — the staff ID and salary marker are glued
# to the next token with no whitespace. SALARY_KEYWORD_RE requires
# ``GAJI\s+`` and ``STAFF\s+SALARY`` (whitespace mandatory), so the
# concat form silently misses. This mirrors the s30 _CORPORATE_SUFFIX_CONCAT_RE
# pattern (commit 7fde7c2) that handled FELCRABERHAD-style concat for
# corporate-suffix detection.
#
# Bounded FP surface:
#   * Lookbehind ``(?<![A-Z])`` prevents matches inside other words
#     (MENGAJI / NGAJI / BERGAJIULAR / OUTSTAFFID5 all fail).
#   * Month alternation accepts only valid BM + English month names
#     (full + abbrev), not arbitrary trailing letters.
#   * STAFFID requires ``\d+`` so STAFFIDABC / STAFFIDNAME don't match.
#   * is_salary_payment's DR-side gate still blocks CR-side hits
#     (PTGNGAJIOKT and BAYPOTONGANGAJI on CR side stay non-C05).
#   * Longer alternatives first so e.g. JANUARI matches before JAN to
#     avoid a JAN-then-failed-lookahead backtrack on JANUARI inputs.
_SALARY_KEYWORD_CONCAT_RE = re.compile(
    r"(?<![A-Z])"
    r"(?:"
        r"GAJI(?:"
            r"BULANAN"
            r"|JANUARI|JAN"
            r"|FEBRUARI|FEB"
            r"|MAC|MAR"
            r"|APRIL|APR"
            r"|MEI|MAY"
            r"|JUN"
            r"|JULAI|JUL"
            r"|OGOS|AUG"
            r"|SEPTEMBER|SEPT|SEP"
            r"|OKTOBER|OCTOBER|OKT|OCT"
            r"|NOVEMBER|NOV"
            r"|DECEMBER|DISEMBER|DEC|DIS"
        r")\d{0,4}"
    r"|STAFFID\d+"
    r")"
    r"(?![A-Z])",
    re.IGNORECASE,
)


COMMISSION_BLOCK_RE = re.compile(
    # \bCOMMS?\b covers bare COMM, bare COMMS, AND the v3.3.1 "PT COMM"
    # / "PT COMMS" prefix forms (word boundaries absorb the PT prefix).
    r"\bCOMMS?\b"
    # COMMISSION (English) plus the misspelt single-S variant COMMISION
    # (v3.3.1 list keeps that verbatim since real banks emit it). The
    # ``(?:S|ED|ING)?`` tail covers the plural / past-tense / gerund
    # forms (COMMISSIONS / COMMISIONS / COMMISSIONED / COMMISSIONING)
    # which v3.3.1 doesn't list literally but are the obvious English
    # equivalents of KOMISYEN and within the policy's intent (broad
    # commission detection to prevent payroll inflation).
    r"|\bCOMMIS(?:S)?ION(?:S|ED|ING)?\b"
    # Malay forms: KOMISEN and KOMISYEN.
    r"|\bKOMIS(?:E|YE)N\b"
    r"|\bHABUAN\b",
    re.IGNORECASE,
)


# Own-account / inter-account markers emitted by bank CIB platforms when a
# transfer's destination is the same customer's own account. HLB MTCE
# (MTC ENGINEERING SDN BHD) surfaced this during the s12 statutory-chain
# calibration: two RM 500K+ DR rows tagged ``OWN ACC TXN`` / ``INTER ACC
# TXN`` with ``SALARY`` in the same description matched C05 even though
# the destination was the company's own account. These are bank-emitted
# machine tags, not free-form description, so a literal block is safe.
# C01 / C02 dispatcher-level own-party detection (blocked on BUG-003)
# would catch the same shape via name comparison; this guard lets C05
# self-suppress until the dispatcher lands.
OWN_ACCOUNT_BLOCK_RE = re.compile(
    r"\bOWN\s+ACC(?:OUNT)?\s+TXN\b"
    r"|\bINTER\s+ACC(?:OUNT)?\s+TXN\b"
    r"|\bOWN\s+ACCOUNT\s+TRANSFER\b",
    re.IGNORECASE,
)


# Bank-emitted fee / charge opcodes. Banks frequently emit a companion fee
# row for each underlying transaction (DuitNow fee, SMS notification fee,
# DR charges, etc.). The Bank Rakyat parser concatenates the underlying
# transaction's counterparty metadata onto the fee row's description, so
# a salary transfer's "StaffID<NNN> GAJI<MONTH><YYYY>" tail surfaces on
# the corresponding RM 0.10 / RM 0.50 fee row. Without this block, the
# concat-form salary regex tags those fee rows as C05 — visible as 1,127
# CIBSMSFEE + 1,126 CIBDRCHARGES + 41 DUITNOWFEE rows on Felcra BR/8.
#
# Cross-bank scope: the surfaced opcodes are BR-specific (CIB* prefix is
# Bank Rakyat CIB platform; DUITNOWFEE is the bank-agnostic interbank
# DuitNow fee shape). Generic ``\bFEE\b`` would risk matching legitimate
# salary rows for employees whose names contain FEE — keep the list
# explicit, extend per surfaced shape rather than broaden.
_BANK_FEE_BLOCK_RE = re.compile(
    r"\b(?:CIBSMSFEE|CIBDRCHARGES|DUITNOWFEE|IBGFEE|SMSFEE)\b",
    re.IGNORECASE,
)


# Bank-Rakyat-specific JomPay utility-merchant detector. BR's DATAPOS
# parser emits ``94804 CIBDRADVICE(JomPA <amount> <BILLER-TOKEN> <ref>``
# for JomPay utility payments. The biller tokens are truncated concat
# shapes — DiGi → DIGITELECOMMUNI, TNB → TENAGANASIONAL, Pengurusan Air
# → PENGURUSANAIRS, Indah Water Konsortium → INDAHWATERKONS, TM
# Technology Services → TMTECHNOLOGYSE. These rows arrive with no
# counterparty extracted by the upstream pipeline AND the truncated
# tokens carry no corporate suffix, so the C26/C27 corporate-suffix
# rung cannot fire and rows silently fall through to UNCLASSIFIED.
#
# Surfaced at: BR/8 (KOPERASIFELCRA), 91 rows post-s32 (DIGI 45 + TNB
# 19 + Pengurusan Air 15 + Indah Water 7 + TM 5).
#
# Cross-bank scope: ``CIBDRADVICE`` is the Bank Rakyat CIB platform's
# debit-advice opcode — no other bank in the corpus emits this string.
# Anchoring the regex on ``CIBDRADVICE\(JomPA`` makes the rung fully
# BR-specific. The 5 biller tokens are BR concat-shape artifacts (other
# banks emit JomPay billers as full strings like ``TENAGA NASIONAL
# BHD``) and were verified absent from BR/7, the Track 2 validation
# JSONs, and all ground-truth corpora as of s33. The token list is
# explicit (not generic ``DIGI|TNB|TM``) to avoid coincidental hits on
# employee names or short tokens — extend per-surfaced shape rather
# than broaden.
_BR_JOMPAY_UTILITY_RE = re.compile(
    r"CIBDRADVICE\(JomPA\b.*\b(?:"
    r"DIGITELECOMMUNI"
    r"|TENAGANASIONAL"
    r"|PENGURUSANAIRS"
    r"|INDAHWATERKONS"
    r"|TMTECHNOLOGYSE"
    r")\b",
    re.IGNORECASE,
)


# Marker stamped by the upstream parser / counterparty_ledger pipeline onto
# counterparty names whose root matches the statement's company. Treated as
# a deterministic signal — when present the row is unambiguously own-party
# and the dispatcher routes it to C01 / C02 directly. The fuller own-party
# detection (company-root extraction + normalisation, RP3 candidate scan)
# stays blocked on BUG-003; this rung covers only the parser-stamped subset.
OWN_PARTY_MARKER_RE = re.compile(r"\(\s*OWN[\s\-_]?PARTY\s*\)", re.IGNORECASE)


# ── Slice 1 of RP foundation port — company-root extraction + own-party
# matching. Used by the dispatcher's non-marker C01/C02 rung when the
# parser hasn't stamped (OWN-PARTY) but the counterparty bucket or
# transaction description literally matches the statement's company name.
# Mirrors Track 1's _company_root + _own_party_match (kredit_lab_classify.py
# L546-685) but defined here because Track 2 must not import from Track 1.
_COMPANY_SUFFIXES_RE = re.compile(
    r"\b(SDN\.?\s*BHD\.?|BERHAD|ENTERPRISE|HOLDINGS|TRADING|SERVICES|"
    r"CORPORATION|GROUP|PRIVATE|LIMITED|LTD|PLT|CORP|INC|BHD)\b",
    re.IGNORECASE,
)
_PAREN_DISAMBIGUATOR_RE = re.compile(r"\s*\([^)]*\)\s*")
# 5-char minimum on the root side mirrors the Mazaa audit (2026-04-28) —
# legitimate 5-char distinctive holder names exist (MAZAA); 4 chars and
# below trip generic substrings.
_COMPANY_ROOT_MIN_LEN = 5
# 6-char minimum on the counterparty-bucket side guards against generic
# bucket-name FPs (UNNAMED, etc.) when the cp itself is inside a longer
# root (the cp_upper in root direction).
_COMPANY_ROOT_CP_MIN_LEN = 6


def _company_root(name: str) -> str:
    """Strip parenthetical disambiguators + corporate suffixes + punctuation
    from ``name`` and return the upper-cased root token sequence used for
    own-party matching. Empty string for falsy or fully-stripped input.

    Mirrors Track 1's ``_company_root`` (kredit_lab_classify.py L561-572),
    with one Track-2-specific addition: leading purely-numeric tokens are
    stripped. SSM registration-prefix names like ``"010 MAZAA SDN BHD"``
    appear in the parser's company-name field but NOT in transaction
    descriptions (which only carry the public name ``"MAZAA SDN BHD"``).
    Without this strip the substring match in ``_own_party_match`` misses
    every legitimate self-transfer (Mazaa s21 Tier-4 audit, fix s22).
    Alphanumeric leading tokens like ``3M`` / ``1MDB`` are preserved.

    Parentheticals are stripped BEFORE suffixes so concat-form holder names
    like ``KOPERASIKAKITANGANFELCRA(M)BERHAD`` reduce to the same root
    (``KOPERASIKAKITANGANFELCRA``) the descriptive bucket name produces.
    """
    if not name:
        return ""
    upper = _PAREN_DISAMBIGUATOR_RE.sub(" ", name.upper())
    cleaned = _COMPANY_SUFFIXES_RE.sub("", upper)
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Strip leading purely-numeric tokens (SSM registration prefix).
    # Repeated leading numeric tokens are stripped greedily; alphanumeric
    # tokens like ``3M`` / ``1MDB`` are preserved because the digit-only
    # anchor (``^\d+\s+``) requires whitespace after the digit run.
    cleaned = re.sub(r"^(?:\d+\s+)+", "", cleaned)
    # If everything that survived is still purely numeric (pathological:
    # the original name was just digit tokens), drop it. The ≥5-char gate
    # in ``_own_party_match`` would filter it anyway; this is for clarity.
    if cleaned.isdigit():
        cleaned = ""
    return cleaned


def _own_party_match(
    cp_upper: str, desc_upper: str, company_roots: list[str]
) -> bool:
    """Bidirectional own-party detection.

    Returns True when any ``company_roots`` entry of length ≥ 5 appears
    inside the transaction description OR the counterparty bucket name,
    with the cp-side keeping a 6-char floor on the *cp_upper inside root*
    direction (prevents generic-bucket FPs like ``UNNAMED`` matching a
    short root).

    Mirrors Track 1's ``_own_party_match`` (kredit_lab_classify.py
    L670-685; audit anchor: Felcra/Waja/KMZ/Principal Gas/Mytutor all
    have roots ≥ 9 chars; Mazaa's 5-char ``MAZAA`` root sweeps 55
    legitimate DuitNow self-transfers per 2026-04-28 review).
    """
    for root in company_roots:
        if not root or len(root) < _COMPANY_ROOT_MIN_LEN:
            continue
        if root in desc_upper:
            return True
        if cp_upper and (
            root in cp_upper
            or (len(cp_upper) >= _COMPANY_ROOT_CP_MIN_LEN and cp_upper in root)
        ):
            return True
    return False


# ── Slice 2 of RP foundation port — RP3 candidate scanner.
# V3-A auto-RP Step 1 (Track 1 design carried verbatim): score each
# counterparty in the upstream ``counterparty_ledger`` against five
# deterministic behavioral signals; total score maps to LOW/MEDIUM/HIGH
# confidence. HIGH-confidence candidates flow into
# ``build_track2_result``'s ``related_parties`` arg without analyst
# intervention so the dispatcher's C03/C04 rung fires on them.
# MEDIUM/LOW would surface in an analyst form (out of scope for headless
# Track 2 — we only act on HIGH).
#
# Mirrors Track 1's constants + helpers from
# kredit_lab_classify.py L292-491. Threshold values come from the
# auto-RP audit (see ``_compute_rp_signals`` docstring for which signal
# fires which weight).

_RP_CONCENTRATION_DR_THRESHOLD = 0.05  # cp gross DR / total gross DR
_RP_RECURRENCE_MIN_MONTHS = 3          # distinct calendar months with DRs
_RP_BIDIRECTIONAL_MIN_SIDE_COUNT = 2   # min(cr_count, dr_count)
_RP_BIDIRECTIONAL_MIN_RATIO = 0.05     # min(total_cr,total_dr)/max(...) — materiality

# Stricter reciprocity floor for the AMBIGUOUS multi-party disambiguation
# escape only. A bucket stamped "(possibly multiple parties)" (single
# first-name token — MOHD / MUHAMMAD / WAN) auto-confirms despite the stamp
# only when its two-way flow is materially reciprocal — a genuine director
# loan-account back-and-forth, not coincidental token refunds across several
# distinct people who happen to share a first name. Corpus RP survey
# (2026-06-06) evidence: ASHRUL ratio 0.92 (real loan account, KEEP) vs MOHD
# 0.088 / WAN 0.066 (thin one-way + token refunds, DROP to analyst-review).
# The general 0.05 ``_RP_BIDIRECTIONAL_MIN_RATIO`` is deliberately left
# untouched so non-ambiguous bucket scoring is unchanged.
_RP_AMBIGUOUS_DISAMBIG_RATIO = 0.20
_RP_ROUND_AMOUNT_FLOOR = 1000.0
_RP_ROUND_AMOUNT_MULTIPLE = 100.0
_RP_ROUND_HITS_MIN = 2
# Sustained round-amount director-draw pattern: ≥5 round DRs across
# ≥4 calendar months upgrades the round-amount signal from weight 1 → 2.
# Captures revolving director advances while leaving the weak
# "≥2 round DRs" tier for one-off vendor refunds.
_RP_ROUND_SUSTAINED_HITS_MIN = 5
_RP_ROUND_SUSTAINED_MONTHS_MIN = 4

# Synthetic / pattern-fallback labels that must never be treated as
# personal names. Equals Track 1's ``_RP_EXCLUDE_NAMES ∪ BUCKET_TO_CATEGORY.keys()``
# (kredit_lab_classify.py L99-114 + L306-312): all the synthetic-bucket
# labels Track 1 filters out before scoring. Kept as a single frozenset
# in Track 2 because we have no ``BUCKET_TO_CATEGORY`` dispatcher map
# (Track 2 uses keyword regex rungs). ``CHEQUE DEPOSIT`` and ``CHEQUE
# ISSUE`` only live in Track 1's BUCKET_TO_CATEGORY — including them here
# preserves identical exclusion behavior.
#
# Rail-label prefixes added in s23 after the Mazaa Tier-4 smoke surfaced
# ``TRSF DR`` (29 unrelated DuitNow DRs aggregated under one synthetic
# bucket label) auto-confirming HIGH. These tokens are bank-rail markers,
# not personal-name prefixes: ``TRSF``/``RMT``/``IBG``/``CHEQ`` are
# transfer-rail abbreviations, ``DEP-ECP``/``DR-ECP`` are Bank Islam
# ECP-channel synthetic buckets. None can legitimately lead a Malaysian
# personal name.
_RP_EXCLUDE_PREFIXES = (
    "UNIDENTIFIED",
    "UNNAMED",
    "TRSF",
    "RMT",
    "IBG",
    "CHEQ",
    "DEP-ECP",
    "DR-ECP",
)
_RP_EXCLUDE_NAMES = frozenset({
    "UNCATEGORIZED", "REVERSAL", "RETURNED CHEQUE", "JANM", "APAYLATER",
    "BULK SALARY", "BANK FEES", "FD/INTEREST", "LOAN REPAYMENT",
    "LOAN DISBURSEMENT", "CASH DEPOSIT", "CASH WITHDRAWAL", "INWARD RETURN",
    "KWSP", "SOCSO", "LHDN", "HRDF",
    "CHEQUE DEPOSIT", "CHEQUE ISSUE",
})

# Memo / rail / facility synthetic-bucket labels that must never auto-confirm
# as a related party. Where ``_RP_EXCLUDE_PREFIXES`` (s23) catches rail tokens
# that LEAD a bucket name, these tokens identify junk buckets where the parser
# could not extract a counterparty and instead aggregated many unrelated
# transactions under a memo / payment-rail / own-facility label appearing
# ANYWHERE in the name. Verified non-party buckets surfaced by the corpus RP
# survey (2026-06-06):
#   * ``BILL PAYMENT DEBIT FUND TRANSFER`` (BankMuamalat) — generic memo
#   * ``PAYPRX DR OPERATING EXPENCES`` / ``DR DUITNOW S/CHRG`` (Maybank) —
#     66 distinct DuitNow payees collapsed under an accounting-category memo
#   * ``CATIS PAYMENT`` / ``REFLEX FTT`` / ``LOANS/FIN CLEAR AUTODEBIT`` (RHB)
#   * ``UOB TRADE BILL FACILITY`` / ``UOB LOAN FACILITY`` / ``LMS SWEEP`` (UOB)
#     — own trade-finance / sweep facility drawdowns, not a counterparty
#   * ``TRADE COLLECTIONS (CA IMPORT)`` (Alliance) / ``REMITT`` (CIMB)
# Every token is distinctive enough that it cannot be part of a Malaysian
# personal name; company-form labels are already dropped by
# ``_looks_like_company``. The scanner gates this on
# ``not has_natural_person_marker`` so a memo-CONTAMINATED real name (e.g.
# ``PAYPRX DR MOHD HAFIZ BIN ARRIF OPERATING EXPENCES``) is left for the
# canonicalisation layer rather than silently dropped here.
_RP_EXCLUDE_LABEL_RE = re.compile(
    r"""
        \bBILL\s+PAYMENT\b
      | \bFUND\s+TRANSFER\b
      | \bPAYPRX\b
      | \bOPERATING\s+EXPEN          # EXPENCES / EXPENSES
      | \bREFLEX\b
      | \bFTT\b
      | S/\s*CHRG                     # S/CHRG service-charge memo
      | \bAUTODEBIT\b
      | \bLOANS?/FIN\b
      | \bSWEEP\b
      | \bFACILITY\b
      | \bTRADE\s+COLLECTIONS\b
      | \bCATIS\b
      | \bREMITT                      # REMITT / REMITTANCE
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Score → confidence tier. Strong signals (weight 2) are concentration,
# personal-keyword sweep, and bidirectional flow. Two strong OR strong+weak
# lands HIGH (3+); a single strong is MEDIUM (2); a single weak is LOW (1).
_RP_HIGH_SCORE = 3
_RP_MEDIUM_SCORE = 2

# Director-like personal-keyword vocabulary used by signal #1
# (personal_keyword_sweep). Substring-matched against the upper-cased
# description; ≥2 hits across a counterparty's DR rows + debit_count ≥ 3
# scores the weight-2 strong signal. Mirrors Track 1's
# ``PERSONAL_KEYWORDS_RP4`` (kredit_lab_classify.py L148-151) verbatim.
PERSONAL_KEYWORDS_RP4 = [
    "ADV FI", "FI ", "CLAIM", "DIVIDEND", "LOAN", "PETTY",
    "HOUSING", "CREDIT CARD", "BONUS", "MEDICAL", "REIMBURSE",
]

# Director-benefit markers — the company settling an INDIVIDUAL's personal
# financial obligation. Unlike the soft personal-keyword sweep (staff
# expense claims), these describe the company paying a person's personal
# liability: vehicle hire-purchase / car loan, housing loan, personal loan,
# or credit-card bill. A business does not pay a vendor's or a rank-and-file
# employee's car loan — it pays the owner's / director's. Gated on a
# natural-person counterparty (``has_natural_person_marker``) so the
# company's OWN asset financing to a bank / finance company never matches.
# Treated as a HARD related-party anchor and overrides salary-exclusion.
_DIRECTOR_BENEFIT_RE = re.compile(
    r"\b(?:HIRE\s*PURCHASE|CAR\s*LOAN|HOUSING\s*LOAN|PERSONAL\s*LOAN|CREDIT\s*CARD)\b",
    re.IGNORECASE,
)

# Finance / money-lender tokens. A counterparty name carrying one of these is
# a financier (the company settling its OWN car loan / hire-purchase to AEON
# CREDIT, RCE CREDIT, a leasing house, etc.) — never a related person. Used to
# keep ``_looks_like_personal_name`` from re-admitting the exact case the
# natural-person gate was built to exclude.
_FINANCE_INSTITUTION_RE = re.compile(
    r"\b(?:CREDIT|FINANCE|FINANCING|LEASING|CAPITAL|INSURANCE|TAKAFUL)\b",
    re.IGNORECASE,
)

# C11 own-facility-repayment financier match (DR side). Narrower than
# _FINANCE_INSTITUTION_RE on purpose — that broader RE is for keeping
# financiers out of the RP person scan, where lumping insurance/takaful with
# lenders is harmless. For C11 the distinction matters: an asset / vehicle
# financier (``SCANIA CREDIT``, ``CARSOME CAPITAL``, ``ORIX LEASING``, ``BMW
# CREDIT``, ``TOYOTA CAPITAL``, ``AEON CREDIT SERVICE``) is debt service, but
# an INSURANCE / TAKAFUL premium or a CREDIT GUARANTEE fee is an operating
# expense, NOT a loan repayment — mislabelling it C11 overstates debt service.
# Shape: a brand token immediately followed by a lender word (CREDIT / CAPITAL
# / LEASING / FINANCE / FINANCING). Requiring the brand prefix keeps a bare
# leading "CREDIT CARD PAYMENT" out, and the exclusion guard below drops
# insurance / guarantee / and raw-extraction noise (ATM / GIRO / SERVICE
# CHARGE rows where "CREDIT" appears only incidentally in the memo).
_FINANCIER_C11_RE = re.compile(
    r"\b[A-Z][A-Z0-9&./-]*\s+(?:CREDIT|CAPITAL|LEASING|FINANCE|FINANCING)\b",
    re.IGNORECASE,
)
_FINANCIER_C11_EXCLUDE_RE = re.compile(
    r"\b(?:INSURANCE|TAKAFUL|GUARANTEE|ATM|GIRO|CASH|WDRWL|WITHDRAW(?:AL)?|"
    r"FEES?|INTERBANK|VELOCITY|RINGKASAN|SAL(?:ARY)?|PAYROLL|DEPOSIT|"
    r"REVERSAL|REFUND)\b|SERVICE\s+CHARGE",
    re.IGNORECASE,
)

# Malay government-body / association / institution markers. A name carrying one
# is an organisation, not an individual — used to keep public bodies (JABATAN
# KASTAM DIRAJA), associations (PERSATUAN ...), councils, boards, cooperatives,
# ministries, foundations, schools/universities out of the person-only advisory
# RP-candidate panel, where ``_looks_like_personal_name`` alone would admit any
# short all-caps phrase.
_PUBLIC_BODY_RE = re.compile(
    r"\b(?:JABATAN|PERSATUAN|PERTUBUHAN|KOPERASI|MAJLIS|LEMBAGA|SURUHANJAYA|"
    r"KEMENTERIAN|KERAJAAN|DIRAJA|PEJABAT|YAYASAN|KELAB|INSTITUT|UNIVERSITI|"
    r"KOLEJ|SEKOLAH|HOSPITAL|KLINIK|KASTAM|PERBADANAN|USAHAWAN|"
    r"EPF|KWSP|SOCSO|PERKESO|LHDN|ZAKAT|BAITULMAL|PTPTN)\b",
    re.IGNORECASE,
)

# Business-noun tokens NOT already in _looks_like_company (which covers SDN/BHD/
# TRADING/SERVICES/TECHNOLOGY/…). A short all-caps phrase carrying one is a firm,
# not an individual — keeps MESO NUTRITION / JURUKUR RESOURCES / RADIUS FUEL
# CARDS / SCB BULK LOGISTICS out of the person-only advisory panel.
_BUSINESS_NOUN_RE = re.compile(
    r"\b(?:RESOURC(?:ES?)?|NUTRITION|LOGISTICS|TECH|SOLUTIONS?|SUPPLY|SUPPLIES|"
    r"MARKETING|CONSTRUCTION|ENGINEERING|VENTURES?|CARDS?|FUEL|NETWORK|"
    r"MOTORS?|HARDWARE|RENTALS?|CATERING|TRANSPORT(?:ATION)?|BUILDERS?|"
    r"DIGITAL|MEDIA|STUDIO|GLOBAL|KILANG|PERNIAGAAN|NIAGA|JURUKUR|BINA|"
    r"ACADEMY|PRINTER|PRINTING|JEWELLER(?:Y)?)\b",
    re.IGNORECASE,
)

# Advisory RP panel tuning — keep it a SHORT, material, person-only review list.
_ADVISORY_RP_MIN_DR = 3000.0   # drop micro-payments (tutors paid RM168 etc.)
_ADVISORY_RP_MAX_ROWS = 25     # cap; renderer notes "top N of M"


def advisory_rp_candidates(
    rp_candidates: list[dict[str, Any]],
    effective_related_parties: list[str] | None,
    company_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter the raw MEDIUM/LOW RP3 candidates down to a trustworthy
    person-only advisory list: individuals (marker or clean name shape), not
    already confirmed, not the statement holder's own company, not a public
    body / firm, and materially active. Sorted by debit value descending.
    Caller caps the display."""
    eff = {str(r).upper() for r in (effective_related_parties or [])}
    own_roots = [
        _company_root(c) for c in (company_names or []) if c and len(_company_root(c)) >= 5
    ]
    out: list[dict[str, Any]] = []
    for c in rp_candidates:
        nm = c.get("name") or ""
        up = nm.upper()
        if c.get("confidence") not in ("MEDIUM", "LOW"):
            continue
        if up in eff:
            continue
        if not (has_natural_person_marker(nm) or _looks_like_personal_name(nm)):
            continue
        if _PUBLIC_BODY_RE.search(nm) or _BUSINESS_NOUN_RE.search(nm):
            continue
        if any(root in up for root in own_roots):  # own company leaked in
            continue
        if float(c.get("total_dr", 0) or 0) < _ADVISORY_RP_MIN_DR:
            continue
        out.append(c)
    out.sort(key=lambda c: -float(c.get("total_dr", 0) or 0))
    return out

# A "clean personal-name shape": 2–4 whitespace tokens, letters only (plus the
# A/L · A/P patronymic particle), no digits. This rescues natural persons whose
# names carry NO BIN/BINTI/A/L marker — Chinese names (KUAN WEI YEE) and
# title-prefixed Malay names (DAYANG SITI RAUDZAH) — which the strict
# ``has_natural_person_marker`` whitelist is structurally blind to.
_PERSONAL_NAME_SHAPE_RE = re.compile(r"^[A-Z]+(?:\s+(?:A/L|A/P|[A-Z]+)){1,3}$")


def _looks_like_personal_name(name: Any) -> bool:
    """True iff ``name`` is shaped like an individual's name and is neither a
    company nor a financier.

    Companion to :func:`has_natural_person_marker` for the director-benefit
    gate: that whitelist only fires on BIN / BINTI / A/L / A/P, so it can never
    rescue a Chinese or title-prefixed Malay name. This accepts a 2–4 word
    all-alphabetic name while excluding corporate markers, corporate suffixes,
    and finance/money-lender tokens (so ``AEON CREDIT`` — the company's own car
    financing — stays out). Digit-bearing memo-ref junk is rejected outright.
    """
    if not isinstance(name, str):
        return False
    n = name.upper().strip()
    if not n or any(ch.isdigit() for ch in n):
        return False
    if not _PERSONAL_NAME_SHAPE_RE.match(n):
        return False
    if _looks_like_company(n) or has_corporate_suffix(n):
        return False
    if _FINANCE_INSTITUTION_RE.search(n):
        return False
    return True


def _has_director_benefit(cp: dict[str, Any]) -> bool:
    """True iff the company pays a NATURAL PERSON's personal liability
    (hire-purchase / car loan / housing loan / personal loan / credit card).

    Strong related-party evidence on its own — surfaced by DCSE's owner
    ``MUHAMMAD ARIF``, whose monthly RM2,627 ``HIRE PURCHASE DCSE`` /
    ``16TH CAR LOAN`` payments are the company settling his personal vehicle
    financing. The natural-person gate keeps the company's own asset
    financing (e.g. ``TR TO AEON CREDIT HIRE PURCHASE``) from matching.

    The gate accepts either an explicit BIN/BINTI/A/L marker OR a clean
    personal-name shape (:func:`_looks_like_personal_name`) — the latter
    rescues marker-less names such as ``DAYANG SITI RAUDZAH`` (BINTI omitted)
    and ``KUAN WEI YEE`` (Chinese names never carry the marker), whose company
    settles their HOUSING LOAN / CREDIT CARD.
    """
    name = cp.get("counterparty_name") or ""
    if not (has_natural_person_marker(name) or _looks_like_personal_name(name)):
        return False
    for tx in cp.get("transactions", []) or []:
        if tx.get("type") != "DEBIT":
            continue
        if _DIRECTOR_BENEFIT_RE.search(str(tx.get("description") or "")):
            return True
    return False


def _looks_like_company(name: str) -> bool:
    """Cheap company-vs-individual heuristic. Returns True iff ``name``
    contains any of the 10 explicit corporate markers, in which case the
    RP scanner skips it (companies are scored via concentration / volume,
    not via the director-like personal-keyword sweep).

    Mirrors Track 1's ``_looks_like_company`` (L487-491) literally —
    the explicit list is broader than the C26/C27 corporate-suffix list
    on purpose (e.g. "BANK" is in here but not in ``_CORPORATE_ENTITY_MARKERS``).
    """
    if not name:
        return False
    upper = name.upper()
    return any(m in upper for m in (
        "SDN", "BHD", "ENTERPRISE", "BANK", "TRADING", "HOLDINGS",
        "SERVICES", "CORPORATION", "GROUP", "PRIVATE",
        # International / foreign-supplier markers (2026-06-07) — keep a
        # one-way overseas trade vendor (XIAMEN ... IMPORT AND EXPO,
        # SHENZHEN ... TECHOLOGY) out of the director-RP scan. These are
        # substring-matched, so the tokens chosen avoid personal-name
        # collisions (e.g. bare "INC"/"LTD" excluded — "INC" hits PRINCIPAL).
        "INTERNATIONAL", "IMPORT", "EXPORT", "EXPO", "TECHNOLOGY",
        "TECHOLOGY", "LIMITED", "PTE LTD", "CO LTD", "GMBH", "INDUSTRIES",
    ))


def _is_salary_recipient(cp: dict[str, Any]) -> bool:
    """True iff a counterparty's debit rows are dominated by C05 salary /
    payroll payments — i.e. the counterparty is an EMPLOYEE, not a related
    party.

    Payroll recipients otherwise auto-confirm as Affiliates: a named staff
    member is paid SALARY every month (``monthly_recurrence``) alongside
    REIMBURSE / CLAIM / MEDICAL / PETTY expense claims
    (``personal_keyword_sweep``), and the two soft signals reach the HIGH
    threshold. Excluding salary-dominated counterparties here removes that
    whole class of false positives at the source.

    Reads ledger-shape rows (``{amount, type, description}``) directly — the
    canonical-schema ``is_salary_payment`` predicate keys on the ``debit`` /
    ``credit`` numeric fields, which the counterparty_ledger transactions do
    not carry. Threshold is conservative: a strict majority of debit *value*
    must be salary AND at least two salary rows, so a real related party who
    happens to carry one stray ``SALARY`` memo is not swept out.
    """
    txs = cp.get("transactions", []) or []
    dr_amt = 0.0
    sal_amt = 0.0
    sal_rows = 0
    for tx in txs:
        if tx.get("type") != "DEBIT":
            continue
        amt = float(tx.get("amount") or 0.0)
        dr_amt += amt
        desc = str(tx.get("description") or "")
        if (
            (SALARY_KEYWORD_RE.search(desc) or _SALARY_KEYWORD_CONCAT_RE.search(desc))
            and not COMMISSION_BLOCK_RE.search(desc)
        ):
            sal_rows += 1
            sal_amt += amt
    return sal_rows >= 2 and dr_amt > 0 and sal_amt >= 0.5 * dr_amt


def _compute_rp_signals(
    cp: dict[str, Any], gross_dr: float
) -> dict[str, Any] | None:
    """Score one counterparty against the five RP signals. Return None if
    no signal fires.

    Signals + weights (matches Track 1 verbatim):
      1. personal_keyword_sweep (weight 2): debit_count ≥ 3 AND ≥ 2 rows
         contain a token from ``PERSONAL_KEYWORDS_RP4``.
      2. concentration_dr (weight 2): cp gross DR ≥ 5% of total gross DR.
      3. monthly_recurrence (weight 1): DR rows span ≥ 3 distinct months.
      4. bidirectional_flow (weight 2): min(cr_count, dr_count) ≥ 2 —
         the textbook director loan-account pattern.
      5. round_amount (weight 1) OR round_amount_sustained (weight 2):
         ≥ 2 round-DR hits (multiple of 100, ≥ 1000) escalates to the
         sustained tier when ≥ 5 hits across ≥ 4 months.

    "(possibly multiple parties)" parser stamp forces LOW unless the cp
    also has bidirectional flow — single-direction ambiguous buckets
    can't be auto-confirmed without analyst disambiguation, but the
    back-and-forth itself disambiguates the bidirectional case (RHB Waja
    ASHRUL precedent).

    Returns a dict with `signals`, `score`, `confidence` (HIGH/MEDIUM/LOW),
    `ambiguous_multi_party`, `evidence`, `total_dr`, `total_cr`,
    `debit_count`, `credit_count` — or None if no signal fires at all.
    Same shape Track 1 emits so downstream consumers see identical data.
    """
    debit_count = int(cp.get("debit_count", 0) or 0)
    credit_count = int(cp.get("credit_count", 0) or 0)
    total_dr = float(cp.get("total_debits", 0.0) or 0.0)
    total_cr = float(cp.get("total_credits", 0.0) or 0.0)
    txs = cp.get("transactions", []) or []
    cp_name = cp.get("counterparty_name") or ""

    # Parser-stamped multi-party suffix forces LOW (see docstring).
    is_ambiguous_multi_party = (
        "(possibly multiple parties)" in cp_name.lower()
    )

    signals: list[tuple[str, int, str]] = []

    # 1. Personal-keyword sweep.
    if debit_count >= 3:
        personal_hits = sum(
            1 for tx in txs
            if any(
                kw in (tx.get("description") or "").upper()
                for kw in PERSONAL_KEYWORDS_RP4
            )
        )
        if personal_hits >= 2:
            signals.append(
                ("personal_keyword_sweep", 2,
                 f"{personal_hits} personal-kw rows")
            )

    # 2. Concentration: cp gross DR ≥ 5% of total gross DR.
    if gross_dr > 0 and total_dr / gross_dr >= _RP_CONCENTRATION_DR_THRESHOLD:
        signals.append((
            "concentration_dr", 2,
            f"DR {100 * total_dr / gross_dr:.1f}% of gross",
        ))

    # 3. Monthly recurrence: DR rows span ≥ 3 distinct calendar months.
    dr_months = {
        (tx.get("date") or "")[:7]
        for tx in txs if tx.get("type") == "DEBIT"
    }
    dr_months.discard("")
    if len(dr_months) >= _RP_RECURRENCE_MIN_MONTHS:
        signals.append((
            "monthly_recurrence", 1, f"DR over {len(dr_months)} months",
        ))

    # 4. Bidirectional flow — director loan-account pattern. Requires both
    # a minimum count on each side AND materiality of the smaller side: a
    # genuine current account has substantial two-way flow, whereas an
    # employee / petty-cash custodian shows large one-way disbursements with
    # a couple of trivial refunds (e.g. RM399 returned against RM22k paid).
    _bidir_max = max(total_cr, total_dr)
    _bidir_material = (
        _bidir_max > 0
        and min(total_cr, total_dr) / _bidir_max >= _RP_BIDIRECTIONAL_MIN_RATIO
    )
    if (
        min(credit_count, debit_count) >= _RP_BIDIRECTIONAL_MIN_SIDE_COUNT
        and _bidir_material
    ):
        signals.append((
            "bidirectional_flow", 2,
            f"{credit_count}CR / {debit_count}DR",
        ))

    # 5. Round-number advances — two tiers.
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
        signals.append((
            "round_amount_advance", 1, f"{round_hits} round DRs",
        ))

    # Director-benefit — company settling a natural person's personal
    # liability (hire-purchase / car / housing / personal loan / credit
    # card). A standalone hard anchor: strong related-party evidence even
    # for an otherwise single-direction bucket.
    if _has_director_benefit(cp):
        signals.append((
            "director_benefit", 2,
            "company pays personal hire-purchase / loan",
        ))

    if not signals:
        return None

    score = sum(weight for _, weight, _ in signals)
    has_bidirectional = any(sig == "bidirectional_flow" for sig, _, _ in signals)
    # An ambiguous first-name bucket disambiguates ONLY on materially
    # reciprocal two-way flow — not on the thin token refunds that merely
    # clear the general 0.05 bidirectional gate (those are coincidental
    # multi-person collisions, not one director's loan account). See
    # _RP_AMBIGUOUS_DISAMBIG_RATIO.
    _recip_max = max(total_cr, total_dr)
    _recip_ratio = (min(total_cr, total_dr) / _recip_max) if _recip_max > 0 else 0.0
    strong_reciprocal = has_bidirectional and _recip_ratio >= _RP_AMBIGUOUS_DISAMBIG_RATIO
    force_low_ambiguous = is_ambiguous_multi_party and not strong_reciprocal

    # HIGH auto-confirm requires a "hard anchor" — bidirectional flow (the
    # director loan-account back-and-forth) OR concentration (a materially
    # large share of total outflow). The soft signals on their own
    # (monthly_recurrence + round_amount_sustained, or personal_keyword_
    # sweep) fire on ordinary single-direction operating payments: fixed
    # monthly rent, recurring sub-contractor retainers, and staff
    # expense-reimbursement buckets. Without a hard anchor those cap at
    # MEDIUM (analyst-review) instead of auto-confirming as Affiliates.
    #
    # concentration_dr stops counting as a hard anchor for a one-way CONFIRMED
    # BUSINESS ENTITY. A one-way, high-volume company / financier / firm is a
    # trade vendor or lender — not an affiliate — and high concentration is the
    # EXPECTED shape of the company's biggest supplier (e.g. PETRON fuel) or
    # asset financier (e.g. SCANIA CREDIT hire-purchase). The bank often
    # truncates the corporate suffix (Maybank's ``*`` cut: ``SCANIA CREDIT
    # (MALA``, ``PETRON FUEL INTL -F``) so the plain company gate can't catch
    # them by SDN BHD; the finance / business-noun / public-body token tests
    # do. Such buckets cap at MEDIUM (advisory panel) instead of auto-
    # confirming. A genuine one-way corporate affiliate is confirmed by the
    # analyst, not by concentration.
    #
    # Demotion is gated on POSITIVE business-entity evidence — NOT on "fails
    # the person test" — so a person whose name shape the marker/clean-name
    # heuristics miss (concatenated Bank Rakyat DATAPOS forms like
    # ``SITINURHAFIZAHBINTIOTHMAN``; ``@``-joined Chinese names like ``LING SOW
    # REUM @ LIN``) keeps concentration as an anchor and is not lost. A
    # counterparty with bidirectional flow is always anchored regardless.
    _concentration_fired = any(sig == "concentration_dr" for sig, _, _ in signals)
    _is_business_entity = bool(
        _looks_like_company(cp_name)
        or has_corporate_suffix(cp_name)
        or _FINANCE_INSTITUTION_RE.search(cp_name)
        or _BUSINESS_NOUN_RE.search(cp_name)
        or _PUBLIC_BODY_RE.search(cp_name)
    )
    _concentration_is_anchor = _concentration_fired and (
        has_bidirectional or not _is_business_entity
    )
    has_hard_anchor = (
        has_bidirectional
        or any(sig == "director_benefit" for sig, _, _ in signals)
        or _concentration_is_anchor
    )

    if force_low_ambiguous:
        confidence = "LOW"
    elif score >= _RP_HIGH_SCORE and has_hard_anchor:
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


def scan_related_party_candidates(
    counterparty_ledger: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Scan ``counterparty_ledger`` for director-like RP candidates.

    Walks every counterparty in the ledger, applies the exclusion filters
    (synthetic-bucket labels, UNIDENTIFIED/UNNAMED prefix, company markers,
    bucket-direct categories), and scores each survivor via
    ``_compute_rp_signals``. Returns a list sorted by confidence
    (HIGH → MEDIUM → LOW) then by ``total_dr`` descending.

    Each result row carries: ``name``, ``method`` (primary signal name,
    backwards-compat), plus all keys ``_compute_rp_signals`` returns
    (``signals``, ``score``, ``confidence``, ``ambiguous_multi_party``,
    ``evidence``, ``total_dr``, ``total_cr``, ``debit_count``,
    ``credit_count``).

    Mirrors Track 1's ``scan_related_party_candidates``
    (kredit_lab_classify.py L441-477). Differs only in input shape — this
    accepts the ledger dict directly while Track 1 takes the parser-
    output ``data`` dict and reads ``data["counterparty_ledger"]``. Same
    output.
    """
    if not counterparty_ledger:
        return []
    cps = counterparty_ledger.get("counterparties", []) or []
    gross_dr = sum(
        float(cp.get("total_debits", 0.0) or 0.0) for cp in cps
    )

    candidates: list[dict[str, Any]] = []
    for cp in cps:
        name = cp.get("counterparty_name") or ""
        if not name:
            continue
        upper = name.upper()
        if _looks_like_company(name):
            continue
        if upper in _RP_EXCLUDE_NAMES:
            continue
        if any(upper.startswith(p) for p in _RP_EXCLUDE_PREFIXES):
            continue
        # Memo / rail / facility synthetic-bucket label appearing anywhere in
        # the name (corpus RP survey 2026-06-06). Gated on the absence of a
        # natural-person marker so a memo-contaminated real name survives for
        # the canonicalisation layer; a pure junk label (no BIN/BINTI/A-L)
        # is dropped before scoring.
        if _RP_EXCLUDE_LABEL_RE.search(upper) and not has_natural_person_marker(name):
            continue
        # Synthetic own-party bucket built by the counterparty_ledger pipeline
        # (e.g. ``PRINCIPAL GAS (OWN-PARTY)``). The dispatcher already routes
        # its rows to C01/C02 via OWN_PARTY_MARKER_RE; the RP3 scanner must
        # not also auto-confirm the bucket label as an Affiliate. Surfaced
        # by the Principal Gas Tier-4 smoke once s22 Fix #2 populated
        # report_info.related_parties (s23 Fix).
        if OWN_PARTY_MARKER_RE.search(upper):
            continue
        # Salary-dominated counterparties are employees, not related
        # parties — payroll + expense-claim rows otherwise trip the
        # soft signals (personal-keyword sweep + recurrence) and
        # auto-confirm every named staff member as an Affiliate. An
        # owner-director who draws salary AND has the company settle a
        # personal liability (director-benefit) is exempt from this skip.
        if _is_salary_recipient(cp) and not _has_director_benefit(cp):
            continue

        scored = _compute_rp_signals(cp, gross_dr)
        if scored is None:
            continue

        candidates.append({
            "name": name,
            "method": scored["signals"][0],
            **scored,
        })

    confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    candidates.sort(
        key=lambda c: (
            confidence_order.get(c["confidence"], 9),
            -c["total_dr"],
        )
    )
    return candidates


def auto_confirmed_related_parties(
    candidates: list[dict[str, Any]],
) -> list[str]:
    """Names from ``scan_related_party_candidates`` whose deterministic
    score reached HIGH confidence. These are merged into the dispatcher's
    ``related_parties`` arg without analyst intervention — V3-A auto-RP
    Step 1 behavior. MEDIUM/LOW candidates require analyst confirmation
    (out of scope for headless Track 2).

    Mirrors Track 1's ``auto_confirmed_related_parties`` (L480-484).
    """
    return [c["name"] for c in candidates if c.get("confidence") == "HIGH"]


def is_salary_payment(row: dict[str, Any]) -> bool:
    """Per-row predicate: is this a C05 salary / payroll payment?

    True iff ALL of:
      * Row side is ``DR`` (employer paying OUT to employees).
      * Description matches ``SALARY_KEYWORD_RE`` (any of the 36 v3.5
        salary_keywords, with ``\\bGAJI\\b`` boundary to avoid the
        MENGAJI / NGAJI tuition-business substring collision) OR
        ``_SALARY_KEYWORD_CONCAT_RE`` (Bank Rakyat DATAPOS concat form —
        ``GAJI<MONTH>[YYYY]`` / ``STAFFID<NNN>`` with no whitespace).
      * Description does NOT match ``COMMISSION_BLOCK_RE`` (the v3.3.1
        commission_policy exclusion — commission agents are independent
        contractors, not employees).
      * Description does NOT match ``OWN_ACCOUNT_BLOCK_RE`` (bank CIB
        own-account / inter-account markers — see the regex comment).
      * Description does NOT match ``_BANK_FEE_BLOCK_RE`` (companion fee
        rows like CIBSMSFEE / DUITNOWFEE / CIBDRCHARGES — Bank Rakyat
        parser concatenates the underlying transaction's metadata into
        the fee row, so the salary tail surfaces on tiny RM 0.10 fees).

    Predicate form so the dispatcher can apply C05 detection per-row
    when ordering priorities against C03 / C04 / C26 / C27.

    Args:
        row: canonical-schema transaction row. Reads ``description``,
            ``credit``, ``debit``.

    Returns:
        True iff the row qualifies as C05 under the FULL_CODE branches.
        The AI_ASSIST CIMB individual branch is NOT applied here.
    """
    description = str(row.get("description") or "")
    if not description:
        return False
    if _row_side(row) != "DR":
        return False
    if not (
        SALARY_KEYWORD_RE.search(description)
        or _SALARY_KEYWORD_CONCAT_RE.search(description)
    ):
        return False
    if COMMISSION_BLOCK_RE.search(description):
        return False
    if OWN_ACCOUNT_BLOCK_RE.search(description):
        return False
    if _BANK_FEE_BLOCK_RE.search(description):
        return False
    return True


def compute_salary_payments(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect C05 salary / payroll payments via the LOCKED v3.5 keyword list.

    Iterates ``transactions`` and applies ``is_salary_payment`` row-by-row.
    Output feeds ``compute_statutory_monthly_amounts``'s ``salary_entries``
    parameter, which in turn drives ``compute_statutory_compliance``'s
    salary-coverage math for Flags 6/7 (EPF / SOCSO coverage).

    See the C05 section comment for the design contract — FULL_CODE
    branches only (salary keyword + commission_policy_v3_3_1 block),
    AI_ASSIST CIMB-individual branch is dispatcher-level.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict:
            salary_payments_count, salary_payments_amount,
            salary_payments_entries.
    """
    entries: list[dict[str, Any]] = []
    total = 0.0
    for row in transactions:
        if not is_salary_payment(row):
            continue
        amount = float(row.get("debit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount
    return {
        "salary_payments_count": len(entries),
        "salary_payments_amount": round(total, 2),
        "salary_payments_entries": entries,
    }


# ---------------------------------------------------------------------------
# Statutory monthly aggregator — bridges the C05/C06/C07/C08/C09 per-row
# detectors to the monthly_amounts dict that compute_statutory_compliance
# (session 3) already consumes. Pure regrouping by YYYY-MM; no detection.
# ---------------------------------------------------------------------------


def compute_statutory_monthly_amounts(
    *,
    salary_entries: list[dict[str, Any]] | None = None,
    epf_entries: list[dict[str, Any]] | None = None,
    socso_entries: list[dict[str, Any]] | None = None,
    lhdn_tax_entries: list[dict[str, Any]] | None = None,
    hrdf_entries: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, float]]:
    """Group statutory detector entries by month into a ``monthly_amounts``
    dict suitable to pass to ``compute_statutory_compliance``.

    Wiring (no detection logic — pure regrouping)::

        salary  = compute_salary_payments(transactions)
        epf     = compute_epf_payments(transactions)
        socso   = compute_socso_payments(transactions)
        lhdn    = compute_lhdn_tax_payments(transactions)
        hrdf    = compute_hrdf_payments(transactions)
        monthly = compute_statutory_monthly_amounts(
            salary_entries=salary["salary_payments_entries"],
            epf_entries=epf["epf_payments_entries"],
            socso_entries=socso["socso_payments_entries"],
            lhdn_tax_entries=lhdn["lhdn_tax_payments_entries"],
            hrdf_entries=hrdf["hrdf_payments_entries"],
        )
        statutory = compute_statutory_compliance(monthly)

    The C05 salary detector covers the FULL_CODE branches per v3.5:
    keyword match + ``commission_policy_v3_3_1`` block. The CIMB
    AI_ASSIST individual branch (TR TO SAVINGS + name + salary keyword)
    is dispatcher-level — when the row has no salary keyword in the
    bank-emitted description, only AI scoring against the extracted
    counterparty name can decide, and that hook lives at the dispatcher
    when AI scoring lands. With FULL_CODE C05 wired, Flags 6 (EPF) and
    7 (SOCSO) now fire on real coverage gaps; before s11 the salary
    input was always None so the salary-empty branch in s3 forced
    100% coverage and Flags 6/7 never lit.

    Schema-field mapping (output keys are what ``compute_statutory_compliance``
    reads):

        salary_entries     -> ``salary_paid``
        epf_entries        -> ``statutory_epf``
        socso_entries      -> ``statutory_socso``
        lhdn_tax_entries   -> ``statutory_tax``  (LHDN bucket: PCB+CP204+SST+stamp+RPGT)
        hrdf_entries       -> ``statutory_hrdf``

    Entry contract: each entry must be a dict carrying ``date``
    (ISO ``YYYY-MM-DD``) and ``amount`` (positive float). This is exactly
    what the C06-C09 detectors produce via ``_to_entry`` (see session 10),
    so detector outputs can be passed in directly with no transformation.
    Entries with missing / non-string / too-short dates, non-numeric or
    non-positive amounts are silently skipped — keeps the function pure
    over already-validated detector outputs and tolerant of unusual rows
    if a caller passes raw transactions by mistake.

    A month appears in the returned dict only if at least one of the five
    buckets had activity in it. Buckets not represented for a given month
    are absent from that month's sub-dict (and read as 0 by the s3
    reducer's ``_amt`` helper) — no zero-padding.

    Args:
        salary_entries: list of ``_to_entry``-shaped dicts from a future
            ``compute_salary_payments`` (C05) detector. Optional.
        epf_entries: list from ``compute_epf_payments(tx)["epf_payments_entries"]``.
        socso_entries: list from ``compute_socso_payments(tx)["socso_payments_entries"]``.
        lhdn_tax_entries: list from
            ``compute_lhdn_tax_payments(tx)["lhdn_tax_payments_entries"]``.
        hrdf_entries: list from ``compute_hrdf_payments(tx)["hrdf_payments_entries"]``.

    Returns:
        dict mapping ``YYYY-MM`` -> dict of monthly totals. Each sub-dict
        contains only the bucket keys that had activity in that month.
        Amounts are rounded to 2dp.
    """
    months: dict[str, dict[str, float]] = {}

    def _add(entries: list[dict[str, Any]] | None, field: str) -> None:
        for entry in entries or ():
            date = entry.get("date") if isinstance(entry, dict) else None
            if not isinstance(date, str) or len(date) < 7:
                continue
            try:
                amount = float(entry.get("amount") or 0)
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            bucket = months.setdefault(date[:7], {})
            bucket[field] = round(bucket.get(field, 0.0) + amount, 2)

    _add(salary_entries, "salary_paid")
    _add(epf_entries, "statutory_epf")
    _add(socso_entries, "statutory_socso")
    _add(lhdn_tax_entries, "statutory_tax")
    _add(hrdf_entries, "statutory_hrdf")

    return months


# ---------------------------------------------------------------------------
# C13 — Reversal credits (CR side, keyword match)
# Source of truth: CLASSIFICATION_RULES_v3_5.json line 864.
# ---------------------------------------------------------------------------

REVERSAL_CREDIT_RE = re.compile(
    r"REVERSAL|REVERSED|REV\s+CR|CREDIT\s+REVERSAL",
    re.IGNORECASE,
)


def compute_reversal_credits(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect bank reversal credits via the LOCKED v3.5 regex.

    Rule: a CR-side row whose description matches ``REVERSAL_CREDIT_RE``
    is a reversal credit (C13) — money credited back to the account
    because a previous debit was erroneous. These are EXCLUDED from Net
    Credits per v3.5 impact note (rules line 873).

    Substring match: the regex's ``REVERSAL`` alternative also catches
    the no-space concatenated form ``REVERSALCR`` (Bank Rakyat /
    Felcra-style: ``94044 REVERSALCR 0.10``) because ``REVERSAL`` is a
    prefix of ``REVERSALCR``.

    Priority interaction documented in v3.5 but NOT enforced here
    (caller / dispatcher responsibility):
      * ``IBG INWARD RETURN`` and ``GIRO INWARD RETURN`` route to C16,
        NOT C13 (rules line 882-885). A row with both keywords (e.g.
        ``IBG INWARD RETURN REVERSAL``) will match BOTH this function
        and ``compute_inward_returns``; the dispatcher must apply
        priority C16 over C13 to route correctly.

    C13 is NOT consumed by any of the 16 risk flags — pure classification
    port that populates the ``reversal_cr`` schema field for the Net
    Credits calculation downstream.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict:
            reversal_cr (float, total amount, schema field),
            reversal_count (int),
            reversal_entries (list).
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not REVERSAL_CREDIT_RE.search(str(description)):
            continue
        if _row_side(row) != "CR":
            continue
        amount = float(row.get("credit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "reversal_cr": round(total, 2),
        "reversal_count": len(entries),
        "reversal_entries": entries,
    }


# ---------------------------------------------------------------------------
# C16 — IBG / GIRO inward returns (CR side, keyword match)
# Source of truth: CLASSIFICATION_RULES_v3_5.json line 927.
# ---------------------------------------------------------------------------

INWARD_RETURN_RE = re.compile(
    r"IBG\s+INWARD\s+RETURN|GIRO\s+INWARD\s+RETURN",
    re.IGNORECASE,
)


def compute_inward_returns(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect IBG / GIRO inward returns via the LOCKED v3.5 regex.

    Rule: a CR-side row whose description matches ``INWARD_RETURN_RE``
    is an inward return (C16) — a failed outward IBG/GIRO transfer that
    was returned to the account because the destination rejected it.
    Per v3.5 (rules line 937), the money never actually left the
    account; these are EXCLUDED from Net Credits.

    Two shape families per v3.5 keywords:
      * ``IBG INWARD RETURN`` — Inter-bank GIRO inward return.
      * ``GIRO INWARD RETURN`` — Bank Negara GIRO inward return.

    Both keywords typically appear in long-form descriptions chained
    with destination details, e.g.:
      ``GIRO INWARD RETURN CREDIT M 1481335P012987 INV-25004 21INVALID
      COMPNY ID``
    The regex finds the keyword anywhere in the description.

    Decoupled from C13 per v3.5 note (rules line 943): "Both excluded
    from net credits. Decision per Q1 (Meesha)."

    C16 is NOT consumed by any of the 16 risk flags — pure classification
    port that populates the ``inward_return_cr`` schema field for the
    Net Credits calculation downstream.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict:
            inward_return_cr (float, total amount, schema field),
            inward_return_count (int),
            inward_return_entries (list).
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not INWARD_RETURN_RE.search(str(description)):
            continue
        if _row_side(row) != "CR":
            continue
        amount = float(row.get("credit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "inward_return_cr": round(total, 2),
        "inward_return_count": len(entries),
        "inward_return_entries": entries,
    }


# ---------------------------------------------------------------------------
# C17 — Cash deposits (CR side, keyword match)
# Source of truth: CLASSIFICATION_RULES_v3_5.json line 947.
# ---------------------------------------------------------------------------

CASH_DEPOSIT_RE = re.compile(
    r"(?:CDM\s+)?CASH\s+DEPOSIT",
    re.IGNORECASE,
)


def compute_cash_deposits(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect physical cash deposits via the LOCKED v3.5 regex.

    Rule: a CR-side row whose description matches ``CASH_DEPOSIT_RE`` is a
    cash deposit (C17). DR-side matches are ignored — Track 1 reserves the
    DR cash path for C18 (cash withdrawals), and the v3.5 exclusion list
    explicitly directs ``CASH CHQ DR`` to C18 not C17.

    Corpus-observed shape variants NOT matched by the LOCKED v3.5 regex
    and therefore intentionally NOT matched here either (preserves Track 1
    parity for the side-by-side validation gate):
      * ``CDM CA DEPOSIT`` — the "CA" token (Bank Rakyat truncation of
        "CASH"?) breaks the ``CASH\\s+DEPOSIT`` requirement. ~26 rows
        observed in corpus.
      * ``CDMCASHDEPOSIT`` — no-space concatenated form (Public Bank
        sample). 1 row observed.
    These should be addressed via a separate Tier-2 keyword improvement
    workstream that updates v3.5 rules + Track 2 simultaneously, not via
    Track 2 drifting ahead.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict with C17 schema fields populated for the 16-flag reducer:
            cash_deposits_count, cash_deposits_amount, cash_deposit_entries.
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not CASH_DEPOSIT_RE.search(str(description)):
            continue
        if _row_side(row) != "CR":
            continue
        amount = float(row.get("credit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "cash_deposits_count": len(entries),
        "cash_deposits_amount": round(total, 2),
        "cash_deposit_entries": entries,
    }


# ---------------------------------------------------------------------------
# C18 — Cash withdrawals (DR side, keyword match)
# Source of truth: CLASSIFICATION_RULES_v3_5.json line 970.
# ---------------------------------------------------------------------------

CASH_WITHDRAWAL_RE = re.compile(
    r"CASH\s+CHQ\s+DR",
    re.IGNORECASE,
)


def compute_cash_withdrawals(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect physical cash withdrawals via the LOCKED v3.5 regex.

    Rule: a DR-side row whose description matches ``CASH_WITHDRAWAL_RE``
    is a cash withdrawal (C18). The v3.5 distinction note (rules line 987)
    locks the prefix-based separation:

      * ``CASH CHQ DR`` -> C18 (cash withdrawal, this function).
      * ``HOUSE CHQ DR`` / ``CLRG CHQ DR`` -> C20 (cheque issue, separate
        function). The regex requires literal ``CASH`` before ``CHQ DR``,
        so HOUSE/CLRG variants are not matched here.

    The regex also matches ``CASH CHQ DR WD`` (the ``WD`` suffix variant
    listed alongside ``CASH CHQ DR`` in v3.5 keywords) because regex
    search is substring-based.

    Corpus-observed shape variants NOT matched by the LOCKED v3.5 regex
    and intentionally NOT matched here either (preserves Track 1 parity
    for the side-by-side validation gate):
      * ``1600 CA CASH CHQ WDWL`` — Bank Rakyat / Felcra-style ``WDWL``
        suffix instead of ``DR``. ~few rows observed.
      * ``CASH CASH CHQ / -`` and ``SERVICE CASH CHQ / -`` — Affin-style
        slash form, no ``DR`` suffix.
    Closing those gaps belongs to a Tier-2 keyword improvement
    workstream that updates v3.5 + Track 2 simultaneously, not Track 2
    drifting ahead.

    C18 is NOT consumed by any of the 16 risk flags — it is a pure
    classification port that populates the ``cash_withdrawals_*`` schema
    fields (BANK_ANALYSIS_SCHEMA_v6_3_5.json) for downstream report
    rendering and monthly_summary aggregation.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict with C18 schema fields:
            cash_withdrawals_count, cash_withdrawals_amount,
            cash_withdrawal_entries.
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not CASH_WITHDRAWAL_RE.search(str(description)):
            continue
        if _row_side(row) != "DR":
            continue
        amount = float(row.get("debit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "cash_withdrawals_count": len(entries),
        "cash_withdrawals_amount": round(total, 2),
        "cash_withdrawal_entries": entries,
    }


# ---------------------------------------------------------------------------
# C19 — Cheque deposits (CR side, keyword match)
# Source of truth: CLASSIFICATION_RULES_v3_5.json line 990.
# ---------------------------------------------------------------------------

CHEQUE_DEPOSIT_RE = re.compile(
    r"HSE\s+CHQ\s+DEPOSIT|2D\s+LOCAL\s+CHQ|CHQ\s+DEPOSIT|CHEQUE\s+DEPOSIT",
    re.IGNORECASE,
)


def compute_cheque_deposits(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect cheque deposits via the LOCKED v3.5 regex.

    Rule: a CR-side row whose description matches ``CHEQUE_DEPOSIT_RE``
    is a cheque deposit (C19). The v3.5 counterparty rule (rules line
    1009) further specifies that the counterparty for these rows is
    ``Unidentified (Cheque)`` — that downstream stamping is the
    counterparty-ledger's responsibility, not this function.

    Per v3.5 keywords, four shape families match: ``HSE CHQ DEPOSIT``
    (House Cheque Deposit), ``2D LOCAL CHQ`` (2-day local clearing),
    ``CHQ DEPOSIT`` (generic), and the long-form ``CHEQUE DEPOSIT`` used
    by Alliance / others. ``CHQ DEPOSIT`` is a substring of ``HSE CHQ
    DEPOSIT`` — both alternatives are listed for explicitness; either
    one matching is sufficient.

    C19 is NOT consumed by any of the 16 risk flags — pure classification
    port populating ``cheque_deposits_*`` schema fields.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict with C19 schema fields:
            cheque_deposits_count, cheque_deposits_amount,
            cheque_deposit_entries.
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not CHEQUE_DEPOSIT_RE.search(str(description)):
            continue
        if _row_side(row) != "CR":
            continue
        amount = float(row.get("credit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "cheque_deposits_count": len(entries),
        "cheque_deposits_amount": round(total, 2),
        "cheque_deposit_entries": entries,
    }


# ---------------------------------------------------------------------------
# C20 — Cheque issues (DR side, keyword match)
# Source of truth: CLASSIFICATION_RULES_v3_5.json line 1012.
# ---------------------------------------------------------------------------

CHEQUE_ISSUE_RE = re.compile(
    r"HOUSE\s+CHQ\s+DR|CLRG\s+CHQ\s+DR|INWARD\s+CLEARING\s+CHQ\s+DEBIT",
    re.IGNORECASE,
)


def compute_cheque_issues(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect issued (cleared) cheques via the LOCKED v3.5 regex.

    Rule: a DR-side row whose description matches ``CHEQUE_ISSUE_RE`` is
    a cheque the company issued that has now cleared the bank (C20). The
    regex deliberately does NOT contain ``CASH CHQ DR`` — that prefix
    routes to C18 (cash withdrawal) per the v3.5 distinction note (rules
    line 987 + 1031). The v3.5 exclusion list also calls out ``CHQ
    PROCESSING FEE -> C24``; that exclusion is enforced upstream by the
    C24 regex matching first / by callers tagging the priority order, not
    by this function.

    Three shape families per v3.5 keywords:
      * ``HOUSE CHQ DR`` — Public Bank / Maybank house-cheque debit form.
      * ``CLRG CHQ DR`` — clearing-cheque debit form.
      * ``INWARD CLEARING CHQ DEBIT`` — long-form variant.

    C20 is NOT consumed by any of the 16 risk flags — pure classification
    port populating ``cheque_issues_*`` schema fields.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict with C20 schema fields:
            cheque_issues_count, cheque_issues_amount, cheque_issue_entries.
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not CHEQUE_ISSUE_RE.search(str(description)):
            continue
        if _row_side(row) != "DR":
            continue
        amount = float(row.get("debit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "cheque_issues_count": len(entries),
        "cheque_issues_amount": round(total, 2),
        "cheque_issue_entries": entries,
    }


# ---------------------------------------------------------------------------
# C10 / C11 — Loan disbursement (CR) / loan repayment (DR), keyword match
# Source of truth: CLASSIFICATION_RULES_v3_5.json lines 706-866
#   (C10 ``two_tier_approach.tier_1.keywords`` + ``factoring_rule``;
#    C11 ``regex`` LOCKED).
#
# C10 (CR) — loan proceeds + factoring advances. Fires on the v3.5 tier-1
# keyword regex (LOAN DISB / FINANCING DISB / TRADE FIN / SCF TRADE /
# FACTORING / INVOICE FIN/DISCOUNT / BILL PURCHAS/DISCOUNT / BANKERS
# ACCEPTANCE / FACILITY DRAWDOWN). Tier-2 (factoring entity match) is
# wired at the dispatcher level — see ``dispatch_transaction``'s C10
# branch — so the regex here covers only the keyword route.
#
# C11 (DR) — own-loan repayments. v3.5's regex is preserved verbatim:
# ``\bTERM LOAN\b|\bLOAN REPAY|\bFINANCING REPAY|\bMONTHLY INSTALMENT\b|
# \bIB2G\s+DR\s+CA\s+CR\s+LN\b|\bTRANSFER TO LOAN\b|\bDD CASA PYMT\b|
# \bFINPAL ISSUER REPAYM`` — mixed-whitespace style is intentional.
# Priority interactions for C11 (OTHER TRANSFER FEE → C24; related-party
# instalment → C04; account-number-only → standalone C11) are enforced
# at the dispatcher level, not here.
# ---------------------------------------------------------------------------

LOAN_DISBURSEMENT_RE = re.compile(
    r"\bLOAN DISB"          # LOAN DISB / LOAN DISBURS / LOAN DISBURSEMENT
    r"|\bFINANCING DISB"    # FINANCING DISB / FINANCING DISBURS(EMENT)
    r"|\bTRADE FINANCE CR\b"
    r"|\bTRADE FIN\b"
    r"|\bSCF TRADE\b"
    r"|\bFACTORING\b"
    r"|\bINVOICE FIN\b"
    r"|\bINVOICE DISCOUNT\b"
    r"|\bBILL PURCHAS"      # BILL PURCHAS / PURCHASE / PURCHASED
    r"|\bBILL DISCOUNT\b"
    r"|\bBANKERS ACCEPTANCE\b"
    r"|\bFACILITY DRAWDOWN\b"
    # RHB own-facility transaction-type token on the CR side = a loan/financing
    # drawdown credited into the account (mirror of the C11 ``LOANS/FIN`` rung;
    # side-disciplined by the CR gate at the dispatcher). 2026-06-14.
    r"|\bLOANS?/FIN\b"
    # P2P / marketplace-lender drawdowns paid out via the platform's trustee
    # account (e.g. Funding Societies via MALAYSIAN TRUSTEES; CapBay; Fundaztic).
    # Cross-bank: these print the lender name in the memo regardless of bank.
    r"|\bFUNDING\s+SOCIET"   # Funding Societies / "Societes" (memo typo)
    r"|\bCAPBAY\b"
    r"|\bFUNDAZTIC\b"
    r"|\bMICROLEAP\b",
    re.IGNORECASE,
)


LOAN_REPAYMENT_RE = re.compile(
    r"\bTERM LOAN\b"
    r"|\bLOAN REPAY"
    r"|\bFINANCING REPAY"
    r"|\bMONTHLY INSTALMENT\b"
    r"|\bIB2G\s+DR\s+CA\s+CR\s+LN\b"
    r"|\bTRANSFER TO LOAN\b"
    r"|\bDD CASA PYMT\b"
    r"|\bFINPAL ISSUER REPAYM"
    # Per-bank own-facility loan-servicing TRANSACTION-TYPE labels (DR side).
    # These key on the bank's transaction-type token, NOT on the word "loan"
    # wherever it appears — so a transfer TO A PERSON that merely mentions a
    # loan in its reference (e.g. RHB ``RFLX <person> / ALZA LOAN``, txn-type
    # ``RFLX``) is NOT matched here and correctly stays a candidate for the
    # RP rung / loan-review net. Each was verified against the source PDF as
    # the statement owner servicing its OWN facility (s, 2026-06-14):
    #   * RHB    — ``011 LOANS/FIN <company> AUTODEBIT`` / ``LOANS/FIN PAYMENT``
    #              / ``LOANS/FIN CLEAR AUTODEBIT`` (own loan-account auto-debit).
    #   * Public Bank — ``AUTOMATED LOAN PYMT TO <loan a/c>`` and
    #              ``LOAN PYMT-ATM/EFT`` (both contain ``LOAN PYMT``).
    #   * Bank Muamalat — ``... REPAYMENT LOAN`` (DR TRF/SAL/MISC/AFT).
    r"|\bLOANS?/FIN\b"
    r"|\bLOAN PYMT\b"
    r"|\bREPAYMENT LOAN\b"
    # Asset / vehicle financing the company repays on its OWN facility.
    # Cross-bank: hire-purchase and "HP LOAN" memos appear under any bank.
    # A natural person's personal HP is caught by the C03/C04 RP rung first
    # (it runs above this), so only the company's own-facility HP reaches C11.
    r"|\bHIRE\s*PURCHASE\b"
    r"|\bHP\s+LOAN\b"
    # P2P / marketplace-lender repayments (DR side) — same lender names as the
    # C10 disbursement regex; the side decides disbursement (CR→C10) vs
    # repayment (DR→C11). Funding Societies routes repayments via its trustee
    # account (MALAYSIAN TRUSTEES) and echoes the lender name in the memo.
    r"|\bFUNDING\s+SOCIET"   # Funding Societies / "Societes" (memo typo)
    r"|\bCAPBAY\b"
    r"|\bFUNDAZTIC\b"
    r"|\bMICROLEAP\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Loan-review safety net (the "silent-zero" guard).
#
# C10/C11 are closed-vocabulary keyword rungs: a loan row is only counted as
# facility activity if its memo matches a phrase we have ALREADY enumerated.
# Two failure modes follow — (a) every bank names loans differently, so a new
# format silently misses; (b) the miss is invisible (total stays 0, the HTML
# prints "No loan activity detected"). This regex is the opposite design: a
# DELIBERATELY BROAD net over loan-shaped transaction-type tokens. It NEVER
# classifies a row (no category, no figure) — it only collects rows that look
# like facility activity but landed in none of C03/C04/C10/C11, so the analyst
# sees "N possible facility rows unclassified — review" instead of silence.
# Because it only surfaces (never books), false positives here are cheap; a
# missed loan is the expensive error, so the net is wide on purpose.
#
# Tuned to avoid marketing / disclosure prose that carries "financing" /
# "pembiayaan" (Maybank PEMBIAYAAN RUMAH ad, Ambank Term Deposit-i footer, HLB
# Kadar Pembiayaan Islamik rate notice) — those are excluded by _LOAN_NOISE_RE.
LOAN_SHAPED_RE = re.compile(
    r"\bLOANS?\b"
    r"|\bLOANS?/FIN\b"
    r"|\bFINANCING\b"
    r"|\bPEMBIAYAAN\b"
    r"|\bANGSURAN\b"
    r"|\bINSTAL?MENT\b"
    r"|\bHIRE\s*PURCHASE\b"
    r"|\bHP\s+LOAN\b"
    r"|\bREPAYMENT\b"
    r"|\bDISBURS",
    re.IGNORECASE,
)

# Marketing / rate-disclosure / product-name prose that mentions a loan word
# but is NOT a transaction. Keeps the review net from flagging footer ads.
_LOAN_NOISE_RE = re.compile(
    r"PEMBIAYAAN\s+RUMAH"
    r"|TERM\s+DEPOSIT"
    r"|FIXED\s+DEPOSIT"
    r"|FINANCE\s+ACT"
    r"|KADAR\s+(?:ASAS|PEMBIAYAAN)"
    r"|ISLAMIK?\s+\("            # "(IFR)" rate-notice fragment
    r"|FINANCIAL\s+(?:SERVICES|MEDIATION|ACTIVITIES|INSTITUTION)"
    r"|FINANCING-?I?\s+PRODUCTS",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# C24 — Bank fees & charges (DR side, keyword match)
# Source of truth: CLASSIFICATION_RULES_v3_5.json line 1084.
# ---------------------------------------------------------------------------

BANK_FEES_RE = re.compile(
    r"AUTOPAY CHARGES"
    r"|OTHER TRANSFER FEE"
    r"|CH(?:Q|EQUE)\s+PROCESS(?:ING)?\s+FEE"
    r"|3RD PARTY CHEQUE"
    r"|ACCOUNT STATUS CONFIRM"
    r"|SERVICE TAX \d+%"
    r"|STAMP DUTY"
    r"|CABLE CHARGE"
    r"|NOSTRO CHARGE"
    r"|MAS SERVICE CHARGE"
    r"|AGENT CHARGES"
    r"|CMS - DR CORP CHG"
    r"|HANDLING\s+CHRG"
    r"|CHEQ(?:UE)?\s+STAMP\s+FEE"
    r"|RFLX\s+INSTANT\s+TRF\s+SC"
    r"|RFLX\s*/\s*CM\d+"
    r"|REFLEX-\s*/\s*CM\d+"
    r"|CHQ\s+SVC"
    r"|SERVICE\s+CASH\s+CHQ"
    r"|SERVICE\s+CHARGES-OTHERS"
    r"|BANKERS\s+REFER\s+CHARGES",
    re.IGNORECASE,
)


def compute_bank_fees(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect bank fees & charges via the LOCKED v3.5 regex.

    Rule: a DR-side row whose description matches ``BANK_FEES_RE`` is a
    bank fee or charge (C24). Twenty-one alternative patterns covering
    autopay / SST / stamp duty / cable / nostro / handling / RFLX /
    SERVICE CHARGES variants — preserved verbatim from v3.5 with their
    mixed literal-space and \\s+ whitespace style.

    Important priority interactions documented in v3.5 but NOT enforced
    here (caller responsibility / Track 2 dispatcher):
      * ``OTHER TRANSFER FEE`` entries with amount <= RM1.00 are ALWAYS
        C24 regardless of trailing purpose text (rules line 1119). This
        function matches the keyword; the amount-bound priority override
        is dispatcher-level.
      * ``CHQ PROCESSING FEE`` is excluded from C20 (cheque issue) per
        the C20 exclusion list. The exclusion is enforced naturally —
        the C20 regex doesn't contain any C24 keyword as a substring.
      * Per the v3.5 keyword list, ``SERVICE CASH CHQ`` (Affin slash-
        form variant we noted as a C18 corpus gap) routes here to C24,
        not to C18.

    C24 has ``schema_fields: null`` in v3.5 — no dedicated schema slot —
    but consumers (monthly_summary, report rendering) still benefit from
    the count + amount totals. Keys are namespaced as ``bank_fees_*``.

    C24 is NOT consumed by any of the 16 risk flags — pure classification
    port.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict:
            bank_fees_count, bank_fees_amount, bank_fees_entries.
    """
    entries: list[dict[str, Any]] = []
    total = 0.0

    for row in transactions:
        description = row.get("description") or ""
        if not BANK_FEES_RE.search(str(description)):
            continue
        if _row_side(row) != "DR":
            continue
        amount = float(row.get("debit") or 0)
        entries.append(_to_entry(row, amount))
        total += amount

    return {
        "bank_fees_count": len(entries),
        "bank_fees_amount": round(total, 2),
        "bank_fees_entries": entries,
    }


# ---------------------------------------------------------------------------
# Data completeness — consolidated reducer over per-month reconciliation
# Source of truth: BANK_ANALYSIS_SCHEMA_v6_3_5.json lines 734-757.
# ---------------------------------------------------------------------------


def compute_data_completeness(
    monthly_reconciliation: list[dict[str, Any]],
    *,
    unkeyworded_return_pair_count: int = 0,
) -> dict[str, Any]:
    """Reduce per-month reconciliation results into consolidated-level keys.

    A month is "gappy" when ``reconciliation_status == 'FAIL'`` OR when
    ``extraction_gaps_count > 0``. Either signal counts the month against
    ``months_with_gaps``. ``data_completeness`` becomes ``'INCOMPLETE'`` if
    any month is gappy, otherwise ``'COMPLETE'``.

    A second extraction-gap signal — ``unkeyworded_return_pair_count`` —
    comes from ``compute_unkeyworded_return_pair_count`` (session 10).
    It captures IBG/DuitNow returns where structural pairing succeeded
    but the parser lost the C16 "INWARD RETURN" keyword token. The count
    is bank-level (not per-month) because the pairing function does not
    surface a month for each pair, so it contributes to
    ``total_extraction_gaps`` and (if > 0) forces ``data_completeness``
    to ``'INCOMPLETE'`` and appends a summary remark to ``data_gaps``.
    ``months_with_gaps`` is NOT incremented from this signal — that
    counter is strictly per-month-reconciliation; unkeyworded-pair
    reporting goes through the remark string instead.

    Args:
        monthly_reconciliation: list of dicts, one per month. Reads (all
            optional, missing keys treated as 0 / 'PASS'):
                ``month``, ``reconciliation_status``
                (``'PASS' | 'FAIL'`` — anything other than ``'PASS'`` is a
                fail signal), ``extraction_gaps_count`` (int),
                ``missing_debit_amount`` (float),
                ``missing_credit_amount`` (float).
        unkeyworded_return_pair_count: bank-level count of paired returns
            whose CR description is missing the C16 inward-return keyword.
            Default 0 (no signal). Negative values are clamped to 0.

    Returns:
        dict with keys ``data_completeness`` (enum 'COMPLETE'|'INCOMPLETE'),
        ``months_with_gaps`` (int), ``total_extraction_gaps`` (int),
        ``total_missing_debits`` (float), ``total_missing_credits`` (float),
        ``data_gaps`` (str — human-readable summary listing affected months
        and, when ``unkeyworded_return_pair_count > 0``, an additional
        clause summarising the unkeyworded-return signal).
    """
    months_with_gaps = 0
    total_extraction_gaps = 0
    total_missing_debits = 0.0
    total_missing_credits = 0.0
    gap_month_descriptions: list[str] = []

    for row in monthly_reconciliation or []:
        status = str(row.get("reconciliation_status") or "PASS").upper()
        gaps = int(row.get("extraction_gaps_count") or 0)
        missing_dr = float(row.get("missing_debit_amount") or 0)
        missing_cr = float(row.get("missing_credit_amount") or 0)
        is_gappy = status != "PASS" or gaps > 0

        total_extraction_gaps += gaps
        total_missing_debits += missing_dr
        total_missing_credits += missing_cr

        if is_gappy:
            months_with_gaps += 1
            month_label = row.get("month") or "?"
            details: list[str] = []
            if status != "PASS":
                details.append(f"reconciliation {status}")
            if gaps > 0:
                details.append(f"{gaps} gap(s)")
            if missing_dr > 0:
                details.append(f"missing DR RM {missing_dr:,.2f}")
            if missing_cr > 0:
                details.append(f"missing CR RM {missing_cr:,.2f}")
            gap_month_descriptions.append(
                f"{month_label} ({', '.join(details)})" if details else str(month_label)
            )

    unkeyworded = max(int(unkeyworded_return_pair_count or 0), 0)
    if unkeyworded > 0:
        total_extraction_gaps += unkeyworded
        gap_month_descriptions.append(
            f"{unkeyworded} unkeyworded IBG/DuitNow return pair(s) "
            f"(parser lost C16 keyword on return CR)"
        )

    data_completeness = (
        "INCOMPLETE"
        if (months_with_gaps > 0 or unkeyworded > 0)
        else "COMPLETE"
    )
    data_gaps = (
        "; ".join(gap_month_descriptions) if gap_month_descriptions else ""
    )

    return {
        "data_completeness": data_completeness,
        "months_with_gaps": months_with_gaps,
        "total_extraction_gaps": total_extraction_gaps,
        "total_missing_debits": round(total_missing_debits, 2),
        "total_missing_credits": round(total_missing_credits, 2),
        "data_gaps": data_gaps,
    }


# ---------------------------------------------------------------------------
# FX detection — Flag 14 input (total_fx_credits / total_fx_debits)
# Source of truth: BANK_ANALYSIS_SCHEMA_v6_3_5.json line 6 ($comment_fx_classification).
# ---------------------------------------------------------------------------

# Negative-list patterns that DEFINITIVELY rule out FX even when other
# triggers are present. Order: this list runs FIRST.
_FX_NEGATIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bRENTAS\b", re.IGNORECASE),
    re.compile(r"\bJANM\b", re.IGNORECASE),
    re.compile(r"\bDUITNOW\b", re.IGNORECASE),
    re.compile(r"\bFPX\b", re.IGNORECASE),
    re.compile(r"\bJOMPAY\b", re.IGNORECASE),
    # IBG/voucher-code internal references — alphabet codes that LOOK like
    # currency prefixes but are payment-rail vouchers (GBPV = General Business
    # Payment Voucher, USDP = USD-Payment internal code, EURK = internal,
    # SGDK / MYRK / etc.). The pattern catches a 3-letter code followed by
    # a single ASCII letter (V/K/P/etc.) attached to digits — a voucher
    # signature, not a currency code.
    re.compile(r"\b(?:GBP|USD|EUR|SGD|MYR|HKD|AUD|JPY|CNY)[A-Z]\d", re.IGNORECASE),
    # IBG/IBFT/INSTANT-PAY rail prefixes — domestic payment rails.
    re.compile(r"\bIBG\b", re.IGNORECASE),
    re.compile(r"\bIBFT\b", re.IGNORECASE),
)

# Positive triggers that strongly indicate genuine FX activity.
_FX_POSITIVE_KEYWORDS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bFOREX\b", re.IGNORECASE),
    re.compile(r"\bFX\s*CONV", re.IGNORECASE),
    re.compile(r"\bFOREIGN\s+EXCHANGE\b", re.IGNORECASE),
    re.compile(r"\bSWIFT\b", re.IGNORECASE),
    re.compile(r"\bCURRENCY\s+CONVER", re.IGNORECASE),
    re.compile(r"\bBOUGHT\s+(?:USD|EUR|GBP|SGD|HKD|AUD|JPY|CNY)", re.IGNORECASE),
    re.compile(r"\bSOLD\s+(?:USD|EUR|GBP|SGD|HKD|AUD|JPY|CNY)", re.IGNORECASE),
)

# A bare currency-code mention (e.g. "USD 1,234.56") that's not blocked by
# the negative list above is treated as FX. Match: 3-letter currency code
# followed by whitespace and a numeric amount, NOT immediately followed by
# a single letter (which would suggest a voucher code).
_FX_CURRENCY_AMOUNT_RE = re.compile(
    r"\b(USD|EUR|GBP|SGD|HKD|AUD|JPY|CNY)(?![A-Z])\s+[\d,]",
    re.IGNORECASE,
)


def is_fx_transaction(row: dict[str, Any]) -> bool:
    """Conservative FX classifier per the schema's FX classification guide.

    Negative list checked FIRST: when any negative pattern matches the
    description, the row is decisively NOT FX, regardless of any positive
    trigger that would otherwise fire (e.g. the bare token ``USD`` inside
    an ``IBG ... USDP-12345`` voucher reference is excluded by the IBG
    rule before the bare-currency-code rule can fire).

    Positive trigger required: at least one of the explicit FX keywords
    or a bare-currency-amount marker. Default is NOT FX.
    """
    description = str(row.get("description") or "")
    if not description:
        return False
    for neg in _FX_NEGATIVE_PATTERNS:
        if neg.search(description):
            return False
    for pos in _FX_POSITIVE_KEYWORDS:
        if pos.search(description):
            return True
    if _FX_CURRENCY_AMOUNT_RE.search(description):
        return True
    return False


def compute_fx_totals(
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sum the credit / debit amounts of rows tagged as FX by ``is_fx_transaction``.

    Args:
        transactions: rows in canonical schema. Reads ``description``,
            ``credit``, ``debit``, ``date``, ``balance``.

    Returns:
        dict with keys ``total_fx_credits`` (float, sum of FX-tagged
        credit amounts), ``total_fx_debits`` (float, sum of FX-tagged
        debit amounts), and ``fx_entries`` (list[transaction_entry]).
    """
    cr_total = 0.0
    dr_total = 0.0
    entries: list[dict[str, Any]] = []
    for row in transactions:
        if not is_fx_transaction(row):
            continue
        side = _row_side(row)
        if side == "CR":
            amount = float(row.get("credit") or 0)
            cr_total += amount
            entries.append(_to_entry(row, amount))
        elif side == "DR":
            amount = float(row.get("debit") or 0)
            dr_total += amount
            entries.append(_to_entry(row, amount))
    return {
        "total_fx_credits": round(cr_total, 2),
        "total_fx_debits": round(dr_total, 2),
        "fx_entries": entries,
    }


# ---------------------------------------------------------------------------
# Per-month aggregation — bank-agnostic, Streamlit-free
# Track 1 equivalent: app.py:calculate_monthly_summary (per-bank Streamlit
# branches). Track 2 owns this clean version derived purely from canonical
# transaction rows; no parser-footer-totals dependency, no st.session_state.
# ---------------------------------------------------------------------------


def _safe_amount(value: Any) -> float:
    """Coerce value to non-negative float, returning 0.0 on missing/invalid."""
    if value is None:
        return 0.0
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return 0.0
    return amount if amount > 0 else 0.0


def _detect_od_balance_convention(
    transactions: list[dict[str, Any]],
) -> str:
    """Decide which OD balance convention an account's rows actually follow.

    Returns ``'signed'`` (modern default: overdrawn = negative balance,
    CR-formula trail) or ``'legacy'`` (positive-magnitude: overdrawn stored
    as a positive drawdown amount, inverted trail formula).

    The old heuristic — "positive balance on an OD account = legacy" — broke
    on accounts that OPEN a month in credit and dip overdrawn mid-month
    (Wung Choon MBB 0651, 2026-06: two months flagged as RM~730k extraction
    gaps that did not exist). The convention is a property of the PARSER
    OUTPUT, not of any single balance's sign, so detect it from the data:

      1. Any negative balance anywhere → ``signed``. The legacy magnitude
         convention cannot emit negatives by construction.
      2. Otherwise vote row-by-row: for each consecutive pair of balances,
         check which formula (``prev + cr - dr`` vs ``prev + dr - cr``)
         reproduces the next balance. Majority wins.
      3. Tie / no usable pairs → ``signed`` (every modern parser since
         2026-04-20; only historic Alliance JSONs are legacy).
    """
    votes_signed = 0
    votes_legacy = 0
    prev: float | None = None
    for row in transactions:
        balance = row.get("balance")
        if balance is None:
            continue
        try:
            b = float(balance)
        except (TypeError, ValueError):
            continue
        if b < 0:
            return "signed"
        credit = _safe_amount(row.get("credit"))
        debit = _safe_amount(row.get("debit"))
        if prev is not None and (credit > 0) != (debit > 0):
            if abs(prev + credit - debit - b) <= 0.02:
                votes_signed += 1
            elif abs(prev + debit - credit - b) <= 0.02:
                votes_legacy += 1
        prev = b
    return "legacy" if votes_legacy > votes_signed else "signed"


def _compute_opening_from_row(
    row: dict[str, Any],
    account_type: str,
    convention: str = "signed",
) -> float | None:
    """Back-compute pre-row balance from the first row of a month.

    Two OD sign conventions exist in the wild:
      * Signed-negative (every modern parser as of 2026-04-20: Maybank,
        Ambank, Alliance, Bank Rakyat, Hong Leong, UOB): overdrawn balance is
        a negative number. Trail math is identical to CR accounts.
      * Positive-magnitude (legacy Alliance pre-2026-04-20): overdrawn
        balance is a positive number representing the drawdown amount.
        Credits reduce the magnitude, debits increase it — formula inverts.

    ``convention`` is the account-wide verdict from
    ``_detect_od_balance_convention`` (the per-row balance sign is NOT a
    reliable signal — a signed-convention account can open a month in
    credit). Only honoured for OD accounts.

    Returns ``None`` when the row's ``balance`` is missing.
    """
    balance = row.get("balance")
    if balance is None:
        return None
    try:
        b = float(balance)
    except (TypeError, ValueError):
        return None
    credit = _safe_amount(row.get("credit"))
    debit = _safe_amount(row.get("debit"))
    if account_type.upper() == "OD" and convention == "legacy":
        # Legacy positive-magnitude OD: opening = balance + credit - debit.
        return round(b + credit - debit, 2)
    # CR accounts AND signed-negative OD: opening = balance - credit + debit.
    return round(b - credit + debit, 2)


def compute_net_totals(
    *,
    gross_credits: float = 0.0,
    gross_debits: float = 0.0,
    own_party_cr: float = 0.0,
    related_party_cr: float = 0.0,
    loan_disbursement_cr: float = 0.0,
    fd_interest_cr: float = 0.0,
    reversal_cr: float = 0.0,
    inward_return_cr: float = 0.0,
    own_party_dr: float = 0.0,
) -> dict[str, Any]:
    """Net credits / net debits per the v6.3.5 schema formula.

    Source of truth: ``BANK_ANALYSIS_SCHEMA_v6_3_5.json`` lines 305-313 +
    621-628.

    Formulas (v6.3.5):
      * net_credits = gross_credits
                      - own_party_cr        (C01)
                      - related_party_cr    (C03)
                      - loan_disbursement_cr (C10)
                      - fd_interest_cr      (C12)
                      - reversal_cr         (C13)
                      - inward_return_cr    (C16)
      * net_debits  = gross_debits - own_party_dr (C02)

    Notable v6.3.5 exclusions that explicitly do NOT subtract from
    net_debits, despite earlier-version intuition or Track 1's
    implementation:
      * C04 related_party_dr — schema v6.3.2 note: "C04 NO LONGER
        excluded — stays in net debits for conservative credit
        assessment. C04 is reporting only."
      * C11 loan_repayment_dr — not in the schema's net_debits formula.
      * Returned cheques — schema note: "naturally net off in the
        statement (debit out + credit back = zero)."

    **Track 1 divergence (intentional — confirmed 2026-05-10):**

    Track 1 (`kredit_lab_classify.py:965-969`, frozen) implements:
      * net_credits exclusion: C01 + C03 + C10 + C13 (MISSING C12, C16)
      * net_debits exclusion: C02 + C04 + C11 (EXTRA C04, C11 vs schema)

    Track 2 follows the v6.3.5 schema, which is correct. Side-by-side
    validation against Track 1 will surface a delta on these specific
    fields — that delta is an intentional Track 2 improvement, not a
    regression. The verify pipeline must whitelist this specific
    divergence.

    All inputs are keyword-only with default 0.0 to keep callers honest:
    naming the exclusion explicitly forces the call site to acknowledge
    which categories are wired up. Inputs not yet ported (C01/C02 own-
    party, C03 related-party, C10/C11 loan, C12 FD/interest) default
    to zero today; they will be plumbed when the corresponding ports
    land.

    Args:
        gross_credits: total credit amount before exclusions.
        gross_debits: total debit amount before exclusions.
        own_party_cr: C01 own-party credit exclusion.
        related_party_cr: C03 related-party credit exclusion.
        loan_disbursement_cr: C10 loan-disbursement credit exclusion.
        fd_interest_cr: C12 FD/interest credit exclusion.
        reversal_cr: C13 reversal credit exclusion.
        inward_return_cr: C16 inward-return credit exclusion.
        own_party_dr: C02 own-party debit exclusion.

    Returns:
        dict:
            gross_credits, gross_debits (round-tripped to 2dp),
            net_credits (float), net_debits (float),
            net_change (net_credits - net_debits),
            net_credits_exclusions (dict of the six CR exclusions),
            net_debits_exclusions (dict of the one DR exclusion).
    """
    net_credits = (
        gross_credits
        - own_party_cr
        - related_party_cr
        - loan_disbursement_cr
        - fd_interest_cr
        - reversal_cr
        - inward_return_cr
    )
    net_debits = gross_debits - own_party_dr

    return {
        "gross_credits": round(gross_credits, 2),
        "gross_debits": round(gross_debits, 2),
        "net_credits": round(net_credits, 2),
        "net_debits": round(net_debits, 2),
        "net_change": round(net_credits - net_debits, 2),
        "net_credits_exclusions": {
            "own_party_cr": round(own_party_cr, 2),
            "related_party_cr": round(related_party_cr, 2),
            "loan_disbursement_cr": round(loan_disbursement_cr, 2),
            "fd_interest_cr": round(fd_interest_cr, 2),
            "reversal_cr": round(reversal_cr, 2),
            "inward_return_cr": round(inward_return_cr, 2),
        },
        "net_debits_exclusions": {
            "own_party_dr": round(own_party_dr, 2),
        },
    }


def compute_monthly_aggregates(
    transactions: list[dict[str, Any]],
    *,
    account_type: str = "CR",
) -> list[dict[str, Any]]:
    """Per-month aggregation derived purely from canonical transaction rows.

    Output is one dict per month present in the input, sorted chronologically
    (earliest first). The function trusts row order WITHIN each date (per
    ``compute_monthly_eod``: statement order is the order the rows were
    parsed) but groups by ``YYYY-MM`` independent of input ordering across
    months — months are emitted sorted regardless of input order.

    Caller responsibilities:
      * Pass single-account input (per-account grouping is the caller's job,
        same precondition as ``compute_monthly_eod``).
      * Pass ``account_type='OD'`` for overdraft accounts so the first-month
        opening-balance back-computation uses the OD sign convention.

    Args:
        transactions: rows in canonical schema. Reads ``date`` (ISO
            ``YYYY-MM-DD``), ``credit``, ``debit``, ``balance``.
        account_type: ``'CR'`` (default) or ``'OD'``. Affects the first
            month's opening-balance back-computation; subsequent months'
            opening = previous month's closing regardless of account type.

    Returns:
        list of per-month dicts with keys:
            month (str, YYYY-MM)
            transaction_count (int)
            credit_count (int), debit_count (int)
            gross_credits (float), gross_debits (float), net_change (float)
            opening_balance (float | None), closing_balance (float | None)
            lowest_balance (float | None), highest_balance (float | None)
            swing (float | None) -- highest minus lowest, or None
            eod_lowest, eod_highest, eod_average (float | None)
            eod_dates_count (int)
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in transactions:
        date = row.get("date")
        if not isinstance(date, str) or len(date) < 7:
            continue
        grouped.setdefault(date[:7], []).append(row)

    od_convention = _detect_od_balance_convention(transactions)

    results: list[dict[str, Any]] = []
    previous_closing: float | None = None
    for index, month in enumerate(sorted(grouped.keys())):
        rows_in_month = grouped[month]

        gross_credits = 0.0
        gross_debits = 0.0
        credit_count = 0
        debit_count = 0
        balances: list[float] = []
        for row in rows_in_month:
            credit = _safe_amount(row.get("credit"))
            debit = _safe_amount(row.get("debit"))
            if credit > 0:
                gross_credits += credit
                credit_count += 1
            if debit > 0:
                gross_debits += debit
                debit_count += 1
            balance = row.get("balance")
            if balance is not None:
                try:
                    balances.append(float(balance))
                except (TypeError, ValueError):
                    pass

        if index == 0:
            opening_balance = _compute_opening_from_row(
                rows_in_month[0], account_type, od_convention
            )
        else:
            opening_balance = previous_closing

        closing_balance = round(balances[-1], 2) if balances else None
        lowest_balance = round(min(balances), 2) if balances else None
        highest_balance = round(max(balances), 2) if balances else None
        swing = (
            round(highest_balance - lowest_balance, 2)
            if (highest_balance is not None and lowest_balance is not None)
            else None
        )

        eod = compute_monthly_eod(rows_in_month, month)

        results.append(
            {
                "month": month,
                "transaction_count": len(rows_in_month),
                "credit_count": credit_count,
                "debit_count": debit_count,
                "gross_credits": round(gross_credits, 2),
                "gross_debits": round(gross_debits, 2),
                "net_change": round(gross_credits - gross_debits, 2),
                "opening_balance": opening_balance,
                "closing_balance": closing_balance,
                "lowest_balance": lowest_balance,
                "highest_balance": highest_balance,
                "swing": swing,
                "eod_lowest": eod["eod_lowest"],
                "eod_highest": eod["eod_highest"],
                "eod_average": (
                    round(eod["eod_average"], 2)
                    if eod["eod_average"] is not None
                    else None
                ),
                "eod_dates_count": eod["eod_dates_count"],
            }
        )

        if closing_balance is not None:
            previous_closing = closing_balance

    return results


# ---------------------------------------------------------------------------
# Session 8: schema validation hard gate.
# ---------------------------------------------------------------------------
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "validation runs - json"
    / "claude ai prompt file"
    / "BANK_ANALYSIS_SCHEMA_v6_3_5.json"
)


def validate_track2_result(
    result: dict[str, Any],
    *,
    schema_path: str | Path | None = None,
) -> tuple[bool, list[str]]:
    """Validate a Track 2 result dict against the v6.3.5 schema.

    Pure function — reads the schema from disk once per call, no other
    side effects. Useful as an integration-test gate when wiring the
    dispatcher to produce full result dicts.

    Args:
        result: candidate Track 2 result dict.
        schema_path: optional override; defaults to the bundled v6.3.5
            schema at ``BANK_ANALYSIS_SCHEMA_v6_3_5.json``.

    Returns:
        ``(is_valid, errors)``. ``errors`` is a list of human-readable
        ``"<json-path>: <message>"`` strings, collected exhaustively (not
        stopping at the first failure). Empty when valid.

    Raises:
        FileNotFoundError: if ``schema_path`` does not exist.
        ImportError: if the ``jsonschema`` package is unavailable.
    """
    import jsonschema  # local import keeps module-load lightweight

    path = Path(schema_path) if schema_path is not None else DEFAULT_SCHEMA_PATH
    with open(path, encoding="utf-8") as f:
        schema = json.load(f)

    validator = jsonschema.Draft7Validator(schema)
    errors: list[str] = []
    for err in sorted(validator.iter_errors(result), key=lambda e: list(e.absolute_path)):
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{location}: {err.message}")
    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# Session 8: JomPAY biller-code-only guard.
#
# Predicate: returns True when a description starts with "JOMPAY" and shows
# only biller / reference codes — no entity name visible. Source rule from
# CLASSIFICATION_RULES_v3_5.json::jompay_rule (line 907 in v3.5):
#   "JomPAY is a payment channel, NOT a payee. Never classify based on
#    JomPAY biller code alone. Only classify when entity name is visible
#    in description. Applies to C06, C07, C08, C09, and C11."
#
# A future dispatcher will skip C06-C09/C11 classification when this
# predicate returns True, leaving the row as a regular expense.
# ---------------------------------------------------------------------------
_JOMPAY_PREFIX_RE = re.compile(r"^\s*JOMPAY\b", re.IGNORECASE)
_JOMPAY_TOKEN_SEPARATORS = re.compile(r"[\s,/\-]+")
_JOMPAY_STOPWORDS = frozenset(
    {
        "JOMPAY",
        "BILL",
        "PAYMENT",
        "PAYMENTS",
        "AT",
        "DIO",
        "DEBIT",
        "CREDIT",
        "TRANSFER",
        "TRANSFERS",
        "TXN",
        "REF",
        "FR",
        "TO",
        "VIA",
        "CR",
        "DR",
        "NO",
        "FOR",
        "OF",
    }
)


def is_jompay_biller_code_only(description: Any) -> bool:
    """Return True iff ``description`` is a JomPAY row with no entity name.

    Conservative — only matches when the description starts with the
    literal ``JOMPAY`` token (case-insensitive). Mid-string JomPAY
    mentions (e.g. ``IWK SDN BHD - JOMPAY BA876095``) are treated as
    "entity visible, JomPAY is just a rail mention" and yield False.

    "Entity name visible" = at least one alphabetic-only token of length
    >= 2 outside the channel/operation stopword set.

    Non-string input returns False (safe default — the dispatcher should
    only suppress when explicitly identified as JomPAY-channel-only).
    """
    if not isinstance(description, str):
        return False
    if not _JOMPAY_PREFIX_RE.match(description):
        return False
    upper = description.upper()
    for token in _JOMPAY_TOKEN_SEPARATORS.split(upper):
        if not token:
            continue
        if token.isalpha() and len(token) >= 2 and token not in _JOMPAY_STOPWORDS:
            return False
    return True


# ---------------------------------------------------------------------------
# Session 8: ghost-verb counterparty suppression.
#
# Source rule: BANK_ANALYSIS_SCHEMA_v6_3_5.json top_parties description ::
#   "Generic verb-only entries (TRANSFER FR A/C, TRANSFER TO A/C,
#    PAYMENT FR A/C, INTER-BANK PAYMENT INTO A/C with no counterparty
#    name attached) MUST be excluded from ranking — they are parser
#    dropouts, not real counterparties. Cheque patterns (HSE CHQ
#    DEPOSIT, CDM CASH DEPOSIT, 2D LOCAL CHQ, CASH CHQ DR) MAY appear
#    only if the analyst explicitly wants unidentified buckets surfaced;
#    default is to exclude."
#
# Predicate matches the whole normalised counterparty NAME (already
# extracted upstream). "TRANSFER FR A/C ALPHA SDN BHD" is a real entity
# and does NOT match — only bare verb prefixes with no name attached.
# ---------------------------------------------------------------------------
_GHOST_VERB_NAMES = frozenset(
    {
        "TRANSFER FR A/C",
        "TRANSFER TO A/C",
        "TRANSFER FROM A/C",
        "PAYMENT FR A/C",
        "PAYMENT TO A/C",
        "PAYMENT FROM A/C",
        "INTER-BANK PAYMENT INTO A/C",
        "INTERBANK PAYMENT INTO A/C",
        "INTER-BANK PAYMENT FR A/C",
        "INTERBANK PAYMENT FR A/C",
        "INTER-BANK PAYMENT FROM A/C",
        "INTERBANK PAYMENT FROM A/C",
    }
)
_CHEQUE_BUCKET_NAMES = frozenset(
    {
        "HSE CHQ DEPOSIT",
        "CDM CASH DEPOSIT",
        "2D LOCAL CHQ",
        "CASH CHQ DR",
    }
)
_WHITESPACE_RUN = re.compile(r"\s+")


def _normalise_counterparty_name(name: str) -> str:
    """Uppercase + collapse internal whitespace + strip outer whitespace."""
    return _WHITESPACE_RUN.sub(" ", name.strip().upper())


def is_ghost_counterparty(
    name: Any,
    *,
    include_cheque_buckets: bool = True,
) -> bool:
    """Return True when ``name`` is a parser dropout, not a real counterparty.

    Suppresses two families per the v6.3.3.2 ``top_parties`` rule:
      * Verb-only prefixes with no name attached (always suppressed).
      * Cheque bucket labels (suppressed by default; set
        ``include_cheque_buckets=False`` to keep them for analysts who
        explicitly want unidentified buckets surfaced).

    Matching is on the whole normalised name (uppercase, whitespace
    collapsed) — so a real entity like ``"TRANSFER FR A/C ALPHA SDN BHD"``
    does NOT match the bare ``"TRANSFER FR A/C"`` ghost.

    Non-string input returns False (safe default — only suppress when
    explicitly identified as a known parser dropout).
    """
    if not isinstance(name, str):
        return False
    normalised = _normalise_counterparty_name(name)
    if not normalised:
        return False
    if normalised in _GHOST_VERB_NAMES:
        return True
    if include_cheque_buckets and normalised in _CHEQUE_BUCKET_NAMES:
        return True
    return False


def filter_ghost_counterparties(
    entries: list[dict[str, Any]],
    *,
    name_key: str = "name",
    include_cheque_buckets: bool = True,
) -> list[dict[str, Any]]:
    """Drop entries whose ``name_key`` value is a ghost counterparty.

    Convenience wrapper for the common case of filtering a top_payers /
    top_payees list before ranking. Entries missing ``name_key`` or with
    non-string values are kept untouched (the predicate handles them
    safely by returning False).
    """
    return [
        entry
        for entry in entries
        if not is_ghost_counterparty(
            entry.get(name_key),
            include_cheque_buckets=include_cheque_buckets,
        )
    ]


# ---------------------------------------------------------------------------
# Session 8: C26 / C27 corporate-suffix detection.
#
# Source rule: CLASSIFICATION_RULES_v3_5.json C26 (Trade Income, CR) and
# C27 (Trade Expense, DR). Triggers list explicitly names the basic
# corporate suffix set:
#
#   "counterparty contains SDN BHD / BHD / BERHAD / ENTERPRISE /
#    TRADING / CORPORATION / GROUP / HOLDINGS / INDUSTRIES"
#
# Per-vertical override (clinic / tuition / services where a sole prop
# might still be a trade partner) stays in the Tier 4 prompt — Track 2
# only does the deterministic suffix detection.
#
# has_natural_person_marker is exported alongside as a related utility,
# but is INDEPENDENT of has_corporate_suffix: a name with both (e.g.
# "MUHAMMAD BIN ABDULLAH SDN BHD") is a registered entity and corporate
# wins. The Tier 4 prompt is responsible for combining the two when
# deciding C26/C27 eligibility for ambiguous sole-proprietor cases.
# ---------------------------------------------------------------------------
_CORPORATE_SUFFIX_RE = re.compile(
    r"(?<![A-Z])"
    r"(?:"
    r"SDN\.?\s*BHD\.?"
    r"|BERHAD"
    r"|BHD\.?"
    r"|ENTERPRISES?"
    r"|TRADING"
    r"|CORPORATION"
    r"|GROUP"
    r"|HOLDINGS"
    r"|INDUSTRIES"
    # International / foreign-supplier entity markers (2026-06-07). Malaysian
    # importers' overseas counterparties carry no SDN BHD suffix — they print
    # the foreign legal form (CO LTD / PTE LTD / LIMITED / GMBH / INC / LLC /
    # PLC) or, when truncated, a trade-business marker (IMPORT / EXPORT / EXPO
    # / TECHNOLOGY, incl. the corpus typo TECHOLOGY). Without these a foreign
    # trade supplier (e.g. XIAMEN ... IMPORT AND EXPO, SHENZHEN ... TECHOLOGY)
    # falls out of C26/C27 and gets mis-scored as a related party.
    r"|CO\.?\s*LTD"
    r"|PTE\.?\s*LTD"
    r"|LIMITED"
    r"|GMBH"
    r"|LLC"
    r"|PLC"
    r"|IMPORT"
    r"|EXPORT"
    r"|EXPO"
    r"|TECHN?OLOGY"
    r")"
    r"(?![A-Z])",
    re.IGNORECASE,
)

# Concat-form corporate suffix: BHD / BERHAD / SDNBHD glued to a letter
# run (no space separator). Triggered by Bank Rakyat DATAPOS extracted
# entities like ``FELCRABERHAD`` / ``FELCRABINASDNBHD`` /
# ``KETUAEKSEKUTIFFELCRABHD-AK`` where the entity-bearing opcode handler
# in app.py concatenates the entity name without spaces (Felcra-style
# PDFs strip spaces from continuation lines — see _br_extract_entity).
#
# Restricted to the three short Malaysian-form suffixes (SDN BHD, BHD,
# BERHAD). The longer suffixes (ENTERPRISE / TRADING / CORPORATION /
# GROUP / HOLDINGS / INDUSTRIES) are excluded — they're less likely to
# appear concat-form in practice and the false-positive surface widens
# considerably. ``[A-Z]{4,}`` prefix guards short-string FPs like
# ``LIMBHD`` / ``ABHD`` from triggering; real Felcra-extracted entities
# all have ≥6 letters in the preceding run.
_CORPORATE_SUFFIX_CONCAT_RE = re.compile(
    r"[A-Z]{4,}"
    r"(?:SDNBHD\.?|BERHAD|BHD\.?)"
    r"(?![A-Z])",
    re.IGNORECASE,
)

# Truncated-form corporate suffix: a trailing ``SDN`` / ``SDN B`` / ``SDN BH``
# (with optional dots) that the bank's beneficiary-name character limit cut
# short of the full ``SDN BHD`` (CIMB ~20-char truncation: ``BBT SOLUTIONS
# SDN B``, ``CHUKAI HARDWARE SDN``, ``LJ MACHINERY SDN BH``, ``AGAMI GROUP
# SDN. BH``). Anchored to end-of-string only — truncation happens at the tail
# — and requires a >=2-letter preceding word so a bare standalone ``SDN`` (junk
# label) never matches.
#
# Anchored on the literal ``SDN`` token by design: ``SDN`` is the Malay
# ``Sendirian`` of ``Sendirian Berhad``, and Malaysian law only allows
# ``Sendirian`` to be followed by ``Berhad`` — so ``SDN BHD`` is the ONLY valid
# completion and the entity is unambiguously a PRIVATE company. This is what
# keeps the rule safe against the public-listed-company case: a standalone
# ``BHD`` / ``BERHAD`` (public limited) is NEVER expanded or rewritten here —
# only forms that explicitly carry ``SDN`` are touched. Full ``SDN BHD`` is
# also not matched (the strict regex above owns it); ``BH?`` stops before the
# ``D`` of ``BHD``.
#
# Policy change s32 (user decision 2026-06-07): truncated SDN forms now count
# as corporate — previously deliberately rejected (see flipped tests).
_CORPORATE_SUFFIX_TRUNC_RE = re.compile(
    r"[A-Z]{2,}\s+"          # a real preceding word (the entity name)
    r"(?:\([^)]*\)\s+)?"     # optional region disambiguator: (M) | (MR) | (KL) ...
    r"SDN\.?"                # SDN | SDN.
    r"(?:\s+BH?\.?)?"        # optional truncated tail: " B" | " BH" | " B." | " BH."
    r"\s*$",                 # end of string only — the truncation point
    re.IGNORECASE,
)

# Strips the trailing truncated tail so it can be rewritten to the canonical
# ``SDN BHD`` (see ``_normalise_truncated_sdn_bhd``). Mirrors the trunc detector
# but captures only the tail (leading whitespace included) for substitution.
_TRUNCATED_SDN_TAIL_RE = re.compile(
    r"\s+SDN\.?(?:\s+BH?\.?)?\s*$", re.IGNORECASE
)

# Abbreviated-form corporate suffix: a trailing ``SB`` — the common Malaysian
# shorthand for ``Sdn Bhd`` (``Sendirian Berhad``). Corpus examples: ``KVC
# INDUSTRIAL SUPPLIES SB``, ``ADVENTURE REALTY SB``, ``SWE ELEKTRIKAL SB``.
#
# Trailing-anchored and requires a >=2-letter preceding word, because ``SB`` is
# a 2-letter token that ALSO appears as non-suffix noise: leading reference
# codes (``SB AM C018148 OCBC ...``), company name initials (``SB ELEKTRIK &
# ELEKTRONIK``), and mid-string memo bleed (``MUDAH MY SB SIM ZIEN YANG``). All
# of those have ``SB`` leading or mid-string — only a FINAL ``SB`` is the
# Sdn Bhd abbreviation, so the ``\s*$`` anchor is what makes the rule safe.
#
# Legal-entity-safe like the SDN rules: ``SB`` is ONLY ever the private
# ``Sdn Bhd`` shorthand — a public company is written ``Bhd`` / ``Berhad``,
# never ``SB`` — so no public-listed-company is ever mislabeled here.
#
# Policy scope #2, s32 (user decision 2026-06-07): added after the truncated-SDN
# scope #1, gated on a clean cross-bank corpus validation run.
_CORPORATE_SUFFIX_SB_RE = re.compile(
    r"[A-Z]{2,}\s+"          # a real preceding word (the entity name)
    r"(?:\([^)]*\)\s+)?"     # optional region disambiguator: (M) | (MR) | (KL) ...
    r"SB"
    r"\s*$",                 # end of string only — trailing SB = Sdn Bhd
    re.IGNORECASE,
)

# Strips the trailing ``SB`` for rewrite to canonical ``SDN BHD``.
_TRAILING_SB_TAIL_RE = re.compile(r"\s+SB\s*$", re.IGNORECASE)

_NATURAL_PERSON_MARKERS = frozenset({"BIN", "BINTI", "A/L", "A/P"})


def _normalise_truncated_sdn_bhd(name: str) -> str:
    """Expand a trailing truncated ``SDN`` / ``SDN B`` / ``SDN BH`` to the
    canonical ``SDN BHD`` so truncation variants of the same private company
    collapse to one bucket and display cleanly.

    No-op unless :data:`_CORPORATE_SUFFIX_TRUNC_RE` matches — which requires a
    real preceding word and the literal ``SDN`` token, so standalone
    ``BHD`` / ``BERHAD`` public-company names and full ``SDN BHD`` names are
    left untouched.
    """
    if not _CORPORATE_SUFFIX_TRUNC_RE.search(name):
        return name
    return _TRUNCATED_SDN_TAIL_RE.sub(" SDN BHD", name).strip()


def _normalise_trailing_sb(name: str) -> str:
    """Expand a trailing ``SB`` abbreviation to the canonical ``SDN BHD`` so
    the ``SB`` form merges with the company's full-form / ``SDN BHD`` buckets.

    No-op unless :data:`_CORPORATE_SUFFIX_SB_RE` matches — trailing-anchored
    with a real preceding word, so leading / mid-string ``SB`` noise is never
    touched. ``SB`` is exclusively the private ``Sdn Bhd`` shorthand, so this
    never mislabels a public ``Bhd`` / ``Berhad``.
    """
    if not _CORPORATE_SUFFIX_SB_RE.search(name):
        return name
    return _TRAILING_SB_TAIL_RE.sub(" SDN BHD", name).strip()


def has_corporate_suffix(name: Any) -> bool:
    """Detect a basic corporate suffix in a counterparty name.

    True iff the name contains one of: ``SDN BHD`` (with optional dots /
    spacing variants), ``BHD``, ``BERHAD``, ``ENTERPRISE``/``ENTERPRISES``,
    ``TRADING``, ``CORPORATION``, ``GROUP``, ``HOLDINGS``, ``INDUSTRIES``.

    Word-boundary-aware via case-insensitive ``[A-Z]`` lookarounds — so
    ``"INDUSTRIESX"`` does not match. ``"SOUTHERN CABLE SDN B"`` does
    not match (truncated, no full ``BHD``).

    Concat-form fallback (``_CORPORATE_SUFFIX_CONCAT_RE``) additionally
    catches Bank Rakyat DATAPOS-style entities where the corporate
    suffix is glued to a letter run with no space separator (e.g.
    ``FELCRABERHAD``, ``FELCRABINASDNBHD``, ``KETUAEKSEKUTIFFELCRABHD``).
    Only the three short Malaysian suffixes (SDN BHD / BHD / BERHAD) are
    covered concat-form; the prefix must be ≥4 letters to suppress
    nonsense like ``LIMBHD``.

    Truncated-form fallback (``_CORPORATE_SUFFIX_TRUNC_RE``) catches a
    trailing ``SDN`` / ``SDN B`` / ``SDN BH`` that the bank's beneficiary
    char-limit cut short of ``SDN BHD`` (``BBT SOLUTIONS SDN B``,
    ``CHUKAI HARDWARE SDN``). Anchored on the literal ``SDN`` token, so it
    only ever recognises PRIVATE companies — a standalone public ``BHD`` /
    ``BERHAD`` is unaffected.

    Abbreviated-form fallback (``_CORPORATE_SUFFIX_SB_RE``) catches a
    trailing ``SB`` — the shorthand for ``Sdn Bhd`` (``KVC INDUSTRIAL
    SUPPLIES SB``). Trailing-anchored with a preceding word, so leading /
    mid-string ``SB`` noise (``SB AM C018148 ...``, ``SB ELEKTRIK ...``) is
    not matched; ``SB`` is exclusively the private shorthand.

    Non-string input returns False.
    """
    if not isinstance(name, str):
        return False
    if _CORPORATE_SUFFIX_RE.search(name):
        return True
    if _CORPORATE_SUFFIX_CONCAT_RE.search(name):
        return True
    if _CORPORATE_SUFFIX_TRUNC_RE.search(name):
        return True
    return bool(_CORPORATE_SUFFIX_SB_RE.search(name))


def has_natural_person_marker(name: Any) -> bool:
    """Detect a Malaysian natural-person marker in a counterparty name.

    True when ``BIN``, ``BINTI``, ``A/L`` (Anak Lelaki), or ``A/P``
    (Anak Perempuan) appears as a whitespace-bounded token.

    Independent of :func:`has_corporate_suffix` — a name with both
    markers (e.g. ``"MUHAMMAD BIN ABDULLAH SDN BHD"``) is a registered
    entity; the caller decides how to combine the two signals.

    Non-string input returns False.
    """
    if not isinstance(name, str):
        return False
    padded = f" {name.upper()} "
    for marker in _NATURAL_PERSON_MARKERS:
        if f" {marker} " in padded:
            return True
    return False


# ---------------------------------------------------------------------------
# Session 8: counterparty name canonicalisation.
#
# Two specific merge rules sourced from the Muhafiz and KDYN sample
# reports (see validation runs - json/Claude AI challenges/):
#
#   1. JANM merge — government treasury branches collapse to parent:
#      "JANM CAWANGAN <branch>" (KUCHING / SHAH ALAM / etc.) -> "JANM".
#      Source: ranking note "All 'JANM CAWANGAN [location]' variants →
#      merge to 'JANM' for top party ranking. Individual branches
#      tracked in transaction detail only."
#
#   2. PLANWORTH GLOBAL merge — factoring-suffix tokens stripped so the
#      same factoring company doesn't split across multiple ranks:
#      "PLANWORTH GLOBAL FAC" / "PLANWORTH GLOBAL FACTORING" ->
#      "PLANWORTH GLOBAL".
#      Source: ranking note "Merge all PLANWORTH GLOBAL variants
#      (PLANWORTH GLOBAL FAC, PLANWORTH GLOBAL FACTORING) → 'PLANWORTH
#      GLOBAL'. Do NOT list 'FAC' as a separate entity."
#
# Future merges can be added by appending to _COUNTERPARTY_CANONICAL_RULES.
# Aggregation of newly-equal names (summing totals, merging month rows)
# is NOT done here — that's the caller's responsibility downstream.
# ---------------------------------------------------------------------------
_COUNTERPARTY_CANONICAL_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^JANM\s+CAWANGAN\b", re.IGNORECASE), "JANM"),
    (
        re.compile(
            r"^PLANWORTH\s+GLOBAL(?:\s+(?:FAC|FACTORING))?\s*$",
            re.IGNORECASE,
        ),
        "PLANWORTH GLOBAL",
    ),
)


def canonicalise_counterparty_name(name: Any) -> Any:
    """Apply known merge rules to a counterparty name.

    Currently merges:
      * ``"JANM CAWANGAN <branch>"`` -> ``"JANM"`` — federal/state
        treasury branches collapse to parent entity.
      * ``"PLANWORTH GLOBAL"`` / ``"PLANWORTH GLOBAL FAC"`` /
        ``"PLANWORTH GLOBAL FACTORING"`` -> ``"PLANWORTH GLOBAL"``.
      * Trailing truncated ``SDN`` / ``SDN B`` / ``SDN BH`` -> ``SDN BHD``
        (private-company truncation; ``SDN``-anchored so public ``BHD`` /
        ``BERHAD`` names are never touched).
      * Trailing ``SB`` abbreviation -> ``SDN BHD`` (trailing-anchored; only
        the final ``SB`` is the Sdn Bhd shorthand).

    Non-matching string input is returned stripped of outer whitespace
    but otherwise unchanged (case preserved). Non-string input is
    returned unchanged.
    """
    if not isinstance(name, str):
        return name
    stripped = name.strip()
    if not stripped:
        return stripped
    for pattern, canonical in _COUNTERPARTY_CANONICAL_RULES:
        if pattern.match(stripped):
            return canonical
    normalised = _normalise_truncated_sdn_bhd(stripped)
    return _normalise_trailing_sb(normalised)


def canonicalise_counterparty_entries(
    entries: list[dict[str, Any]],
    *,
    name_key: str = "name",
) -> list[dict[str, Any]]:
    """Return a new list with each entry's name canonicalised.

    Does NOT aggregate / sum entries that collapse to the same canonical
    name — the caller decides how to combine totals / month rows for
    merged entries.

    Entries missing ``name_key`` are passed through unchanged.
    """
    out: list[dict[str, Any]] = []
    for entry in entries:
        if name_key in entry:
            new_entry = dict(entry)
            new_entry[name_key] = canonicalise_counterparty_name(entry[name_key])
            out.append(new_entry)
        else:
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Special-bucket digit-noise stripper + ledger dedup pass.
#
# The upstream ``_extract_counterparty`` (app.py) sometimes glues amount,
# balance, and bank-reference noise onto the end of a special-bucket name.
# The Huahub HLB ledger had:
#   ``INTEREST 37 16 50 431 54``                   (should be ``INTEREST``)
#   ``INTEREST 35 74 84 051 56 2 8 9 0 0 0 3 2 7 6 1 _ 9 9``
#   ``UNIDENTIFIED (CHEQUE)``  ×3 (one per bank's cheque bucket)
#   ``BANK FEES``               ×3 (different banks, identical name)
#   ``HUAHUB MARKETING (OWN-PARTY)`` ×2
#
# All five are extraction artefacts that should collapse to a single
# ledger row each. The cleaner is intentionally CONSERVATIVE — only fires
# when the leading token is on a known special-bucket allowlist, so a
# legitimate name+number combo like ``PMG PHARMACY 24/7`` is never
# truncated.
# ---------------------------------------------------------------------------


# Buckets that should never carry a trailing digit-run. Any trailing
# digits/underscores/spaces/dots/dashes after the bucket name (optionally
# with a single parenthesised qualifier preserved) get stripped. Each entry
# is the FULL bucket name as it appears in the ledger after upper-casing.
_SPECIAL_BUCKET_NAMES: tuple[str, ...] = (
    "PROFIT PAID",
    "PROFIT CHARGED",
    "FD INTEREST",
    "BANK FEES",
    "BANK FEE",
    "SERVICE CHARGE",
    "UNIDENTIFIED",
    "INTEREST",
)


_TRAILING_DIGIT_NOISE_RE = re.compile(r"^[\d\s\._\-]+$")
_BUCKET_QUALIFIER_RE = re.compile(r"^(\([^)]+\))\s*[\d\s\._\-]*$")


# ---------------------------------------------------------------------------
# Pattern B (L1 + L2-with-anchor + L3 + L5) — rail-label narrative cleanup.
#
# Empirical survey of post-s26 Huahub ledger (228 entries) showed 49 entries
# with name length >= 35 chars. Of those, ~5 are bank-machine narratives
# that should fold into existing special buckets (L1), ~15-20 are rail-label
# narratives whose real counterparty can be extracted via either a
# parenthesised qualifier or a corporate-suffix anchor (L2+L3), and the
# remaining ~23 had no inline anchor and required L5.
#
# Conservative design: each layer only fires on a tight, deterministic
# anchor. L5 is the trailing-uppercase fallback — it only runs AFTER L2
# confirms a rail-label prefix matched and BOTH paren/corp-suffix anchors
# failed, so the cross-bank false-positive surface is the rail-label gate
# itself. Cross-corpus audit (18,786 unique counterparty names across
# 71 JSONs) showed L5 fires on exactly 22 names, all Huahub, zero
# accidental matches elsewhere.
# ---------------------------------------------------------------------------


# L1 — narratives that have no real counterparty content, so route to the
# existing special bucket whose semantics they actually belong to. The
# dedup pass downstream then merges these into the canonical bucket entry.
_SPECIAL_BUCKET_ROUTES: tuple[tuple[re.Pattern[str], str], ...] = (
    # HLB local/house cheque-machine deposits — analytically equivalent to
    # any other UNIDENTIFIED (CHEQUE) row; the (RPC) AT <branch> tail is
    # operational, not counterparty info.
    (
        re.compile(
            r"^(?:LOCAL|HOUSE)\s+CHEQUE(?:\s+RETURNED)?\s*\(RPC\)\s+AT\b",
            re.IGNORECASE,
        ),
        "UNIDENTIFIED (CHEQUE)",
    ),
    # HLB cash deposit machine — mirrors the (CHEQUE) convention with a
    # (CASH) qualifier. Creates a new bucket the first time it appears; the
    # dedup pass then aggregates all CDM rows under one entry.
    (
        re.compile(r"^CDM\s+DEPOSIT\s+AT\b", re.IGNORECASE),
        "UNIDENTIFIED (CASH)",
    ),
    # CIMB statement-printing fee narrative — folds into BANK FEES.
    (
        re.compile(
            r"^SCREEN\s+PRINT\s+FOR\s+STATEMENT\s+CHARGE\b",
            re.IGNORECASE,
        ),
        "BANK FEES",
    ),
)


# L2 — known rail-label prefixes that bank CIB platforms prepend to
# inter-bank-transfer narratives. List is HLB-heavy because HLB's narrative
# style is the noisiest in the corpus; cross-bank false-positive risk is
# minimal because these literal strings don't appear in non-rail-label
# descriptions. Add new prefixes only with cross-bank-corpus validation.
_RAIL_LABEL_PREFIX_RE = re.compile(
    r"^(?:"
    r"CR\s+ADV[-\s]+INTERBANK\s+GIRO\s+AT\s+KLM"
    r"|FUND\s+TRF\s+FR\s+CA\s+TO\s+CA[-\s]+INTERNET"
    r")\b",
    re.IGNORECASE,
)


# L3 — recipient-bank suffix tails that some rail-label narratives carry.
# Stripped INSIDE the L2 path so anchor detection sees the clean tail; not
# applied to arbitrary names. Pattern is anchored at end-of-string to avoid
# accidental mid-string matches.
_BANK_SUFFIX_RE = re.compile(
    r"\s*(?:"
    r"HONG\s+LEONG\s+BANK\s+BERHAD\(\d+-\w+\)"
    r"|MAYBANK\s+(?:ISLAMIC\s+)?BERHAD"
    r"|CIMB\s+(?:ISLAMIC\s+)?BANK\s+BERHAD"
    r"|PUBLIC\s+BANK\s+BERHAD"
    r"|RHB\s+(?:ISLAMIC\s+)?BANK\s+BERHAD"
    r"|AMBANK\s+\(M\)\s+BERHAD"
    r"|BANK\s+ISLAM\s+MALAYSIA\s+BERHAD"
    r")\s*$",
    re.IGNORECASE,
)


# Inline anchor patterns for L2. Both anchors cap the leading word group at
# at most 1 word (so up to 2 words total including the anchor's word). The
# tight cap is empirical: every paren- or suffix-anchored counterparty in
# the Huahub corpus is a 2-3 word name; allowing more leading words causes
# the leftmost-search to grab unrelated prefix words like ``FUND TRANSFER``
# or ``NSA`` (location qualifier) that happen to appear immediately before
# the real CP. See tests/test_track2_pattern_b.py for the locked cases.
#   (a) trailing parenthesised qualifier — e.g. "PMG PHARMACY (OUG)".
_PAREN_ANCHOR_RE = re.compile(
    r"((?:[A-Z][A-Z&/\-]*\s+)?[A-Z][A-Z&/\-]*)\s*(\([A-Z][A-Z0-9 /\-]*\))\s*[\d\s_]*$"
)
#   (b) trailing corporate suffix — e.g. "MEDIXTRA PLT", "APEX CONSULTANCY
#       SERVICES" (where SERVICES is one of the _COMPANY_SUFFIXES_RE words).
_CORP_SUFFIX_ANCHOR_RE = re.compile(
    r"((?:[A-Z][A-Z&/\-]*\s+)?[A-Z][A-Z&/\-]*)\s+"
    r"(SDN\.?\s*BHD\.?|BERHAD|ENTERPRISE|HOLDINGS|TRADING|SERVICES"
    r"|CORPORATION|GROUP|PRIVATE|LIMITED|LTD|PLT|CORP|INC|BHD)"
    r"\s*[\d\s_]*$",
    re.IGNORECASE,
)


# L5 — trailing-uppercase fallback. Fires only after L2 confirms a known
# rail-label prefix and both paren/corp-suffix anchors fail. Walks the
# remainder right-to-left, taking consecutive UPPERCASE tokens up to
# ``_L5_MAX_TOKENS`` and stopping at any non-uppercase or digit token. Then
# strips operational stop-words from both ends; requires >= ``_L5_MIN_KEPT``
# non-stop-word tokens or returns None (pass-through). The rail-label gate
# means cross-bank FP risk is bounded to whatever the L2 prefix list admits.
_L5_UPPER_TOKEN_RE = re.compile(r"^[A-Z][A-Z&/\-]*$")
_L5_TRAILING_NOISE_RE = re.compile(r"[\d\s_]+$")
_L5_MAX_TOKENS = 5
_L5_MIN_KEPT = 2
_L5_STOP_WORDS = frozenset({
    # Operational / banking-narrative noise
    "PAYMENT", "PAYMENTS", "INVOICE", "INVOICES", "INVOIS",
    "TRANSFER", "TRANSFERS", "TRSF", "FUND", "FUNDS",
    "WAGES", "SALARY", "SALARIES", "PROMOTION", "PROMOTIONS", "MARKETING",
    "DEPOSIT", "DEPOSITS", "DEPOSITED",
    "INV", "PV", "REF",
    # English months (short + full)
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "SEPT",
    "OCT", "NOV", "DEC",
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "JUNE", "JULY", "AUGUST",
    "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    # Malay months
    "JANUARI", "FEBRUARI", "MAC", "MEI", "JULAI", "OGOS", "OKTOBER", "DISEMBER",
    # Connectives + leftover rail words
    "AND", "OR", "FOR", "TO", "FROM", "THE", "OF", "BY", "AT",
    "CR", "DR", "ADV", "GIRO", "INTERBANK", "INTERNET",
})


def _extract_trailing_uppercase(remainder: str) -> str | None:
    """L5: trailing-uppercase fallback for rail-label narratives.

    Caller (``_extract_after_rail_label_prefix``) has already confirmed a
    known rail-label prefix matched and the paren / corp-suffix anchors
    found nothing. We do a right-to-left walk on the remainder:

      1. Strip trailing digit / underscore / whitespace noise.
      2. Tokenise on whitespace.
      3. From the right, collect consecutive UPPERCASE tokens (matching
         ``_L5_UPPER_TOKEN_RE``) up to ``_L5_MAX_TOKENS``.
      4. Stop at the first non-uppercase or digit-bearing token.
      5. Drop leading + trailing stop-word tokens from the captured run.
      6. Require >= ``_L5_MIN_KEPT`` non-stop-word tokens AND total length
         >= 4 chars; else return None (caller falls through to pass-through).

    A known trade-off: digit tokens interleaved INSIDE the trailing name
    (e.g. ``F IMAN 3 FARMASI IMAN``) truncate the capture, so two
    occurrences of the same real CP may extract to different cleaned
    forms. Acceptable: shorter clean names dedupe partially rather than
    leaking full 80-char narratives.
    """
    stripped = _L5_TRAILING_NOISE_RE.sub("", remainder).rstrip()
    if not stripped:
        return None
    tokens = stripped.split()
    trailing: list[str] = []
    for tok in reversed(tokens):
        if len(trailing) >= _L5_MAX_TOKENS:
            break
        if _L5_UPPER_TOKEN_RE.match(tok):
            trailing.append(tok)
        else:
            break
    if not trailing:
        return None
    trailing.reverse()
    while trailing and trailing[-1] in _L5_STOP_WORDS:
        trailing.pop()
    while trailing and trailing[0] in _L5_STOP_WORDS:
        trailing.pop(0)
    kept_non_stop = [t for t in trailing if t not in _L5_STOP_WORDS]
    if len(kept_non_stop) < _L5_MIN_KEPT:
        return None
    candidate = " ".join(trailing)
    if len(candidate) < 4:
        return None
    return candidate


def _route_to_special_bucket(name: str) -> str | None:
    """L1: route bank-machine narratives to existing special buckets.

    Returns the target bucket name if the input matches one of the
    deterministic narrative patterns in ``_SPECIAL_BUCKET_ROUTES``;
    otherwise returns None.
    """
    for pattern, target in _SPECIAL_BUCKET_ROUTES:
        if pattern.match(name):
            return target
    return None


def _extract_after_rail_label_prefix(name: str) -> str | None:
    """L2+L3+L5: strip a known rail-label prefix + bank suffix, then extract
    the real counterparty via an inline anchor (paren qualifier or corporate
    suffix), falling back to a trailing-uppercase capture if neither anchor
    fires.

    Returns the extracted counterparty name if (a) a known rail-label
    prefix was matched AND (b) either an anchor was found OR the L5
    trailing-uppercase fallback yielded a viable candidate. Returns None
    otherwise — caller passes through unchanged.
    """
    if not _RAIL_LABEL_PREFIX_RE.match(name):
        return None

    # Strip the rail-label prefix.
    remainder = _RAIL_LABEL_PREFIX_RE.sub("", name, count=1).strip()

    # L3 inline: strip trailing recipient-bank suffix so anchor regex sees
    # the clean tail.
    remainder = _BANK_SUFFIX_RE.sub("", remainder).strip()

    if not remainder:
        return None

    # (a) Parenthesised-qualifier anchor — e.g. "PMG PHARMACY (OUG)".
    paren_match = _PAREN_ANCHOR_RE.search(remainder)
    if paren_match:
        return f"{paren_match.group(1).strip()} {paren_match.group(2)}"

    # (b) Corporate-suffix anchor — e.g. "MEDIXTRA PLT".
    suffix_match = _CORP_SUFFIX_ANCHOR_RE.search(remainder)
    if suffix_match:
        return f"{suffix_match.group(1).strip()} {suffix_match.group(2).upper()}"

    # (c) L5 trailing-uppercase fallback — e.g. "AEON CO", "PASTEL CARE".
    return _extract_trailing_uppercase(remainder)


# Pattern B L4 — invoice-ref prefix strip. Some statement narratives lead with a
# varying invoice / payment-voucher reference and the real counterparty name
# FOLLOWS it (Alliance KYDN payroll + vendor rows: "@ MAG PV/YN/2502-171
# SHARAVAANNAN A/L RAJ", "SOP47418 PV/YN/2503-125 INSAN BAKTI", "1007603935
# PV/YN/2507-084 ZUELLIG PHARMA"). The leading ref changes every transaction, so
# one entity fragments into a bucket per invoice. Anchoring on the highly
# distinctive "PV/YN/####-###" ref token and keeping the plausible name AFTER it
# collapses the fragments.
#
# The trailing-ref shape ("STATIONERY CLAIM PV/YN/2506-153" — ref at the END,
# only memo before it) is deliberately NOT handled: nothing plausible follows
# the ref, so the guard returns None and the name passes through untouched.
# Doubled forms (two PV/YN tokens) also pass through. The "PV/YN/" anchor is so
# distinctive it carries no cross-bank FP risk — it cannot fire on data that
# doesn't contain this exact ref scheme.
#
# Corpus-validated (s32, 2026-06-07): 564 -> 189 distinct names, every collision
# a true same-entity merge, 0 wrong-person fusions.
_INVOICE_REF_PREFIX_RE = re.compile(
    r"^.*PV/YN/\d{3,4}-\d{2,4}\s+", re.IGNORECASE | re.DOTALL
)


def _strip_invoice_ref_prefix(name: str) -> str | None:
    """L4: return the counterparty name FOLLOWING a leading 'PV/YN/####-###'
    invoice-ref prefix, or None if the pattern doesn't apply.

    The remainder must look like a real name — >=2 alpha-bearing tokens, >=6
    chars, >70% alphabetic/space, and no further PV/YN token (doubled form) —
    else None (pass-through). This keeps the trailing-ref shape and memo-only
    strings untouched.
    """
    m = _INVOICE_REF_PREFIX_RE.match(name)
    if not m:
        return None
    remainder = name[m.end():].strip()
    if not remainder or "PV/YN/" in remainder.upper():
        return None
    tokens = [tok for tok in remainder.split() if re.search(r"[A-Za-z]", tok)]
    alpha = sum(c.isalpha() or c.isspace() for c in remainder)
    if len(tokens) >= 2 and len(remainder) >= 6 and alpha / max(len(remainder), 1) > 0.7:
        return remainder
    return None


# Pattern B L6 — DuitNow "/IB <name> DESC" extraction. OCBC DuitNow instant-
# transfer narratives carry the counterparty name between the "/IB " field marker
# and the " DESC" memo marker, with per-transaction DESC/REF memo AFTER it
# ("DUITNOW(INST TRF) CR /IB CALVIN PROFESSIONAL DESC REF ATOME", "...DR /IB
# ALICE ANAK DULAH DESC JULY..."). The varying trailing memo splits one entity
# into a bucket per transaction; extracting the bounded /IB..DESC name collapses
# them. Both delimiters are present so fusion risk is low; the name region is
# bank-truncated (~20 chars) so variants merge among themselves (same truncation
# ceiling as the raw data, not a new fusion).
#
# Anchored on the distinctive "DUITNOW(INST TRF)" rail label, so the layer is
# inert on any other narrative. Stripping the rail label to surface the clean
# entity name is consistent with the "no rail labels in bucket names" rule.
# Corpus-validated (s32, 2026-06-07): 430 -> 127 distinct names, every collision
# a true same-entity merge, 0 wrong-person fusions.
_DUITNOW_IB_RE = re.compile(
    r"^DUITNOW\(INST TRF\).*?/IB\s+(.*?)(?:\s+DESC\b|$)", re.IGNORECASE | re.DOTALL
)


def _strip_duitnow_ib_prefix(name: str) -> str | None:
    """L6: extract the counterparty name from an OCBC DuitNow "/IB <name> DESC"
    narrative, or None if the pattern doesn't apply.

    The captured name must look real — >=2 alpha-bearing tokens, >=4 chars —
    else None (pass-through).
    """
    m = _DUITNOW_IB_RE.match(name)
    if not m:
        return None
    captured = m.group(1).strip()
    tokens = [tok for tok in captured.split() if re.search(r"[A-Za-z]", tok)]
    if len(tokens) >= 2 and len(captured) >= 4:
        return captured
    return None


# Pattern B L7 — "CREDIT TRANSFER <name> SENT FROM AMONLINE" extraction. AmBank
# AMOnline credit-transfer narratives fence the counterparty name between the
# "CREDIT TRANSFER " prefix and the " SENT FROM AMONLINE" marker, with
# per-transaction memo after it ("CREDIT TRANSFER MST GLOBAL MOTORSPORT SENT FROM
# AMONLINE BOOKING VELLFIRE"). The trailing memo splits one entity per
# transaction; extracting the fenced name collapses them. Back-fence present so
# fusion risk is low. Small scheme (one corpus customer); the no-AMONLINE subset
# (no back fence) is left untouched.
#
# Corpus-validated (s32, 2026-06-07): the MST GLOBAL MOTORSPORT cluster collapses
# 6 -> 1; 0 wrong-entity fusions.
_CREDIT_TRANSFER_RE = re.compile(
    r"^CREDIT TRANSFER\s+(.*?)\s+SENT FROM AMONLINE\b", re.IGNORECASE | re.DOTALL
)


def _strip_credit_transfer_prefix(name: str) -> str | None:
    """L7: extract the counterparty name fenced between 'CREDIT TRANSFER ' and
    ' SENT FROM AMONLINE', or None if the pattern doesn't apply.

    Requires the back fence ('SENT FROM AMONLINE') so the name is bounded on
    both sides. The captured name must be >=2 alpha-bearing tokens / >=4 chars.
    """
    m = _CREDIT_TRANSFER_RE.match(name)
    if not m:
        return None
    captured = m.group(1).strip()
    tokens = [tok for tok in captured.split() if re.search(r"[A-Za-z]", tok)]
    if len(tokens) >= 2 and len(captured) >= 4:
        return captured
    return None


def clean_counterparty_name(name: Any) -> Any:
    """Strip parser-leaked digit noise from special-bucket counterparty names.

    Conservative cleanup that fires only when the leading token matches a
    known ``_SPECIAL_BUCKET_NAMES`` entry. For all other names the input is
    returned unchanged (after outer-whitespace strip) so free-form
    counterparties like ``PMG PHARMACY (OUG)`` or ``2026 INVOICES`` are
    preserved.

    Behavior:
      * ``"INTEREST 37 16 50 431 54"``        -> ``"INTEREST"``
      * ``"UNIDENTIFIED (CHEQUE) 12 34"``     -> ``"UNIDENTIFIED (CHEQUE)"``
      * ``"BANK FEES"``                       -> ``"BANK FEES"`` (no change)
      * ``"PMG PHARMACY (OUG)"``              -> unchanged (not in bucket list)
      * Non-string input                      -> returned unchanged
    """
    if not isinstance(name, str):
        return name
    stripped = name.strip()
    if not stripped:
        return stripped

    # Pattern B L1: bank-machine narratives → existing special buckets.
    routed = _route_to_special_bucket(stripped)
    if routed is not None:
        return routed

    # Pattern B L2+L3+L5: rail-label prefix strip + anchor extraction +
    # trailing-uppercase fallback.
    extracted = _extract_after_rail_label_prefix(stripped)
    if extracted is not None:
        return extracted

    # Pattern B L4: invoice-ref prefix strip — name follows a leading PV/YN ref.
    deref = _strip_invoice_ref_prefix(stripped)
    if deref is not None:
        return deref

    # Pattern B L6: DuitNow "/IB <name> DESC" extraction.
    duitnow = _strip_duitnow_ib_prefix(stripped)
    if duitnow is not None:
        return duitnow

    # Pattern B L7: "CREDIT TRANSFER <name> SENT FROM AMONLINE" extraction.
    credit_xfer = _strip_credit_transfer_prefix(stripped)
    if credit_xfer is not None:
        return credit_xfer

    # Pattern A: digit-noise strip on known special buckets.
    upper = stripped.upper()
    for bucket in _SPECIAL_BUCKET_NAMES:
        if upper == bucket:
            return bucket
        if not upper.startswith(bucket + " "):
            continue
        rest = upper[len(bucket):].lstrip()
        if _TRAILING_DIGIT_NOISE_RE.match(rest):
            return bucket
        m = _BUCKET_QUALIFIER_RE.match(rest)
        if m:
            return f"{bucket} {m.group(1)}"
        # Bucket prefix matched but the tail looks like real content
        # (parenthesised qualifier mixed with words, or non-digit
        # extension). Leave alone — better to under-clean than corrupt.
        return stripped
    return stripped


# ---------------------------------------------------------------------------
# Natural-person bucket merge — compensate for the CIMB beneficiary-field
# truncation + memo bleed.
#
# CIMB statements truncate the beneficiary name to ~20 chars and the
# extractor sometimes glues narrative/reference text onto the name (both as
# a leading prefix and a trailing suffix). One person therefore fragments
# into many counterparty buckets, e.g. DCSE owner MUHAMMAD ARIF splits into
# ~19: `MUHAMMAD ARIF BIN NO REIMBURSE`, `...BIN NO HIRE PURCHASE DCSE`,
# `...BIN N STARTUP WALLET`, `REFUND PETTY CASH MUHAMMAD ARIF BIN NO`, ...
#
# This pre-pass groups person buckets by the two given-name tokens before the
# BIN/BINTI/A-L/A-P marker (which survives both prefix bleed and surname
# truncation) and merges only the buckets whose surname region is memo-
# polluted or truncated (≤2 chars / empty) — clustered by surname prefix-
# compatibility so `BIN N` folds into `BIN NO` but two distinct full
# surnames never merge.
#
# CROSS-CORPUS VALIDATED (CIMB DCSE + PBB/Rakyat/RHB/BIMB regression
# corpora, ~5,000 names): the ONLY non-CIMB merges produced are correct
# same-person merges (honorific prefix, transaction-ref suffix). Full-
# surname rosters like BIMB Mytutor (six distinct "SITI HAJAR BINTI <X>")
# are left untouched because their surnames are clean and distinct. The
# memo-token set below deliberately contains operational/admin words only —
# never Malay patronymic name tokens — so a real father-name first token is
# never mistaken for memo.
# ---------------------------------------------------------------------------


_PERSON_MERGE_MARKERS = ("BINTI", "BINTE", "BIN", "BI", "A/L", "A/P")

# Marker truncations folded to their canonical form so a name whose marker
# CIMB truncated (e.g. "BAKHTIAR SAFFUAN BI Aug 24" — "BIN" cut to "BI" by the
# ~20-char beneficiary-name limit) groups with the same person's full-marker
# siblings ("BAKHTIAR SAFFUAN BIN" on the credit side). Only unambiguous
# truncations: "BI"->"BIN", "BINTE"->"BINTI".
_PERSON_MARKER_CANONICAL = {"BI": "BIN", "BINTE": "BINTI"}

# Month stamps glued onto the surname region as a trailing suffix (CIMB
# "...BI Aug 24"). Distinct from the LEADING month-prefix bleed the given2
# window already strips — these sit AFTER the marker and would otherwise be
# mistaken for a surname fragment, splitting one person across months. Treated
# as post-marker memo only (never evaluated on the pre-marker name body), and
# no Malay patronymic surname is a month word, so this cannot collide with a
# real surname.
_PERSON_MONTH_RE = re.compile(
    r"^(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)$"
)

_PERSON_MEMO_TOKENS = frozenset({
    "REIMBURSE", "CLAIM", "PETTY", "ADV", "ADDITIONAL", "HIRE", "PURCHASE",
    "TECHNICAL", "SERVICE", "EMAIL", "DCSE", "FLIGHT", "HOTEL", "MOB", "FUEL",
    "INV", "STARTUP", "WALLET", "PROJECT", "MANAGER", "MEDICAL", "OFFSHORE",
    "ADMINISTRATION", "INTERN", "HOMESTAY", "PARKING", "PARKRITE", "TOUCH",
    "DINNER", "MEETIN", "MEETING", "CLIENT", "TRAVELLING", "FINAL", "QAQC",
    "CHECKER", "PEST", "CONTROL", "WATCHES", "BIL", "BILL", "EXTEND", "LOAN",
    "ELECTRIC", "PANTRY", "CAR", "NO", "OPENING", "BANK", "REFUND", "CASH",
    "ABZ", "FUND",
})

_PERSON_ORDINAL_RE = re.compile(r"^\d+(?:ST|ND|RD|TH)$")


def _person_token_is_memo(token: str) -> bool:
    """True if a post-marker token is operational memo rather than a name
    fragment — a memo keyword, an ordinal (``30TH``), or any token carrying
    a digit (reference codes / month stamps)."""
    if token in _PERSON_MEMO_TOKENS:
        return True
    if _PERSON_ORDINAL_RE.match(token):
        return True
    if _PERSON_MONTH_RE.match(token):
        return True
    return any(ch.isdigit() for ch in token)


def _person_merge_core(name: Any) -> tuple[str, str, str, list[str]] | None:
    """Extract ``(given2_key, marker, surname_first, after_tokens)`` from a
    natural-person counterparty name, or None if it carries no person marker.

    ``given2_key`` is the last two tokens before the marker (survives leading
    memo bleed); ``surname_first`` is the first post-marker token only when it
    looks like a real surname fragment (not memo / digit), else ``""``.
    """
    cleaned = clean_counterparty_name(canonicalise_counterparty_name(name))
    normalised = _normalise_counterparty_name(cleaned) if isinstance(cleaned, str) else ""
    if not normalised:
        return None
    tokens = normalised.split()
    marker_idx = next(
        (i for i, t in enumerate(tokens) if t in _PERSON_MERGE_MARKERS), None
    )
    if marker_idx is None or marker_idx == 0:
        return None
    given = tokens[max(0, marker_idx - 2):marker_idx]
    if not given:
        return None
    given2_key = " ".join(given[-2:])
    marker = _PERSON_MARKER_CANONICAL.get(tokens[marker_idx], tokens[marker_idx])
    after = tokens[marker_idx + 1:]
    surname_first = (
        after[0]
        if after and not _person_token_is_memo(after[0])
        else ""
    )
    return given2_key, marker, surname_first, after


def _person_surname_mergeable(surname_first: str, after: list[str]) -> bool:
    """True if a person bucket's surname region is truncated or memo-polluted
    (so it should fold into a sibling), rather than a clean full surname."""
    if surname_first == "":
        return True
    if len(surname_first) <= 2:
        return True
    return any(_person_token_is_memo(t) for t in after)


def _person_prefix_compatible(a: str, b: str) -> bool:
    """Two surname-first tokens are compatible when one is a character prefix
    of the other (truncation) — empty matches anything."""
    if not a or not b:
        return True
    return a.startswith(b) or b.startswith(a)


def _apply_person_name_merge(
    entries: list[dict[str, Any]], *, name_key: str
) -> list[dict[str, Any]]:
    """Rewrite the ``name_key`` of fragmented natural-person buckets to a
    single canonical display name so the downstream dedup aggregates them.

    Non-person entries and clean full-surname people are returned unchanged.
    Only memo-polluted / truncated siblings sharing the same two given-name
    tokens + marker (and prefix-compatible surnames) are rewritten.
    """
    groups: dict[tuple[str, str], list[tuple[int, str, str, list[str]]]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or name_key not in entry:
            continue
        core = _person_merge_core(entry.get(name_key))
        if core is None:
            continue
        given2_key, marker, surname_first, after = core
        groups.setdefault((given2_key, marker), []).append(
            (idx, surname_first, " ".join(after), after)
        )

    rewrites: dict[int, str] = {}
    for (given2_key, marker), members in groups.items():
        mergeable = [
            m for m in members if _person_surname_mergeable(m[1], m[3])
        ]
        if len(mergeable) >= 2:
            clusters: list[list[tuple[int, str, str, list[str]]]] = []
            for m in mergeable:
                for cluster in clusters:
                    if all(_person_prefix_compatible(m[1], c[1]) for c in cluster):
                        cluster.append(m)
                        break
                else:
                    clusters.append([m])
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                best_surname = max((c[1] for c in cluster), key=len)
                display = f"{given2_key} {marker} {best_surname}".strip()
                for idx, _surname, _after_str, _after in cluster:
                    rewrites[idx] = display

        # Case B — fold a BARE proper-prefix surname truncation into its unique
        # longer sibling. "NOOR AZLAN BIN MOHAM" (surname MOHAM, nothing clean
        # after it) is a truncation of "NOOR AZLAN BIN MOHAMED ISA" — the ≤2-char
        # rule above misses it because MOHAM is 5 clean chars. Guarded against
        # over-merge: the short surname must be a STRICT character-prefix, carry
        # no distinguishing clean token after it, and resolve to EXACTLY ONE
        # distinct longer sibling full-name (so MOHAMED ALI vs MOHAMED ISA stays
        # split — ambiguous targets are left for the analyst).
        for idx, surname, _after_str, after in members:
            if idx in rewrites or not surname:
                continue
            if surname in _MALAY_GIVEN_PREFIXES:
                # A common standalone name component (ABDUL / SITI / NUR ...)
                # can extend many ways (ABDUL RAHMAN / AZIZ / KARIM), so a bare
                # "BINTI ABDUL" truncation is NOT unambiguously "...ABDULLAH".
                # Skip — only genuinely mid-word fragments (MOHAM⊂MOHAMED) fold.
                continue
            if any(not _person_token_is_memo(t) for t in after[1:]):
                continue  # has a distinguishing clean token → full surname
            targets = {
                c[2] for c in members
                if len(c[1]) > len(surname) and c[1].startswith(surname)
            }
            if len(targets) != 1:
                continue
            target_after = next(iter(targets))
            display = f"{given2_key} {marker} {target_after}".strip()
            rewrites[idx] = display
            for c in members:
                if c[2] == target_after:
                    rewrites.setdefault(c[0], display)

    if not rewrites:
        return entries
    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if idx in rewrites and isinstance(entry, dict):
            new_entry = dict(entry)
            new_entry[name_key] = rewrites[idx]
            out.append(new_entry)
        else:
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Marker-less person-name merge (Cluster 3 Case C).
#
# Some Malay names carry NO patronymic marker (no BIN/BINTI/A-L/A-P) — the
# person simply wrote "MOHD KHAWAZATUL FAEZ". The marker-keyed pre-pass above
# cannot anchor on them, so CIMB memo-bleed fragments one person into many
# buckets (EXPENSES CLAIM MOHD KHAWAZATUL / PETTYCASH REIMBURSMT MOHD
# KHAWAZATUL / TRIP ALLOWANCE MOHD KHAWAZATUL / MOHD KHAWAZATUL FAEZ ...).
#
# Safe key = (Malay given-name prefix, the RARE distinctive token after it):
# ("MOHD", "KHAWAZATUL"). Because the rare name token is always in the key,
# two DIFFERENT people (MOHD KHAWAZATUL vs MOHD KHAIRUL) get different keys and
# never merge — only genuine same-spelling fragments collapse. The given-name
# anchor also excludes companies (a "... SOLUTIONS" bucket has no Malay given
# name, so it is never considered). Leading memo is skipped by taking the run
# from the first given-name token; un-stoplisted memo words elsewhere cannot
# cause an over-merge because they never displace the rare name token from the
# key. Cross-corpus validated by the over-merge canary.
# ---------------------------------------------------------------------------

_MALAY_GIVEN_PREFIXES = frozenset({
    "MOHD", "MOHAMAD", "MOHAMED", "MOHAMMAD", "MUHAMMAD", "MUHAMAD", "MAT",
    "AHMAD", "ABDUL", "ABD", "NUR", "NURUL", "NOOR", "NOR", "SITI", "WAN",
    "TENGKU", "SYED", "SHARIFAH", "NIK", "CHE", "KU", "FATIN", "FARAH",
    "AINA", "AISYAH", "AIMAN",
})

# Leading honorifics + operational memo words that legitimately PRECEDE a name
# in a marker-less bucket. The run is accepted only when EVERY token before the
# first given-name prefix is one of these (or a digit/ref/month token) — so a
# bucket beginning with an unexpected real name ("INTAN NUR MAISARAH") is
# rejected rather than silently truncated to "NUR MAISARAH" and mis-merged.
# Failure mode is under-merge (a person stays fragmented), never over-merge.
_MARKERLESS_HONORIFICS = frozenset({
    "CIK", "PUAN", "ENCIK", "EN", "TUAN", "MS", "MR", "MRS", "DR", "HJ", "HJH",
    "HAJI", "MISS", "MADAM", "TUN", "DATO", "DATUK", "DATIN",
})
_MARKERLESS_LEAD_MEMO = frozenset({
    "EXPENSES", "ALLOWANCE", "TRIP", "PUMP", "MRC", "SAMB", "REPAYMENT",
    "POSLAJU", "BALANCE", "SUBCON", "ADVANCE", "INTERN", "PETTYCASH",
    "REIMBURSMT", "WAGES", "OT", "EXPENSE", "TRANSPORT", "RENTAL", "RENT",
    "COMMISSION", "INCENTIVE", "BONUS",
})

# Distinctive name token: long enough to be near-unique. 6 chars keeps
# KHAWAZATUL/ANSARUTHEEN-class names while excluding common short fragments.
_MARKERLESS_DISTINCTIVE_MIN_LEN = 6

# Trailing bank-code tokens CIMB/AmBank glue after the beneficiary name
# ("MOHD KHAWAZATUL MBB"). Trimmed only by EXACT match — never by length —
# so a truncated distinguishing name token (MUHAMMAD DANIAL "AQI" vs "HAK",
# also 3 chars) is preserved and keeps distinct people apart.
_MARKERLESS_BANK_CODES = frozenset({
    "MBB", "MAYB", "CIMB", "BIMB", "RHB", "PBB", "HLB", "HLBB", "UOB", "OCBC",
    "BSN", "AMFB", "AMBB", "AFFIN", "AGRO", "BKRM", "KFH", "MBSB", "BMMB",
})


def _markerless_name_run(name: Any) -> tuple[str, tuple[str, ...]] | None:
    """For a marker-LESS person-name counterparty, return
    ``(given_prefix, run_tokens)`` or None.

    ``run_tokens`` is the name-token run from the first Malay given-name token
    onward, trailing noise (memo / digits / bank codes ≤3 chars) trimmed. The
    run must contain at least one DISTINCTIVE token — length ≥
    ``_MARKERLESS_DISTINCTIVE_MIN_LEN``, not a memo word, and not itself a
    common given-name prefix (so "MUHAMMAD"/"MOHD" never qualify; "KHAWAZATUL"
    does). Returns None for marked names (handled upstream), companies,
    ambiguous multi-party buckets, or runs with no distinctive token.
    """
    if not isinstance(name, str):
        return None
    if OWN_PARTY_MARKER_RE.search(name.upper()) or "(possibly multiple parties)" in name.lower():
        return None
    if _looks_like_company(name):
        return None
    cleaned = clean_counterparty_name(canonicalise_counterparty_name(name))
    normalised = _normalise_counterparty_name(cleaned) if isinstance(cleaned, str) else ""
    if not normalised:
        return None
    tokens = normalised.split()
    if any(t in _PERSON_MERGE_MARKERS for t in tokens):
        return None
    start = next((i for i, t in enumerate(tokens) if t in _MALAY_GIVEN_PREFIXES), None)
    if start is None:
        return None
    # Every token BEFORE the name must be a known honorific / memo / ref token;
    # an unexpected leading real-name token means we cannot trust the run start.
    for t in tokens[:start]:
        if not (
            _person_token_is_memo(t)
            or t in _MARKERLESS_HONORIFICS
            or t in _MARKERLESS_LEAD_MEMO
        ):
            return None
    run = tokens[start:]
    while len(run) > 1 and (
        _person_token_is_memo(run[-1]) or run[-1] in _MARKERLESS_BANK_CODES
    ):
        run.pop()
    has_distinctive = any(
        len(t) >= _MARKERLESS_DISTINCTIVE_MIN_LEN
        and not _person_token_is_memo(t)
        and t not in _MALAY_GIVEN_PREFIXES
        for t in run
    )
    if not has_distinctive:
        return None
    return tokens[start], tuple(run)


def _markerless_run_compatible(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """True iff the two runs are the same person.

    * Same token count: all tokens but the last must match EXACTLY; the last
      may be a character-prefix either way (pure ~18-char final-token
      truncation — "NUR FARHANA HAN" ⊑ "NUR FARHANA HANIM").
    * Different token count: the shorter must be an EXACT token-prefix of the
      longer (clean extension — "MOHD KHAWAZATUL" ⊑ "MOHD KHAWAZATUL FAEZ").
      No char-prefix tolerance across differing lengths, so a concatenated
      token (SITI "NURAFIFAH") never matches a common prefix (SITI "NUR" ...)
      of a longer, different name.
    """
    if len(a) == len(b):
        for i in range(len(a) - 1):
            if a[i] != b[i]:
                return False
        return a[-1].startswith(b[-1]) or b[-1].startswith(a[-1])
    short, long = (a, b) if len(a) < len(b) else (b, a)
    return all(short[i] == long[i] for i in range(len(short)))


def _apply_markerless_name_merge(
    entries: list[dict[str, Any]], *, name_key: str
) -> list[dict[str, Any]]:
    """Fold marker-less person-name fragments (no BIN/BINTI) onto one canonical
    name. Candidates (those with a distinctive name token) are grouped by their
    given-name prefix, then clustered by run-prefix-compatibility so only true
    truncation/extension siblings of the SAME person merge — distinct people
    sharing a common given name (MUHAMMAD DANIAL AQI vs ...HAK) never fuse.
    See the module comment above. No-op for marked / company / ambiguous /
    non-distinctive entries."""
    by_given: dict[str, list[tuple[int, tuple[str, ...]]]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or name_key not in entry:
            continue
        parsed = _markerless_name_run(entry.get(name_key))
        if parsed is None:
            continue
        given, run = parsed
        by_given.setdefault(given, []).append((idx, run))

    rewrites: dict[int, str] = {}
    for members in by_given.values():
        if len(members) < 2:
            continue
        clusters: list[list[tuple[int, tuple[str, ...]]]] = []
        for m in members:
            for cluster in clusters:
                if all(_markerless_run_compatible(m[1], c[1]) for c in cluster):
                    cluster.append(m)
                    break
            else:
                clusters.append([m])
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            display = " ".join(max((c[1] for c in cluster), key=lambda r: (len(r), sum(map(len, r)))))
            for idx, _run in cluster:
                rewrites[idx] = display

    if not rewrites:
        return entries
    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if idx in rewrites and isinstance(entry, dict):
            new_entry = dict(entry)
            new_entry[name_key] = rewrites[idx]
            out.append(new_entry)
        else:
            out.append(entry)
    return out


def dedup_counterparty_entries(
    entries: list[dict[str, Any]],
    *,
    name_key: str = "counterparty_name",
) -> list[dict[str, Any]]:
    """Merge ledger entries whose cleaned + normalised name matches.

    Cleanup applied in order: ``canonicalise_counterparty_name`` (JANM,
    PLANWORTH GLOBAL), then ``clean_counterparty_name`` (special-bucket
    digit-noise strip). Two entries belong to the same group iff the
    upper-cased, whitespace-collapsed form of their cleaned name is equal.

    Aggregation per merged group:
      * Name: the cleaned form (post-canonicalise + post-strip)
      * Sum: ``total_credits``, ``total_debits``, ``credit_count``,
        ``debit_count``, ``transaction_count``
      * Recompute: ``net_position = total_credits - total_debits``
      * Concat: ``transactions`` (preserves chronological order if input
        is already chronological)
      * Other fields: inherit from the first entry in the group

    Solo entries (group of one) still get name cleanup applied. Entries
    missing ``name_key`` pass through unchanged in original order.
    Preserves the position of each group's earliest occurrence.
    """
    if not entries:
        return entries

    # Pre-pass: fold fragmented natural-person buckets (CIMB name truncation
    # + memo bleed) onto a shared canonical name so the grouping below
    # aggregates them. No-op for non-person / clean-surname entries.
    entries = _apply_person_name_merge(entries, name_key=name_key)
    # Second pre-pass: fold marker-LESS person fragments (names with no
    # BIN/BINTI patronymic — "MOHD KHAWAZATUL FAEZ") keyed on the rare
    # distinctive name token. See _apply_markerless_name_merge.
    entries = _apply_markerless_name_merge(entries, name_key=name_key)

    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    order: list[str] = []
    passthrough: list[tuple[int, dict[str, Any]]] = []

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or name_key not in entry:
            passthrough.append((idx, entry))
            continue
        raw = entry.get(name_key)
        canonical = canonicalise_counterparty_name(raw)
        cleaned = clean_counterparty_name(canonical)
        if isinstance(cleaned, str):
            normalized = _normalise_counterparty_name(cleaned)
        else:
            normalized = ""
        if normalized not in groups:
            groups[normalized] = []
            order.append(normalized)
        groups[normalized].append((cleaned, entry))

    merged: list[dict[str, Any]] = []
    for key in order:
        bucket = groups[key]
        first_cleaned, first_entry = bucket[0]
        new_entry = dict(first_entry)
        new_entry[name_key] = first_cleaned
        if len(bucket) > 1:
            total_cr = sum(float(e.get("total_credits") or 0) for _, e in bucket)
            total_dr = sum(float(e.get("total_debits") or 0) for _, e in bucket)
            cr_count = sum(int(e.get("credit_count") or 0) for _, e in bucket)
            dr_count = sum(int(e.get("debit_count") or 0) for _, e in bucket)
            tx_count = sum(int(e.get("transaction_count") or 0) for _, e in bucket)
            transactions: list[Any] = []
            for _, e in bucket:
                transactions.extend(e.get("transactions") or [])
            new_entry["total_credits"] = round(total_cr, 2)
            new_entry["total_debits"] = round(total_dr, 2)
            new_entry["credit_count"] = cr_count
            new_entry["debit_count"] = dr_count
            new_entry["transaction_count"] = tx_count
            new_entry["net_position"] = round(total_cr - total_dr, 2)
            new_entry["transactions"] = transactions
        merged.append(new_entry)

    # Restore passthrough entries at their original index when possible.
    if not passthrough:
        return merged
    for idx, entry in passthrough:
        insert_at = min(idx, len(merged))
        merged.insert(insert_at, entry)
    return merged


# ---------------------------------------------------------------------------
# Session 9: IBG / DuitNow return pairing (±N business day window).
#
# Detects pairs of outward IBG/DuitNow/GIRO debits and matching credit
# returns that arrive within N business days. The pairs feed:
#   * Flag 13 Data Quality — count of paired returns where the parser
#     did NOT extract a C16 (IBG/GIRO INWARD RETURN) keyword tells you
#     how many returns came in without a keyword, i.e. data-extraction
#     gaps the parser missed.
#   * Future dispatcher — optional augmentation to C16 detection so a
#     return without the literal keyword can still be excluded from
#     net credits when it pairs cleanly with an outward DR.
#
# This function intentionally does NOT mutate the input rows or emit
# category tags — it just identifies pairs. Tagging is the dispatcher's
# job once the rest of the per-row classifier lands.
# ---------------------------------------------------------------------------
_OUTWARD_IBG_DUITNOW_GIRO_RE = re.compile(
    r"\b(?:IBG|DUITNOW|GIRO)\b",
    re.IGNORECASE,
)
_INWARD_OR_RETURN_RE = re.compile(
    r"\b(?:INWARD|INCOMING|RETURN|RETURNED|REFUND)\b",
    re.IGNORECASE,
)


def _is_outward_ibg_duitnow_giro(description: Any) -> bool:
    """True when a description looks like an outward IBG/DuitNow/GIRO transfer.

    A DR row description qualifies when:
      - It contains a word-boundary IBG, DUITNOW, or GIRO token.
      - It does NOT contain inward / return / refund tokens (those rows
        are themselves potential return candidates, not outward sends).

    The caller is expected to also gate on ``debit > 0`` — this helper
    only judges the description shape.
    """
    if not isinstance(description, str):
        return False
    if not _OUTWARD_IBG_DUITNOW_GIRO_RE.search(description):
        return False
    if _INWARD_OR_RETURN_RE.search(description):
        return False
    return True


def _parse_iso_date(value: Any) -> _date | None:
    """Parse a ``YYYY-MM-DD`` string into a ``date``; return None on failure."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return _date.fromisoformat(value[:10])
    except ValueError:
        return None


def _business_days_between(
    start: _date,
    end: _date,
    holidays: frozenset[str],
) -> int:
    """Inclusive-start, inclusive-end count of business days between two dates.

    Returns a non-negative int. ``start > end`` returns 0 (callers should
    check direction themselves before calling). Mon-Fri count as business
    days. Holidays (``YYYY-MM-DD`` strings) are excluded even when they
    fall on a weekday.

    Counting convention: same-day returns 0 (no business days "between"
    the start and end of a same-day pair); adjacent business days
    return 1; etc.
    """
    if end <= start:
        return 0
    count = 0
    current = start + timedelta(days=1)
    while current <= end:
        if current.weekday() < 5 and current.isoformat() not in holidays:
            count += 1
        current += timedelta(days=1)
    return count


def pair_ibg_duitnow_returns(
    transactions: list[dict[str, Any]],
    *,
    max_business_days: int = 5,
    holidays: frozenset[str] = frozenset(),
    amount_tolerance: float = 0.01,
) -> list[dict[str, Any]]:
    """Pair outward IBG/DuitNow/GIRO debits with matching credit returns.

    Pairing criteria:
      * DR row: ``debit > 0`` AND description matches an IBG / DUITNOW /
        GIRO outward pattern AND does NOT contain inward/return tokens.
      * CR row: ``credit > 0`` AND amount matches the DR within
        ``amount_tolerance`` AND CR date is the same as or after the DR
        date, within ``max_business_days`` business days.

    Pairing is greedy by date: DRs are processed in date order, and each
    DR claims the earliest still-unclaimed qualifying CR. Each CR can be
    matched to at most one DR; each DR to at most one CR.

    Args:
        transactions: canonical-schema rows. Each row needs ``date``
            (ISO ``YYYY-MM-DD``), ``debit`` and ``credit`` (non-negative
            floats), and ``description`` (string).
        max_business_days: window upper bound; CRs more than this many
            business days after the DR are not eligible. Default 5.
        holidays: optional set of ``YYYY-MM-DD`` strings to exclude from
            the business-day count (e.g. Malaysian public holidays).
        amount_tolerance: absolute float tolerance for amount matching.
            Default ``0.01`` (one cent).

    Returns:
        list of pair dicts, each with:
            ``dr_index`` (int), ``cr_index`` (int) — indices into the
                input ``transactions`` list.
            ``amount`` (float) — the DR's debit amount, rounded to 2dp.
            ``dr_date`` (str), ``cr_date`` (str) — ISO dates.
            ``business_days_apart`` (int) — 0 for same-day, up to
                ``max_business_days``.

        Pairs are sorted by ``dr_index`` ascending. The function never
        mutates the input rows.
    """
    drs: list[tuple[int, _date, float]] = []
    crs: list[tuple[int, _date, float]] = []
    for index, row in enumerate(transactions):
        d = _parse_iso_date(row.get("date"))
        if d is None:
            continue
        debit = row.get("debit") or 0.0
        credit = row.get("credit") or 0.0
        try:
            debit = float(debit)
            credit = float(credit)
        except (TypeError, ValueError):
            continue
        if debit > 0 and _is_outward_ibg_duitnow_giro(row.get("description")):
            drs.append((index, d, debit))
        elif credit > 0:
            crs.append((index, d, credit))

    drs.sort(key=lambda t: (t[1], t[0]))
    crs.sort(key=lambda t: (t[1], t[0]))

    consumed_cr_indices: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for dr_index, dr_date, dr_amount in drs:
        for cr_index, cr_date, cr_amount in crs:
            if cr_index in consumed_cr_indices:
                continue
            if cr_date < dr_date:
                continue
            # Round diff to 2dp first so float artefacts like
            # ``100.00 - 99.99 == 0.0100000000000051`` don't push a
            # legitimately-equal pair past a 0.01 cent tolerance.
            if round(abs(cr_amount - dr_amount), 2) > amount_tolerance:
                continue
            days_apart = _business_days_between(dr_date, cr_date, holidays)
            if days_apart > max_business_days:
                continue
            pairs.append(
                {
                    "dr_index": dr_index,
                    "cr_index": cr_index,
                    "amount": round(dr_amount, 2),
                    "dr_date": dr_date.isoformat(),
                    "cr_date": cr_date.isoformat(),
                    "business_days_apart": days_apart,
                }
            )
            consumed_cr_indices.add(cr_index)
            break

    pairs.sort(key=lambda p: p["dr_index"])
    return pairs


# ---------------------------------------------------------------------------
# Session 10: Flag 13 Data Quality wiring — extraction-gap signal from
# IBG/DuitNow return pairs whose CR description does NOT contain a C16
# inward-return keyword. Such pairs indicate the parser observed the
# return on the credit side but lost the "INWARD RETURN" keyword token,
# so the row would not be excluded from net credits via the C16 path.
#
# This function composes `pair_ibg_duitnow_returns` (session 9) with the
# `INWARD_RETURN_RE` keyword (session 7 C16) — no new pairing logic, no
# row mutation. The count it returns feeds `compute_data_completeness`
# via its new `unkeyworded_return_pair_count` parameter, which flips
# `data_completeness` to ``INCOMPLETE`` and surfaces a remark that the
# existing Flag 13 reducer in `compute_risk_flags` consumes.
# ---------------------------------------------------------------------------


def compute_unkeyworded_return_pair_count(
    transactions: list[dict[str, Any]],
    *,
    max_business_days: int = 5,
    holidays: frozenset[str] = frozenset(),
    amount_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Count IBG/DuitNow/GIRO return pairs missing a C16 inward-return keyword.

    Runs ``pair_ibg_duitnow_returns`` over ``transactions``, then for each
    pair inspects the credit-side row at ``cr_index``: if its description
    does NOT match ``INWARD_RETURN_RE`` (the LOCKED v3.5 C16 keyword),
    the pair is counted as an extraction gap. A return-of-funds-shape pair
    without the keyword tells you the parser lost the "INWARD RETURN"
    token — the row will not be excluded from net credits via the C16
    classifier even though structurally it is a returned outward transfer.

    Pairing kwargs (``max_business_days`` / ``holidays`` /
    ``amount_tolerance``) are forwarded verbatim so the caller can keep
    the same business-day window the rest of the pipeline uses.

    Output is consumed by ``compute_data_completeness``'s new
    ``unkeyworded_return_pair_count`` parameter, which in turn drives
    Flag 13 (Data Quality) via the existing summary fields the Flag 13
    reducer in ``compute_risk_flags`` already reads.

    Corpus-observed false-positive rate (2026-05-12): the s9 pairing
    function has NO CR-side keyword filter — pairing is purely structural
    on amount + same-or-after date within the business-day window.
    Legitimate same-amount inter-account round-trip business activity
    therefore also pairs (and is counted here as "unkeyworded"). On a
    spot-check of corpus full_report.json files, this surfaced:
      * Real extraction gaps the C16 regex missed — e.g. Plentitude
        ``DuitNow TRF /MISC CREDIT, BOND MOBILE SOLUTION, refund, IBG``
        is a legitimate refund whose keyword ("refund") is not in the
        LOCKED C16 regex (which only matches ``IBG INWARD RETURN`` /
        ``GIRO INWARD RETURN``). That's the signal Flag 13 wants.
      * Same-amount same-day inter-account transfers — e.g. UOB GWE
        Food Pack BOREINTERNATIONAL <-> GOODWILL EVEREST flows. Not
        returns, but match the structural pairing rule.
    Tightening either side (CR keyword gate, counterparty disjointness,
    or RP / own-party exclusion) is a future refinement and would be a
    deliberate change to s9 + this function. Until then Flag 13 is a
    "look at the unkeyworded pairs" attention signal rather than a hard
    "C16 was missed" assertion.

    Args:
        transactions: canonical-schema rows. Reads ``date``, ``debit``,
            ``credit``, ``description``.
        max_business_days: pairing window upper bound. Default 5.
        holidays: optional set of ``YYYY-MM-DD`` strings to exclude from
            the business-day count.
        amount_tolerance: absolute float tolerance for amount matching.
            Default ``0.01``.

    Returns:
        dict:
            unkeyworded_return_pair_count (int) — number of pairs whose
                CR description does NOT match ``INWARD_RETURN_RE``.
            unkeyworded_return_pair_entries (list[dict]) — one record
                per unkeyworded pair, copying the pair shape from
                ``pair_ibg_duitnow_returns`` plus ``cr_description`` for
                downstream diagnostics. Sorted by ``dr_index`` ascending.
    """
    pairs = pair_ibg_duitnow_returns(
        transactions,
        max_business_days=max_business_days,
        holidays=holidays,
        amount_tolerance=amount_tolerance,
    )

    entries: list[dict[str, Any]] = []
    for pair in pairs:
        cr_row = transactions[pair["cr_index"]]
        description = str(cr_row.get("description") or "")
        if INWARD_RETURN_RE.search(description):
            continue
        entries.append({**pair, "cr_description": description})

    return {
        "unkeyworded_return_pair_count": len(entries),
        "unkeyworded_return_pair_entries": entries,
    }


# ---------------------------------------------------------------------------
# Session 13 — per-row dispatcher (Slice A).
#
# Source of truth: CLASSIFICATION_RULES_v3_5.json
#   ``global_rules.classification_order`` (LOCKED).
#
# Slice A wires the UNBLOCKED categories only. Remaining blocked items
# (after RP foundation slices 1-3 landed in s17-s19) are TODO hooks for
# the next dispatcher slice once their dependencies land:
#
#   * C12 — FD / interest credit. No Track 2 detector ported yet.
#
# C14/C15 (returned cheques in/out) and C16 (inward return) and C17-C20
# (cash/cheque) and C24 (bank fees) all share the same shape: a regex
# match plus the correct side. The dispatcher inlines that check rather
# than calling each ``compute_*`` because the ``compute_*`` functions
# produce aggregate entry lists; calling them per-row would be O(N²).
# The regex constants are imported by name from each detector section
# above and reused here, so any v3.5 keyword tightening updates both
# paths automatically.
#
# Counterparty-driven C26/C27 are wired conditionally: when the caller
# extracts a counterparty name and threads it via the ``counterparty_name``
# kwarg, the dispatcher applies ``has_corporate_suffix`` /
# ``has_natural_person_marker`` to decide. When no name is provided (the
# common case until per-row counterparty extraction is wired into the
# Track 2 pipeline), C26/C27 simply do not fire and the row falls through
# to unclassified — which per v3.5 ``no_unknown_bucket`` stays in net
# credits/debits.
# ---------------------------------------------------------------------------

CANONICAL_CATEGORIES: tuple[str, ...] = (
    "C25", "C01", "C02", "C05", "C03", "C04",
    "C06", "C07", "C08", "C09",
    "C10", "C11", "C12",
    "C13", "C14", "C15", "C16",
    "C17", "C18", "C19", "C20",
    "C24", "C26", "C27",
)

DISPATCHER_BLOCKED_CATEGORIES: frozenset[str] = frozenset({"C12"})

_BALANCE_ROW_RE = re.compile(
    r"^\s*(?:OPENING|CLOSING|PREVIOUS|STATEMENT)\s+BALANCE\b"
    r"|^\s*BAL(?:ANCE)?\s*[BC]/F\b"
    r"|^\s*B/F\b|^\s*C/F\b"
    r"|^\s*(?:BROUGHT|CARRIED)\s+FORWARD\b",
    re.IGNORECASE,
)


def _is_balance_row(row: dict[str, Any]) -> bool:
    """Detect C25 balance / opening / closing rows.

    Parser-set markers (``is_opening_balance`` / ``is_statement_balance``)
    are authoritative when present. Description-regex fallback covers
    parsers that emit a synthetic balance row without marker fields.
    """
    if row.get("is_opening_balance") or row.get("is_statement_balance"):
        return True
    description = row.get("description")
    if isinstance(description, str) and _BALANCE_ROW_RE.search(description):
        return True
    return False


# Counterparty-bucket → category mapping. The parser (app.py) stamps
# certain rows with synthetic bucket names like "CHEQUE DEPOSIT" or
# "BULK SALARY" instead of an extracted counterparty entity. When that
# happens the bucket name itself encodes the classification — e.g. PBB
# DEP-ECP rows are stamped CHEQUE DEPOSIT and should route to C19 even
# though the description carries no CHEQUE keyword.
#
# Mirrors Track 1's BUCKET_TO_CATEGORY at kredit_lab_classify.py L99-114
# verbatim. Paired with ``_CATEGORY_SIDES`` for side-validation per
# Track 1 L737-740 (a CR row mis-bucketed into BULK SALARY would
# otherwise fire C05 here too; the side check rejects it).
_BUCKET_TO_CATEGORY: dict[str, str] = {
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

# Canonical side for each category. Used by the bucket-direct rung to
# reject rows that were mis-bucketed onto the wrong side. ``ANY`` allows
# either side. Mirrors Track 1's ``_CATEGORY_SIDES`` at
# kredit_lab_classify.py L137-146.
_CATEGORY_SIDES: dict[str, str] = {
    "C01": "CR", "C02": "DR", "C03": "CR", "C04": "DR",
    "C05": "DR", "C06": "DR", "C07": "DR", "C08": "DR", "C09": "DR",
    "C10": "CR", "C11": "DR", "C12": "CR", "C13": "CR",
    "C14": "DR", "C15": "CR", "C16": "CR",
    "C17": "CR", "C18": "DR", "C19": "CR", "C20": "DR",
    "C24": "ANY",
    "C25": "ANY",
    "C26": "CR", "C27": "DR",
}


def dispatch_transaction(
    row: dict[str, Any],
    *,
    counterparty_name: str | None = None,
    company_names: list[str] | None = None,
    related_parties: list[str] | None = None,
    factoring_entities: list[str] | None = None,
) -> dict[str, Any]:
    """Apply the v3.5 ``classification_order`` priority ladder to one row.

    Slice A wires the unblocked categories: C25, C05, C06-C09, C13-C20, C24,
    plus C26/C27 when ``counterparty_name`` is supplied. C01/C02 fires in
    two tiers: (a) parser-stamped marker — counterparty_name carrying the
    ``(OWN-PARTY)`` suffix; (b) company-root match — when ``company_names``
    is supplied, any root of length ≥ 5 appearing in the counterparty
    bucket or description fires the row. C03/C04 fires when
    ``related_parties`` supplies an analyst-confirmed (or auto-confirmed
    HIGH) name and either the counterparty bucket OR description contains
    it (case-insensitive substring match, mirrors Track 1). C10 fires on
    the v3.5 tier-1 keyword regex OR on a factoring-entity match in
    ``factoring_entities``. C11 fires on the v3.5 LOCKED keyword regex
    with a BANK_FEES_RE short-circuit (OTHER TRANSFER FEE → C24 not C11).
    C12 (FD / interest) remains blocked — the row falls through to the
    next rung.

    Per the v3.5 ``jompay_rule``, biller-code-only JomPAY rows are
    suppressed before C06-C09 keyword matching (and would be before C11
    if it ever needed it; the LOAN_REPAYMENT_RE list has no JOMPAY
    overlap so the guard is not required at the C11 rung). The check
    uses the s8 ``is_jompay_biller_code_only`` predicate (matches only
    when ``JOMPAY`` is the leading token and no entity name is otherwise
    visible).

    Args:
        row: canonical-schema transaction row.
        counterparty_name: optional pre-extracted counterparty name. When
            provided, drives C26/C27 trade-in/out classification via
            ``has_corporate_suffix`` / ``has_natural_person_marker``, and
            participates in the non-marker C01/C02 own-party rung.
        company_names: statement-owner company names (typically
            ``data["summary"]["company_names"]``). When provided each
            entry is normalised via ``_company_root`` and the dispatcher
            fires C01/C02 on rows whose counterparty or description
            matches any root (length ≥ 5). Marker-stamped rows are
            handled separately and unconditionally.
        related_parties: case-insensitive name list (analyst-confirmed
            plus auto-confirmed HIGH-score candidates from
            ``scan_related_party_candidates``). Each entry is upper-cased
            once per dispatch call and substring-matched against the
            counterparty bucket and description; CR side fires C03, DR
            side fires C04. Per v3.5 ``classification_order``, RP runs
            AFTER own-party and salary so a director who's also been
            paid a "SALARY" stays C05, and an own-party transfer stays
            C01/C02 even if the director name appears in
            ``related_parties``.
        factoring_entities: analyst-confirmed factoring company names. When
            provided, every entry's upper-cased form is substring-matched
            against the counterparty bucket and description on CR-side
            rows; a hit fires C10 (tier-2 factoring rule). Mirrors Track 1's
            ``decisions.factoring_entities`` flow (kredit_lab_classify.py
            L707-745) — no separate ADVANCE-keyword gate, the analyst-
            confirmed list is treated as authoritative.

    Returns:
        dict with four keys:
            primary: category string ``"C##"`` or ``None`` when no rung
                matched. ``None`` means the row is unclassified per v3.5
                ``no_unknown_bucket`` (stays in net credits/debits).
            side: ``"CR"`` / ``"DR"`` / ``None``.
            reason: short human-readable rationale.
            mode: ``"FULL_CODE"`` for keyword/regex matches;
                ``"HEURISTIC"`` for trade-in/out marker-based fires;
                ``None`` when primary is None.
    """
    description = str(row.get("description") or "")
    side = _row_side(row)

    def _result(
        primary: str | None,
        reason: str,
        mode: str | None = "FULL_CODE",
    ) -> dict[str, Any]:
        return {
            "primary": primary,
            "side": side,
            "reason": reason,
            "mode": mode if primary else None,
        }

    # C25 — balance / opening / closing row. Wins outright per v3.5 order.
    if _is_balance_row(row):
        return _result("C25", "Statement / opening / closing balance row")

    # C01 / C02 — own-party (parser-stamped marker subset). The upstream
    # counterparty_ledger pipeline appends "(OWN-PARTY)" to names whose
    # root matches the statement's company. We honour that stamp directly:
    # CR → C01, DR → C02. Runs BEFORE bucket / keyword rungs so own-account
    # transfers carrying e.g. "PAYMENT" or trade keywords don't mis-route
    # (notably C26/C27, which would otherwise capture them on the corporate-
    # suffix rung).
    if (
        side
        and counterparty_name
        and OWN_PARTY_MARKER_RE.search(counterparty_name)
    ):
        if side == "CR":
            return _result(
                "C01",
                f"Own-party (parser-stamped marker): {counterparty_name}",
            )
        return _result(
            "C02",
            f"Own-party (parser-stamped marker): {counterparty_name}",
        )

    # C01 / C02 — own-party (company-root match, non-marker subset). Fires
    # when ``company_names`` supplies the statement-owner's company and
    # either the counterparty bucket name OR transaction description
    # literally contains a root of length ≥ 5. Sits AFTER the marker rung
    # so a stamped row never reaches this path; sits BEFORE bucket /
    # keyword rungs so an own-account row carrying "PAYMENT" / trade
    # keywords doesn't fall through to C26/C27. RP3 scanner + C03/C04
    # related-party stay blocked on the rest of the RP foundation sprint.
    if side and company_names:
        cp_upper = (counterparty_name or "").upper()
        desc_upper = description.upper()
        company_roots = [_company_root(c) for c in company_names if c]
        if _own_party_match(cp_upper, desc_upper, company_roots):
            # Memo-echo guard: a third-party LOAN disbursement / repayment
            # routed through a financier or trustee commonly echoes the
            # borrower's own name in the memo (e.g. "FUNDING SOCIETES ZAIM
            # EXPRESS SDN BHD" paid by MALAYSIAN TRUSTEES). When the company
            # root appears ONLY in the description (not in the extracted
            # counterparty) AND the row carries an explicit loan keyword, the
            # row is a facility movement, not an own-account transfer — let it
            # fall through to the C10/C11 rung. A genuine own-account transfer
            # never carries LOAN DISB / FUNDING SOCIET / TERM LOAN / HP LOAN.
            #
            # Gated on a REAL extracted third-party counterparty: the company
            # settling its OWN loan ("TR IBG <OWN CO> Term loan", no separate
            # counterparty) stays C02 per the v3.5 dual-tag rule — only a row
            # naming a distinct non-own counterparty (the trustee / financier)
            # diverts to the facility rung.
            cp_owns = _own_party_match(cp_upper, "", company_roots)
            cp_present = bool(cp_upper.strip())
            is_loan_memo = (
                LOAN_DISBURSEMENT_RE.search(description)
                or LOAN_REPAYMENT_RE.search(description)
            )
            if not (cp_present and not cp_owns and is_loan_memo):
                label = counterparty_name or description
                if side == "CR":
                    return _result(
                        "C01",
                        f"Own-party (company-root match): {label}",
                    )
                return _result(
                    "C02",
                    f"Own-party (company-root match): {label}",
                )

    # C05 — salary (FULL_CODE branch, predicate already gates own-account
    # via OWN_ACCOUNT_BLOCK_RE and commission_policy via COMMISSION_BLOCK_RE).
    if is_salary_payment(row):
        return _result("C05", "Salary keyword + commission/own-account guard")

    # C03 / C04 — related-party. Fires when ``related_parties`` supplies a
    # name (analyst-confirmed or auto-confirmed HIGH from the RP3 scanner)
    # whose upper-cased form appears as a substring in either the
    # counterparty bucket OR the description. CR → C03, DR → C04. Runs
    # AFTER own-party (a director who shows up on the own-account stays
    # C01/C02) and AFTER salary (a "SALARY" payment to a director stays
    # C05). Mirrors Track 1's order at L730-732.
    if side and related_parties:
        cp_upper = (counterparty_name or "").upper()
        desc_upper = description.upper()
        related_upper = [r.upper() for r in related_parties if r]
        if any(r in cp_upper or r in desc_upper for r in related_upper):
            if side == "CR":
                return _result(
                    "C03",
                    "Related-party (analyst-confirmed or auto-confirmed HIGH)",
                )
            return _result(
                "C04",
                "Related-party (analyst-confirmed or auto-confirmed HIGH)",
            )

    # Bucket-direct dispatch — when the parser-stamped counterparty bucket
    # matches a known bucket name, route straight to the mapped category.
    # Mirrors Track 1's BUCKET_TO_CATEGORY shortcut (kredit_lab_classify.py
    # L99-114 + L736-741). Two behaviours this rung enables that the
    # keyword-only rungs miss:
    #   * PBB DEP-ECP / DR-ECP: app.py's PBB-gated branch stamps these as
    #     CHEQUE DEPOSIT / CHEQUE ISSUE; the description carries no CHEQUE
    #     keyword so without bucket dispatch the rows stay UNCLASSIFIED.
    #   * Cross-bank opcode-coded rows where the parser identified the
    #     bucket but the description is shorthand (no keyword to match).
    # Runs AFTER own-party + salary + RP per Track 2's design choice
    # (salary > bucket > keyword); BEFORE all keyword rungs so the bucket
    # name beats stale-keyword false positives (e.g. BANK FEES bucket
    # carrying a "TERM LOAN" memo would otherwise fire C11 instead of C24).
    if counterparty_name and side:
        cp_upper_full = counterparty_name.upper().strip()
        candidate = _BUCKET_TO_CATEGORY.get(cp_upper_full)
        if candidate is not None:
            expected = _CATEGORY_SIDES.get(candidate, "ANY")
            if expected == "ANY" or expected == side:
                return _result(
                    candidate, f"Counterparty bucket: {cp_upper_full}"
                )

    # C06 - C09 — statutory (DR side, with JomPAY-biller-only short-circuit).
    if side == "DR" and not is_jompay_biller_code_only(description):
        if EPF_PAYMENT_RE.search(description):
            return _result("C06", "EPF / KWSP keyword")
        if SOCSO_PAYMENT_RE.search(description):
            return _result("C07", "SOCSO / PERKESO keyword")
        if LHDN_TAX_PAYMENT_RE.search(description):
            return _result("C08", "LHDN / Lembaga Hasil keyword")
        if HRDF_PAYMENT_RE.search(description):
            return _result("C09", "HRDF / PSMB keyword")

    # C10 — loan disbursement / factoring (CR side). Two routes:
    #   1. Tier-1 keyword regex (``LOAN_DISBURSEMENT_RE``): LOAN DISB,
    #      FINANCING DISB, TRADE FIN, SCF TRADE, FACTORING, INVOICE FIN /
    #      DISCOUNT, BILL PURCHAS / DISCOUNT, BANKERS ACCEPTANCE, FACILITY
    #      DRAWDOWN.
    #   2. Tier-2 factoring rule — when ``factoring_entities`` supplies an
    #      analyst-confirmed entity name and that name appears as a
    #      substring of either the counterparty bucket OR the description,
    #      the row fires C10. Mirrors Track 1's dispatcher rung at
    #      L742-745 verbatim: the analyst-confirmed list is treated as
    #      authoritative, so no separate ADVANCE-keyword gate is applied
    #      (Track 2 follows Track 1 over v3.5's stricter planworth_rule).
    if side == "CR":
        if LOAN_DISBURSEMENT_RE.search(description):
            return _result("C10", "Loan disbursement / factoring keyword")
        if factoring_entities:
            factoring_upper = [f.upper() for f in factoring_entities if f]
            if factoring_upper:
                cp_upper = (counterparty_name or "").upper()
                desc_upper = description.upper()
                if any(f in cp_upper or f in desc_upper for f in factoring_upper):
                    return _result(
                        "C10",
                        "Factoring disbursement (analyst-confirmed entity)",
                    )

    # C11 — loan repayment (DR side). Fires on the v3.5 LOCKED regex (TERM
    # LOAN / LOAN REPAY / FINANCING REPAY / MONTHLY INSTALMENT / IB2G DR
    # CA CR LN / TRANSFER TO LOAN / DD CASA PYMT / FINPAL ISSUER REPAYM).
    # Priority guards enforced here because Track 2 has no
    # ``BUCKET_TO_CATEGORY`` shortcut:
    #   * OTHER TRANSFER FEE precedence — BANK_FEES_RE matches "OTHER
    #     TRANSFER FEE Term loan" and similar; per v3.5 line 817 those
    #     rows are ALWAYS C24, not C11. Track 1 absorbs them via the
    #     BANK FEES bucket firing BEFORE description-keyword. Without a
    #     bucket map, Track 2 short-circuits C11 here when BANK_FEES_RE
    #     also matches the description.
    #   * Director's personal loan (related-party + Instalment) → C04,
    #     not C11 (v3.5 line 831). Naturally satisfied — the C03/C04
    #     rung above runs first and returns.
    #   * Account-number-only sub-rule (v3.5 ``account_number_only_rule_
    #     v3_5_3``, e.g. ``TRANSFER TO LOAN 12345678L``) — fires as
    #     standalone C11 here because the C01/C02 company-root rung
    #     above CANNOT match an account-number-only description (the
    #     root is alphabetic). Track 2 emits a single primary so the
    #     v3.5 "C11 standalone, NOT C02+C11" wording is naturally met.
    if (
        side == "DR"
        and LOAN_REPAYMENT_RE.search(description)
        and not BANK_FEES_RE.search(description)
    ):
        return _result("C11", "Loan repayment keyword")

    # C11 — finance-institution counterparty (DR side, name-based). A debit
    # paid to a counterparty whose NAME is a "<brand> CREDIT/CAPITAL/LEASING/
    # FINANCE" financier and is not a natural person is the company servicing
    # its OWN facility — vehicle / asset hire-purchase, leasing, or a term
    # loan. These rows routinely carry only a contract / agreement number in
    # the memo (no LOAN keyword), so the keyword rung above misses them — the
    # whole Scania Credit HP book on Zaim Express (~RM533k, 22 rows) was
    # invisible. The bank often truncates the corporate suffix (``SCANIA CREDIT
    # (MALA``), so this keys on the lender token, not on a SDN BHD suffix. The
    # exclusion guard drops insurance / takaful premiums and credit-guarantee
    # fees (operating expense, not debt) plus raw-extraction noise (ATM / GIRO
    # / SERVICE CHARGE rows where "CREDIT" only appears incidentally). Runs
    # AFTER the keyword rung (keyword wins for precision) and AFTER C03/C04 (a
    # director settling a financier on the company's behalf stays related-
    # party). BANK_FEES_RE → C24 short-circuit kept for finance-co fee rows.
    if (
        side == "DR"
        and counterparty_name
        and _FINANCIER_C11_RE.search(counterparty_name)
        and not _FINANCIER_C11_EXCLUDE_RE.search(counterparty_name)
        and not has_natural_person_marker(counterparty_name)
        and not BANK_FEES_RE.search(description)
    ):
        return _result(
            "C11",
            f"Loan repayment — finance-institution counterparty: {counterparty_name}",
            mode="HEURISTIC",
        )

    # C12 — FD / interest credit. BLOCKED (no Track 2 detector ported yet).

    # C13 — reversal credit (CR side).
    if side == "CR" and REVERSAL_CREDIT_RE.search(description):
        return _result("C13", "Reversal credit keyword")

    # C14 / C15 — returned cheque (side discriminates inward vs outward).
    if RETURNED_CHEQUE_RE.search(description):
        if side == "DR":
            return _result("C14", "Returned cheque — inward (deposit bounced)")
        if side == "CR":
            return _result("C15", "Returned cheque — outward (issued bounced)")

    # C16 — inward return (CR side; outward transfer rejected and returned).
    if side == "CR" and INWARD_RETURN_RE.search(description):
        return _result("C16", "Inward return keyword")

    # C17 / C18 — cash deposit (CR) / cash withdrawal (DR).
    if side == "CR" and CASH_DEPOSIT_RE.search(description):
        return _result("C17", "Cash deposit keyword")
    if side == "DR" and CASH_WITHDRAWAL_RE.search(description):
        return _result("C18", "Cash withdrawal keyword")

    # C19 / C20 — cheque deposit (CR) / cheque issue (DR).
    if side == "CR" and CHEQUE_DEPOSIT_RE.search(description):
        return _result("C19", "Cheque deposit keyword")
    if side == "DR" and CHEQUE_ISSUE_RE.search(description):
        return _result("C20", "Cheque issue keyword")

    # C24 — bank fees (DR side).
    if side == "DR" and BANK_FEES_RE.search(description):
        return _result("C24", "Bank fees keyword")

    # C27 — Bank Rakyat JomPay utility merchant (DR side). Sits between C24
    # and the C26/C27 corporate-counterparty fallback so utility billers
    # whose truncated concat token isn't visible to the corporate-suffix
    # detector still route to trade-expense. See ``_BR_JOMPAY_UTILITY_RE``
    # for the cross-bank scope rationale.
    if side == "DR" and _BR_JOMPAY_UTILITY_RE.search(description):
        return _result(
            "C27",
            "BR JomPay utility merchant (DiGi/TNB/Pengurusan Air/Indah Water/TM)",
        )

    # C26 / C27 — trade income / trade expense. Requires a counterparty name
    # (parser- or caller-extracted) carrying a corporate suffix without a
    # natural-person marker. The ghost-verb shapes ("TRANSFER FR A/C" etc.)
    # are excluded inside ``has_corporate_suffix`` already, so no extra
    # guard needed here.
    if (
        side
        and counterparty_name
        and has_corporate_suffix(counterparty_name)
        and not has_natural_person_marker(counterparty_name)
    ):
        if side == "CR":
            return _result(
                "C26", f"Trade income — corporate counterparty: {counterparty_name}",
                mode="HEURISTIC",
            )
        return _result(
            "C27", f"Trade expense — corporate counterparty: {counterparty_name}",
            mode="HEURISTIC",
        )

    # Unclassified — stays in net credits/debits per v3.5 ``no_unknown_bucket``.
    return _result(None, "No rung matched")


def classify_transactions(
    transactions: list[dict[str, Any]],
    *,
    counterparty_lookup: dict[int, str] | None = None,
    company_names: list[str] | None = None,
    related_parties: list[str] | None = None,
    factoring_entities: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run ``dispatch_transaction`` across every row in ``transactions``.

    Returns shallow copies of each row with a ``classification`` sub-dict
    attached (shape: ``{"primary", "side", "reason", "mode"}``). The
    input list is not mutated.

    Args:
        transactions: canonical-schema rows.
        counterparty_lookup: optional ``index -> counterparty name`` map.
            Rows whose enumerated index appears here get their entry
            passed to ``dispatch_transaction(counterparty_name=...)``,
            enabling the C26/C27 branch. When the lookup is absent the
            dispatcher's C26/C27 rungs simply do not fire and rows fall
            through to unclassified.
        company_names: threaded through to ``dispatch_transaction``'s
            C01/C02 own-party company-root rung (RP foundation Slice 1).
        related_parties: threaded through to ``dispatch_transaction``'s
            C03/C04 rung (RP foundation Slice 2). Analyst-confirmed names
            plus auto-confirmed HIGH candidates from
            ``scan_related_party_candidates`` should be merged before the
            call; ``build_track2_result`` does this automatically.
        factoring_entities: threaded through to ``dispatch_transaction``'s
            C10 tier-2 factoring rung (RP foundation Slice 3).

    Returns:
        list of row dicts, in original order, each with a
        ``classification`` field.
    """
    counterparty_lookup = counterparty_lookup or {}
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(transactions):
        counterparty_name = counterparty_lookup.get(idx)
        classification = dispatch_transaction(
            row,
            counterparty_name=counterparty_name,
            company_names=company_names,
            related_parties=related_parties,
            factoring_entities=factoring_entities,
        )
        out.append({**row, "classification": classification})
    return out


# ---------------------------------------------------------------------------
# Session 14 — Dispatcher Slice B: counterparty-lookup wiring.
#
# ``build_counterparty_ledger`` (in ``app.py``) groups canonical transactions
# by extracted counterparty name and returns a ``counterparty_ledger`` dict.
# The per-row dispatcher needs the same name attached back to each row's
# enumeration index so the C26/C27 trade-in/out rung can fire on real corpus
# data. This helper performs the join.
#
# Design notes:
#   * The join key is ``(date, description, rounded amount, "CREDIT"|"DEBIT")``
#     — the same shape ``build_counterparty_ledger`` emits per ledger
#     transaction. Two canonical rows that share this key map to the SAME
#     counterparty group by construction, so collisions are not a correctness
#     issue (any duplicate fingerprint resolves to the same name).
#   * Synthetic / protected labels are dropped at the build stage instead of
#     being filtered inside ``has_corporate_suffix``. This keeps the lookup
#     semantic: "this row has an identified third-party counterparty name".
#     The dispatcher's existing C26/C27 guard
#     (``has_corporate_suffix and not has_natural_person_marker``) is left
#     untouched.
#   * The helper is intentionally ledger-shape-only — it does NOT call
#     ``build_counterparty_ledger`` itself. Callers thread the upstream
#     ledger in. This preserves Track 2's "no app.py / no Track 1" import
#     hygiene and lets scripts feed a hand-built ledger for testing.
# ---------------------------------------------------------------------------


# Synthetic / protected counterparty labels that should NOT be threaded into
# the dispatcher as a real counterparty name. Mirrors the relevant portion of
# app.py's ``_OWN_PARTY_PROTECTED_LABELS`` (snapshot 2026-05-13) plus the
# rail-agnostic UNNAMED <BANK> TRANSFER (CR|DR) and CARD POS (...) families
# which are matched by the regex below rather than enumerated. If app.py adds
# new operational buckets that contain a corporate suffix in the literal
# (e.g. a hypothetical "LOAN REPAYMENT BERHAD"), extend this set OR the regex
# — without that, ``has_corporate_suffix`` would accept the bucket label as a
# real corporate counterparty and the C26/C27 rung would fire spuriously.
_SYNTHETIC_COUNTERPARTY_LABELS: frozenset[str] = frozenset(
    {
        "UNIDENTIFIED",
        "UNCATEGORIZED",
        "CASH DEPOSIT",
        "CASH WITHDRAWAL",
        "BANK FEES",
        "BULK SALARY",
        "FD/INTEREST",
        "LOAN REPAYMENT",
        "LOAN DISBURSEMENT",
        "KWSP",
        "SOCSO",
        "LHDN",
        "HRDF",
        "REVERSAL",
        "RETURNED CHEQUE",
        "INWARD RETURN",
        "JANM",
        "APAYLATER",
    }
)

# Names carried on the synthetic/protected label set for *classification*
# purposes (so they stay out of the C26/C27 counterparty name-map and keep
# their existing bucket) but which are nonetheless real external entities that
# SHOULD appear in top_parties ranking. JANM (Jabatan Akauntan Negara — the
# federal Accountant-General / government paymaster) is the canonical case:
# for a government contractor it is the single largest real payer, and its
# sibling channel "AKAUNTAN NEGARA" already ranks, so suppressing JANM was
# inconsistent. Allow-listing here is ranking-visibility ONLY — these names
# do NOT re-enter the classification name-map, so their rows stay in whatever
# bucket the dispatcher already assigned (e.g. unclassified). Matched on the
# normalised (upper-cased, whitespace-collapsed) name. See
# ``_build_top_parties_track2``.
_RANKABLE_DESPITE_SYNTHETIC: frozenset[str] = frozenset({"JANM"})

# Catches UNNAMED <BANK> TRANSFER (CR|DR), UNNAMED INTERNAL PAYROLL (CR|DR),
# CARD POS (...), Unidentified (Cheque), and any UNIDENTIFIED-prefixed
# variant (UNIDENTIFIED 1234, etc.).
_SYNTHETIC_COUNTERPARTY_RE = re.compile(
    r"^UNNAMED\s+.+?\s+TRANSFER\s*\((?:CR|DR)\)\s*$"
    r"|^UNNAMED\s+INTERNAL\s+PAYROLL\s*\((?:CR|DR)\)\s*$"
    r"|^CARD\s+POS\s*\([A-Z]+\)\s*$"
    r"|^UNIDENTIFIED(?:\s.*)?$"
    r"|^Unidentified\s+\(Cheque\)\s*$",
    re.IGNORECASE,
)


def _is_synthetic_counterparty_label(name: Any) -> bool:
    """Return True when ``name`` is an upstream synthetic / protected bucket
    label rather than an identifiable third-party counterparty."""
    if not isinstance(name, str):
        return True
    stripped = name.strip()
    if not stripped:
        return True
    if stripped.upper() in _SYNTHETIC_COUNTERPARTY_LABELS:
        return True
    if _SYNTHETIC_COUNTERPARTY_RE.match(stripped):
        return True
    return False


def _ledger_join_key(row: dict[str, Any]) -> tuple[Any, Any, float, str | None]:
    """Derive a ledger-shape join key from a canonical transaction row.

    Matches the per-ledger-transaction shape emitted by
    ``build_counterparty_ledger`` in app.py:
    ``(date, description, rounded(max(debit, credit), 2), "CREDIT"|"DEBIT")``.

    Balance-only rows (both sides zero) get a ``None`` type slot and are
    therefore unmatchable against the ledger — which is correct: the upstream
    ledger skips them too.
    """
    debit = float(row.get("debit") or 0.0)
    credit = float(row.get("credit") or 0.0)
    if credit > 0:
        return (row.get("date"), row.get("description"), round(credit, 2), "CREDIT")
    if debit > 0:
        return (row.get("date"), row.get("description"), round(debit, 2), "DEBIT")
    return (row.get("date"), row.get("description"), 0.0, None)


def build_counterparty_lookup_track2(
    transactions: list[dict[str, Any]],
    counterparty_ledger: dict[str, Any] | None,
    *,
    include_synthetic: bool = False,
) -> dict[int, str]:
    """Join an upstream ``counterparty_ledger`` back to per-row enumeration
    indices for use as the ``counterparty_lookup`` kwarg of
    :func:`classify_transactions`.

    Args:
        transactions: canonical-schema rows — the SAME list (in order) that
            the caller will pass to ``classify_transactions``.
        counterparty_ledger: the ``counterparty_ledger`` dict produced by
            ``build_counterparty_ledger`` (or the corresponding section of a
            ``full_report.json`` export). ``None`` / empty / shapeless inputs
            return ``{}``.
        include_synthetic: when True, synthetic / protected bucket labels
            (BANK FEES, FD/INTEREST, CASH DEPOSIT, CHEQUE DEPOSIT, …) are
            kept in the lookup so the dispatcher's bucket-direct rung can
            fire on them. Default False preserves the original "real
            entity name only" contract — used by callers that thread the
            lookup directly to a C26/C27-only consumer. ``build_track2_
            result`` passes True so synthetic bucket → category dispatch
            (Mazaa CHEQUE DEPOSIT → C19, Felcra FD/INTEREST → C12, etc.)
            takes effect. The dispatcher's existing ``has_corporate_
            suffix`` guard prevents spurious C26/C27 firings even when
            synthetic names appear in the lookup, because no synthetic
            label in the current set carries an embedded corporate suffix.

    Returns:
        ``dict[int, str]`` mapping ``enumerate(transactions)`` index →
        counterparty name. With ``include_synthetic=False`` (default), rows
        whose counterparty is a synthetic / protected label (UNIDENTIFIED,
        BULK SALARY, UNNAMED ... TRANSFER, CARD POS, …) or whose ledger
        fingerprint does not match any row are absent from the map.
        Absence is the signal for the dispatcher to skip C26/C27 for that
        row. With ``include_synthetic=True``, all matched names appear —
        the bucket-direct rung relies on this.

    See module-level note above for join semantics and rationale.
    """
    if not isinstance(counterparty_ledger, dict):
        return {}

    counterparties = counterparty_ledger.get("counterparties") or []
    if not counterparties:
        return {}

    name_by_key: dict[tuple[Any, Any, float, str | None], str] = {}
    for cp in counterparties:
        if not isinstance(cp, dict):
            continue
        name = cp.get("counterparty_name")
        if not include_synthetic and _is_synthetic_counterparty_label(name):
            continue
        for ltx in cp.get("transactions") or []:
            if not isinstance(ltx, dict):
                continue
            try:
                amount = round(float(ltx.get("amount") or 0.0), 2)
            except (TypeError, ValueError):
                continue
            key = (
                ltx.get("date"),
                ltx.get("description"),
                amount,
                ltx.get("type"),
            )
            name_by_key[key] = name

    if not name_by_key:
        return {}

    lookup: dict[int, str] = {}
    for idx, row in enumerate(transactions):
        key = _ledger_join_key(row)
        matched = name_by_key.get(key)
        if matched:
            lookup[idx] = matched
    return lookup


# ---------------------------------------------------------------------------
# Session 15 — Dispatcher Slice C Part 1: v6.3.5 result orchestrator.
#
# ``build_track2_result(transactions, ...)`` composes the per-row dispatcher,
# the counterparty-lookup join, and the s1-s12 compute helpers into a single
# top-level dict matching ``BANK_ANALYSIS_SCHEMA_v6_3_5.json``. The result
# passes ``validate_track2_result`` (Draft-7 jsonschema). This is the public
# surface that side-by-side validation against Track 1 will consume.
#
# Part 1 scope (session 15):
#   * All 15 top-level keys present with correct shape and types.
#   * Sections wired from existing compute helpers:
#       - report_info: derived from transactions.
#       - accounts: per-account aggregation.
#       - monthly_analysis: per-(account, month) entries with all 58 keys
#         populated from per-row classification + compute helpers (FX,
#         round-figure, high-value). Categories whose dispatcher rungs are
#         BLOCKED (C01-C04, C10-C12) stay at 0 by construction — when the RP
#         foundation sprint lands, those buckets light up automatically.
#       - consolidated: aggregated across monthly_analysis.
#       - top_parties: top 10 payers / payees from counterparty_ledger.
#       - large_credits: from compute_large_credits.
#       - unclassified_transactions: rows whose classification.primary is None.
#       - flags.indicators: from compute_risk_flags via aggregated summary.
#       - parsing_metadata: per-account-month reconciliation slots.
#       - counterparty_ledger: pass-through.
#       - pdf_integrity: pass-through (already sanitised upstream).
#       - classification_config: Track 2 constants snapshot.
#
# Part 2 wiring (session 16):
#   * observations.positive / observations.concerns populated from
#     consolidated + statutory + flags (capped at 8 each per v6.3.4).
#   * own_related_transactions.transactions[] lists every C01-C04 row.
#     The list is empty until those rungs fire (RP foundation), but the
#     builder is in place.
#   * loan_transactions.disbursements / .repayments[] list C10 / C11 rows.
#     Same — empty until the rungs fire.
#
# Still deferred (RP foundation or later):
#   * report_info.related_parties is [] unless the caller passes them in
#     via the ``related_parties`` kwarg.
#   * accounts[*].account_type_determination is a conservative default
#     ('UNDETERMINED' / 0.0 confidence) until parser meta is threaded.
#
# Single-account vs multi-account: the orchestrator groups transactions by
# ``account_no``. Rows with missing account_no are bucketed under the
# placeholder key ``"_unknown"`` — same convention as the parser monthly
# summary builders. This keeps the orchestrator stable on parsers that don't
# stamp account_no per row (older Bank Islam / RHB samples).
# ---------------------------------------------------------------------------

_TRACK2_SCHEMA_VERSION = "6.3.5"  # schema 'const' — no v-prefix
_TRACK2_RULES_VERSION = "v3.5"
_TRACK2_CLASSIFIER_VERSION = "track2-slice-c1"
_TRACK2_EXECUTION_MODE = "FULL_CODE"
_TRACK2_LARGE_CREDIT_THRESHOLD = 100000.0
_TRACK2_UNCLASSIFIED_LISTING_THRESHOLD = 10000.0
_TRACK2_ACCOUNT_NO_FALLBACK = "_unknown"

# v6.3.5 statutory_compliance.overall_status only allows three values; the
# Track 2 s12 calibration introduced two additional verdicts that don't fit
# (SUB_THRESHOLD / CHANNEL_BLIND). Map them onto the schema's three for
# serialisation only — the rich extension fields (subthreshold_employer,
# channel_blind_employer) survive on the object and carry the real verdict.
_OVERALL_STATUS_SCHEMA_MAP = {
    "SUB_THRESHOLD": "COMPLIANT",
    "CHANNEL_BLIND": "GAPS_DETECTED",
}

# v6.3.5 PDF-integrity layer enum, mirrors Track 1's _PDF_LAYER_VALID +
# _PDF_LAYER_MAP. The fraud detector emits eight layer kinds; the schema
# only accepts five. Map the extras onto 'structural'.
_TRACK2_PDF_LAYER_VALID = frozenset(
    {"fonts", "visual", "bank_profile", "structural", "arithmetic"}
)
_TRACK2_PDF_LAYER_MAP = {
    "text_layers": "structural",
    "text_layer": "structural",
    "metadata": "structural",
    "cross_validation": "structural",
    "cross-validation": "structural",
}


def _coerce_account_no(row: dict[str, Any]) -> str:
    for key in ("account_no", "account_number"):
        value = row.get(key)
        if value:
            return str(value)
    return _TRACK2_ACCOUNT_NO_FALLBACK


def _sanitize_statutory_compliance_for_schema(
    statutory_compliance: dict[str, Any],
) -> dict[str, Any]:
    """Project Track 2's extended ``overall_status`` values onto the v6.3.5
    schema enum. The original verdict survives on the
    ``subthreshold_employer`` / ``channel_blind_employer`` extension fields,
    so analysts retain full fidelity even after the top-level mapping."""
    if not isinstance(statutory_compliance, dict):
        return statutory_compliance
    status = statutory_compliance.get("overall_status")
    if status in _OVERALL_STATUS_SCHEMA_MAP:
        return {
            **statutory_compliance,
            "overall_status": _OVERALL_STATUS_SCHEMA_MAP[status],
        }
    return statutory_compliance


def _sanitize_pdf_integrity_for_schema(
    pdf_integrity: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Coerce parser-emitted ``pdf_integrity`` findings into a v6.3.5-valid
    shape. Maps non-enum layer values onto ``'structural'``, drops
    ``detail`` entries that aren't dict-shaped (schema requires
    ``detail`` to be an object when present), and derives the per-file
    severity counts (``finding_count`` / ``high_count`` / ``medium_count``
    / ``low_count``) from the findings list when the upstream payload
    didn't supply them."""
    if pdf_integrity is None:
        return None
    if not isinstance(pdf_integrity, dict):
        return None
    out: dict[str, Any] = {}
    for fname, payload in pdf_integrity.items():
        if not isinstance(payload, dict):
            continue
        sanitized = dict(payload)
        findings = []
        high = medium = low = 0
        for f in payload.get("findings") or []:
            if not isinstance(f, dict):
                continue
            layer = f.get("layer")
            if layer not in _TRACK2_PDF_LAYER_VALID:
                layer = _TRACK2_PDF_LAYER_MAP.get(layer, "structural")
            cleaned = {**f, "layer": layer}
            if "detail" in cleaned and not isinstance(cleaned["detail"], dict):
                cleaned.pop("detail")
            severity = cleaned.get("severity")
            if severity == "HIGH":
                high += 1
            elif severity == "MEDIUM":
                medium += 1
            else:
                low += 1
            findings.append(cleaned)
        sanitized["findings"] = findings
        sanitized.setdefault("finding_count", len(findings))
        sanitized.setdefault("high_count", high)
        sanitized.setdefault("medium_count", medium)
        sanitized.setdefault("low_count", low)
        sanitized.setdefault("overall_risk", "LOW")
        out[fname] = sanitized
    return out


def _empty_monthly_entry(month: str, account_no: str, bank_name: str) -> dict[str, Any]:
    """Produce a v6.3.5 monthly_analysis item pre-filled with zero-values for
    the 58 required keys (plus a handful of optional Track 2 extension
    fields). Caller mutates the entry in place during per-row aggregation."""
    return {
        "month": month,
        "account_number": account_no,
        "bank_name": bank_name,
        "gross_credits": 0.0,
        "gross_debits": 0.0,
        "net_credits": 0.0,
        "net_debits": 0.0,
        "credit_count": 0,
        "debit_count": 0,
        "own_party_cr": 0.0,
        "own_party_dr": 0.0,
        "related_party_cr": 0.0,
        "related_party_dr": 0.0,
        "reversal_cr": 0.0,
        "returned_cheques_inward_count": 0,
        "returned_cheques_inward_amount": 0.0,
        "returned_cheques_outward_count": 0,
        "returned_cheques_outward_amount": 0.0,
        "loan_disbursement_cr": 0.0,
        "fd_interest_cr": 0.0,
        "round_figure_cr": 0.0,
        "high_value_cr": 0.0,
        "cash_deposits_count": 0,
        "cash_deposits_amount": 0.0,
        "cash_withdrawals_count": 0,
        "cash_withdrawals_amount": 0.0,
        "cheque_deposits_count": 0,
        "cheque_deposits_amount": 0.0,
        "cheque_issues_count": 0,
        "cheque_issues_amount": 0.0,
        "loan_repayment_dr": 0.0,
        "salary_paid": 0.0,
        "statutory_epf": 0.0,
        "statutory_socso": 0.0,
        "statutory_tax": 0.0,
        "statutory_hrdf": 0.0,
        "eod_lowest": 0.0,
        "eod_highest": 0.0,
        "eod_average": 0.0,
        "opening_balance": 0.0,
        "closing_balance": 0.0,
        "fx_credit_count": 0,
        "fx_credit_amount": 0.0,
        "fx_debit_count": 0,
        "fx_debit_amount": 0.0,
        "fx_currencies": [],
        "reconciliation_status": "PASS",
        "reconciliation_delta": 0.0,
        "extraction_gaps": 0,
        "missing_debit_amount": 0.0,
        "missing_credit_amount": 0.0,
        "data_quality_note": None,
        "own_party_cr_count": 0,
        "own_party_dr_count": 0,
        "related_party_cr_count": 0,
        "related_party_dr_count": 0,
        "loan_repayment_count": 0,
        "inward_return_cr": 0.0,
        "trade_income_count": 0,
        "trade_income_amount": 0.0,
        "trade_expense_count": 0,
        "trade_expense_amount": 0.0,
        "unclassified_cr_count": 0,
        "unclassified_cr_amount": 0.0,
        "unclassified_dr_count": 0,
        "unclassified_dr_amount": 0.0,
    }


_CATEGORY_TO_MONTHLY_BUCKET: dict[str, tuple[str, str | None]] = {
    # primary -> (amount_key, optional_count_key)
    "C01": ("own_party_cr", "own_party_cr_count"),
    "C02": ("own_party_dr", "own_party_dr_count"),
    "C03": ("related_party_cr", "related_party_cr_count"),
    "C04": ("related_party_dr", "related_party_dr_count"),
    "C05": ("salary_paid", None),
    "C06": ("statutory_epf", None),
    "C07": ("statutory_socso", None),
    "C08": ("statutory_tax", None),
    "C09": ("statutory_hrdf", None),
    "C10": ("loan_disbursement_cr", None),
    "C11": ("loan_repayment_dr", "loan_repayment_count"),
    "C12": ("fd_interest_cr", None),
    "C13": ("reversal_cr", None),
    "C16": ("inward_return_cr", None),
    "C17": ("cash_deposits_amount", "cash_deposits_count"),
    "C18": ("cash_withdrawals_amount", "cash_withdrawals_count"),
    "C19": ("cheque_deposits_amount", "cheque_deposits_count"),
    "C20": ("cheque_issues_amount", "cheque_issues_count"),
    "C26": ("trade_income_amount", "trade_income_count"),
    "C27": ("trade_expense_amount", "trade_expense_count"),
}


def _aggregate_classified_into_monthly(
    monthly_by_key: dict[tuple[str, str], dict[str, Any]],
    classified: list[dict[str, Any]],
) -> None:
    """Walk classified rows once and accumulate into per-(account, month)
    entries. Mutates ``monthly_by_key`` in place."""
    for row in classified:
        date = row.get("date") or ""
        if not isinstance(date, str) or len(date) < 7:
            continue
        month = date[:7]
        account_no = _coerce_account_no(row)
        key = (account_no, month)
        entry = monthly_by_key.get(key)
        if entry is None:
            continue

        classification = row.get("classification") or {}
        primary = classification.get("primary")
        side = classification.get("side")
        credit = _safe_amount(row.get("credit"))
        debit = _safe_amount(row.get("debit"))

        if primary == "C25":
            # Balance rows do not contribute to gross / category totals.
            continue

        if credit > 0:
            entry["gross_credits"] += credit
            entry["credit_count"] += 1
            amount = credit
        elif debit > 0:
            entry["gross_debits"] += debit
            entry["debit_count"] += 1
            amount = debit
        else:
            amount = 0.0

        bucket = _CATEGORY_TO_MONTHLY_BUCKET.get(primary) if primary else None
        if bucket is not None:
            amount_key, count_key = bucket
            entry[amount_key] += amount
            if count_key is not None:
                entry[count_key] += 1
        elif primary == "C14":  # inward returned cheque (DR side)
            entry["returned_cheques_inward_count"] += 1
            entry["returned_cheques_inward_amount"] += amount
        elif primary == "C15":  # outward returned cheque (CR side)
            entry["returned_cheques_outward_count"] += 1
            entry["returned_cheques_outward_amount"] += amount
        elif primary is None:
            if side == "CR":
                entry["unclassified_cr_count"] += 1
                entry["unclassified_cr_amount"] += amount
            elif side == "DR":
                entry["unclassified_dr_count"] += 1
                entry["unclassified_dr_amount"] += amount
        # C24 bank fees is intentionally NOT bucketed into monthly_analysis —
        # the v6.3.5 schema has no per-month bank-fees field; it surfaces via
        # the unclassified / gross totals and the C24 indicator separately.


def _round_monthly_entry(entry: dict[str, Any]) -> None:
    """Round all float fields on a monthly_analysis entry to 2 decimals."""
    for key in (
        "gross_credits", "gross_debits", "net_credits", "net_debits",
        "own_party_cr", "own_party_dr", "related_party_cr", "related_party_dr",
        "reversal_cr", "returned_cheques_inward_amount",
        "returned_cheques_outward_amount", "loan_disbursement_cr",
        "fd_interest_cr", "round_figure_cr", "high_value_cr",
        "cash_deposits_amount", "cash_withdrawals_amount",
        "cheque_deposits_amount", "cheque_issues_amount", "loan_repayment_dr",
        "salary_paid", "statutory_epf", "statutory_socso", "statutory_tax",
        "statutory_hrdf", "eod_lowest", "eod_highest", "eod_average",
        "opening_balance", "closing_balance", "fx_credit_amount",
        "fx_debit_amount", "reconciliation_delta", "missing_debit_amount",
        "missing_credit_amount", "inward_return_cr",
        "trade_income_amount", "trade_expense_amount",
        "unclassified_cr_amount", "unclassified_dr_amount",
    ):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            entry[key] = round(float(value), 2)


def _build_monthly_for_account(
    account_no: str,
    bank_name: str,
    account_type: str,
    transactions: list[dict[str, Any]],
    classified: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the per-month entries for a single account. ``transactions`` and
    ``classified`` are the SAME rows (in identical order) — the orchestrator
    classifies once over the full input list and slices here."""
    aggregates = compute_monthly_aggregates(transactions, account_type=account_type)
    if not aggregates:
        return []
    fx_summary = compute_fx_totals(transactions)
    rf_summary = compute_round_figure_credits(transactions)
    hv_summary = compute_high_value_credits(transactions)
    fx_entries = fx_summary.get("fx_entries") or []
    rf_entries = rf_summary.get("round_figure_entries") or []
    hv_entries = hv_summary.get("high_value_entries") or []

    monthly_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for agg in aggregates:
        month = agg["month"]
        entry = _empty_monthly_entry(month, account_no, bank_name)
        entry["opening_balance"] = round(float(agg.get("opening_balance") or 0.0), 2)
        entry["closing_balance"] = round(float(agg.get("closing_balance") or 0.0), 2)
        entry["eod_lowest"] = round(float(agg.get("eod_lowest") or 0.0), 2)
        entry["eod_highest"] = round(float(agg.get("eod_highest") or 0.0), 2)
        if agg.get("eod_average") is not None:
            entry["eod_average"] = round(float(agg["eod_average"]), 2)
        monthly_by_key[(account_no, month)] = entry

    _aggregate_classified_into_monthly(monthly_by_key, classified)

    # FX, round-figure, high-value overlays (per-month from the compute helpers)
    for fx_row in fx_entries:
        date = fx_row.get("date") or ""
        if not isinstance(date, str) or len(date) < 7:
            continue
        month = date[:7]
        entry = monthly_by_key.get((account_no, month))
        if entry is None:
            continue
        amount = round(float(fx_row.get("amount") or 0.0), 2)
        currency = fx_row.get("currency")
        if fx_row.get("type") == "CREDIT":
            entry["fx_credit_count"] += 1
            entry["fx_credit_amount"] += amount
        elif fx_row.get("type") == "DEBIT":
            entry["fx_debit_count"] += 1
            entry["fx_debit_amount"] += amount
        if currency and isinstance(currency, str):
            currencies = entry["fx_currencies"]
            if currency not in currencies:
                currencies.append(currency)
    for rf_row in rf_entries:
        date = rf_row.get("date") or ""
        if not isinstance(date, str) or len(date) < 7:
            continue
        entry = monthly_by_key.get((account_no, date[:7]))
        if entry is None:
            continue
        entry["round_figure_cr"] += float(rf_row.get("amount") or 0.0)
    for hv_row in hv_entries:
        date = hv_row.get("date") or ""
        if not isinstance(date, str) or len(date) < 7:
            continue
        entry = monthly_by_key.get((account_no, date[:7]))
        if entry is None:
            continue
        entry["high_value_cr"] += float(hv_row.get("amount") or 0.0)

    # Reconciliation: PASS when |opening + gross_credits - gross_debits -
    # closing| <= RM 1.00. OD accounts in the signed-negative convention
    # (every modern parser as of 2026-04-20) use the same formula as CR.
    # Only OD in the legacy positive-magnitude convention (Alliance
    # pre-2026-04-20) inverts the sign. Detected account-wide from the
    # balance trail (NOT from the sign of one month's opening — a signed-
    # convention OD account can open a month in credit and dip overdrawn
    # mid-month, which the old sign-of-opening check misread as legacy and
    # reported as a phantom extraction gap).
    od_convention = _detect_od_balance_convention(transactions)
    for entry in monthly_by_key.values():
        opening = float(entry["opening_balance"])
        closing = float(entry["closing_balance"])
        gross_credits = float(entry["gross_credits"])
        gross_debits = float(entry["gross_debits"])
        if account_type == "OD" and od_convention == "legacy":
            # Legacy positive-magnitude OD.
            expected_closing = opening + gross_debits - gross_credits
        else:
            expected_closing = opening + gross_credits - gross_debits
        delta = round(closing - expected_closing, 2)
        entry["reconciliation_delta"] = delta
        if abs(delta) > 1.00:
            entry["reconciliation_status"] = "FAIL"
            entry["data_quality_note"] = (
                f"Balance trail mismatch: expected {expected_closing:.2f}, "
                f"got {closing:.2f} (delta {delta:+.2f})."
            )

    # Net totals per v6.3.5: net = gross minus identity-bucket flows.
    for entry in monthly_by_key.values():
        nets = compute_net_totals(
            gross_credits=float(entry["gross_credits"]),
            gross_debits=float(entry["gross_debits"]),
            own_party_cr=float(entry["own_party_cr"]),
            related_party_cr=float(entry["related_party_cr"]),
            loan_disbursement_cr=float(entry["loan_disbursement_cr"]),
            fd_interest_cr=float(entry["fd_interest_cr"]),
            reversal_cr=float(entry["reversal_cr"]),
            inward_return_cr=float(entry["inward_return_cr"]),
            own_party_dr=float(entry["own_party_dr"]),
        )
        entry["net_credits"] = nets["net_credits"]
        entry["net_debits"] = nets["net_debits"]
        _round_monthly_entry(entry)

    return sorted(
        monthly_by_key.values(), key=lambda r: (r["account_number"], r["month"])
    )


def _build_consolidated_track2(
    monthly: list[dict[str, Any]],
    statutory_compliance: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate monthly_analysis into the consolidated section per v6.3.5."""
    if not monthly:
        return {
            "gross_credits": 0.0, "gross_debits": 0.0,
            "net_credits": 0.0, "net_debits": 0.0,
            "annualized_net_credits": 0.0, "annualized_net_debits": 0.0,
            "total_own_party_cr": 0.0, "total_own_party_dr": 0.0,
            "total_related_party_cr": 0.0, "total_related_party_dr": 0.0,
            "total_reversal_cr": 0.0,
            "total_returned_cheques_inward": 0.0,
            "total_returned_cheques_outward": 0.0,
            "total_loan_disbursement_cr": 0.0, "total_fd_interest_cr": 0.0,
            "total_round_figure_cr": 0.0, "total_high_value_cr": 0.0,
            "total_cash_deposits": 0.0, "total_cash_withdrawals": 0.0,
            "total_cheque_deposits": 0.0, "total_cheque_issues": 0.0,
            "total_loan_repayment_dr": 0.0, "total_salary_paid": 0.0,
            "total_statutory_epf": 0.0, "total_statutory_socso": 0.0,
            "total_statutory_tax": 0.0, "total_statutory_hrdf": 0.0,
            "total_trade_income_cr": 0.0, "total_trade_income_count": 0,
            "total_trade_expense_dr": 0.0, "total_trade_expense_count": 0,
            "eod_lowest": 0.0, "eod_highest": 0.0, "eod_average": 0.0,
            "total_fx_credits": 0.0, "total_fx_debits": 0.0,
            "data_completeness": "COMPLETE", "months_with_gaps": 0,
            "total_extraction_gaps": 0,
            "total_missing_debits": 0.0, "total_missing_credits": 0.0,
            "total_inward_return_cr": 0.0,
            "total_unclassified_cr": 0.0, "total_unclassified_dr": 0.0,
            "statutory_compliance": statutory_compliance,
        }

    months_count = len({m["month"] for m in monthly}) or 1
    s = lambda key: round(sum(float(m.get(key, 0) or 0) for m in monthly), 2)
    sc = lambda key: int(sum(int(m.get(key, 0) or 0) for m in monthly))

    return {
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
        "total_trade_income_cr": s("trade_income_amount"),
        "total_trade_income_count": sc("trade_income_count"),
        "total_trade_expense_dr": s("trade_expense_amount"),
        "total_trade_expense_count": sc("trade_expense_count"),
        "eod_lowest": min((float(m.get("eod_lowest") or 0) for m in monthly), default=0.0),
        "eod_highest": max((float(m.get("eod_highest") or 0) for m in monthly), default=0.0),
        "eod_average": round(
            sum(float(m.get("eod_average") or 0) for m in monthly) / months_count, 2
        ),
        "total_fx_credits": s("fx_credit_amount"),
        "total_fx_debits": s("fx_debit_amount"),
        "data_completeness": (
            "COMPLETE"
            if all(m.get("reconciliation_status") == "PASS" for m in monthly)
            else "INCOMPLETE"
        ),
        "months_with_gaps": sum(
            1 for m in monthly if m.get("reconciliation_status") != "PASS"
        ),
        "total_extraction_gaps": sc("extraction_gaps"),
        "total_missing_debits": s("missing_debit_amount"),
        "total_missing_credits": s("missing_credit_amount"),
        "total_inward_return_cr": s("inward_return_cr"),
        "total_unclassified_cr": s("unclassified_cr_amount"),
        "total_unclassified_dr": s("unclassified_dr_amount"),
        "statutory_compliance": statutory_compliance,
    }


def _build_accounts_track2(
    monthly: list[dict[str, Any]],
    transactions_by_account: dict[str, list[dict[str, Any]]],
    account_meta: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build the per-account summary rows per v6.3.5 accounts schema."""
    monthly_by_account: dict[str, list[dict[str, Any]]] = {}
    for m in monthly:
        monthly_by_account.setdefault(m["account_number"], []).append(m)

    accounts: list[dict[str, Any]] = []
    for account_no, rows in transactions_by_account.items():
        months = monthly_by_account.get(account_no, [])
        bank_name = next(
            (r.get("bank") for r in rows if r.get("bank")),
            "Unknown",
        )
        company = next(
            (r.get("company_name") for r in rows if r.get("company_name")),
            "",
        )
        period_start = ""
        period_end = ""
        dates = sorted(r.get("date") for r in rows if isinstance(r.get("date"), str))
        if dates:
            period_start = dates[0]
            period_end = dates[-1]
        meta = (account_meta or {}).get(account_no, {})
        account_type = str(meta.get("account_type", "Current"))
        is_od = bool(meta.get("is_od", account_type == "OD"))
        od_limit = meta.get("od_limit")
        if months:
            opening_balance = float(months[0].get("opening_balance") or 0.0)
            closing_balance = float(months[-1].get("closing_balance") or 0.0)
            total_credits = sum(float(m.get("gross_credits") or 0) for m in months)
            total_debits = sum(float(m.get("gross_debits") or 0) for m in months)
            transaction_count = sum(
                int(m.get("credit_count") or 0) + int(m.get("debit_count") or 0)
                for m in months
            )
        else:
            opening_balance = 0.0
            closing_balance = 0.0
            total_credits = 0.0
            total_debits = 0.0
            transaction_count = 0
        accounts.append(
            {
                "bank_name": bank_name,
                "account_number": account_no,
                "account_holder": str(company),
                "account_type": account_type,
                "is_od": is_od,
                "od_limit": od_limit,
                "account_type_determination": {
                    "tested_formulas": [],
                    "locked_type": "OD" if is_od else "CR",
                    "confidence": "LOW",
                    "locked_rationale": "Track 2 default — parser meta not threaded.",
                },
                "period_start": period_start,
                "period_end": period_end,
                "opening_balance": round(opening_balance, 2),
                "closing_balance": round(closing_balance, 2),
                "total_credits": round(total_credits, 2),
                "total_debits": round(total_debits, 2),
                "transaction_count": transaction_count,
            }
        )
    accounts.sort(key=lambda a: a["account_number"])
    return accounts


def _monthly_breakdown_for_cp(
    cp_transactions: list[dict[str, Any]], ttype: str
) -> list[dict[str, Any]]:
    """Per-month aggregate of one counterparty's ledger transactions of a
    given type (``'CREDIT'`` or ``'DEBIT'``)."""
    buckets: dict[str, dict[str, float | int]] = {}
    for ltx in cp_transactions or []:
        if not isinstance(ltx, dict) or ltx.get("type") != ttype:
            continue
        date = ltx.get("date") or ""
        if not isinstance(date, str) or len(date) < 7:
            continue
        month = date[:7]
        b = buckets.setdefault(month, {"month": month, "amount": 0.0, "count": 0})
        try:
            b["amount"] = float(b["amount"]) + float(ltx.get("amount") or 0.0)
        except (TypeError, ValueError):
            continue
        b["count"] = int(b["count"]) + 1
    return [
        {
            "month": str(b["month"]),
            "amount": round(float(b["amount"]), 2),
            "count": int(b["count"]),
        }
        for b in sorted(buckets.values(), key=lambda x: x["month"])
    ]


def _build_top_parties_track2(
    counterparty_ledger: dict[str, Any] | None,
    *,
    top_n: int = 10,
    related_parties: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build top_payers / top_payees lists from the upstream ledger.

    Each entry conforms to ``$defs/party`` in the v6.3.5 schema:
    ``{rank, party_name, total_amount, transaction_count, is_related_party,
       monthly_breakdown}``.
    """
    if not isinstance(counterparty_ledger, dict):
        return {"top_payers": [], "top_payees": []}
    counterparties = counterparty_ledger.get("counterparties") or []
    related_upper = {str(p).upper() for p in (related_parties or []) if p}

    candidates_cr: list[dict[str, Any]] = []
    candidates_dr: list[dict[str, Any]] = []
    for cp in counterparties:
        if not isinstance(cp, dict):
            continue
        name = str(cp.get("counterparty_name") or "")
        if not name:
            continue
        if (
            _is_synthetic_counterparty_label(name)
            and _normalise_counterparty_name(name) not in _RANKABLE_DESPITE_SYNTHETIC
        ):
            continue
        is_rp = name.upper() in related_upper
        cr_amount = float(cp.get("total_credits") or 0.0)
        dr_amount = float(cp.get("total_debits") or 0.0)
        cr_count = int(cp.get("credit_count") or 0)
        dr_count = int(cp.get("debit_count") or 0)
        ltxs = cp.get("transactions") or []
        if cr_amount > 0:
            candidates_cr.append(
                {
                    "party_name": name,
                    "total_amount": round(cr_amount, 2),
                    "transaction_count": cr_count,
                    "is_related_party": is_rp,
                    "monthly_breakdown": _monthly_breakdown_for_cp(ltxs, "CREDIT"),
                }
            )
        if dr_amount > 0:
            candidates_dr.append(
                {
                    "party_name": name,
                    "total_amount": round(dr_amount, 2),
                    "transaction_count": dr_count,
                    "is_related_party": is_rp,
                    "monthly_breakdown": _monthly_breakdown_for_cp(ltxs, "DEBIT"),
                }
            )
    candidates_cr.sort(key=lambda x: x["total_amount"], reverse=True)
    candidates_dr.sort(key=lambda x: x["total_amount"], reverse=True)

    payers = [
        {**c, "rank": idx + 1} for idx, c in enumerate(candidates_cr[:top_n])
    ]
    payees = [
        {**c, "rank": idx + 1} for idx, c in enumerate(candidates_dr[:top_n])
    ]
    return {"top_payers": payers, "top_payees": payees}


def _build_unclassified_track2(
    classified: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """List unclassified rows above ``threshold`` per v6.3.0 listing rule."""
    out: list[dict[str, Any]] = []
    for row in classified:
        classification = row.get("classification") or {}
        if classification.get("primary") is not None:
            continue
        credit = _safe_amount(row.get("credit"))
        debit = _safe_amount(row.get("debit"))
        amount = credit if credit > 0 else debit
        if amount < threshold:
            continue
        out.append(
            {
                "date": str(row.get("date") or ""),
                "description": str(row.get("description") or ""),
                "amount": round(amount, 2),
                "type": "CREDIT" if credit > 0 else "DEBIT",
                "account_number": _coerce_account_no(row),
                "bank_name": str(row.get("bank") or "Unknown"),
                "reason": "No dispatcher rung matched",
            }
        )
    return out


def _build_parsing_metadata_track2(
    monthly: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-account-month reconciliation summary.

    ``total_transactions_extracted`` is the sum of the per-month
    money-moving counts (credit_count + debit_count) so it equals the
    transaction figure surfaced in the header, accounts[], and the
    counterparty ledger. It deliberately does NOT use the raw
    ``len(transactions)`` intake count, which includes zero-amount /
    dateless rows (balance-carried-forward and informational lines) that
    never group into a month — that mismatch previously showed e.g. 1,204
    here vs 1,192 everywhere else.
    """
    checks = []
    passed = 0
    for m in monthly:
        ok = m.get("reconciliation_status") == "PASS"
        if ok:
            passed += 1
        opening = float(m.get("opening_balance") or 0.0)
        closing = float(m.get("closing_balance") or 0.0)
        gross_credits = float(m.get("gross_credits") or 0.0)
        gross_debits = float(m.get("gross_debits") or 0.0)
        expected_closing = round(opening + gross_credits - gross_debits, 2)
        delta = float(m.get("reconciliation_delta") or 0.0)
        tx_count = int(m.get("credit_count") or 0) + int(m.get("debit_count") or 0)
        checks.append(
            {
                "month": m["month"],
                "account_number": m["account_number"],
                "bank_name": str(m.get("bank_name") or "Unknown"),
                "opening_balance": opening,
                "closing_balance": closing,
                "gross_credits": gross_credits,
                "gross_debits": gross_debits,
                "expected_closing": expected_closing,
                "reconciliation_delta": delta,
                "passed": ok,
                "transactions_extracted": tx_count,
            }
        )
    total_checks = len(checks)
    return {
        "overall_success_rate": round(passed / total_checks, 4) if total_checks else 1.0,
        "total_transactions_extracted": sum(
            c["transactions_extracted"] for c in checks
        ),
        "total_balance_checks": total_checks,
        "total_balance_checks_passed": passed,
        "account_month_checks": checks,
        "extraction_gaps": [],
    }


def _summary_for_flags_track2(
    consolidated: dict[str, Any],
    classified: list[dict[str, Any]],
    large_credits: list[dict[str, Any]],
    *,
    round_figure_count: int = 0,
) -> dict[str, Any]:
    """Project a consolidated dict into the summary shape ``compute_risk_flags``
    expects. Counts that consolidated doesn't carry (returned cheque counts,
    cash-deposit count) are derived from the classified list.

    ``round_figure_count`` is passed in from the single canonical detector
    ``compute_round_figure_credits`` (the same source as the
    ``round_figure_credits`` detail list and the per-month
    ``round_figure_cr`` amounts) so the Flag-3 count can never diverge from
    the detail table. It is NOT re-derived here — an earlier independent
    re-derivation used ``% 1000`` and drifted from the rulebook C21 rule
    (``amount >= 10000 AND amount % 10000 == 0``), reporting one extra hit.
    """
    returned_in_count = sum(
        1
        for r in classified
        if (r.get("classification") or {}).get("primary") == "C14"
    )
    returned_out_count = sum(
        1
        for r in classified
        if (r.get("classification") or {}).get("primary") == "C15"
    )
    cash_deposit_count = sum(
        1
        for r in classified
        if (r.get("classification") or {}).get("primary") == "C17"
    )
    high_value_count = sum(
        1 for r in classified if _safe_amount(r.get("credit")) >= 100000.0
    )
    return {
        "returned_cheques_inward_count": returned_in_count,
        "returned_cheques_inward_amount": consolidated.get("total_returned_cheques_inward", 0.0),
        "returned_cheques_outward_count": returned_out_count,
        "returned_cheques_outward_amount": consolidated.get("total_returned_cheques_outward", 0.0),
        "round_figure_cr": consolidated.get("total_round_figure_cr", 0.0),
        "round_figure_count": round_figure_count,
        "high_value_cr": consolidated.get("total_high_value_cr", 0.0),
        "high_value_count": high_value_count,
        "eod_unreliable": False,
        "cash_deposits_count": cash_deposit_count,
        "cash_deposits_amount": consolidated.get("total_cash_deposits", 0.0),
        "gross_credits": consolidated.get("gross_credits", 0.0),
        "gross_debits": consolidated.get("gross_debits", 0.0),
        "large_credits": large_credits or [],
        "own_party_cr": consolidated.get("total_own_party_cr", 0.0),
        "own_party_dr": consolidated.get("total_own_party_dr", 0.0),
        "related_party_cr": consolidated.get("total_related_party_cr", 0.0),
        "related_party_dr": consolidated.get("total_related_party_dr", 0.0),
        "related_party_names": [],
        "loan_disbursement_cr": consolidated.get("total_loan_disbursement_cr", 0.0),
        "loan_repayment_dr": consolidated.get("total_loan_repayment_dr", 0.0),
        "data_completeness": consolidated.get("data_completeness", "COMPLETE"),
        "data_gaps": "",
        "total_fx_credits": consolidated.get("total_fx_credits", 0.0),
        "total_fx_debits": consolidated.get("total_fx_debits", 0.0),
        "salary_months_active": (consolidated.get("statutory_compliance") or {})
        .get("salary_months_active", 0),
    }


# ---------------------------------------------------------------------------
# Session 16 — Dispatcher Slice C Part 2: orchestrator list builders.
#
# Three helpers fill the sections stubbed in s15:
#   * own_related_transactions.transactions[] — C01/C02/C03/C04 rows.
#   * loan_transactions.disbursements / .repayments[] — C10 / C11 rows.
#   * observations.positive / .concerns[] — human-readable strings (≤ 8).
#
# All three are pure functions over the classified-row stream produced by
# ``classify_transactions`` (canonical row + ``{classification: {primary,
# side, reason, mode}}``). They re-derive amount and side from the
# canonical ``credit`` / ``debit`` fields rather than relying on Track 1's
# private ``_amount`` / ``_side`` shape.
#
# RP foundation slices 1-3 (s17-s19) unblocked C01-C04 and C10/C11 in
# the dispatcher. The list builders below were wired ahead of the
# dispatcher rungs so the sections lit up the moment each rung started
# emitting; no additional orchestrator work is needed.
# ---------------------------------------------------------------------------

# C01 = own DR, C02 = own CR, C03 = related DR, C04 = related CR.
_OWN_RELATED_PRIMARIES: frozenset[str] = frozenset({"C01", "C02", "C03", "C04"})
_OWN_PARTY_PRIMARIES: frozenset[str] = frozenset({"C01", "C02"})


def _row_amount_and_side(row: dict[str, Any]) -> tuple[float, str | None]:
    """Return (positive amount, 'CR'|'DR'|None) from a canonical row.

    Uses the classification's recorded side when present; otherwise infers
    from whichever of credit/debit is non-zero. Returns (0.0, None) when
    neither side has a positive amount.
    """
    credit = _safe_amount(row.get("credit"))
    debit = _safe_amount(row.get("debit"))
    side = (row.get("classification") or {}).get("side")
    if side == "CR" or (side is None and credit > 0):
        return credit, "CR" if credit > 0 else None
    if side == "DR" or (side is None and debit > 0):
        return debit, "DR" if debit > 0 else None
    return 0.0, None


def _build_own_related_transactions_list_track2(
    classified: list[dict[str, Any]],
    counterparty_lookup: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Build ``own_related_transactions.transactions[]`` per v6.3.5 schema.

    Lists every row whose classification primary is C01/C02/C03/C04.
    Each entry carries the v6.3.5-required keys ``date``, ``description``,
    ``amount``, ``type`` (CREDIT|DEBIT), and ``party_type`` (OWN|RELATED).
    ``party_name`` is populated from ``counterparty_lookup`` when the
    row's index has a non-synthetic name; otherwise omitted.

    Pure function. Empty list until C01-C04 rungs fire (RP foundation).
    """
    counterparty_lookup = counterparty_lookup or {}
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(classified):
        classification = row.get("classification") or {}
        primary = classification.get("primary")
        if primary not in _OWN_RELATED_PRIMARIES:
            continue
        amount, side = _row_amount_and_side(row)
        if side is None:
            continue
        entry: dict[str, Any] = {
            "date": str(row.get("date") or ""),
            "description": str(row.get("description") or ""),
            "amount": round(amount, 2),
            "type": "CREDIT" if side == "CR" else "DEBIT",
            "party_type": "OWN" if primary in _OWN_PARTY_PRIMARIES else "RELATED",
            "account_number": _coerce_account_no(row),
        }
        party_name = counterparty_lookup.get(idx)
        if isinstance(party_name, str) and party_name.strip():
            entry["party_name"] = party_name.strip()
        out.append(entry)
    return out


# Map dispatcher primary → ``transaction_entry.category`` label so the
# downstream HTML / analyst can tell a C10-disbursement entry apart from
# any future loan rung that lands in the same bucket. Schema-allowed —
# ``transaction_entry`` is open-ended on ``category``.
_LOAN_CATEGORY_LABELS: dict[str, str] = {
    "C10": "loan_disbursement",
    "C11": "loan_repayment",
}


def _build_loan_transactions_track2(
    classified: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build ``loan_transactions.disbursements`` / ``.repayments`` lists.

    C10 → disbursements (CR side); C11 → repayments (DR side). Each entry
    conforms to the ``$defs/transaction_entry`` shape: ``date``,
    ``description``, ``amount``; plus the optional ``category`` (always
    set) and ``balance`` (when the row carries one). The v6.3.2
    ``exclusion_note`` field is only relevant for dual-tagged C02+C11
    rows; Track 2's dispatcher emits a single primary, so the note is
    not set here.

    Pure function. Empty lists until C10/C11 rungs fire (RP foundation).
    """
    disbursements: list[dict[str, Any]] = []
    repayments: list[dict[str, Any]] = []
    for row in classified:
        primary = (row.get("classification") or {}).get("primary")
        if primary not in _LOAN_CATEGORY_LABELS:
            continue
        amount, _ = _row_amount_and_side(row)
        if amount <= 0:
            continue
        entry: dict[str, Any] = {
            "date": str(row.get("date") or ""),
            "description": str(row.get("description") or ""),
            "amount": round(amount, 2),
            "category": _LOAN_CATEGORY_LABELS[primary],
        }
        balance = row.get("balance")
        if isinstance(balance, (int, float)):
            entry["balance"] = float(balance)
        if primary == "C10":
            disbursements.append(entry)
        else:
            repayments.append(entry)
    return {"disbursements": disbursements, "repayments": repayments}


def _build_loan_review_track2(
    classified: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Loan-shaped rows the classifier did NOT book as facility/RP activity.

    The "silent-zero" safety net (see ``LOAN_SHAPED_RE``). A row is collected
    when its description looks like facility activity (loan / financing /
    instalment / hire-purchase / repayment / disbursement) but its primary is
    NOT in ``_LOAN_REVIEW_SETTLED`` — i.e. it was neither booked C10/C11 nor
    routed to the related-party rung. These are exactly the rows a closed
    keyword vocabulary misses; surfacing them turns an invisible miss into an
    analyst-visible "review" line. Marketing / rate-disclosure prose is
    dropped via ``_LOAN_NOISE_RE`` and balance rows (C25) are skipped.

    Each entry mirrors the ``transaction_entry`` shape used by the
    disbursement / repayment lists, plus ``side`` and ``account_no`` so the
    analyst can locate the row. Pure function.
    """
    review: list[dict[str, Any]] = []
    for row in classified:
        primary = (row.get("classification") or {}).get("primary")
        # Only TRULY UNCLASSIFIED rows (primary is None) are "silently missed"
        # loans — the exact failure this net guards. A row that received ANY
        # category was a deliberate classifier decision and is NOT a hidden
        # facility: in particular C24 bank-fee rows like "OTHER TRANSFER FEE
        # Monthly Instalment" describe what a *fee* was for, not a repayment,
        # and must not pollute the review list (corpus pass: CIMB 116→ the
        # genuine person-loan remainder once the C24 fee noise is excluded).
        if primary is not None:
            continue
        description = str(row.get("description") or "")
        if not LOAN_SHAPED_RE.search(description):
            continue
        if _LOAN_NOISE_RE.search(description):
            continue
        amount, side = _row_amount_and_side(row)
        if amount <= 0:
            continue
        entry: dict[str, Any] = {
            "date": str(row.get("date") or ""),
            "description": description,
            "amount": round(amount, 2),
            "side": side,
            "account_no": _coerce_account_no(row),
        }
        balance = row.get("balance")
        if isinstance(balance, (int, float)):
            entry["balance"] = float(balance)
        review.append(entry)
    return review


_OBSERVATION_MAX_ITEMS = 8  # v6.3.4 raised the cap from 5 → 8.


def _build_observations_track2(
    consolidated: dict[str, Any],
    flags: list[dict[str, Any]],
    statutory_compliance: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Derive ``observations.positive`` and ``observations.concerns`` strings.

    Reads from the consolidated dict (already-sanitised statutory section)
    and the 16-flag indicator array. Each list is capped at
    ``_OBSERVATION_MAX_ITEMS`` (= 8).

    Track 2-specific extensions over Track 1's observation set:
      * sub-threshold employer → positive ("statutory N/A at this salary").
      * channel-blind employer → concern ("compliance unverifiable").
    Both surface from the unsanitised ``subthreshold_employer`` /
    ``channel_blind_employer`` extension fields kept by the sanitiser.
    """
    consolidated = consolidated or {}
    statutory_compliance = (
        statutory_compliance
        if statutory_compliance is not None
        else consolidated.get("statutory_compliance") or {}
    )
    flags = flags or []

    positive: list[str] = []
    concerns: list[str] = []

    # --- Reconciliation ---------------------------------------------------
    months_with_gaps = int(consolidated.get("months_with_gaps") or 0)
    if consolidated.get("data_completeness") == "COMPLETE":
        positive.append("All months reconciled to bank statements within tolerance.")
    elif months_with_gaps > 0:
        concerns.append(
            f"Reconciliation gaps in {months_with_gaps} months — "
            "investigate before relying on totals."
        )

    # --- Statutory --------------------------------------------------------
    salary_months = int(statutory_compliance.get("salary_months_active") or 0)
    epf_pct = statutory_compliance.get("epf_coverage_pct", 0)
    socso_pct = statutory_compliance.get("socso_coverage_pct", 0)
    overall_status = statutory_compliance.get("overall_status", "COMPLIANT")
    subthreshold = (statutory_compliance.get("subthreshold_employer") or {}).get(
        "is_subthreshold"
    )
    channel_blind = (statutory_compliance.get("channel_blind_employer") or {}).get(
        "is_channel_blind"
    )
    if subthreshold:
        positive.append(
            "Sub-threshold employer — statutory contributions not required "
            "at this salary level."
        )
    elif channel_blind:
        concerns.append(
            "Statutory payments not visible in this channel — compliance "
            "unverifiable from these statements alone."
        )
    elif overall_status == "COMPLIANT" and salary_months > 0:
        positive.append(
            f"EPF and SOCSO fully covered across {salary_months} salary months."
        )
    elif overall_status == "CRITICAL":
        concerns.append(
            f"Statutory compliance CRITICAL — EPF {epf_pct}% / SOCSO {socso_pct}%."
        )
    elif overall_status == "GAPS_DETECTED":
        concerns.append(
            f"Statutory gaps — EPF {epf_pct}% / SOCSO {socso_pct}%."
        )

    # --- Cashflow ---------------------------------------------------------
    net_credits = float(consolidated.get("net_credits") or 0.0)
    if net_credits > 0:
        positive.append(
            f"Net credits of RM {net_credits:,.2f} indicate active turnover."
        )

    # --- Unclassified residue --------------------------------------------
    unc_cr = float(consolidated.get("total_unclassified_cr") or 0.0)
    unc_dr = float(consolidated.get("total_unclassified_dr") or 0.0)
    if unc_cr + unc_dr > 0:
        concerns.append(
            f"Unclassified rows: CR RM {unc_cr:,.2f} / DR RM {unc_dr:,.2f} "
            "— V1 deferred categories."
        )

    # --- Flag-driven concerns (Track 1 parity) ---------------------------
    flag_concern_names = {
        "Returned Cheques (Inward)",
        "Returned Cheques (Outward)",
        "Round Figure Credits (AML)",
        "Cash Deposits (AML)",
    }
    for f in flags:
        if (
            f.get("detected")
            and f.get("name") in flag_concern_names
            and len(concerns) < _OBSERVATION_MAX_ITEMS
        ):
            concerns.append(f"{f.get('name')}: {f.get('remarks')}")

    return {
        "positive": positive[:_OBSERVATION_MAX_ITEMS],
        "concerns": concerns[:_OBSERVATION_MAX_ITEMS],
    }


def account_meta_from_determinations(
    determinations: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Project ``core_utils.determine_account_type`` outputs into the
    ``{account_no: {is_od, account_type, od_limit}}`` shape
    ``build_track2_result`` expects.

    ``app.py`` accumulates one determination per processed PDF in
    ``st.session_state.account_type_determinations``. Each entry carries the
    PDF's ``account_no``, the locked ``CR | OD | UNDETERMINED`` label, and
    the parsed ``facility_limits_in_header`` ({"overdraft": [...],
    "cashline": [...]}). This helper folds that list into the per-account
    metadata the engine consumes.

    Multi-PDF same-account semantics: when two PDFs for the same account
    disagree on facility type (one CR, one OD), **OD wins**. Same facility
    across different month ranges — losing the OD signal would mis-route
    the engine's monthly_analysis facility math. od_limit is the maximum
    seen across PDFs (header rounding can shift the printed limit by a
    few ringgit; take the largest).

    Pure / Streamlit-free so it can be unit-tested directly. Used by the
    Track 2 wire-through in ``app.py`` and by any headless caller mapping
    parser output → engine input.
    """
    meta: dict[str, dict[str, Any]] = {}
    for det in determinations or []:
        if not isinstance(det, dict):
            continue
        acct = str(det.get("account_no") or "").strip()
        if not acct:
            continue
        is_od = det.get("locked_type") == "OD"
        limits = det.get("facility_limits_in_header") or {}
        od_limit_vals = (
            list(limits.get("overdraft") or [])
            + list(limits.get("cashline") or [])
        )
        od_limit = max(od_limit_vals) if od_limit_vals else None
        existing = meta.get(acct)
        if existing is None:
            meta[acct] = {
                "is_od": is_od,
                "account_type": "OD" if is_od else "Current",
                "od_limit": od_limit,
            }
            continue
        if is_od and not existing["is_od"]:
            existing["is_od"] = True
            existing["account_type"] = "OD"
        existing_limit = existing.get("od_limit")
        if od_limit is not None and (
            existing_limit is None or od_limit > existing_limit
        ):
            existing["od_limit"] = od_limit
    return meta


def build_track2_result(
    transactions: list[dict[str, Any]],
    *,
    counterparty_ledger: dict[str, Any] | None = None,
    pdf_integrity: dict[str, Any] | None = None,
    company_names: list[str] | None = None,
    related_parties: list[str] | None = None,
    factoring_entities: list[str] | None = None,
    account_meta: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compose Track 2 compute helpers, dispatcher, and counterparty-lookup
    into a single v6.3.5-schema result dict.

    Args:
        transactions: canonical-schema rows. May span multiple accounts.
        counterparty_ledger: upstream ledger from
            ``build_counterparty_ledger`` (or the corresponding section of
            a ``full_report.json`` export). Used to (a) drive the C26/C27
            dispatcher rung via ``build_counterparty_lookup_track2`` and
            (b) populate top_parties.
        pdf_integrity: upstream pdf_fraud_detector output (sanitised). Passed
            through unchanged.
        company_names: threaded through to the dispatcher's C01/C02 own-
            party company-root rung (RP foundation Slice 1).
        related_parties: analyst-confirmed names. The orchestrator auto-
            merges HIGH-confidence candidates from
            ``scan_related_party_candidates`` before classification (RP
            foundation Slice 2), so callers can pass an empty list to
            still get auto-confirmed RP fires.
        factoring_entities: analyst-confirmed factoring company names.
            Threaded through to the dispatcher's C10 tier-2 factoring
            rung (RP foundation Slice 3). Caller-supplied names are
            authoritative — there is no scanner equivalent for
            factoring entities yet.
        account_meta: optional per-account-no metadata
            (``{account_no: {"account_type": "Current"|"Savings"|"OD",
            "is_od": bool, "od_limit": float|None}}``). When omitted, every
            account is treated as Current / non-OD with no facility limit.

    Returns:
        v6.3.5-schema dict with all 15 top-level keys. Passes
        ``validate_track2_result`` (Draft-7 jsonschema).

        ``observations`` is populated from the consolidated + flags
        signals available today. ``own_related_transactions.transactions``
        and ``loan_transactions.*`` are built from C01-C04 (RP slices 1-2)
        and C10/C11 (RP slice 3) rows respectively.
    """
    transactions = transactions or []

    # ── Stage 0a — clean + dedup the upstream counterparty_ledger ─────────
    # ``build_counterparty_ledger`` (app.py) emits per-bucket per-bank rows
    # without a normalisation pass. The Huahub HLB ledger surfaced five
    # extraction artefacts that should collapse to one row each:
    # ``UNIDENTIFIED (CHEQUE)`` ×3, ``INTEREST`` ×5 (with trailing
    # digit-noise glued on), ``BANK FEES`` ×3, ``HUAHUB MARKETING
    # (OWN-PARTY)`` ×2, ``MARKETING`` ×2. Dedup here (before RP scanning,
    # top-parties, and the result's ledger field) so every downstream
    # consumer sees the canonical view.
    #
    # Note: the rail-label prefix problem (long CR ADV-INTERBANK GIRO ...
    # PMG PHARMACY (OUG) strings) is a separate, harder problem and is
    # NOT addressed here. See TRACK_2_HANDOFF_AFTER_SESSION_26.md.
    if isinstance(counterparty_ledger, dict):
        cps = counterparty_ledger.get("counterparties")
        if isinstance(cps, list) and cps:
            deduped = dedup_counterparty_entries(cps)
            if len(deduped) != len(cps) or any(
                deduped[i].get("counterparty_name")
                != cps[i].get("counterparty_name")
                for i in range(min(len(deduped), len(cps)))
            ):
                counterparty_ledger = dict(counterparty_ledger)
                counterparty_ledger["counterparties"] = deduped
                counterparty_ledger["total_counterparties"] = len(deduped)

    # ── Stage 0 — auto-confirm HIGH-confidence RP candidates ───────────────
    # Run the RP3 scanner over the counterparty_ledger; names whose
    # deterministic score reaches HIGH confidence are merged into
    # ``related_parties`` so the dispatcher's C03/C04 rung fires on them
    # without analyst intervention (V3-A auto-RP Step 1). Caller-supplied
    # ``related_parties`` (analyst-confirmed) take precedence and are
    # preserved verbatim; auto-confirmed names are appended (deduped by
    # case-insensitive identity).
    rp_candidates = scan_related_party_candidates(counterparty_ledger)
    auto_rp = auto_confirmed_related_parties(rp_candidates)
    if related_parties or auto_rp:
        seen_upper: set[str] = set()
        merged_rp: list[str] = []
        for name in (related_parties or []):
            if not name:
                continue
            key = name.upper()
            if key in seen_upper:
                continue
            seen_upper.add(key)
            merged_rp.append(name)
        for name in auto_rp:
            key = name.upper()
            if key in seen_upper:
                continue
            seen_upper.add(key)
            merged_rp.append(name)
        effective_related_parties = merged_rp
    else:
        effective_related_parties = None

    # ── Stage 1 — classify every row once ──────────────────────────────────
    # ``include_synthetic=True`` so synthetic bucket labels (BANK FEES,
    # FD/INTEREST, CHEQUE DEPOSIT, …) reach the dispatcher's bucket-direct
    # rung. The C26/C27 ``has_corporate_suffix`` guard prevents spurious
    # trade firings on synthetic names that lack a corp suffix.
    counterparty_lookup = build_counterparty_lookup_track2(
        transactions, counterparty_ledger, include_synthetic=True
    )
    classified = classify_transactions(
        transactions,
        counterparty_lookup=counterparty_lookup,
        company_names=company_names,
        related_parties=effective_related_parties,
        factoring_entities=factoring_entities,
    )

    # ── Stage 2 — group by account ─────────────────────────────────────────
    transactions_by_account: dict[str, list[dict[str, Any]]] = {}
    classified_by_account: dict[str, list[dict[str, Any]]] = {}
    for raw, cls in zip(transactions, classified):
        account_no = _coerce_account_no(raw)
        transactions_by_account.setdefault(account_no, []).append(raw)
        classified_by_account.setdefault(account_no, []).append(cls)

    # ── Stage 3 — monthly_analysis per account, then concatenate ──────────
    monthly: list[dict[str, Any]] = []
    for account_no, account_rows in transactions_by_account.items():
        meta = (account_meta or {}).get(account_no, {})
        account_type = "OD" if meta.get("is_od") or meta.get("account_type") == "OD" else "CR"
        bank_name = next(
            (r.get("bank") for r in account_rows if r.get("bank")),
            "Unknown",
        )
        monthly.extend(
            _build_monthly_for_account(
                account_no,
                str(bank_name),
                account_type,
                account_rows,
                classified_by_account[account_no],
            )
        )
    monthly.sort(key=lambda m: (m["account_number"], m["month"]))

    # ── Stage 4 — statutory compliance from full classified stream ────────
    salary = compute_salary_payments(transactions)
    epf = compute_epf_payments(transactions)
    socso = compute_socso_payments(transactions)
    lhdn = compute_lhdn_tax_payments(transactions)
    hrdf = compute_hrdf_payments(transactions)
    monthly_amounts = compute_statutory_monthly_amounts(
        salary_entries=salary.get("salary_payments_entries") or [],
        epf_entries=epf.get("epf_payments_entries") or [],
        socso_entries=socso.get("socso_payments_entries") or [],
        lhdn_tax_entries=lhdn.get("lhdn_tax_payments_entries") or [],
        hrdf_entries=hrdf.get("hrdf_payments_entries") or [],
    )
    statutory_compliance = _sanitize_statutory_compliance_for_schema(
        compute_statutory_compliance(monthly_amounts, transactions=transactions)
    )

    # ── Stage 5 — consolidated, large_credits, top_parties, accounts ──────
    consolidated = _build_consolidated_track2(monthly, statutory_compliance)
    large_credits_summary = compute_large_credits(transactions)
    large_credits = large_credits_summary.get("large_credits") or []
    round_figure_credits = (
        compute_round_figure_credits(transactions).get("round_figure_entries") or []
    )
    accounts = _build_accounts_track2(monthly, transactions_by_account, account_meta)
    top_parties = _build_top_parties_track2(
        counterparty_ledger, related_parties=related_parties
    )

    # ── Stage 6 — flags, unclassified, parsing_metadata ───────────────────
    summary_for_flags = _summary_for_flags_track2(
        consolidated, classified, large_credits,
        round_figure_count=len(round_figure_credits),
    )
    # Flags 8/16 print LHDN/HRDF tx counts + totals via ``lhdn_count`` /
    # ``lhdn_total`` / ``hrdf_count`` / ``hrdf_total`` — keys
    # compute_statutory_compliance does not carry (it tracks months, not tx).
    # Feed the flag builder an augmented COPY so the stored
    # statutory_compliance object stays schema-canonical.
    statutory_for_flags = {
        **statutory_compliance,
        "lhdn_count": lhdn.get("lhdn_tax_payments_count") or 0,
        "lhdn_total": lhdn.get("lhdn_tax_payments_amount") or 0.0,
        "hrdf_count": hrdf.get("hrdf_payments_count") or 0,
        "hrdf_total": hrdf.get("hrdf_payments_amount") or 0.0,
    }
    # Loan-review safety net: loan-shaped rows booked as neither facility
    # (C10/C11) nor related-party (C03/C04). Computed before the flags so
    # Flag 12 can surface the count (turns a silent miss into a review line).
    loan_review = _build_loan_review_track2(classified)
    summary_for_flags["loan_review_count"] = len(loan_review)
    indicators = compute_risk_flags(
        summary_for_flags,
        monthly_analysis=monthly,
        statutory_compliance=statutory_for_flags,
    )
    unclassified_listing = _build_unclassified_track2(
        classified, _TRACK2_UNCLASSIFIED_LISTING_THRESHOLD
    )
    parsing_metadata = _build_parsing_metadata_track2(monthly)

    # ── Stage 6b — Slice C Part 2 list builders ────────────────────────────
    own_related_list = _build_own_related_transactions_list_track2(
        classified, counterparty_lookup=counterparty_lookup
    )
    loan_lists = _build_loan_transactions_track2(classified)
    # Attach the loan-review net (computed above for Flag 12) alongside the
    # booked disbursement / repayment lists so the renderer can surface it.
    loan_lists["review"] = loan_review
    observations = _build_observations_track2(
        consolidated, indicators, statutory_compliance
    )

    # ── Stage 7 — report_info ─────────────────────────────────────────────
    period_dates = sorted(
        r.get("date") for r in transactions if isinstance(r.get("date"), str)
    )
    period_start = period_dates[0] if period_dates else ""
    period_end = period_dates[-1] if period_dates else ""
    primary_company = ""
    if company_names:
        primary_company = company_names[0]
    else:
        for row in transactions:
            cn = row.get("company_name")
            if cn:
                primary_company = str(cn)
                break

    _advisory_rp = advisory_rp_candidates(
        rp_candidates, effective_related_parties, company_names
    )

    report_info = {
        "schema_version": _TRACK2_SCHEMA_VERSION,
        "company_name": primary_company,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period_start": period_start,
        "period_end": period_end,
        "total_accounts": len(transactions_by_account),
        "total_months": len({m["month"] for m in monthly}),
        # Surface the *effective* RP list (analyst-supplied + auto-confirmed
        # HIGH-confidence RP3 candidates) as schema-compliant
        # ``{name, relationship}`` objects so analysts see the parties
        # driving the C03/C04 totals. Auto-confirmed names default to
        # ``Affiliate`` (most-neutral enum value); the Tier-4 analyst can
        # reclass via the pre-analysis template. The original
        # ``related_parties`` arg would omit auto-confirmed names AND
        # emit plain strings (schema-invalid); fixed in s22 after the
        # Principal Gas Tier-4 smoke caught the gap.
        "related_parties": [
            {"name": str(rp), "relationship": "Affiliate"}
            for rp in (effective_related_parties or [])
            if rp
        ],
        # Advisory only — MEDIUM/LOW RP3 candidates that did NOT auto-confirm
        # (no hard anchor) and are NOT in the effective list. They change no
        # figure and exclude nothing; they exist so the analyst can SEE the
        # near-misses (e.g. a relative drawing funds with no loan memo —
        # Muhafiz's SAMSI BIN IBRAHIM) instead of hunting them in the ledger.
        # Filtered to a short, material, person-only list; the analyst confirms
        # or dismisses each in the web flow.
        "related_party_candidates": [
            {
                "name": c["name"],
                "confidence": c["confidence"],
                "score": c.get("score"),
                "evidence": c.get("evidence", ""),
                "total_dr": c.get("total_dr", 0.0),
                "total_cr": c.get("total_cr", 0.0),
                "debit_count": c.get("debit_count", 0),
                "credit_count": c.get("credit_count", 0),
            }
            for c in _advisory_rp[:_ADVISORY_RP_MAX_ROWS]
        ],
        "related_party_candidates_total": len(_advisory_rp),
    }

    # ── Stage 8 — assemble result ─────────────────────────────────────────
    return {
        "report_info": report_info,
        "accounts": accounts,
        "monthly_analysis": monthly,
        "consolidated": consolidated,
        "top_parties": top_parties,
        "large_credits": large_credits,
        # Per-transaction round-figure (AML Flag 3) detail so the analyst can
        # see *which* credits drove the aggregate round_figure_cr and where in
        # the statement (date / description / amount / balance). The aggregate
        # alone (count + total) can't be traced back to source rows.
        "round_figure_credits": round_figure_credits,
        "own_related_transactions": {
            "summary": {
                "own_party_cr": consolidated.get("total_own_party_cr", 0.0),
                "own_party_dr": consolidated.get("total_own_party_dr", 0.0),
                "related_party_cr": consolidated.get("total_related_party_cr", 0.0),
                "related_party_dr": consolidated.get("total_related_party_dr", 0.0),
            },
            "transactions": own_related_list,
        },
        "loan_transactions": loan_lists,
        "flags": {"indicators": indicators},
        "observations": observations,
        "parsing_metadata": parsing_metadata,
        "unclassified_transactions": unclassified_listing,
        "classification_config": {
            "rulebook_version": _TRACK2_RULES_VERSION,
            "large_credit_threshold": _TRACK2_LARGE_CREDIT_THRESHOLD,
            "unclassified_listing_threshold": _TRACK2_UNCLASSIFIED_LISTING_THRESHOLD,
            "known_factoring_entities": list(factoring_entities or []),
            "execution_mode": _TRACK2_EXECUTION_MODE,
        },
        "pdf_integrity": _sanitize_pdf_integrity_for_schema(pdf_integrity),
        "counterparty_ledger": counterparty_ledger,
    }
