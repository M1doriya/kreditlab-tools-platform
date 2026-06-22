#!/usr/bin/env python3
"""
Financial Statement Analyzer — Claude API engine.

Pipeline:
    .txt (OCR'd financial statements)
        -> Claude API + KreditLab v7.9 framework (system prompt, cached)
        -> JSON
        -> validation (reusing existing validators from the Streamlit renderer)
        -> [self-correction loop on errors, reusing cached document]
        -> valid JSON

Caching: framework cached on the system block; input documents cached on the
user turn. Validation retries reuse both at ~10% read cost.

Goal: produce JSON that matches what web-Claude produces from the same .txt
inputs, so the existing Streamlit renderer ingests it unchanged.

Usage:
    # Free dry-run: count tokens + estimate cost. No API call.
    python analyze.py samples/muhafiz/*.txt --dry-run

    # Real run on Haiku (pipeline test, ~$0.10):
    python analyze.py samples/muhafiz/*.txt --model haiku --no-thinking --out out.json

    # Real run on Sonnet (~$0.30):
    python analyze.py samples/muhafiz/*.txt --model sonnet --out out.json

    # Parity test against the golden JSON (~$0.50–$0.80 on Opus w/ thinking):
    python analyze.py samples/muhafiz/*.txt --model opus --effort high \\
        --out out.json --compare samples/muhafiz/muhafiz_security_kreditlab_v7_9.expected.json

Cost guardrails:
    --max-cost-usd N    Refuse to call if pre-call estimate exceeds N (default 2.00)
    --confirm           Bypass that ceiling for a single run
    --max-retries N     Cap on self-correction passes (default 1, clamped 0..3)

Every call's actual usage and cost is appended to samples/.runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

# -----------------------------------------------------------------------------
# Configuration

REPO = Path(__file__).resolve().parent
# Version policy: framework prompts are IMMUTABLE once released. Never edit a
# released file in place — copy it to KreditLab_v7_9_X.txt, bump the changelog,
# and point this constant at the new file. ("KreditLab_v7_9 copy.txt" is the
# legacy in-place-edited file, frozen as KreditLab_v7_9_2.txt; kept for reference.)
FRAMEWORK_PATH = REPO / "KreditLab_v7_9_6.txt"
RUNS_DIR = REPO / "samples" / ".runs"

# Pricing per 1M tokens (USD). Source: Anthropic pricing page, verified 2026-06-02.
# cache_write_5m = 1.25x input; cache_read = 0.1x input.
# Opus 4.8 (released 2026-05-28) uses identical base pricing to 4.7.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-8":   {"in": 5.00,  "out": 25.00, "cache_write_5m": 6.25, "cache_read": 0.50},
    "claude-opus-4-7":   {"in": 5.00,  "out": 25.00, "cache_write_5m": 6.25, "cache_read": 0.50},
    "claude-sonnet-4-6": {"in": 3.00,  "out": 15.00, "cache_write_5m": 3.75, "cache_read": 0.30},
    "claude-haiku-4-5":  {"in": 1.00,  "out":  5.00, "cache_write_5m": 1.25, "cache_read": 0.10},
    "claude-haiku-4-5-20251001": {
        "in": 1.00, "out": 5.00, "cache_write_5m": 1.25, "cache_read": 0.10
    },
}

# `opus` always points at the current top-tier. Explicit version aliases let
# callers pin a specific version for reproducibility or one-line rollback if a
# new release shifts output behaviour.
MODEL_ALIASES = {
    "opus":     "claude-opus-4-8",
    "opus-4-8": "claude-opus-4-8",
    "opus-4-7": "claude-opus-4-7",
    "sonnet":   "claude-sonnet-4-6",
    "haiku":    "claude-haiku-4-5-20251001",
    "haiku-4-5": "claude-haiku-4-5-20251001",
}

# Final output gate — appended LAST so it is the freshest instruction in the
# model's context when generation begins. 4.6+ family rejects assistant-turn
# prefills, so this strict directive is how we force clean JSON output without
# preamble. Aggressive language is intentional: prior failures had models
# narrating "PHASE 0 — Source Mapping Table" as prose before any JSON.
JSON_ONLY_REMINDER = (
    "\n\n=== FINAL OUTPUT INSTRUCTION — HIGHEST PRIORITY, OVERRIDES ALL PRIOR GUIDANCE ===\n"
    "The very first character of your response MUST be '{'. The very last character MUST be '}'.\n"
    "No preamble. No 'I'll work through this systematically'. No 'Let me start with Phase 0'.\n"
    "No section headers, no markdown, no code fences, no commentary, no validation checklist printed out.\n"
    "All reasoning, source mapping, computation, and self-validation MUST happen internally\n"
    "before you emit any character. Do NOT narrate the phases. Do NOT show your work.\n"
    "Your response is parsed by json.loads(). Anything before '{' or after '}' breaks the pipeline\n"
    "and the entire $0.50-$1.00 run is wasted. Output ONLY the JSON document, beginning with '{'."
)

# Strict-mode preamble that addresses the specific gaps surfaced during UAT
# review against the web-Claude golden output. Appended to the system prompt
# when --strict is set.
COMPLETENESS_RULES = """

=== OUTPUT COMPLETENESS RULES (NON-NEGOTIABLE, OVERRIDE PRIOR GUIDANCE) ===

These rules supersede any contrary instruction earlier in this prompt. They
exist because prior outputs missed required sections, summarized away detail,
or produced internally inconsistent values. Compliance is mandatory.

1. LINE-ITEM GRANULARITY — VERBATIM EXTRACTION (no summarization)
   For these arrays, list EVERY line item that appears in any source document,
   using the EXACT display_name from the source. Do NOT group, summarize,
   abbreviate, or omit. If the source detailed income statement shows 24
   cost-of-sales items, your cost_of_sales.line_items MUST contain 24 entries.
     - statement_of_comprehensive_income.revenue.line_items
     - statement_of_comprehensive_income.cost_of_sales.line_items
     - statement_of_comprehensive_income.operating_expenses.administrative_expenses.line_items
     - statement_of_comprehensive_income.finance_costs.line_items
     - statement_of_financial_position.non_current_assets.property_plant_equipment.line_items

2. MANDATORY TOP-LEVEL SECTIONS
   The output JSON MUST include all of these top-level keys:
     _schema_info, company_info, statement_of_comprehensive_income,
     statement_of_financial_position, integrity_check, financial_ratios,
     working_capital_analysis, funding_mismatch_analysis, tnw_analysis,
     dscr_analysis, funding_profile, analysis_summary, report_footer.
   ADDITIONALLY, if any analyzed period is a Management Account (MA):
     _ma_limitations is MANDATORY. It must enumerate every MA-specific quirk
     observed: missing depreciation, improper equity carryforward, HP/TL shown
     as negative balances, net negative cash from overdrawn accounts mixed
     with cash, income tax embedded in admin expenses, missing trade-payables
     breakdown, etc. Use the format {"fy{year}_notes": [string, ...]}.

3. WORKING CAPITAL COMPONENTS — FILL EVERY PERIOD
   working_capital_analysis.operating_working_capital.components MUST be
   populated for EVERY period in periods_analyzed, with trade_receivables,
   inventory, and trade_payables broken out individually (integers in RM).
   An empty {} for any period is a failure.

4. TRADE PAYABLES — NO PROXYING FROM OTHER PAYABLES
   If the audited balance sheet reports only "Other Payables" (lumped) and
   has NO separate Trade Payables line: set trade_payables = 0 for that
   period. Do NOT use Other Payables as a Trade Payables proxy — that
   inflates Creditor Days and corrupts CCC. The framework's purpose is to
   SURFACE this disclosure gap (note it in _ma_limitations or
   areas_of_concern), not paper over it. Creditor Days = 0 is the correct
   result when there is no trade-payables disclosure.

5. DSCR CROSS-SECTION CONSISTENCY (Phase 3.5 reinforcement)
   The values at financial_ratios.leverage_ratios.dscr.values.{period} MUST
   exactly equal dscr_analysis.calculation.{period}.dscr. Both must use the
   same denominator: Term Loan Current + Hire Purchase Current + Total
   Finance Costs. Hire Purchase / Finance Lease is ALWAYS a term facility —
   never revolving. Before emitting, verify these two values match for every
   period; if they don't, recompute and overwrite the inconsistent one.

6. NARRATIVE DEPTH
   Every entry in analysis_summary.key_observations,
   analysis_summary.positive_indicators, analysis_summary.areas_of_concern,
   and analysis_summary.recommendations MUST be at least 2 complete sentences
   AND cite at least one specific RM figure (or ratio with unit). One-line
   observations are insufficient. Tie each narrative to specific computed
   numbers from elsewhere in this output.

7. PRE-EMIT VALIDATION CHECKLIST
   Before producing your final JSON, mentally verify:
     [ ] Every mandatory top-level section is present.
     [ ] _ma_limitations exists if any period is MA.
     [ ] Every period in periods_analyzed has populated WC components.
     [ ] DSCR values match across both locations for every period.
     [ ] No line-item array has been summarized to fewer items than the source shows.
     [ ] Trade Payables is 0 (not Other Payables) where no separate disclosure exists.
   If any check fails, fix before emitting. There is no second chance — the
   downstream renderer consumes whatever you emit.
"""


# -----------------------------------------------------------------------------
# Loading


def clean_text(text: str, label: str = "") -> str:
    """Universal mojibake guard — used on every text crossing into or out of
    the engine. Cleans patterns like 'Ã¢â€ â€™' back to '→' using ftfy. Silent
    no-op if text is already clean. Safe on JSON-escaped strings.

    Defense-in-depth: applied at all three boundaries:
      1. Framework file load        (analyze.py -> Claude)
      2. Input document file load   (analyze.py -> Claude)
      3. Model JSON response        (Claude -> analyze.py -> disk)
    So no matter where bad encoding originates (Windows file save, OCR vendor,
    or rare Claude output drift), it never reaches the JSON or the renderer.
    """
    try:
        import ftfy
    except ImportError:
        return text  # graceful degrade if ftfy unavailable
    cleaned = ftfy.fix_text(text)
    if cleaned != text and label:
        delta = abs(len(text) - len(cleaned))
        print(f"  [encoding] cleaned mojibake in {label}: {delta or '?'} char diff")
    return cleaned


def load_framework() -> str:
    if not FRAMEWORK_PATH.exists():
        sys.exit(f"[fatal] Framework not found at {FRAMEWORK_PATH}")
    return clean_text(FRAMEWORK_PATH.read_text(encoding="utf-8"), label="framework")


def load_input_files(paths: list[str]) -> str:
    """Concatenate input .txt files with delimiters so the model knows the boundaries.
    Each file goes through clean_text() to defang any OCR-introduced mojibake."""
    parts = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            sys.exit(f"[fatal] Input not found: {p}")
        body = clean_text(path.read_text(encoding="utf-8"), label=path.name)
        parts.append(f"===== SOURCE FILE: {path.name} =====\n\n{body}")
    return "\n\n".join(parts)


# -----------------------------------------------------------------------------
# Request building


def build_system(framework: str, strict: bool = False) -> list[dict]:
    """System prompt: framework -> [completeness rules] -> JSON_ONLY (always last).

    JSON_ONLY must be the freshest instruction in context at generation time,
    otherwise the model may follow earlier guidance to "narrate phases" or
    "show validation steps" and emit prose before the JSON.
    """
    text = framework
    if strict:
        text += COMPLETENESS_RULES
    text += JSON_ONLY_REMINDER  # always last; do not move
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def build_initial_messages(input_text: str) -> list[dict]:
    """First-turn user message: the document content, cached for retry reuse."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": input_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]


def build_correction_messages(input_text: str, prior_json_text: str, errors: list[str]) -> list[dict]:
    """Retry turn: documents (cached) + prior assistant JSON + error feedback."""
    err_lines = "\n".join(f"  - {e}" for e in errors)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": input_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "assistant", "content": prior_json_text},
        {
            "role": "user",
            "content": (
                "Your previous JSON output failed validation. Fix EVERY issue below and re-emit "
                "the complete corrected JSON document. Output the entire JSON with corrections "
                "applied — do not skip unchanged sections.\n\n"
                f"Validation errors ({len(errors)}):\n{err_lines}\n\n"
                "BALANCE-SHEET FIX PROTOCOL (read before correcting any SUM/IDENTITY error):\n"
                "  1. The source balance sheet ALREADY balances in its own presentation "
                "(often a 'net current assets / financed by' layout). Re-derive each section "
                "total by ANCHORING to the source's PRINTED subtotal, not from your prior output.\n"
                "  2. A section whose line items fall short of the printed subtotal is missing a "
                "line — re-scan the source and ADD the missing item. Do NOT change the subtotal.\n"
                "  3. A genuine rounding / control-account residual (e.g. SST control account) "
                "gets ONE explicit labelled line, on the SAME side and section it sits in the "
                "source. NEVER move a residual across the assets <-> equity+liabilities boundary, "
                "and NEVER drop from one side while adding to the other — that doubles the gap.\n"
                "  4. total_assets MUST equal total_equity_and_liabilities to the sen. Do NOT plug. "
                "If the source genuinely cannot be made to balance, keep the true figures and state "
                "this in the limitation note instead of forcing agreement."
                + JSON_ONLY_REMINDER
            ),
        },
    ]


# -----------------------------------------------------------------------------
# Cost accounting


def cost_from_usage(usage: dict, model: str) -> float:
    p = PRICING[model]
    return (
        usage.get("input_tokens", 0) * p["in"] / 1e6
        + usage.get("output_tokens", 0) * p["out"] / 1e6
        + usage.get("cache_creation_input_tokens", 0) * p["cache_write_5m"] / 1e6
        + usage.get("cache_read_input_tokens", 0) * p["cache_read"] / 1e6
    )


def usage_to_dict(usage: Any) -> dict:
    """SDK usage object -> plain dict. Cache fields can be None on responses without cache hits."""
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


# -----------------------------------------------------------------------------
# API call


def call_claude(
    client: anthropic.Anthropic,
    model: str,
    system: list[dict],
    messages: list[dict],
    max_tokens: int,
    thinking: bool,
    effort: str | None,
) -> tuple[str, Any]:
    """Streamed call. Returns (assistant_text, final_message_object)."""
    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    if thinking:
        # Adaptive: model decides when/how much to think. Off by default in our pipeline.
        kwargs["thinking"] = {"type": "adaptive"}
    if effort:
        kwargs["output_config"] = {"effort": effort}

    chunks: list[str] = []
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            chunks.append(text)
            # Progress dot per chunk to show liveness on long responses.
            sys.stdout.write(".")
            sys.stdout.flush()
        final = stream.get_final_message()
    sys.stdout.write("\n")
    return "".join(chunks), final


# -----------------------------------------------------------------------------
# JSON extraction & validation


def _scrub_mojibake_in_place(obj):
    """Recursively walk a parsed JSON structure and run clean_text() on every
    string value. Catches any mojibake the model produced regardless of source.
    """
    if isinstance(obj, dict):
        return {k: _scrub_mojibake_in_place(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_mojibake_in_place(v) for v in obj]
    if isinstance(obj, str):
        return clean_text(obj)
    return obj


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction. Handles markdown fences, leading/trailing prose,
    and LLM-style malformations (unescaped quotes, missing commas, trailing commas).
    Every extracted string is run through clean_text() — the final defense layer
    that catches any mojibake regardless of where it originated."""
    text = text.strip()
    parsed = None
    # Direct parse first (the happy path when the model obeys).
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    # Markdown fence (occasionally slips through despite instructions).
    if parsed is None:
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    # First '{' to last '}'.
    snippet = None
    if parsed is None:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError:
                pass
    # LLM-aware repair (handles unescaped quotes, missing commas, trailing commas).
    # Saves the spend when Opus emits slightly malformed JSON on long outputs.
    if parsed is None:
        target = snippet or text
        try:
            import json_repair  # type: ignore
        except ImportError:
            json_repair = None  # type: ignore
        if json_repair is not None:
            try:
                repaired = json_repair.repair_json(target, return_objects=True)
                if isinstance(repaired, dict):
                    parsed = repaired
                    print("  [extract_json] repaired malformed JSON via json_repair")
            except Exception:
                pass
    if parsed is None:
        # Re-raise with the snippet's offset so the caller still sees a useful error.
        if snippet is not None:
            try:
                json.loads(snippet)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON parse failed at offset {e.pos}: {e.msg}\n"
                    f"  first 200 chars: {text[:200]!r}\n"
                    f"  last 200 chars:  {text[-200:]!r}"
                )
    if parsed is None:
        raise ValueError(f"No JSON object found in response. First 500 chars:\n{text[:500]}")
    return _scrub_mojibake_in_place(parsed)


_PERIOD_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _canonical_period_key(key: str) -> str:
    """Reduce a period key to a comparison-friendly canonical form.

    Captures the meaningful year + qualifier while stripping casing, separators,
    and bookkeeping suffixes that vary between the ratios table and the
    dscr_analysis section.

    Examples:
      "fy2025"        -> "2025"
      "FY2025"        -> "2025"
      "fy2025_ma"     -> "2025ma"
      "FY_Dec_2025"   -> "dec2025"      (preserves period context)
      "ytd_sep2025"   -> "ytdsep2025"
      "fy2025_audited" -> "2025audited"

    Two keys that refer to the same actual period under different naming
    conventions ("fy2025" / "FY2025" / "FY_2025") all collapse to the same
    canonical form so the validator can compare them. Keys that refer to
    DIFFERENT periods ("fy2025" vs "ytd_sep2025") stay distinct.
    """
    if not isinstance(key, str):
        return ""
    lo = key.lower().strip()
    # Strip a leading "fy" only if directly followed by digits — keep "ytd_sep2025" intact.
    if lo.startswith("fy"):
        rest = lo[2:].lstrip("_- ")
        if rest and rest[0].isdigit():
            lo = rest
    return _PERIOD_NORMALIZE_RE.sub("", lo)


def check_dscr_cross_section_consistency(data: dict, tolerance: float = 0.02) -> list[str]:
    """Diff DSCR between the ratios table and the dedicated DSCR section.

    Opus drifts between financial_ratios.leverage_ratios.dscr.values.{period}
    and dscr_analysis.calculation.{period}.dscr — typically by mis-summing the
    denominator in one location (e.g. omitting Hire Purchase Current from the
    ratios-table compute). Strict-mode rule 5 forbids this, but lands soft
    through the API. Confirmed live on HuaHub fy2025 (1.78 vs 1.86) and Muhafiz
    fy2024 (1.02 vs 0.97).

    Period-key matching is canonicalised so "fy2025" matches "FY2025" matches
    "fy_dec_2025" — without this, the 2026-06-02 HuaHub prod run drifted
    silently because the two sections used different casing/suffixes for the
    same period, and the original `set(A) & set(B)` intersection was empty.

    Tolerance of 0.02 absorbs benign one-decimal rounding (e.g. 5.388 → 5.39 vs
    5.4) while still catching real denominator disagreements.
    """
    issues: list[str] = []
    ratio_values = get_nested(data, "financial_ratios", "leverage_ratios", "dscr", "values")
    calc = get_nested(data, "dscr_analysis", "calculation")
    if not isinstance(ratio_values, dict) or not isinstance(calc, dict):
        return issues

    # Build canonical -> original-key maps so the error message still names the
    # actual JSON keys the model emitted (helpful when the model is correcting).
    ratio_canon = {_canonical_period_key(k): k for k in ratio_values.keys()}
    calc_canon = {_canonical_period_key(k): k for k in calc.keys()}

    matched = sorted(set(ratio_canon) & set(calc_canon))
    ratio_only = sorted(set(ratio_canon) - set(calc_canon))
    calc_only = sorted(set(calc_canon) - set(ratio_canon))
    if ratio_only or calc_only:
        # ANY exclusive canonical key on either side is a naming-mismatch bug.
        # Surface it so the model can fix the keys themselves rather than have
        # the validator silently skip the period (which is exactly what bit
        # HuaHub on 2026-06-02: the FY2025 drift survived because the two
        # sections used different suffixes for the same period).
        rk = [ratio_canon[c] for c in ratio_only]
        ck = [calc_canon[c] for c in calc_only]
        issues.append(
            f"DSCR cross-section period-key mismatch. "
            f"financial_ratios.leverage_ratios.dscr.values has periods "
            f"{sorted(ratio_values.keys())}; dscr_analysis.calculation has "
            f"{sorted(calc.keys())}. Unmatched in ratios: {rk}; unmatched in "
            f"dscr_analysis: {ck}. The two sections MUST use IDENTICAL period "
            f"keys (e.g. both 'fy2025' or both 'fy2025_ma'). Pick one form and "
            f"apply it consistently throughout the JSON so cross-section "
            f"consistency can be verified for every period."
        )

    for canon in matched:
        r_key = ratio_canon[canon]
        c_key = calc_canon[canon]
        r = ratio_values.get(r_key)
        c_block = calc.get(c_key)
        c = c_block.get("dscr") if isinstance(c_block, dict) else None
        if not isinstance(r, (int, float)) or not isinstance(c, (int, float)):
            continue
        if abs(r - c) > tolerance:
            issues.append(
                f"DSCR cross-section inconsistency for period {r_key}/{c_key}: "
                f"financial_ratios.leverage_ratios.dscr.values.{r_key} = {r} "
                f"but dscr_analysis.calculation.{c_key}.dscr = {c} "
                f"(diff {abs(r - c):.2f} > tolerance {tolerance}). "
                f"Both fields MUST agree. Recompute using the dscr_analysis denominator "
                f"(Term Loan Current + Hire Purchase Current + Total Finance Costs) and "
                f"overwrite the inconsistent value in both locations with the correct result."
            )
    return issues


def check_dscr_formula_consistency(data: dict, tolerance: float = 0.02) -> list[str]:
    """Verify stored dscr_analysis.calculation.{period}.dscr ≈ ebitda / total_debt_service.

    The cross-section validator catches HORIZONTAL drift (ratios table vs
    dscr_analysis section). This catches VERTICAL drift inside dscr_analysis
    itself: the stored `dscr` field disagreeing with EBITDA divided by the
    Total Debt Service from the same period's own components. Surfaced on
    HuaHub run3 2026-06-02 where fy2023 was stored as 7.92 but
    394,326 / 45,426 = 8.68 (diff 0.76). Cross-section consistent (same wrong
    number in both locations), so the existing validator missed it.
    """
    issues: list[str] = []
    calc = get_nested(data, "dscr_analysis", "calculation")
    if not isinstance(calc, dict):
        return issues
    for period, block in calc.items():
        if not isinstance(block, dict):
            continue
        stored = block.get("dscr")
        ebitda = block.get("ebitda")
        ds = get_nested(block, "debt_service", "total_debt_service")
        if not all(isinstance(x, (int, float)) for x in (stored, ebitda, ds)):
            continue
        if ds == 0:
            continue
        computed = round(ebitda / ds, 2)
        if abs(stored - computed) > tolerance:
            issues.append(
                f"DSCR formula inconsistency for {period}: stored dscr={stored} "
                f"but EBITDA / Total Debt Service = {ebitda:,} / {ds:,} = {computed} "
                f"(diff {abs(stored - computed):.2f} > tolerance {tolerance}). "
                f"Stored DSCR MUST equal EBITDA divided by Total Debt Service per "
                f"the dscr_analysis components. Recompute and overwrite the stored "
                f"dscr field, and propagate the corrected value to "
                f"financial_ratios.leverage_ratios.dscr.values.{period}."
            )
    return issues


# Name patterns used to identify amortizing term facilities vs hire purchase in
# free-form line-item keys / display names. The model emits these keys
# dynamically (the schema only fixes the "total" node), so we match on intent.
_TERM_LOAN_NAME_RE = re.compile(
    r"term[\s_]*loan|bank[\s_]*loan|bridging|term[\s_]*facilit|\bbfi\b", re.I
)
_HP_NAME_RE = re.compile(r"hire[\s_]*purchase|finance[\s_]*lease|\bhp\b", re.I)


def _term_loan_balances_by_period(data: dict) -> dict:
    """Sum the outstanding term loan balance per period from the balance sheet.

    Scans BOTH non_current_liabilities and current_liabilities so that a
    reclassification of part of the loan from long-term to current does not look
    like a repayment — only genuine amortization moves the combined total.
    Hire purchase / finance lease lines are excluded (they carry their own
    disclosed current portion which the model already books).
    """
    balances: dict = {}
    bsp = get_nested(data, "statement_of_financial_position")
    if not isinstance(bsp, dict):
        return balances
    for section_name in ("non_current_liabilities", "current_liabilities"):
        section = bsp.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, item in section.items():
            if key == "total" or not isinstance(item, dict):
                continue
            hay = f"{key} {item.get('display_name', '')}"
            if _HP_NAME_RE.search(hay) or not _TERM_LOAN_NAME_RE.search(hay):
                continue
            vals = item.get("values")
            if not isinstance(vals, dict):
                continue
            for pk, v in vals.items():
                if isinstance(v, (int, float)):
                    balances[pk] = balances.get(pk, 0.0) + v
    return balances


def _captured_term_principal(block: dict) -> float:
    """Best estimate of the term loan principal already inside this period's
    debt service. Sums principal_repayment entries named like a term facility;
    if none are named but a total_principal exists, credits the residual left
    after removing hire-purchase-named principal (i.e. term principal bundled
    into the total)."""
    pr = get_nested(block, "debt_service", "principal_repayment")
    if not isinstance(pr, dict):
        return 0.0
    term = 0.0
    hp = 0.0
    term_keys = 0
    for k, v in pr.items():
        if k == "total_principal" or not isinstance(v, (int, float)):
            continue
        if _HP_NAME_RE.search(k):
            hp += v
        elif _TERM_LOAN_NAME_RE.search(k):
            term += v
            term_keys += 1
    if term_keys == 0:
        total_principal = pr.get("total_principal")
        if isinstance(total_principal, (int, float)):
            residual = total_principal - hp
            if residual > term:
                term = residual
    return term


def check_dscr_term_principal_completeness(data: dict, tolerance: float = 0.02) -> list[str]:
    """Catch DSCR denominators that omit real term loan principal amortization.

    The two existing DSCR validators only check INTERNAL consistency (ratios
    table vs dscr_analysis, and stored dscr vs ebitda/total_debt_service). They
    pass even when the denominator is wrong, as long as it is uniformly wrong.

    The blind spot: when an auditor discloses no current portion for a term
    loan (e.g. Standard Line FY2024 split hire purchase into current/non-current
    in Note 7 but dumped the entire term loan into non-current liabilities), the
    model finds 0 term principal and books only the hire purchase current
    portion. The loan is plainly amortizing — visible in the year-on-year drop
    in its balance sheet carrying amount — but that principal never reaches the
    DSCR. The ratio is then materially overstated (Standard Line FY2025:
    reported 1.56x vs ~0.83x once the ~RM167,737 term loan repayment is
    included).

    This check derives scheduled term principal from the decline in the term
    loan balance and flags any period where the captured principal falls short
    by enough to move the DSCR beyond `tolerance`. Balance INCREASES (new
    drawdowns) are never flagged.
    """
    issues: list[str] = []
    calc = get_nested(data, "dscr_analysis", "calculation")
    if not isinstance(calc, dict):
        return issues
    balances = _term_loan_balances_by_period(data)
    if len(balances) < 2:
        return issues  # need a prior period to measure amortization against

    bal_canon = {_canonical_period_key(k): v for k, v in balances.items()}
    calc_canon = {_canonical_period_key(k): k for k in calc.keys()}
    ordered = sorted(bal_canon.keys())  # canonical keys lead with the year

    for prev_c, curr_c in zip(ordered, ordered[1:]):
        delta = bal_canon[prev_c] - bal_canon[curr_c]  # +ve => amortized
        if delta <= 0:
            continue  # increase => drawdown; flat => nothing repaid
        c_key = calc_canon.get(curr_c)
        if c_key is None:
            continue
        block = calc.get(c_key)
        if not isinstance(block, dict):
            continue
        ebitda = block.get("ebitda")
        ds = get_nested(block, "debt_service", "total_debt_service")
        stored = block.get("dscr")
        if not all(isinstance(x, (int, float)) for x in (ebitda, ds, stored)):
            continue
        shortfall = delta - _captured_term_principal(block)
        if shortfall <= 0:
            continue
        corrected_ds = ds + shortfall
        if corrected_ds <= 0:
            continue
        corrected_dscr = round(ebitda / corrected_ds, 2)
        if abs(stored - corrected_dscr) <= tolerance:
            continue  # immaterial to the ratio
        issues.append(
            f"DSCR term-principal omission for {c_key}: the term loan balance "
            f"fell {bal_canon[prev_c]:,.0f} -> {bal_canon[curr_c]:,.0f} "
            f"(principal repaid ~{delta:,.0f}), but dscr_analysis.calculation."
            f"{c_key}.debt_service captures only {ds:,.0f} of total debt service "
            f"and credits ~{delta - shortfall:,.0f} of term principal. Add the "
            f"scheduled term loan principal (derive it from the year-on-year "
            f"decline in the loan's balance-sheet carrying amount, or the cash "
            f"flow statement's loan-repayment line, when the auditor discloses "
            f"no current portion). Corrected total debt service ~{corrected_ds:,.0f} "
            f"=> DSCR ~{corrected_dscr} (stored {stored}). Update "
            f"debt_service.principal_repayment, total_debt_service, the stored "
            f"dscr, and financial_ratios.leverage_ratios.dscr.values.{c_key}; "
            f"note in the assessment that the principal is derived, not disclosed."
        )
    return issues


def run_validators(data: dict) -> tuple[list[str], list[str]]:
    """Run the existing Streamlit-side validators plus local cross-section checks.
    Returns (errors, warnings)."""
    try:
        # Defer the import — streamlit_financial_report_v7_7 imports streamlit at top.
        sys.path.insert(0, str(REPO))
        from streamlit_financial_report_v7_7 import (  # type: ignore
            check_mathematical_integrity,
            run_financial_consistency_checks,
            validate_json_structure,
        )
    except Exception as e:
        print(f"[warn] could not import validators ({e}); skipping validation")
        return (check_dscr_cross_section_consistency(data)
                + check_dscr_formula_consistency(data)
                + check_dscr_term_principal_completeness(data)), []
    is_valid, errors, warnings = validate_json_structure(data)
    math_issues = check_mathematical_integrity(data)
    # Financial consistency suite: recomputes every total / identity / ratio /
    # cross-section figure from raw line items. Errors feed the retry loop so
    # the model self-corrects (same mechanism as the DSCR validators).
    fin_errors, fin_warnings = run_financial_consistency_checks(data)
    dscr_issues = (check_dscr_cross_section_consistency(data)
                   + check_dscr_formula_consistency(data)
                   + check_dscr_term_principal_completeness(data))
    return (list(errors) + fin_errors + dscr_issues,
            list(warnings) + list(math_issues) + fin_warnings)


# -----------------------------------------------------------------------------
# Parity diff


def get_nested(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def compare_to_expected(actual: dict, expected: dict) -> str:
    """Lightweight parity check on key numeric and structural fields."""
    lines = []

    a_top = set(actual.keys()) if isinstance(actual, dict) else set()
    e_top = set(expected.keys()) if isinstance(expected, dict) else set()
    lines.append(f"  top-level keys: actual={len(a_top)} expected={len(e_top)} match={a_top == e_top}")
    missing = sorted(e_top - a_top)
    extra = sorted(a_top - e_top)
    if missing:
        lines.append(f"  missing in actual: {missing}")
    if extra:
        lines.append(f"  extra in actual:   {extra}")

    # Spot-check the load-bearing numbers — these are what the renderer relies on
    # and what a credit analyst would eyeball first.
    checks = [
        ("Company legal name",   ("company_info", "legal_name")),
        ("Revenue FY2024",       ("statement_of_comprehensive_income", "revenue", "total", "values", "fy2024")),
        ("Revenue FY2025",       ("statement_of_comprehensive_income", "revenue", "total", "values", "fy2025")),
        ("Gross profit FY2024",  ("statement_of_comprehensive_income", "gross_profit", "values", "fy2024")),
        ("NPAT FY2024",          ("statement_of_comprehensive_income", "net_profit_after_tax", "values", "fy2024")),
        ("EBITDA FY2024",        ("statement_of_comprehensive_income", "ebitda", "values", "fy2024")),
        ("Total assets FY2024",  ("statement_of_financial_position", "total_assets", "values", "fy2024")),
        ("Total equity FY2024",  ("statement_of_financial_position", "equity", "total", "values", "fy2024")),
        ("Current ratio FY2024", ("financial_ratios", "liquidity_ratios", "current_ratio", "values", "fy2024")),
        ("DSCR FY2024",          ("financial_ratios", "leverage_ratios", "dscr", "values", "fy2024")),
        ("CCC FY2024",           ("financial_ratios", "efficiency_ratios", "cash_conversion_cycle", "values", "fy2024")),
    ]
    lines.append("")
    lines.append("  spot-check (numeric/structural):")
    matches = 0
    for label, keys in checks:
        a = get_nested(actual, *keys)
        e = get_nested(expected, *keys)
        # Allow small numeric drift on ratios (rounding to 2 dp can differ at the boundary).
        ok = a == e or (
            isinstance(a, (int, float)) and isinstance(e, (int, float)) and abs(a - e) <= 0.02
        )
        if ok:
            matches += 1
        mark = "OK" if ok else "DIFF"
        lines.append(f"    [{mark}] {label}: actual={a!r} expected={e!r}")
    lines.append("")
    lines.append(f"  parity: {matches}/{len(checks)} spot-checks match")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Run logging


def write_run_log(meta: dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fp = RUNS_DIR / f"{ts}.json"
    fp.write_text(json.dumps(meta, indent=2, default=str))
    return fp


# -----------------------------------------------------------------------------
# CLI


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Claude analysis on .txt financial statements (parity engine).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+", help="One or more .txt files (OCR'd financial statements)")
    parser.add_argument(
        "--model",
        default="haiku",
        help="Model alias: haiku | sonnet | opus (or a full ID). Default: haiku (cheap pipeline test).",
    )
    parser.add_argument("--out", default=None, help="Path to write the resulting JSON (default: stdout)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count tokens + estimate cost via the free count_tokens endpoint. No API call.",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=2.00,
        help="Pre-call cost ceiling per attempt (USD). Refuses to call if estimate exceeds. Default: 2.00",
    )
    parser.add_argument("--confirm", action="store_true", help="Bypass the cost ceiling for this run")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Self-correction attempts on validation failure (0..3). Default: 1",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64000,
        help="Output token cap. Default: 64000 (strict mode needs ~30k+ for full line-item granularity).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable adaptive thinking (cheaper, faster, less reasoning depth)",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=None,
        help="output_config.effort (Sonnet/Opus only; Haiku errors). Default: model default.",
    )
    parser.add_argument("--compare", default=None, help="Path to expected JSON for parity diff")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Append completeness-rules preamble (forces line-item granularity, mandatory "
             "sections, DSCR cross-section consistency, etc). Address UAT-found gaps.",
    )
    args = parser.parse_args()

    args.max_retries = max(0, min(args.max_retries, 3))
    model = MODEL_ALIASES.get(args.model, args.model)
    if model not in PRICING:
        sys.exit(f"[fatal] Unknown model: {model}. Known: {list(PRICING)} or aliases {list(MODEL_ALIASES)}")

    # Hard-disable effort on Haiku (the API rejects it there).
    if args.effort and model == "claude-haiku-4-5":
        print("[warn] --effort is not supported on Haiku; ignoring.")
        args.effort = None
    # Hard-disable adaptive thinking on Haiku 4.5 — only 4.6+ models support it,
    # and the API returns 400 otherwise.
    thinking_enabled = not args.no_thinking
    if thinking_enabled and model == "claude-haiku-4-5":
        print("[warn] adaptive thinking not supported on Haiku 4.5; disabling.")
        thinking_enabled = False

    print(f"  [config] model={model}  thinking={thinking_enabled}  effort={args.effort or 'default'}  strict={args.strict}")
    print(f"           max_retries={args.max_retries}  max_cost=${args.max_cost_usd:.2f}  max_tokens={args.max_tokens}")

    # Load framework + inputs
    framework = load_framework()
    input_text = load_input_files(args.inputs)
    print(f"  [inputs] framework: {len(framework):,} chars   docs: {len(input_text):,} chars across {len(args.inputs)} files")

    system = build_system(framework, strict=args.strict)
    messages = build_initial_messages(input_text)

    # API client (will read ANTHROPIC_API_KEY from env)
    try:
        client = anthropic.Anthropic()
    except anthropic.AnthropicError as e:
        sys.exit(f"[fatal] API client init failed: {e}")

    # Free token count
    print("  [tokens] counting input tokens via free count_tokens endpoint...")
    try:
        tc = client.messages.count_tokens(model=model, system=system, messages=messages)
        input_tokens = tc.input_tokens
    except anthropic.APIError as e:
        sys.exit(f"[fatal] count_tokens failed: {e}")

    # Pre-call estimate. Assume the framework + docs all hit cache_creation on the
    # first call (1.25x). Output worst case = the full max_tokens ceiling — this is
    # what we actually risk paying for if thinking explodes or the model rambles.
    # A "likely" estimate (golden JSON ~5.5k + some thinking ~6k) is also shown.
    est_output_likely = 12_000
    est_output_worst = args.max_tokens
    est_likely = cost_from_usage(
        {"cache_creation_input_tokens": input_tokens, "output_tokens": est_output_likely}, model
    )
    est_worst = cost_from_usage(
        {"cache_creation_input_tokens": input_tokens, "output_tokens": est_output_worst}, model
    )
    print(f"  [tokens] input: {input_tokens:,}  likely output: ~{est_output_likely:,}  max output: {est_output_worst:,}")
    print(f"  [cost]   likely: ${est_likely:.4f}   worst (max_tokens hit): ${est_worst:.4f}")
    # Guardrail uses the honest worst case.
    est_cost = est_worst

    if args.dry_run:
        print("\n  [dry-run] No API call made. Done.")
        return 0

    # Cost guardrail (per-call estimate)
    if est_cost > args.max_cost_usd and not args.confirm:
        print(f"\n[!] Estimated cost ${est_cost:.4f} exceeds --max-cost-usd ${args.max_cost_usd:.2f}.")
        print(f"[!] Re-run with --confirm to proceed, or raise --max-cost-usd.")
        return 3

    # -------------------------------------------------------------------------
    # First call

    call_log: list[dict] = []
    total_cost = 0.0

    print(f"\n=== Attempt 1 ({model}) ===")
    t0 = time.monotonic()
    response_text, final = call_claude(
        client, model, system, messages,
        max_tokens=args.max_tokens,
        thinking=thinking_enabled,
        effort=args.effort,
    )
    elapsed = time.monotonic() - t0
    usage = usage_to_dict(final.usage)
    cost = cost_from_usage(usage, model)
    total_cost += cost
    call_log.append({
        "attempt": 1, "elapsed_sec": round(elapsed, 1),
        "stop_reason": final.stop_reason,
        "usage": usage, "cost_usd": round(cost, 4),
        "response_len_chars": len(response_text),
    })
    print(f"  [done] {elapsed:.1f}s  stop={final.stop_reason}  usage={usage}  cost=${cost:.4f}")

    if final.stop_reason == "max_tokens":
        print("[!] Response was truncated at max_tokens. JSON is likely incomplete.")
    if final.stop_reason == "refusal":
        print("[!] Model refused. Cannot continue.")
        write_run_log({"model": model, "inputs": args.inputs, "calls": call_log, "total_cost_usd": round(total_cost, 4), "outcome": "refusal"})
        return 5

    # Parse first JSON
    try:
        data = extract_json(response_text)
    except ValueError as e:
        print(f"[!] JSON extraction failed: {e}")
        raw_path = RUNS_DIR / f"raw-{datetime.now():%Y%m%d-%H%M%S}.txt"
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(response_text)
        print(f"[!] Raw response saved to {raw_path}")
        write_run_log({"model": model, "inputs": args.inputs, "calls": call_log, "total_cost_usd": round(total_cost, 4), "outcome": "json_parse_failed"})
        return 4

    # -------------------------------------------------------------------------
    # Validation / self-correction loop

    errors, warnings = run_validators(data)
    for attempt_idx in range(args.max_retries):
        if not errors:
            break
        prev_error_count = len(errors)
        print(f"\n  [validate] {len(errors)} error(s), {len(warnings)} warning(s)")
        for e in errors[:6]:
            print(f"    - {e}")
        if len(errors) > 6:
            print(f"    ... and {len(errors) - 6} more")

        attempt_num = attempt_idx + 2
        print(f"\n=== Attempt {attempt_num} (correction; cached document reused) ===")
        retry_msgs = build_correction_messages(input_text, json.dumps(data), errors)

        t0 = time.monotonic()
        response_text, final = call_claude(
            client, model, system, retry_msgs,
            max_tokens=args.max_tokens,
            thinking=thinking_enabled,
            effort=args.effort,
        )
        elapsed = time.monotonic() - t0
        usage = usage_to_dict(final.usage)
        cost = cost_from_usage(usage, model)
        total_cost += cost
        call_log.append({
            "attempt": attempt_num, "elapsed_sec": round(elapsed, 1),
            "stop_reason": final.stop_reason,
            "usage": usage, "cost_usd": round(cost, 4),
            "response_len_chars": len(response_text),
        })
        print(f"  [done] {elapsed:.1f}s  stop={final.stop_reason}  usage={usage}  cost=${cost:.4f}")

        if final.stop_reason == "max_tokens":
            print("[!] Retry response truncated at max_tokens. JSON likely incomplete.")

        try:
            data = extract_json(response_text)
        except ValueError as e:
            print(f"[!] Retry JSON extraction failed: {e}")
            write_run_log({"model": model, "inputs": args.inputs, "calls": call_log, "total_cost_usd": round(total_cost, 4), "outcome": "retry_json_parse_failed"})
            return 4

        errors, warnings = run_validators(data)

        # EARLY-STOP: if a correction pass did not REDUCE the error count, the
        # model is stuck (e.g. a residual it cannot reconcile). Further retries
        # just re-pay full price for the same failure, so stop now. The final
        # validation report below still records the remaining error(s).
        if errors and len(errors) >= prev_error_count:
            print(
                f"\n  [validate] correction did not reduce errors "
                f"({prev_error_count} -> {len(errors)}); stopping retries to avoid wasted cost."
            )
            break

    # Final validation report
    if errors:
        print(f"\n  [validate] FINAL: {len(errors)} error(s) remain after {len(call_log)} attempt(s).")
        for e in errors[:6]:
            print(f"    - {e}")
    else:
        print(f"\n  [validate] PASS  ({len(warnings)} warning(s))")

    # -------------------------------------------------------------------------
    # Output + compare + log

    if args.out:
        out_path = Path(args.out)
        # ensure_ascii=False: writes proper UTF-8 chars directly, not \uXXXX escapes.
        # Makes the file greppable for mojibake and readable to humans.
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  [out] JSON written: {out_path}  ({len(json.dumps(data, ensure_ascii=False)):,} chars)")
    else:
        print()
        print(json.dumps(data, indent=2))

    parity_report = None
    if args.compare:
        cmp_path = Path(args.compare)
        if not cmp_path.exists():
            print(f"\n[warn] --compare path not found: {cmp_path}")
        else:
            print(f"\n=== Parity vs {cmp_path.name} ===")
            expected = json.loads(cmp_path.read_text())
            parity_report = compare_to_expected(data, expected)
            print(parity_report)

    # Summary
    print(f"\n=== Summary ===")
    print(f"  calls:      {len(call_log)}")
    print(f"  total time: {sum(c['elapsed_sec'] for c in call_log):.1f}s")
    print(f"  total cost: ${total_cost:.4f}")
    print(f"  validation: {'PASS' if not errors else f'{len(errors)} error(s)'}")

    log_path = write_run_log({
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "inputs": args.inputs,
        "thinking": thinking_enabled,
        "effort": args.effort,
        "calls": call_log,
        "total_cost_usd": round(total_cost, 4),
        "validation_errors": len(errors),
        "validation_warnings": len(warnings),
        "outcome": "success" if not errors else "validation_failed",
        "parity_report": parity_report,
    })
    print(f"  log:        {log_path}")
    return 0 if not errors else 6


if __name__ == "__main__":
    sys.exit(main())
