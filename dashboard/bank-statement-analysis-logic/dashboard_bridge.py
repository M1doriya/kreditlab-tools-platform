from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class _SessionState(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class _Noop:
    def __enter__(self) -> "_Noop":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def __call__(self, *_args: Any, **_kwargs: Any) -> "_Noop":
        return self

    def __getattr__(self, _name: str) -> "_Noop":
        return self

    def progress(self, *_args: Any, **_kwargs: Any) -> "_Noop":
        return self

    def metric(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _FakeStreamlit(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state = _SessionState({"is_authenticated": True})

    def set_page_config(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def selectbox(self, _label: str, options: Any, *_args: Any, **_kwargs: Any) -> Any:
        return list(options)[0] if options else None

    def file_uploader(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def text_input(self, *_args: Any, **_kwargs: Any) -> str:
        return ""

    def button(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def form_submit_button(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def columns(self, spec: Any, *_args: Any, **_kwargs: Any) -> list[_Noop]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_Noop() for _ in range(count)]

    def tabs(self, labels: Any, *_args: Any, **_kwargs: Any) -> list[_Noop]:
        return [_Noop() for _ in labels]

    def form(self, *_args: Any, **_kwargs: Any) -> _Noop:
        return _Noop()

    def expander(self, *_args: Any, **_kwargs: Any) -> _Noop:
        return _Noop()

    def empty(self, *_args: Any, **_kwargs: Any) -> _Noop:
        return _Noop()

    def progress(self, *_args: Any, **_kwargs: Any) -> _Noop:
        return _Noop()

    def stop(self) -> None:
        raise RuntimeError("Streamlit stop called during dashboard bridge import")

    def rerun(self) -> None:
        return None

    def cache_data(self, func: Any = None, **_kwargs: Any) -> Any:
        if func is None:
            return lambda wrapped: wrapped
        return func

    def __getattr__(self, _name: str) -> Any:
        def _noop(*_args: Any, **_kwargs: Any) -> _Noop:
            return _Noop()

        return _noop


def _install_streamlit_stub() -> _FakeStreamlit:
    fake = _FakeStreamlit()
    sys.modules["streamlit"] = fake
    os.environ.setdefault("BASIC_AUTH_USER", "dashboard")
    os.environ.setdefault("BASIC_AUTH_PASS", "dashboard")
    return fake


_STREAMLIT = _install_streamlit_stub()

import app as bank_app  # noqa: E402
import pandas as pd  # noqa: E402
import renderer_core  # noqa: E402
from kredit_lab_classify_track2 import (  # noqa: E402
    account_meta_from_determinations,
    build_track2_result,
    validate_track2_result,
)


BANK_NAMES = tuple(bank_app.PARSERS.keys())


def analyze_request(request: dict[str, Any]) -> dict[str, Any]:
    bank_name = str(request.get("bankName") or "").strip()
    if bank_name not in BANK_NAMES:
        raise ValueError(f"Unsupported bankName: {bank_name}")

    files = request.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("At least one PDF file is required")

    pdf_password = str(request.get("pdfPassword") or "")
    company_name_override = str(request.get("companyNameOverride") or "").strip()

    _reset_bank_session(bank_name)
    all_transactions: list[dict[str, Any]] = []
    file_results: list[dict[str, Any]] = []

    for file_item in files:
        if not isinstance(file_item, dict):
            raise ValueError("File item is malformed")

        file_path = Path(str(file_item.get("path") or ""))
        file_name = str(file_item.get("fileName") or file_path.name or "statement.pdf")
        if not file_path.exists():
            raise ValueError(f"PDF file not found: {file_path}")

        file_result = _process_pdf_file(
            bank_name=bank_name,
            file_path=file_path,
            file_name=file_name,
            pdf_password=pdf_password,
            company_name_override=company_name_override,
        )
        all_transactions.extend(file_result["transactions"])
        file_results.append(
            {
                "fileName": file_name,
                "transactionCount": len(file_result["transactions"]),
                "companyName": file_result.get("companyName"),
                "accountNo": file_result.get("accountNo"),
                "encrypted": file_result.get("encrypted", False),
            }
        )

    _apply_batch_integrity_checks()
    all_transactions = bank_app.dedupe_transactions(all_transactions)

    for index, transaction in enumerate(all_transactions):
        transaction.setdefault("__row_order", index)

    all_transactions = sorted(all_transactions, key=_transaction_sort_key)
    bank_app.st.session_state.results = all_transactions

    report_bundle = _build_reports(bank_name, all_transactions, file_results)
    return _clean_jsonable(report_bundle)


def export_request(request: dict[str, Any]) -> dict[str, Any]:
    report_payload = request.get("report")
    export_format = str(request.get("format") or "").strip().lower()

    if export_format not in {"html", "excel", "json"}:
        raise ValueError("Export format must be html, excel, or json")
    if not isinstance(report_payload, dict):
        raise ValueError("Report payload is required")

    base_name = _safe_base_name(str(request.get("fileName") or "bank-analysis"))

    if export_format == "json":
        content = report_payload.get("full_report") or report_payload.get("analysis_json") or report_payload
        raw = json.dumps(_clean_jsonable(content), indent=2).encode("utf-8")
        return {
            "fileName": f"{base_name}.json",
            "contentType": "application/json",
            "contentBase64": base64.b64encode(raw).decode("ascii"),
        }

    analysis_json = _extract_analysis_json(report_payload)

    if export_format == "html":
        html = report_payload.get("html")
        if not isinstance(html, str) or not html.strip():
            html = renderer_core.generate_interactive_html(analysis_json)
        raw = html.encode("utf-8")
        return {
            "fileName": f"{base_name}.html",
            "contentType": "text/html; charset=utf-8",
            "contentBase64": base64.b64encode(raw).decode("ascii"),
        }

    workbook = renderer_core.generate_excel(analysis_json)
    if workbook is None:
        raise RuntimeError("Bank statement Excel renderer returned no workbook")
    raw_excel = workbook.getvalue() if hasattr(workbook, "getvalue") else workbook
    return {
        "fileName": f"{base_name}.xlsx",
        "contentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "contentBase64": base64.b64encode(raw_excel).decode("ascii"),
    }


def _reset_bank_session(bank_name: str) -> None:
    bank_app.bank_choice = bank_name
    state = bank_app.st.session_state
    state.status = "running"
    state.results = []
    state.affin_statement_totals = []
    state.affin_file_transactions = {}
    state.ambank_statement_totals = []
    state.ambank_file_transactions = {}
    state.cimb_statement_totals = []
    state.rhb_statement_totals = []
    state.cimb_file_transactions = {}
    state.rhb_file_transactions = {}
    state.bank_islam_file_month = {}
    state.file_company_name = {}
    state.file_account_no = {}
    state.pdf_integrity_results = {}
    state.pdf_raw_bytes = {}
    state.account_type_determinations = []


def _process_pdf_file(
    *,
    bank_name: str,
    file_path: Path,
    file_name: str,
    pdf_password: str,
    company_name_override: str,
) -> dict[str, Any]:
    state = bank_app.st.session_state
    pdf_bytes = file_path.read_bytes()
    pdf_pw = pdf_password or None
    encrypted = False

    try:
        encrypted = bool(bank_app.is_pdf_encrypted(pdf_bytes))
    except Exception:
        encrypted = False

    if encrypted:
        try:
            pdf_bytes = bank_app.decrypt_pdf_bytes(pdf_bytes, pdf_pw)
            pdf_pw = None
        except Exception:
            pass

    try:
        integrity_result = bank_app.analyze_pdf_integrity(
            pdf_bytes, file_name, bank_hint=bank_name
        )
        state.pdf_integrity_results[file_name] = integrity_result
        state.pdf_raw_bytes[file_name] = pdf_bytes
    except Exception as exc:
        state.pdf_integrity_results[file_name] = {
            "overall_risk": "UNKNOWN",
            "finding_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "all_findings": [
                {
                    "layer": "dashboard_bridge",
                    "severity": "LOW",
                    "message": "PDF integrity check skipped",
                    "detail": str(exc),
                }
            ],
        }

    company_name = None
    account_no = None
    try:
        with bank_app.bytes_to_pdfplumber(pdf_bytes, password=pdf_pw) as meta_pdf:
            company_name = bank_app.extract_company_name(meta_pdf, max_pages=2)
    except Exception:
        company_name = None

    try:
        with bank_app.bytes_to_pdfplumber(pdf_bytes, password=pdf_pw) as meta_pdf:
            account_no = bank_app.extract_account_number(meta_pdf, max_pages=2)
    except Exception:
        account_no = None

    if company_name_override:
        company_name = company_name_override

    state.file_company_name[file_name] = company_name
    state.file_account_no[file_name] = account_no

    tx_raw = _parse_transactions(bank_name, pdf_bytes, pdf_pw, file_name)
    tx_norm = bank_app.normalize_transactions(
        tx_raw,
        default_bank=bank_name,
        source_file=file_name,
    )

    for transaction in tx_norm:
        transaction["company_name"] = company_name
        transaction["account_no"] = account_no

    determination = None
    for transaction in tx_norm:
        payload = transaction.pop("_account_type_determination", None)
        if determination is None and isinstance(payload, dict):
            determination = payload
    if determination is None:
        determination = bank_app.determine_account_type([])

    state.account_type_determinations.append(
        {
            "source_file": file_name,
            "bank": bank_name,
            "company_name": company_name,
            "account_no": account_no,
            **determination,
        }
    )

    if bank_name == "Affin Bank":
        state.affin_file_transactions[file_name] = tx_norm
    elif bank_name == "Ambank":
        state.ambank_file_transactions[file_name] = tx_norm
    elif bank_name == "CIMB Bank":
        state.cimb_file_transactions[file_name] = tx_norm
    elif bank_name == "RHB Bank":
        state.rhb_file_transactions[file_name] = tx_norm

    return {
        "transactions": tx_norm,
        "companyName": company_name,
        "accountNo": account_no,
        "encrypted": encrypted,
    }


def _parse_transactions(
    bank_name: str,
    pdf_bytes: bytes,
    pdf_password: str | None,
    file_name: str,
) -> list[dict[str, Any]]:
    state = bank_app.st.session_state

    if bank_name == "Affin Bank":
        with bank_app.bytes_to_pdfplumber(pdf_bytes, password=pdf_password) as pdf:
            totals = bank_app.extract_affin_statement_totals(pdf, file_name)
            state.affin_statement_totals.append(totals)
            return bank_app.parse_affin_bank(pdf, file_name) or []

    if bank_name == "Ambank":
        with bank_app.bytes_to_pdfplumber(pdf_bytes, password=pdf_password) as pdf:
            totals = bank_app.extract_ambank_statement_totals(pdf, file_name)
            state.ambank_statement_totals.append(totals)
            return bank_app.parse_ambank(pdf, file_name) or []

    if bank_name == "CIMB Bank":
        with bank_app.bytes_to_pdfplumber(pdf_bytes, password=pdf_password) as pdf:
            totals = bank_app.extract_cimb_statement_totals(pdf, file_name)
            state.cimb_statement_totals.append(totals)
            return bank_app.parse_transactions_cimb(pdf, file_name) or []

    if bank_name == "RHB Bank":
        with bank_app.bytes_to_pdfplumber(pdf_bytes, password=pdf_password) as pdf:
            totals = bank_app.extract_rhb_statement_totals(pdf, file_name)
            state.rhb_statement_totals.append(totals)
        return bank_app.PARSERS[bank_name](pdf_bytes, file_name) or []

    if bank_name == "Bank Islam":
        with bank_app.bytes_to_pdfplumber(pdf_bytes, password=pdf_password) as pdf:
            transactions = bank_app.parse_bank_islam(pdf, file_name) or []
            statement_month = bank_app.extract_bank_islam_statement_month(pdf)
            if statement_month:
                state.bank_islam_file_month[file_name] = statement_month
            return transactions

    return bank_app.PARSERS[bank_name](pdf_bytes, file_name) or []


def _apply_batch_integrity_checks() -> None:
    state = bank_app.st.session_state
    try:
        if len(state.pdf_raw_bytes) < 2:
            return

        batch_extra = bank_app.compare_pdf_batch(
            state.pdf_integrity_results,
            state.pdf_raw_bytes,
        )
        for file_name, extra_findings in batch_extra.items():
            if not extra_findings or file_name not in state.pdf_integrity_results:
                continue

            result = state.pdf_integrity_results[file_name]
            result.setdefault("all_findings", []).extend(extra_findings)
            result.setdefault("layer_results", {}).setdefault(
                "batch_comparison", []
            ).extend(extra_findings)
            result["finding_count"] = len(result["all_findings"])
            result["high_count"] = sum(
                1 for finding in result["all_findings"] if finding.get("severity") == "HIGH"
            )
            result["medium_count"] = sum(
                1 for finding in result["all_findings"] if finding.get("severity") == "MEDIUM"
            )
            result["low_count"] = sum(
                1 for finding in result["all_findings"] if finding.get("severity") == "LOW"
            )
            severities = [finding.get("severity", "LOW") for finding in result["all_findings"]]
            if "HIGH" in severities:
                result["overall_risk"] = "HIGH"
            elif "MEDIUM" in severities:
                result["overall_risk"] = "MEDIUM"
    except Exception:
        pass
    finally:
        state.pdf_raw_bytes = {}


def _transaction_sort_key(transaction: dict[str, Any]) -> tuple[Any, int, int, int]:
    parsed_date = bank_app.parse_any_date_for_summary(transaction.get("date"))
    page = _safe_int(transaction.get("page"), 10**9)
    sequence = _safe_int(transaction.get("seq"), 10**9)
    row_order = _safe_int(transaction.get("__row_order"), 10**12)
    sort_date = parsed_date if pd.notna(parsed_date) else pd.Timestamp.max
    return sort_date, page, sequence, row_order


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _build_reports(
    bank_name: str,
    transactions: list[dict[str, Any]],
    file_results: list[dict[str, Any]],
) -> dict[str, Any]:
    state = bank_app.st.session_state
    dataframe = pd.DataFrame(transactions)
    ledger_transactions = dataframe.to_dict(orient="records") if not dataframe.empty else []
    counterparty_ledger = bank_app.build_counterparty_ledger(ledger_transactions)

    monthly_summary_raw = bank_app.calculate_monthly_summary(state.results)
    monthly_summary = bank_app.present_monthly_summary_standard(monthly_summary_raw)
    records = _sanitize_records(dataframe.to_dict(orient="records")) if not dataframe.empty else []

    date_min = dataframe["date"].min() if "date" in dataframe.columns and not dataframe.empty else None
    date_max = dataframe["date"].max() if "date" in dataframe.columns and not dataframe.empty else None
    company_names = _nonempty_unique(
        [item.get("companyName") for item in file_results]
        + ([*dataframe.get("company_name", pd.Series([], dtype=object)).dropna().astype(str).tolist()] if not dataframe.empty else [])
    )
    account_nos = _nonempty_unique(
        [item.get("accountNo") for item in file_results]
        + ([*dataframe.get("account_no", pd.Series([], dtype=object)).dropna().astype(str).tolist()] if not dataframe.empty else [])
    )

    pdf_integrity = _build_integrity_report(state.pdf_integrity_results)

    full_report = {
        "summary": {
            "total_transactions": int(len(records)),
            "date_range": f"{date_min} to {date_max}" if date_min and date_max else None,
            "total_files_processed": len(file_results),
            "company_names": company_names,
            "account_nos": account_nos,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "pdf_integrity": pdf_integrity,
        "account_type_determinations": list(state.account_type_determinations),
        "monthly_summary": monthly_summary,
        "counterparty_ledger": counterparty_ledger,
        "transactions": records,
    }

    account_meta = account_meta_from_determinations(
        list(state.account_type_determinations)
    )
    analysis_json = build_track2_result(
        transactions=records,
        counterparty_ledger=counterparty_ledger,
        pdf_integrity=pdf_integrity,
        company_names=company_names,
        related_parties=[],
        factoring_entities=[],
        account_meta=account_meta,
    )
    try:
        validation_ok, validation_errors = validate_track2_result(analysis_json)
    except Exception as exc:
        validation_ok = False
        validation_errors = [f"Validation skipped: {exc}"]
    analysis_view = renderer_core.normalize_claude_v635(
        json.loads(json.dumps(_clean_jsonable(analysis_json)))
    )
    html = renderer_core.generate_interactive_html(analysis_view)

    return {
        "success": True,
        "tool": "bank_statement",
        "bank": bank_name,
        "html": html,
        "report": analysis_view,
        "analysis_json": analysis_view,
        "full_report": full_report,
        "source_files": file_results,
        "stats": {
            "totalTransactions": len(records),
            "totalFilesProcessed": len(file_results),
            "totalCounterparties": counterparty_ledger.get("total_counterparties", 0),
            "companyNames": company_names,
            "accountNos": account_nos,
        },
        "validation": {
            "ok": bool(validation_ok),
            "errors": list(validation_errors or [])[:20],
        },
    }


def _build_integrity_report(pdf_integrity_results: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for file_name, result in (pdf_integrity_results or {}).items():
        output[file_name] = {
            "overall_risk": result.get("overall_risk"),
            "finding_count": result.get("finding_count"),
            "high_count": result.get("high_count"),
            "medium_count": result.get("medium_count"),
            "low_count": result.get("low_count"),
            "findings": [
                {
                    "layer": finding.get("layer"),
                    "severity": finding.get("severity"),
                    "message": finding.get("message"),
                    "detail": finding.get("detail"),
                }
                for finding in result.get("all_findings", [])
            ],
        }
    return output


def _sanitize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_clean_jsonable(record) for record in records]


def _clean_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_jsonable(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(pd, "Timestamp") and isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _nonempty_unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return sorted(output)


def _extract_analysis_json(report_payload: dict[str, Any]) -> dict[str, Any]:
    candidate = report_payload.get("analysis_json")
    if not isinstance(candidate, dict):
        candidate = report_payload.get("report")
    if not isinstance(candidate, dict):
        candidate = report_payload

    if "summary" in candidate and "transactions" in candidate:
        candidate = renderer_core.adapt_to_v6(candidate)

    schema_version = ((candidate.get("report_info") or {}).get("schema_version") or "")
    if schema_version in {"6.3.4", "6.3.5"}:
        return renderer_core.normalize_claude_v635(
            json.loads(json.dumps(_clean_jsonable(candidate)))
        )
    return candidate


def _safe_base_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80] or "bank-analysis"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("analyze", "export"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        if args.mode == "analyze":
            result = analyze_request(request)
        else:
            result = export_request(request)
        Path(args.output).write_text(
            json.dumps(_clean_jsonable(result), ensure_ascii=False),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        error = {
            "success": False,
            "error": str(exc),
            "errorType": exc.__class__.__name__,
        }
        Path(args.output).write_text(json.dumps(error), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
