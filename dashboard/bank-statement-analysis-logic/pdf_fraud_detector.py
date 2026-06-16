"""
pdf_fraud_detector.py

Eight-layer PDF integrity analysis for detecting tampered bank statements.
Runs on raw PDF bytes BEFORE transaction extraction.
Always flags — never blocks.

Layers:
  1. Metadata    — editing software, date gaps, incremental saves, XMP,
                   timezone anomalies, line endings, metadata string artifacts
  2. Fonts       — dominant font fingerprint vs per-amount font comparison
  3. Text layers — overlapping text, invisible text, multiple content streams
  4. Visual      — page dimension consistency, per-page render hashes
  5. Cross-validation — PyMuPDF vs pdfplumber text disagreement
  6. Bank profile — creator/producer/font fingerprint vs known bank profiles,
                    PDF version, encryption level, font architecture checks
  7. Structural  — file size ratio, duplicate text blocks, PDF version anomalies
  8. Arithmetic  — running balance verification, cross-month balance continuity

Plus: cross-file batch comparison (compare_batch) for multi-statement uploads.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"

_SEVERITY_ORDER = {LOW: 1, MEDIUM: 2, HIGH: 3}


def _worst_severity(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return LOW
    return max(
        (f.get("severity", LOW) for f in findings),
        key=lambda s: _SEVERITY_ORDER.get(s, 0),
    )


def _finding(layer: str, severity: str, message: str, detail: Any = None) -> Dict[str, Any]:
    f = {"layer": layer, "severity": severity, "message": message}
    if detail is not None:
        f["detail"] = detail
    return f


# ---------------------------------------------------------------------------
# Known PDF editing software signatures (consumer editors only)
# ---------------------------------------------------------------------------
# IMPORTANT: Server-side PDF generation libraries that banks legitimately use
# must NOT appear here. These are safe/expected:
#   iText, ReportLab, wkhtmltopdf, JasperReports, Crystal Reports,
#   TCPDF, FPDF, Prince, WeasyPrint, Apache FOP, Stimulsoft,
#   mPDF, BIRT, Telerik, DevExpress, PDFsharp, QuestPDF
_EDITOR_SIGNATURES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"adobe\s*acrobat",
        r"adobe\s*indesign",
        r"foxit",
        r"nitro\s*p",
        r"pdfelement",
        r"wondershare",
        r"phantompdf",
        r"pdf[\-\s]?xchange",
        r"libreoffice",
        r"openoffice",
        r"microsoft\s*word",
        r"canva",
        r"inkscape",
        r"scribus",
        r"smallpdf",
        r"sejda",
        r"pdfsam",
        r"master\s*pdf",
        r"pdf\s*architect",
        r"nuance",
        r"able2extract",
        r"soda\s*pdf",
        r"pdf\s*expert",
        r"preview",  # macOS Preview can edit
    ]
]


def _is_editor(value: str) -> Optional[str]:
    """Return matched editor name if value looks like PDF editing software."""
    if not value:
        return None
    for rx in _EDITOR_SIGNATURES:
        m = rx.search(value)
        if m:
            return m.group(0)
    return None


# ---------------------------------------------------------------------------
# Layer 1 — Metadata
# ---------------------------------------------------------------------------

def _layer_metadata(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    layer = "metadata"

    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    except Exception as e:
        findings.append(_finding(layer, MEDIUM, f"Cannot read PDF metadata: {e}"))
        return findings

    info = reader.metadata or {}

    # --- Creator / Producer ---
    creator = str(info.get("/Creator", "") or "").strip()
    producer = str(info.get("/Producer", "") or "").strip()

    for label, value in [("Creator", creator), ("Producer", producer)]:
        editor = _is_editor(value)
        if editor:
            findings.append(_finding(
                layer, HIGH,
                f"{label} field contains PDF editing software: '{editor}' — "
                "banking systems do not produce statements with consumer PDF editors.",
                {"field": label, "value": value, "matched_editor": editor},
            ))

    # --- Creation vs Modification date gap ---
    creation_date = info.get("/CreationDate")
    mod_date = info.get("/ModificationDate")

    def _parse_pdf_date(d: Any) -> Optional[datetime]:
        if not d:
            return None
        s = str(d).strip()
        # PDF date format: D:YYYYMMDDHHmmSS
        m = re.match(r"D?:?(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", s)
        if not m:
            return None
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0),
            )
        except Exception:
            return None

    dt_created = _parse_pdf_date(creation_date)
    dt_modified = _parse_pdf_date(mod_date)

    if dt_created and dt_modified:
        gap = abs((dt_modified - dt_created).total_seconds())
        if gap > 86400:  # > 1 day
            days = round(gap / 86400, 1)
            sev = HIGH if days > 30 else MEDIUM
            findings.append(_finding(
                layer, sev,
                f"Creation-to-modification gap: {days} days — "
                "genuine bank statements are generated and not later modified.",
                {"created": str(dt_created), "modified": str(dt_modified), "gap_days": days},
            ))

    # --- Incremental saves ---
    try:
        raw = pdf_bytes
        incremental_count = raw.count(b"%%EOF")
        # Many bank PDF generators produce 2 %%EOF markers as standard practice
        # (e.g. Hong Leong, some CIMB variants). Only flag 3+ as suspicious.
        if incremental_count > 2:
            sev = HIGH if incremental_count > 3 else MEDIUM
            findings.append(_finding(
                layer, sev,
                f"PDF has {incremental_count} %%EOF markers (incremental saves) — "
                "indicates the document was modified and re-saved multiple times.",
                {"eof_count": incremental_count},
            ))
    except Exception:
        pass

    # --- Timezone anomaly in creation date ---
    # Genuine Malaysian bank PDFs always use +08:00 (Malaysia timezone).
    # A recreation tool on a European machine may produce +01:00 or +02:00.
    try:
        raw_creation = str(creation_date or "")
        raw_mod = str(mod_date or "")
        tz_re = re.compile(r"([+-])(\d{2})'(\d{2})'")
        create_tz = tz_re.search(raw_creation)
        mod_tz = tz_re.search(raw_mod)

        if create_tz and mod_tz:
            create_offset = create_tz.group(0)
            mod_offset = mod_tz.group(0)
            # Timezone mismatch between creation and modification
            if create_offset != mod_offset:
                findings.append(_finding(
                    layer, HIGH,
                    f"Timezone mismatch: CreationDate uses {create_offset} but "
                    f"ModificationDate uses {mod_offset}. Genuine bank PDFs have "
                    "consistent timezone across both dates.",
                    {"creation_tz": create_offset, "mod_tz": mod_offset},
                ))
            # Non-Malaysian timezone
            elif create_tz.group(2) != "08":
                findings.append(_finding(
                    layer, MEDIUM,
                    f"CreationDate timezone is {create_offset} — Malaysian bank "
                    "servers use +08:00. Non-local timezone suggests recreation "
                    "on a machine outside Malaysia.",
                    {"creation_tz": create_offset},
                ))
    except Exception:
        pass

    # --- Line ending anomaly ---
    # Genuine iText PDFs use Unix LF (\n). Windows CRLF (\r\n) suggests
    # the PDF was rebuilt on a Windows machine with a different tool.
    try:
        header_chunk = pdf_bytes[:200]
        if b"\r\n" in header_chunk:
            # Check if this is genuinely a Windows-generated PDF
            # by looking at creator. Known bank generators use Unix LF.
            if creator and re.search(r"maybank|itext|jasper|elixir", creator, re.IGNORECASE):
                findings.append(_finding(
                    layer, MEDIUM,
                    "PDF uses Windows CRLF line endings, but the claimed creator "
                    f"'{creator}' normally produces Unix LF endings. "
                    "This suggests the PDF was re-saved or recreated on Windows.",
                    {"line_endings": "CRLF", "creator": creator},
                ))
    except Exception:
        pass

    # --- Metadata string artifacts ---
    # Some PDF manipulation tools alter metadata strings (add quotes, strip whitespace).
    try:
        raw_info = reader.metadata or {}
        keywords = str(raw_info.get("/Keywords", "") or "")
        subject = str(raw_info.get("/Subject", "") or "")
        # Quoted keywords — genuine PDFs don't wrap keywords in double quotes
        if keywords.startswith('"') and keywords.endswith('"'):
            findings.append(_finding(
                layer, MEDIUM,
                "Keywords field is wrapped in double quotes — this is an artifact "
                "of PDF re-processing tools. Genuine bank PDFs don't quote keywords.",
                {"keywords": keywords},
            ))
    except Exception:
        pass

    # --- XMP metadata cross-check ---
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        xmp_raw = doc.metadata or {}
        xmp_creator = (xmp_raw.get("creator") or "").strip()
        xmp_producer = (xmp_raw.get("producer") or "").strip()

        # Check XMP for editors too
        for label, value in [("XMP Creator", xmp_creator), ("XMP Producer", xmp_producer)]:
            editor = _is_editor(value)
            if editor and not any(f.get("detail", {}).get("field") == label for f in findings):
                findings.append(_finding(
                    layer, HIGH,
                    f"{label} contains editing software: '{editor}'.",
                    {"field": label, "value": value},
                ))

        # Cross-check: if /Info creator != XMP creator, flag inconsistency
        if creator and xmp_creator and creator.lower() != xmp_creator.lower():
            findings.append(_finding(
                layer, MEDIUM,
                "Metadata inconsistency: /Info Creator differs from XMP Creator.",
                {"info_creator": creator, "xmp_creator": xmp_creator},
            ))

        doc.close()
    except Exception:
        pass

    return findings


# ---------------------------------------------------------------------------
# Layer 2 — Fonts (the "killer feature")
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(r"-?[\d,]{1,15}\.\d{2}")


def _layer_fonts(pdf_bytes: bytes,
                 bank_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    layer = "fonts"

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        findings.append(_finding(layer, MEDIUM, f"Cannot open PDF for font analysis: {e}"))
        return findings

    # Collect font usage across the entire document
    all_font_spans: List[Dict] = []  # every text span
    money_spans: List[Dict] = []     # spans containing monetary amounts

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        except Exception:
            continue

        for block in blocks:
            if block.get("type") != 0:  # text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    raw_font = span.get("font", "unknown")
                    font_size = round(span.get("size", 0), 1)
                    font_key = f"{raw_font}|{font_size}"
                    # Font family key: strip Bold/Italic/Light weight suffixes
                    # so "Helvetica-Bold" and "Helvetica" are the same family
                    font_family = re.sub(
                        r"[-,](Bold|Italic|Light|Medium|Regular|Semibold|Thin|Black|Heavy|Condensed|Oblique|BoldItalic|BoldOblique)+",
                        "", raw_font, flags=re.IGNORECASE,
                    )
                    family_key = f"{font_family}|{font_size}"
                    family_only = font_family  # size-agnostic: just the font family
                    span_info = {
                        "page": page_idx + 1,
                        "font": raw_font,
                        "size": font_size,
                        "font_key": font_key,
                        "family_key": family_key,
                        "family_only": family_only,
                        "text": text[:80],
                        "color": span.get("color", 0),
                    }

                    all_font_spans.append(span_info)

                    if _MONEY_RE.search(text):
                        money_spans.append(span_info)

    doc.close()

    if not all_font_spans:
        return findings

    # Determine dominant font across the whole document
    font_counter = Counter(s["font_key"] for s in all_font_spans)
    dominant_font_key = font_counter.most_common(1)[0][0]

    # Determine dominant font FAMILY for monetary amounts specifically
    # Using family_only (ignores Bold/Italic AND size) to avoid false positives.
    # Banks legitimately use different sizes for headers, body, footers, and
    # different weights for totals vs regular amounts. Only a completely
    # DIFFERENT font family (e.g. Arial vs Times) is suspicious.
    if money_spans:
        money_family_counter = Counter(s["family_only"] for s in money_spans)
        dominant_money_family = money_family_counter.most_common(1)[0][0]
        # For display, show the most common exact font_key
        money_font_counter = Counter(s["font_key"] for s in money_spans)
        dominant_money_font = money_font_counter.most_common(1)[0][0]
        total_money = len(money_spans)

        # BUG-003 (2026-05-05): bank-aware allowance. UOB BIBPlus statements
        # legitimately render monetary amounts in 3+ font families (Latin
        # body in ArialUnicodeMS, CJK fragments in FangSong, totals/labels
        # in OpenSans-Bold). These show as "different family" against the
        # single dominant family but are NOT fraud signals. Look up the
        # bank's expected_fonts (already curated in _BANK_PROFILES) and
        # treat any monetary span whose family is in that set as part of
        # the dominant for consistency-scoring purposes. Falls back to the
        # original behaviour when bank is unknown or has no expected_fonts.
        bank_known_families: set = set()
        try:
            _detected_bank = bank_hint or _detect_bank(pdf_bytes)
            if _detected_bank:
                for _profile in _BANK_PROFILES.get(_detected_bank, []):
                    for _f in _profile.get("expected_fonts", set()) or set():
                        # Strip variant suffix to match family_only
                        bank_known_families.add(
                            re.sub(
                                r"-(Regular|Bold|Italic|BoldItalic|Light|"
                                r"Medium|Heavy|Black|Semibold|Thin|"
                                r"Condensed|Oblique)$",
                                "", _f, flags=re.IGNORECASE,
                            )
                        )
        except Exception:
            pass

        if bank_known_families:
            effective_dominant = sum(
                1 for s in money_spans
                if s["family_only"] == dominant_money_family
                or s["family_only"] in bank_known_families
            )
            consistency_pct = round(100.0 * effective_dominant / total_money, 1)
        else:
            dominant_count = money_family_counter[dominant_money_family]
            consistency_pct = round(100.0 * dominant_count / total_money, 1)

        # Flag any monetary amounts using a completely DIFFERENT font family
        # (and NOT in the bank's known-legitimate font set).
        anomalous_font = [
            s for s in money_spans
            if s["family_only"] != dominant_money_family
            and s["family_only"] not in bank_known_families
        ]

        # --- Color anomaly: monetary amounts in a different color ---
        money_color_counter = Counter(s["color"] for s in money_spans)
        dominant_money_color = money_color_counter.most_common(1)[0][0]
        anomalous_color = [s for s in money_spans if s["color"] != dominant_money_color]

        # Combine font + color anomalies (union of suspicious spans)
        anomalous_texts = {s["text"] for s in anomalous_font} | {s["text"] for s in anomalous_color}
        all_anomalous = [s for s in money_spans if s["text"] in anomalous_texts]

        # Deduplicate by (page, text)
        seen = set()
        deduped_anomalous: List[Dict] = []
        for s in all_anomalous:
            key = (s["page"], s["text"])
            if key not in seen:
                seen.add(key)
                deduped_anomalous.append(s)

        if deduped_anomalous and consistency_pct < 85:
            # Only flag when consistency is below 85% — multilingual bank
            # statements (e.g. UOB using FangSong for CJK + Helvetica for Latin)
            # naturally have 5-15% of amounts in a different font family.
            sev = HIGH if len(deduped_anomalous) <= 10 else MEDIUM
            reasons = []
            if anomalous_font:
                reasons.append(f"{len(anomalous_font)} with different font/size")
            if anomalous_color:
                reasons.append(f"{len(anomalous_color)} with different color")

            findings.append(_finding(
                layer, sev,
                f"SUSPICIOUS AMOUNTS DETECTED: {len(deduped_anomalous)} of {total_money} "
                f"monetary amounts are anomalous ({', '.join(reasons)}). "
                f"Font consistency: {consistency_pct}%. "
                f"Dominant money font: {dominant_money_font}. "
                "When someone edits an amount in a PDF editor, the replacement text almost "
                "always uses a slightly different font, size, or color.",
                {
                    "dominant_money_font": dominant_money_font,
                    "dominant_money_color": dominant_money_color,
                    "font_consistency_pct": consistency_pct,
                    "total_money_spans": total_money,
                    "anomalous_count": len(deduped_anomalous),
                    "anomalous_amounts": [
                        {
                            "page": s["page"],
                            "text": s["text"],
                            "font": s["font"],
                            "size": s["size"],
                            "color": s["color"],
                            "font_matches_dominant": s["font_key"] == dominant_money_font,
                            "color_matches_dominant": s["color"] == dominant_money_color,
                        }
                        for s in deduped_anomalous[:20]  # cap detail
                    ],
                },
            ))
        else:
            # Clean — all monetary amounts are consistent
            findings.append(_finding(
                layer, LOW,
                f"Font consistency: {consistency_pct}% — all {total_money} monetary amounts "
                f"use the same font ({dominant_money_font}) and color. No anomalies detected.",
                {
                    "dominant_money_font": dominant_money_font,
                    "font_consistency_pct": consistency_pct,
                    "total_money_spans": total_money,
                    "anomalous_count": 0,
                    "status": "CLEAN",
                },
            ))

    # Flag if the document uses an unusually high number of distinct fonts
    distinct_fonts = len(font_counter)
    if distinct_fonts > 8:
        findings.append(_finding(
            layer, LOW,
            f"Document uses {distinct_fonts} distinct font/size combinations — "
            "bank statements typically use 2-5.",
            {"distinct_fonts": distinct_fonts},
        ))

    return findings


# ---------------------------------------------------------------------------
# Layer 3 — Text layers (overlapping, invisible, multi-stream)
# ---------------------------------------------------------------------------

def _layer_text_layers(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    layer = "text_layers"

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        findings.append(_finding(layer, MEDIUM, f"Cannot open PDF for text layer analysis: {e}"))
        return findings

    pages_with_overlap = []
    pages_with_invisible = []
    pages_with_multi_stream = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1

        # --- Overlapping text detection ---
        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            text_positions: List[Tuple[float, float, float, float, str]] = []

            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        # Skip very short text (single chars, punctuation)
                        # — these cause false overlaps with adjacent spans
                        if not text or len(text) < 3:
                            continue
                        bbox = span.get("bbox") or line.get("bbox")
                        if bbox:
                            text_positions.append((bbox[0], bbox[1], bbox[2], bbox[3], text))

            # Check for overlapping bounding boxes with DIFFERENT content.
            # Many banks (e.g. Maybank) render table columns as separate text
            # layers that technically overlap bounding boxes — that is normal
            # formatting, NOT tampering.  We only flag when overlapping spans
            # contain different text (i.e. one value placed on top of another).
            suspicious_overlaps: List[Tuple[str, str]] = []
            for i in range(len(text_positions)):
                for j in range(i + 1, len(text_positions)):
                    ax0, ay0, ax1, ay1, text_a = text_positions[i]
                    bx0, by0, bx1, by1, text_b = text_positions[j]

                    # Require substantial overlap (not just column edges touching)
                    overlap_x = max(0, min(ax1, bx1) - max(ax0, bx0))
                    overlap_y = max(0, min(ay1, by1) - max(ay0, by0))

                    # Minimum overlap: 30% of the smaller span's width AND height
                    width_a = ax1 - ax0
                    width_b = bx1 - bx0
                    height_a = ay1 - ay0
                    height_b = by1 - by0
                    min_width = min(width_a, width_b) if min(width_a, width_b) > 0 else 1
                    min_height = min(height_a, height_b) if min(height_a, height_b) > 0 else 1

                    if overlap_x / min_width < 0.3 or overlap_y / min_height < 0.3:
                        continue

                    # Same or very similar text = normal formatting (e.g. table cell re-render)
                    norm_a = re.sub(r"\s+", "", text_a.lower())
                    norm_b = re.sub(r"\s+", "", text_b.lower())
                    if norm_a == norm_b:
                        continue
                    # One is a substring of the other = partial re-render, not tampering
                    if norm_a in norm_b or norm_b in norm_a:
                        continue

                    # Skip garbled/non-readable text — CID font encoding
                    # artifacts from certain PDF generators (e.g. Bank Islam's
                    # openhtmltopdf) produce Unicode replacement chars,
                    # sequential ASCII codes (123456789:;<=>), and random
                    # letter sequences that overlap visually but are just
                    # font mapping issues, not tampered content.
                    # Filter out CID font encoding artifacts (e.g. Bank
                    # Islam openhtmltopdf) that produce garbage overlaps.
                    #
                    # Financial PDF tampering = changing amounts. The original
                    # amount is hidden beneath the replacement. Require BOTH
                    # sides to be monetary AND on the same row baseline —
                    # otherwise narrow-column table layouts from absolute-
                    # positioning generators (e.g. Ambank omsgen) produce
                    # description↔amount bbox overlaps that look suspicious
                    # but are just normal formatting.
                    _amt_re = re.compile(r"\d[\d,]*\.\d{2}")
                    if not (_amt_re.search(text_a) and _amt_re.search(text_b)):
                        continue
                    # Real amount substitution happens on the same row baseline
                    # (within a few points — tampering tools may offset slightly).
                    if abs(ay1 - by1) > 5.0:
                        continue

                    suspicious_overlaps.append((text_a[:40], text_b[:40]))
                    if len(suspicious_overlaps) >= 3:
                        break
                if len(suspicious_overlaps) >= 3:
                    break

            if suspicious_overlaps:
                pages_with_overlap.append({
                    "page": page_num,
                    "examples": [{"text_a": a, "text_b": b} for a, b in suspicious_overlaps[:3]],
                })

        except Exception:
            pass

        # --- Invisible / white text detection ---
        # Banks legitimately use white text for security hashes, account-number
        # watermarks, and page identifiers. These are short, single-value strings
        # (one hash, one account number). Real tampering hides TRANSACTION data
        # — amounts with decimal points that overlap visible content.
        # Only flag when invisible text contains actual monetary amounts AND
        # there are multiple such spans on the page (not just one watermark).
        try:
            invisible_money_spans = 0
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        color = span.get("color", 0)
                        size = span.get("size", 12)
                        if color == 16777215 or size < 0.5:
                            # Only count if text is a monetary amount (not a hash/ID)
                            if _MONEY_RE.search(text) and len(text) < 20:
                                invisible_money_spans += 1
            # Flag only if there are multiple hidden monetary amounts
            if invisible_money_spans >= 2:
                if page_num not in pages_with_invisible:
                    pages_with_invisible.append(page_num)
        except Exception:
            pass

        # --- Multiple content streams ---
        try:
            xref = page.xref
            page_obj = doc.xref_object(xref)
            # Count /Contents references — array = multiple streams
            contents_count = page_obj.count("/Contents")
            if contents_count > 1:
                pages_with_multi_stream.append(page_num)
            elif "/Contents" in page_obj:
                # Check if Contents is an array
                m = re.search(r"/Contents\s*\[([^\]]+)\]", page_obj)
                if m:
                    refs = re.findall(r"\d+\s+\d+\s+R", m.group(1))
                    if len(refs) > 1:
                        pages_with_multi_stream.append(page_num)
        except Exception:
            pass

    doc.close()

    if pages_with_overlap:
        page_nums = [p["page"] if isinstance(p, dict) else p for p in pages_with_overlap]
        findings.append(_finding(
            layer, HIGH,
            f"Overlapping DIFFERENT text detected on {len(pages_with_overlap)} page(s) — "
            "text with different content placed over existing content is a strong sign of "
            "PDF editing (original value hidden beneath replacement).",
            {"pages": pages_with_overlap, "page_numbers": page_nums},
        ))

    if pages_with_invisible:
        findings.append(_finding(
            layer, HIGH,
            f"Invisible/white text found on {len(pages_with_invisible)} page(s) — "
            "hidden original values beneath edited replacement text.",
            {"pages": pages_with_invisible},
        ))

    if pages_with_multi_stream:
        findings.append(_finding(
            layer, LOW,
            f"Multiple content streams on {len(pages_with_multi_stream)} page(s) — "
            "common in bank statements with layered layouts.",
            {"pages": pages_with_multi_stream},
        ))

    return findings


# ---------------------------------------------------------------------------
# Layer 4 — Visual (page dimensions, render hashes)
# ---------------------------------------------------------------------------

def _layer_visual(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    layer = "visual"

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        findings.append(_finding(layer, MEDIUM, f"Cannot open PDF for visual analysis: {e}"))
        return findings

    if len(doc) == 0:
        doc.close()
        return findings

    # --- Page dimension consistency ---
    dimensions: List[Tuple[int, float, float]] = []
    for page_idx in range(len(doc)):
        rect = doc[page_idx].rect
        w = round(rect.width, 1)
        h = round(rect.height, 1)
        dimensions.append((page_idx + 1, w, h))

    dim_counter = Counter((d[1], d[2]) for d in dimensions)
    if len(dim_counter) > 1:
        dominant_dim = dim_counter.most_common(1)[0][0]
        mismatched = [d for d in dimensions if (d[1], d[2]) != dominant_dim]
        findings.append(_finding(
            layer, MEDIUM,
            f"Page dimension inconsistency: {len(mismatched)} page(s) differ from the dominant "
            f"size ({dominant_dim[0]}x{dominant_dim[1]}pt) — "
            "may indicate replaced pages from a different source.",
            {
                "dominant_dimensions": {"width": dominant_dim[0], "height": dominant_dim[1]},
                "mismatched_pages": [
                    {"page": d[0], "width": d[1], "height": d[2]} for d in mismatched
                ],
            },
        ))

    # --- Per-page render hashes (for comparison) ---
    page_hashes: List[Dict[str, Any]] = []
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=72)  # low DPI for speed
            h = hashlib.sha256(pix.samples).hexdigest()[:16]
            page_hashes.append({"page": page_idx + 1, "hash": h})
    except Exception:
        pass

    # We don't flag on hashes alone — they're included in the report for
    # downstream comparison (e.g., comparing the same statement month from
    # different sources).
    if page_hashes:
        findings.append(_finding(
            layer, LOW,
            "Page render hashes computed for cross-document comparison.",
            {"page_hashes": page_hashes},
        ))

    doc.close()
    return findings


# ---------------------------------------------------------------------------
# Layer 5 — Cross-validation (PyMuPDF vs pdfplumber)
# ---------------------------------------------------------------------------

def _normalize_for_comparison(text: str) -> str:
    """Collapse whitespace and strip for fuzzy comparison."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _layer_cross_validation(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    layer = "cross_validation"

    # Extract text with PyMuPDF
    fitz_texts: Dict[int, str] = {}
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            fitz_texts[i + 1] = _normalize_for_comparison(doc[i].get_text())
        doc.close()
    except Exception as e:
        findings.append(_finding(layer, LOW, f"PyMuPDF extraction failed: {e}"))
        return findings

    # Extract text with pdfplumber
    plumber_texts: Dict[int, str] = {}
    try:
        pdf = pdfplumber.open(BytesIO(pdf_bytes))
        for i, page in enumerate(pdf.pages):
            plumber_texts[i + 1] = _normalize_for_comparison(page.extract_text())
        pdf.close()
    except Exception as e:
        findings.append(_finding(layer, LOW, f"pdfplumber extraction failed: {e}"))
        return findings

    # Compare page by page
    all_pages = sorted(set(fitz_texts.keys()) | set(plumber_texts.keys()))
    disagreement_pages: List[Dict[str, Any]] = []

    for page_num in all_pages:
        ft = fitz_texts.get(page_num, "")
        pt = plumber_texts.get(page_num, "")

        if not ft and not pt:
            continue

        # Calculate similarity (simple character-level)
        if ft == pt:
            continue

        # Use length-based heuristic: if lengths differ significantly, flag
        len_ft = len(ft)
        len_pt = len(pt)

        if len_ft == 0 or len_pt == 0:
            if abs(len_ft - len_pt) > 20:
                disagreement_pages.append({
                    "page": page_num,
                    "fitz_len": len_ft,
                    "plumber_len": len_pt,
                    "reason": "One engine extracted text, the other found none",
                })
            continue

        # Check for significant differences
        ratio = min(len_ft, len_pt) / max(len_ft, len_pt)
        if ratio < 0.85:
            disagreement_pages.append({
                "page": page_num,
                "fitz_len": len_ft,
                "plumber_len": len_pt,
                "length_ratio": round(ratio, 3),
                "reason": f"Text length ratio {round(ratio, 3)} — significant extraction disagreement",
            })

    if disagreement_pages:
        total_pages = len(all_pages)
        disagree_ratio = len(disagreement_pages) / total_pages if total_pages else 0

        # If ALL or nearly all pages disagree (>70%), this is a systematic
        # engine/encoding difference (e.g. Bank Islam openhtmltopdf, encrypted
        # PDFs, CJK fonts) — not targeted tampering.  Tampering affects
        # specific pages, not every single one.  70% catches 3/4-page
        # statements where 3 pages disagree systematically.
        if disagree_ratio > 0.7:
            sev = LOW
            msg = (
                f"Text extraction disagreement on {len(disagreement_pages)}/{total_pages} "
                "pages between PyMuPDF and pdfplumber. Since nearly ALL pages differ, "
                "this is a systematic engine/encoding difference (common with certain "
                "PDF generators), not evidence of tampering."
            )
        else:
            sev = HIGH if len(disagreement_pages) >= 3 else MEDIUM
            msg = (
                f"Text extraction disagreement on {len(disagreement_pages)} page(s) between "
                "PyMuPDF and pdfplumber — may indicate hidden layers, non-standard encoding, "
                "or injected content not visible to all renderers."
            )
        findings.append(_finding(layer, sev, msg, {"pages": disagreement_pages}))

    return findings


# ---------------------------------------------------------------------------
# Layer 6 — Bank profile matching
# ---------------------------------------------------------------------------
# Known legitimate PDF generation profiles for Malaysian banks.
# Each bank may have multiple valid profiles (e.g. Islamic vs conventional).
# Fields: creator_pattern, producer_pattern, expected_fonts (set of base names).

_BANK_PROFILES: Dict[str, List[Dict[str, Any]]] = {
    # ── Profiled from real PDFs in Bank-Statement/ folder ──
    "maybank": [
        {
            "name": "Maybank2u.com (conventional/Islamic)",
            "creator_re": re.compile(r"maybank2u\.com", re.IGNORECASE),
            "producer_re": re.compile(r"itext\s*2\.1\.3", re.IGNORECASE),
            "expected_fonts": {"Tahoma", "NSimSun", "MicrosoftSansSerif"},
            "expected_pdf_version": "1.4",
            "expected_font_type": "Type0",  # genuine uses Type0 CIDFont
            "max_fonts_per_page": 4,  # genuine uses 3 fonts/page
            "forbidden_fonts": {"Calibri", "TimesNewRomanPSMT"},
        },
        {
            "name": "Maybank Islamic (Elixir/iText)",
            "creator_re": re.compile(r"elixir\s*report", re.IGNORECASE),
            "producer_re": re.compile(r"itext\s*2\.1\.7", re.IGNORECASE),
            "expected_fonts": {"ArialUnicodeMS"},
        },
    ],
    "cimb": [
        {
            "name": "CIMB JasperReports",
            "creator_re": re.compile(r"jasperreports", re.IGNORECASE),
            "producer_re": re.compile(r"itext\s*1\.4", re.IGNORECASE),
            "expected_fonts": {"Helvetica"},
        },
        {
            "name": "CIMB Vault Rendering",
            "creator_re": re.compile(r"vault\s*rendering", re.IGNORECASE),
            "producer_re": re.compile(r"rendering\s*engine", re.IGNORECASE),
            "expected_fonts": {"Helvetica"},
        },
    ],
    "public bank": [
        {
            "name": "Public Bank iTextSharp",
            "creator_re": re.compile(r".*", re.IGNORECASE),  # empty creator is normal
            "producer_re": re.compile(r"itextsharp", re.IGNORECASE),
            "expected_fonts": {"Helvetica"},
        },
    ],
    "rhb": [
        {
            "name": "RHB JasperReports",
            "creator_re": re.compile(r"jasperreports", re.IGNORECASE),
            "producer_re": re.compile(r"itext", re.IGNORECASE),
            "expected_fonts": {"Helvetica"},
        },
        {
            "name": "RHB PDFium",
            "creator_re": re.compile(r".*"),  # empty creator
            "producer_re": re.compile(r"pdfium", re.IGNORECASE),
            "expected_fonts": {"Calibri"},
        },
        {
            "name": "RHB Vault Rendering",
            "creator_re": re.compile(r"vault", re.IGNORECASE),
            "producer_re": re.compile(r"vault", re.IGNORECASE),
            "expected_fonts": set(),  # uses proprietary CZG fonts
        },
    ],
    "hong leong": [
        {
            "name": "Hong Leong iText (Dax fonts)",
            "creator_re": re.compile(r"5\.7", re.IGNORECASE),  # creator is "5.7.0"
            "producer_re": re.compile(r"itext\s*1\.4", re.IGNORECASE),
            "expected_fonts": {"Dax-Bold", "Dax-Regular"},
        },
    ],
    "bank rakyat": [
        {
            "name": "Bank Rakyat PoDoFo",
            "creator_re": re.compile(r".*"),  # empty creator
            "producer_re": re.compile(r"podofo", re.IGNORECASE),
            "expected_fonts": {"Helvetica"},
        },
        {
            "name": "Bank Rakyat iText",
            "creator_re": re.compile(r".*"),  # empty creator
            "producer_re": re.compile(r"itext\s*2\.1", re.IGNORECASE),
            "expected_fonts": {"Helvetica"},
        },
    ],
    "bank islam": [
        {
            "name": "Bank Islam openhtmltopdf",
            "creator_re": re.compile(r".*"),
            "producer_re": re.compile(r"openhtmltopdf", re.IGNORECASE),
            "expected_fonts": set(),
        },
        {
            "name": "Bank Islam iText",
            "creator_re": re.compile(r".*"),
            "producer_re": re.compile(r"itext\s*2\.1", re.IGNORECASE),
            "expected_fonts": set(),
        },
        {
            "name": "Bank Islam PoDoFo",
            "creator_re": re.compile(r".*"),
            "producer_re": re.compile(r"podofo", re.IGNORECASE),
            "expected_fonts": set(),
        },
        {
            "name": "Bank Islam PDFium",
            "creator_re": re.compile(r".*"),
            "producer_re": re.compile(r"pdfium", re.IGNORECASE),
            "expected_fonts": set(),
        },
        {
            "name": "Bank Islam Acrobat Distiller",
            "creator_re": re.compile(r".*"),
            "producer_re": re.compile(r"acrobat\s*distiller", re.IGNORECASE),
            "expected_fonts": set(),
        },
        {
            "name": "Bank Islam Microsoft Print to PDF",
            "creator_re": re.compile(r".*"),
            "producer_re": re.compile(r"microsoft.*print\s*to\s*pdf", re.IGNORECASE),
            "expected_fonts": set(),
        },
    ],
    "affin": [
        {
            "name": "Affin Bank (image-based)",
            "creator_re": re.compile(r".*"),  # empty creator/producer is normal for Affin
            "producer_re": re.compile(r".*"),
            "expected_fonts": set(),  # image-based PDFs have no fonts
        },
    ],
    "alliance": [
        {
            "name": "Alliance Bank Quadient/Inspire",
            "creator_re": re.compile(r"quadient|inspire", re.IGNORECASE),
            "producer_re": re.compile(r"quadient|inspire", re.IGNORECASE),
            "expected_fonts": {"ArialMT"},
        },
    ],
    "ambank": [
        {
            "name": "AmBank Streamline Pdfgen",
            "creator_re": re.compile(r"streamline\s*pdfgen", re.IGNORECASE),
            "producer_re": re.compile(r"compugr", re.IGNORECASE),
            "expected_fonts": set(),  # uses Arial/Verdana variants
        },
        {
            "name": "AmBank omsgen",
            "creator_re": re.compile(r"omsgen", re.IGNORECASE),
            "producer_re": re.compile(r".*"),
            "expected_fonts": set(),  # uses Helvetica/Times variants
        },
    ],
    "agrobank": [
        {
            "name": "AgroBank Microsoft Word",
            "creator_re": re.compile(r"microsoft.*word", re.IGNORECASE),
            "producer_re": re.compile(r".*"),
            "expected_fonts": {"Calibri", "Tahoma"},
        },
        {
            "name": "AgroBank WPS Writer",
            "creator_re": re.compile(r"wps\s*writer", re.IGNORECASE),
            "producer_re": re.compile(r".*"),
            "expected_fonts": set(),
        },
    ],
    "bsn": [
        {
            "name": "BSN e-Statement",
            "creator_re": re.compile(r"bsn|jasper", re.IGNORECASE),
            "producer_re": re.compile(r"itext|jasper", re.IGNORECASE),
            "expected_fonts": set(),
        },
    ],
    "muamalat": [
        {
            "name": "Bank Muamalat PlanetPress",
            "creator_re": re.compile(r".*"),  # empty creator
            "producer_re": re.compile(r"planetpress", re.IGNORECASE),
            "expected_fonts": {"Calibri-Italic", "CourierNewPS-BoldMT", "ArialNarrow"},
        },
    ],
    "ocbc": [
        {
            "name": "OCBC Streamline Pdfgen",
            "creator_re": re.compile(r"streamline\s*pdfgen", re.IGNORECASE),
            "producer_re": re.compile(r"compugr", re.IGNORECASE),
            "expected_fonts": {"Arial"},
        },
        {
            "name": "OCBC omsgen",
            "creator_re": re.compile(r"omsgen", re.IGNORECASE),
            "producer_re": re.compile(r"omsgen", re.IGNORECASE),
            "expected_fonts": {"Arial"},
        },
    ],
    "uob": [
        {
            "name": "UOB JasperReports",
            "creator_re": re.compile(r"jasperreports", re.IGNORECASE),
            "producer_re": re.compile(r"itext\s*2\.1", re.IGNORECASE),
            # BUG-003 (2026-05-05): added FangSong (UOB embeds it for CJK
            # characters in some monetary amounts) so the multi-script
            # UOB BIBPlus layout doesn't trigger font-mismatch findings.
            "expected_fonts": {
                "Helvetica", "OpenSans", "ArialUnicodeMS", "FangSong",
            },
        },
        {
            # BUG-003 (2026-05-05): UOB occasionally emits BIBPlus statements
            # via a secondary PDFium-based generator path (March-2026 7357
            # observed). Same font palette as JasperReports; both are
            # legitimate UOB outputs.
            "name": "UOB PDFium",
            "creator_re": re.compile(r"pdfium", re.IGNORECASE),
            "producer_re": re.compile(r"pdfium", re.IGNORECASE),
            "expected_fonts": {
                "Helvetica", "OpenSans", "ArialUnicodeMS", "FangSong",
            },
        },
    ],
}

# Consumer/scanning tools that should NOT produce bank statements
_NON_BANK_PRODUCERS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"adobe\s*scan",
        r"camscanner",
        r"genius\s*scan",
        r"microsoft\s*lens",
        r"google\s*drive",
        r"scanbot",
        r"tiny\s*scanner",
        r"clear\s*scanner",
        r"scanner\s*pro",
        r"adobe\s*photoshop",
        r"gimp",
        r"paint",
        r"canva",
    ]
]


# Keywords per bank for text-content detection. Longer/more-specific keywords
# get higher weight per occurrence so that 'maybank islamic' (one mention,
# len=15) beats a stray short reference.
# IMPORTANT: 'maybank islamic' is intentionally included under 'maybank'
# so a Maybank Islamic statement is identified as Maybank.
# Short bare tokens (rhb/uob/bsn) are matched on WORD BOUNDARIES, not with a
# trailing space — Wung Choon RHB Reflex statements (2026-06) only carry
# 'rhb' at line ends ('JOMPAY NON RHB\n') and inside 'www.rhbgroup.com',
# which the old 'rhb ' literal never hit, so the bank came back None.
_BANK_TEXT_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("maybank", ["maybank islamic", "maybankislamicberhad", "maybank2u",
                 "malayan banking", "maybank"]),
    ("hong leong", ["hong leong bank", "hong leong islamic", "hong leong"]),
    ("public bank", ["public bank", "public berhad"]),
    ("bank rakyat", ["bank rakyat"]),
    ("bank islam", ["bank islam"]),
    ("alliance", ["alliance bank"]),
    ("agrobank", ["agrobank", "agro bank"]),
    ("bsn", ["bank simpanan nasional", "bsn"]),
    ("muamalat", ["muamalat"]),
    ("affin", ["affin bank", "affin islamic", "affin"]),
    ("ambank", ["ambank", "ammb holdings"]),
    ("cimb", ["cimb bank", "cimb islamic", "cimb group", "cimb"]),
    # 'reflex cash management' / 'rhbgroup' — RHB's corporate cash-management
    # platform and web domain; Reflex transaction statements carry no other
    # printable RHB branding.
    ("rhb", ["rhb bank", "rhb islamic", "reflex cash management",
             "rhbgroup", "rhb"]),
    ("ocbc", ["ocbc bank", "ocbc al-amin", "ocbc"]),
    # BUG-003 (2026-05-05): added 'uobm' (UOB Malaysia branch code,
    # printed in 'Account Branch: UOBM Kepong' on every UOB BIBPlus
    # statement) so low-activity 2-page statements with only one or
    # two 'uob' tokens are still recognised as UOB.
    ("uob", ["uob malaysia", "uobm", "uob"]),
]

# First chunk of page 1 — where the issuing bank's letterhead lives. A hit
# here is worth far more than repeated mentions inside transaction rows
# (e.g. dozens of 'CIMB CHQ DEP' cheque-deposit descriptions on a Maybank
# statement must not outvote the 'Maybank Islamic Berhad' letterhead).
_HEADER_CHARS = 600
_HEADER_HIT_BONUS = 25.0


def _count_keyword(text: str, kw: str) -> int:
    """Occurrences of ``kw`` in ``text`` on word boundaries (both sides)."""
    return len(re.findall(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])",
                          text))


def _detect_bank_from_text(pdf_bytes: bytes) -> Optional[str]:
    """Identify the bank from the first 2 pages of text content.

    Uses frequency-weighted scoring plus a strong letterhead bonus: the bank
    named in the first ~600 chars of page 1 (the statement header) wins over
    banks that merely appear often inside transaction descriptions.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for i in range(min(2, len(doc))):
            text += doc[i].get_text().lower()
        doc.close()
    except Exception:
        return None

    header = text[:_HEADER_CHARS]

    scores: Dict[str, float] = {}
    for bank, keywords in _BANK_TEXT_KEYWORDS:
        best = 0.0
        for kw in keywords:
            count = _count_keyword(text, kw)
            if count > 0:
                weight = count * (1.0 + len(kw) / 10.0)
                if _count_keyword(header, kw) > 0:
                    weight += _HEADER_HIT_BONUS
                if weight > best:
                    best = weight
        if best > 0:
            scores[bank] = scores.get(bank, 0.0) + best

    if not scores:
        return None
    return max(scores, key=scores.get)


def _metadata_bank_candidates(creator: str, producer: str) -> Dict[str, int]:
    """Banks whose profiles SPECIFICALLY match the creator/producer strings.

    Returns ``{bank: score}`` where score is the number of non-wildcard
    regexes the bank's best profile matches (1–2). Wildcard patterns
    (``.*``) are ignored — several profiles use them for "empty creator is
    normal", which previously made EVERY pdf with any metadata match half
    the profile table and forced the text fallback even when the creator
    literally said 'Maybank2u.com' (Wung Choon 2026-06: a genuine Maybank
    statement full of 'CIMB CHQ DEP' cheque-deposit rows was profiled as
    CIMB by the text fallback).

    NOTE: a higher score does NOT mean a more trustworthy bank — shared
    generators (omsgen → AmBank+OCBC, PDFium → RHB+UOB+Bank Islam) differ
    only in how tightly each bank's profile happened to be written. Scores
    are used as a last-resort tiebreak only.
    """
    best_by_bank: Dict[str, int] = {}
    for bank, profiles in _BANK_PROFILES.items():
        for profile in profiles:
            score = 0
            for field, pattern in ((creator, profile["creator_re"]),
                                   (producer, profile["producer_re"])):
                if pattern.pattern == ".*":
                    continue  # wildcard carries no identification signal
                if field and pattern.search(field):
                    score += 1
            if score > best_by_bank.get(bank, 0):
                best_by_bank[bank] = score
    return best_by_bank


def _detect_bank(pdf_bytes: bytes) -> Optional[str]:
    """Identify the bank using metadata first, falling back to text content.

    Cascade:
      1. Exactly one bank's non-wildcard profile regexes match the
         creator/producer → trust it (transaction descriptions routinely
         name OTHER banks, so text must not override specific metadata).
      2. Several banks share the generator (omsgen, PDFium, Jasper+iText) →
         text detection; accepted if it lands on a metadata candidate.
      3. Text picked an outsider or nothing → re-score text restricted to
         the candidates (a poisoned full-text vote like 59 'CIMB CHQ DEP'
         rows can't elect CIMB if CIMB doesn't match the metadata).
      4. Still nothing → highest metadata specificity if unique, else
         whatever text said.
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        info = reader.metadata or {}
        creator = str(info.get("/Creator", "") or "").strip()
        producer = str(info.get("/Producer", "") or "").strip()
    except Exception:
        creator = producer = ""

    candidates = _metadata_bank_candidates(creator, producer)

    if len(candidates) == 1:
        return next(iter(candidates))

    text_bank = _detect_bank_from_text(pdf_bytes)
    if not candidates:
        return text_bank
    if text_bank in candidates:
        return text_bank

    # Metadata narrowed it to a few banks but full text scoring picked an
    # outsider (or nothing) — re-score text among the candidates only.
    keywords_by_bank = dict(_BANK_TEXT_KEYWORDS)
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(doc[i].get_text().lower()
                       for i in range(min(2, len(doc))))
        doc.close()
    except Exception:
        text = ""
    best_bank, best_hits = None, 0
    for bank in candidates:
        hits = sum(_count_keyword(text, kw)
                   for kw in keywords_by_bank.get(bank, []))
        if hits > best_hits:
            best_bank, best_hits = bank, hits
    if best_bank:
        return best_bank

    # No candidate is named in the text at all — fall back to the most
    # specific metadata match if it is unique, else to the text verdict.
    top = max(candidates.values())
    tops = [b for b, s in candidates.items() if s == top]
    if len(tops) == 1:
        return tops[0]
    return text_bank


def _extract_font_names(pdf_bytes: bytes) -> set:
    """Extract all font base names used in the PDF."""
    fonts = set()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            for font in page.get_fonts():
                # font[3] is the base font name; strip subset prefix (e.g. ABCDEF+)
                name = font[3] or ""
                if "+" in name:
                    name = name.split("+", 1)[1]
                if name:
                    fonts.add(name)
        doc.close()
    except Exception:
        pass
    return fonts


def _layer_bank_profile(pdf_bytes: bytes,
                        bank_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check if PDF metadata matches known legitimate bank generation profiles."""
    findings: List[Dict[str, Any]] = []
    layer = "bank_profile"

    # The caller's bank selection (the parser that will read this file) is
    # authoritative when provided; otherwise detect from metadata/text.
    bank = bank_hint or _detect_bank(pdf_bytes)
    if not bank:
        findings.append(_finding(layer, LOW, "Could not identify bank from text content."))
        return findings

    profiles = _BANK_PROFILES.get(bank, [])
    if not profiles:
        findings.append(_finding(layer, LOW, f"No profile database for '{bank}' yet."))
        return findings

    # Get PDF metadata
    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        info = reader.metadata or {}
        creator = str(info.get("/Creator", "") or "").strip()
        producer = str(info.get("/Producer", "") or "").strip()
    except Exception:
        creator = producer = ""

    # Get fonts
    pdf_fonts = _extract_font_names(pdf_bytes)

    # Check against non-bank producers first (scanning apps, image editors)
    for rx in _NON_BANK_PRODUCERS:
        for label, value in [("Creator", creator), ("Producer", producer)]:
            if rx.search(value):
                findings.append(_finding(
                    layer, HIGH,
                    f"PDF was produced by scanning/consumer software: '{value}' — "
                    f"genuine {bank.title()} statements are server-generated, not scanned or "
                    "recreated. This is a strong indicator of a fabricated document.",
                    {"field": label, "value": value, "detected_bank": bank},
                ))

    # Match against known profiles (prefer creator match, then producer-only)
    matched_profile = None
    for profile in profiles:
        creator_ok = profile["creator_re"].search(creator) if creator else False
        if creator_ok:
            matched_profile = profile
            break
    if not matched_profile:
        for profile in profiles:
            producer_ok = profile["producer_re"].search(producer) if producer else False
            if producer_ok:
                matched_profile = profile
                break

    if matched_profile:
        # Matched a known profile — check font consistency if we have expected fonts.
        # BUG-003 (2026-05-05): family-aware comparison. Banks subset-embed
        # variant weights (OpenSans-Regular, OpenSans-Bold, ArialUnicodeMS-Bold,
        # …) but the profile lists base families (OpenSans, ArialUnicodeMS).
        # Strip the variant suffix from the PDF's actual fonts so the
        # set-difference checks family presence rather than exact-string
        # match. Generic improvement — every existing bank profile benefits
        # without needing per-bank changes.
        expected = matched_profile.get("expected_fonts", set())
        if expected and pdf_fonts:
            pdf_families = {
                re.sub(
                    r"-(Regular|Bold|Italic|BoldItalic|Light|Medium|"
                    r"Heavy|Black|Semibold|Thin|Condensed|Oblique)$",
                    "", f, flags=re.IGNORECASE,
                )
                for f in pdf_fonts
            }
            missing = expected - pdf_families
            extra = pdf_fonts - expected
            if missing:
                findings.append(_finding(
                    layer, MEDIUM,
                    f"Font mismatch vs {matched_profile['name']} profile: "
                    f"missing expected fonts {missing}.",
                    {"profile": matched_profile["name"], "missing": sorted(missing),
                     "extra": sorted(extra), "found": sorted(pdf_fonts)},
                ))
            if not missing:
                findings.append(_finding(
                    layer, LOW,
                    f"PDF matches known {matched_profile['name']} profile "
                    f"(creator/producer and fonts consistent).",
                    {"profile": matched_profile["name"], "status": "MATCH"},
                ))

        # --- PDF version check against profile ---
        expected_version = matched_profile.get("expected_pdf_version")
        if expected_version:
            try:
                version_line = pdf_bytes[:20].decode("latin-1", errors="replace")
                vm = re.search(r"%PDF-(\d+\.\d+)", version_line)
                if vm and vm.group(1) != expected_version:
                    findings.append(_finding(
                        layer, HIGH,
                        f"PDF version {vm.group(1)} does not match expected "
                        f"{expected_version} for {matched_profile['name']}. "
                        f"Genuine {bank.title()} uses PDF {expected_version}. "
                        "A version upgrade indicates the PDF was re-processed.",
                        {"expected_version": expected_version,
                         "actual_version": vm.group(1),
                         "profile": matched_profile["name"]},
                    ))
            except Exception:
                pass

        # --- Font architecture check ---
        # Genuine Maybank uses Type0 CIDFont (3 fonts/page). Tampered files
        # decompose into 5-9 TrueType+Type0 mixed fonts per page.
        max_fonts = matched_profile.get("max_fonts_per_page")
        if max_fonts:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_idx in range(min(2, len(doc))):  # check first 2 pages
                    page_fonts = doc[page_idx].get_fonts()
                    if len(page_fonts) > max_fonts:
                        findings.append(_finding(
                            layer, HIGH,
                            f"Page {page_idx+1} has {len(page_fonts)} fonts "
                            f"(expected max {max_fonts} for {matched_profile['name']}). "
                            "Genuine statements embed fewer, larger CIDFonts. "
                            "Excess fonts indicate PDF reconstruction.",
                            {"page": page_idx+1, "font_count": len(page_fonts),
                             "expected_max": max_fonts,
                             "fonts": [f[3] for f in page_fonts]},
                        ))
                        break  # one finding is enough
                doc.close()
            except Exception:
                pass

        # --- Forbidden fonts check ---
        forbidden = matched_profile.get("forbidden_fonts", set())
        if forbidden and pdf_fonts:
            found_forbidden = forbidden & pdf_fonts
            if found_forbidden:
                findings.append(_finding(
                    layer, HIGH,
                    f"Foreign fonts detected: {sorted(found_forbidden)}. "
                    f"Genuine {bank.title()} statements never use these fonts. "
                    "They were introduced by the tool used to recreate the PDF.",
                    {"forbidden_fonts": sorted(found_forbidden),
                     "all_fonts": sorted(pdf_fonts),
                     "profile": matched_profile["name"]},
                ))

        # --- Font subset prefix pattern check ---
        # Genuine iText embeds fonts with random 6-letter prefixes (e.g. RAZIHG+).
        # Some re-creation tools use sequential prefixes like BCDEEE+, BCDFEE+.
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            all_prefixes = []
            for page in doc:
                for font in page.get_fonts():
                    name = font[3] or ""
                    if "+" in name:
                        prefix = name.split("+")[0]
                        if len(prefix) == 6 and prefix.isalpha():
                            all_prefixes.append(prefix)
            doc.close()
            if len(all_prefixes) >= 4:
                unique_prefixes = set(all_prefixes)
                # Check for sequential BCD-style pattern
                if len(unique_prefixes) >= 3:
                    sorted_prefixes = sorted(unique_prefixes)
                    # Check if prefixes share a common 3+ char root
                    common_root = sorted_prefixes[0][:3]
                    sharing_root = sum(1 for p in sorted_prefixes if p.startswith(common_root))
                    if sharing_root >= 3 and sharing_root == len(unique_prefixes):
                        findings.append(_finding(
                            layer, MEDIUM,
                            f"Font subset prefixes share sequential pattern "
                            f"(all start with '{common_root}'): {sorted_prefixes}. "
                            "Genuine bank PDFs use random prefixes per generation run. "
                            "Sequential prefixes suggest a single re-embedding tool.",
                            {"prefixes": sorted_prefixes, "common_root": common_root},
                        ))
        except Exception:
            pass

        # --- Encryption level check ---
        # Genuine Maybank2u uses V4R4 AES-128 (for older) or V5R6 AES-256.
        # But the version must be consistent: PDF 1.4 cannot have V5R6 encryption.
        try:
            version_line = pdf_bytes[:20].decode("latin-1", errors="replace")
            vm = re.search(r"%PDF-(\d+\.\d+)", version_line)
            if vm:
                pdf_ver = vm.group(1)
                # V5R6 AES-256 requires PDF 1.7+. If expected PDF version is 1.4
                # but encryption was upgraded, that confirms re-processing.
                if expected_version and pdf_ver != expected_version:
                    reader2 = PdfReader(BytesIO(pdf_bytes), strict=False)
                    if reader2.is_encrypted:
                        encrypt_dict = reader2.trailer.get("/Encrypt")
                        if encrypt_dict:
                            v_val = encrypt_dict.get("/V")
                            r_val = encrypt_dict.get("/R")
                            if v_val and int(str(v_val)) >= 5:
                                findings.append(_finding(
                                    layer, HIGH,
                                    f"Encryption upgraded to V{v_val}R{r_val} (AES-256) "
                                    f"but genuine {bank.title()} uses PDF {expected_version} "
                                    f"with V4R4 (AES-128). The encryption level was changed "
                                    "during PDF re-processing.",
                                    {"encryption_v": str(v_val), "encryption_r": str(r_val),
                                     "pdf_version": pdf_ver,
                                     "expected_version": expected_version},
                                ))
        except Exception:
            pass

        # --- File size anomaly vs bank profile ---
        # Genuine Maybank2u statements are ~25-30KB/page. Tampered ones are 48-73KB/page.
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            n_pages = len(doc)
            doc.close()
            if n_pages > 0:
                kb_per_page = len(pdf_bytes) / n_pages / 1024
                # For Maybank2u.com specifically, genuine is ~25-30KB/page
                if matched_profile.get("expected_pdf_version") == "1.4" and kb_per_page > 45:
                    findings.append(_finding(
                        layer, MEDIUM,
                        f"File size {round(kb_per_page)}KB/page is unusually large for "
                        f"{matched_profile['name']} (expected ~25-30KB/page). "
                        "Larger size indicates font re-embedding or reconstruction.",
                        {"kb_per_page": round(kb_per_page, 1),
                         "expected_kb_per_page": "25-30"},
                    ))
        except Exception:
            pass
    else:
        # No profile matched — flag as suspicious
        if creator or producer:
            sev = HIGH if (creator and not any(p["creator_re"].search(creator) for p in profiles)) else MEDIUM
            findings.append(_finding(
                layer, sev,
                f"PDF creator/producer does NOT match any known {bank.title()} "
                f"generation profile. Creator='{creator}', Producer='{producer}'. "
                f"Genuine {bank.title()} statements use specific server-side software. "
                "A mismatched profile suggests the PDF was recreated from scratch.",
                {"detected_bank": bank, "creator": creator, "producer": producer,
                 "known_profiles": [p["name"] for p in profiles],
                 "found_fonts": sorted(pdf_fonts)},
            ))
        else:
            findings.append(_finding(
                layer, MEDIUM,
                f"PDF has no creator/producer metadata — genuine {bank.title()} "
                "statements always include generation metadata.",
                {"detected_bank": bank},
            ))

    return findings


# ---------------------------------------------------------------------------
# Layer 7 — Structural anomalies
# ---------------------------------------------------------------------------

def _layer_structural(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Detect structural anomalies: file size ratio, duplicate text, PDF version."""
    findings: List[Dict[str, Any]] = []
    layer = "structural"

    file_size = len(pdf_bytes)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        findings.append(_finding(layer, MEDIUM, f"Cannot open PDF: {e}"))
        return findings

    num_pages = len(doc)
    if num_pages == 0:
        doc.close()
        return findings

    # --- File size per page ratio ---
    # Genuine bank statements with embedded logos are typically 50-400 KB per page.
    # Scanned/recreated PDFs are often 400KB+ per page (full raster images).
    # Only flag when clearly excessive (> 500KB/page for multi-page docs).
    size_per_page = file_size / num_pages
    threshold = 500_000 if num_pages > 1 else 800_000  # single-page = more lenient
    if size_per_page > threshold:
        findings.append(_finding(
            layer, MEDIUM,
            f"Unusually large file: {round(file_size/1024)}KB for {num_pages} pages "
            f"({round(size_per_page/1024)}KB/page). Genuine bank statements are typically "
            "50-400KB/page. Very large size suggests scanned images or recreated PDF.",
            {"file_size_kb": round(file_size/1024), "pages": num_pages,
             "kb_per_page": round(size_per_page/1024)},
        ))

    # --- PDF version check ---
    # Most bank statements use PDF 1.4-1.7. PDF 1.3 or lower is unusual.
    try:
        version_line = pdf_bytes[:20].decode("latin-1", errors="replace")
        version_match = re.search(r"%PDF-(\d+\.\d+)", version_line)
        if version_match:
            version = version_match.group(1)
            findings.append(_finding(
                layer, LOW,
                f"PDF version: {version}",
                {"pdf_version": version},
            ))
    except Exception:
        pass

    # --- Duplicate structural text detection ---
    # Catches copy-paste artifacts: repeated STRUCTURAL elements (full-width footers,
    # disclaimers, "THANK YOU" blocks) appearing multiple times on the same page.
    # Recurring transaction descriptions (payee names, transfer types) naturally
    # repeat and must NOT be flagged.  Only flag text that:
    #   (a) spans > 50% of page width (structural, not table cell), AND
    #   (b) appears 2+ times with > 200pt vertical spread, AND
    #   (c) is long enough to be a structural element (> 40 chars)
    _STRUCTURAL_MIN_LEN = 40
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_width = page.rect.width

        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            text_lines: List[Tuple[str, float, float]] = []  # (text, y, width)
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    bbox = line.get("bbox", [0, 0, 0, 0])
                    line_width = bbox[2] - bbox[0]
                    if len(line_text) >= _STRUCTURAL_MIN_LEN and line_width > page_width * 0.5:
                        text_lines.append((line_text, bbox[1], line_width))

            text_counter = Counter(t[0] for t in text_lines)
            for text, count in text_counter.items():
                if count >= 2:
                    positions = [t[1] for t in text_lines if t[0] == text]
                    pos_spread = max(positions) - min(positions)
                    if pos_spread > 200:
                        findings.append(_finding(
                            layer, HIGH,
                            f"Duplicate structural text on page {page_num}: "
                            f"'{text[:60]}' appears {count} times at different sections "
                            f"(spread: {round(pos_spread)}pt). "
                            "This is a sign of copy-paste PDF construction.",
                            {"page": page_num, "text": text[:80], "count": count,
                             "position_spread_pt": round(pos_spread)},
                        ))
                        break
        except Exception:
            pass

    # --- Image-only page detection ---
    # If pages have images but very little text, it's likely a scan
    text_sparse_pages = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        images = page.get_images()
        text = page.get_text().strip()
        if images and len(text) < 50:
            text_sparse_pages.append(page_idx + 1)

    if text_sparse_pages:
        findings.append(_finding(
            layer, MEDIUM,
            f"Image-only pages detected (pages {text_sparse_pages[:5]}) — "
            "pages contain images but almost no extractable text. "
            "May indicate a scanned document rather than native PDF.",
            {"pages": text_sparse_pages},
        ))

    doc.close()
    return findings


# ---------------------------------------------------------------------------
# Layer 8 — Arithmetic validation
# ---------------------------------------------------------------------------

# Captures optional trailing DR/CR tag to detect overdraft accounts.
# DR after a balance means the account is overdrawn (negative balance).
_BALANCE_RE = re.compile(
    r"(BEGINNING\s+BALANCE|ENDING\s+BALANCE|LEDGER\s+BALANCE|"
    r"TOTAL\s+DEBIT|TOTAL\s+CREDIT)\s*:?\s*([\d,]+\.\d{2})\s*(DR|CR)?",
    re.IGNORECASE,
)

_TXN_LINE_RE = re.compile(
    r"(\d{1,2}/\d{2})\s+.+?\s+([\d,]*\.?\d{1,2})([+-])\s+([\d,]*\.?\d{1,2})\s*$",
    re.MULTILINE,
)


def _parse_amount(s: str) -> float:
    """Convert '1,234.56' to float."""
    return float(s.replace(",", ""))


def _layer_arithmetic(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Validate running balance arithmetic and internal consistency."""
    findings: List[Dict[str, Any]] = []
    layer = "arithmetic"

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        findings.append(_finding(layer, LOW, f"Cannot open PDF: {e}"))
        return findings

    # Extract all text for balance parsing
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()

    # --- Extract summary values ---
    # Track DR/CR tags to detect overdraft (OD) accounts.
    # In OD accounts, BEGINNING/ENDING BALANCE are tagged "DR" meaning
    # the balance is negative (the customer owes the bank).
    summaries: Dict[str, float] = {}
    summary_tags: Dict[str, str] = {}  # key -> "DR"/"CR"/""
    for m in _BALANCE_RE.finditer(full_text):
        key = m.group(1).upper().strip()
        val = _parse_amount(m.group(2))
        tag = (m.group(3) or "").upper()
        summaries[key] = val
        summary_tags[key] = tag

    beginning = summaries.get("BEGINNING BALANCE")
    ending = summaries.get("ENDING BALANCE")
    total_debit = summaries.get("TOTAL DEBIT")
    total_credit = summaries.get("TOTAL CREDIT")

    # Detect OD account: balances tagged DR mean negative (overdrawn).
    # For OD accounts the formula is:
    #   -beginning + credits - debits = -ending  (when both are DR)
    #   i.e. beginning_signed + credits - debits = ending_signed
    is_od = summary_tags.get("BEGINNING BALANCE") == "DR" or summary_tags.get("ENDING BALANCE") == "DR"
    beginning_signed = beginning
    ending_signed = ending
    if beginning is not None and summary_tags.get("BEGINNING BALANCE") == "DR":
        beginning_signed = -beginning
    if ending is not None and summary_tags.get("ENDING BALANCE") == "DR":
        ending_signed = -ending

    # --- Check: beginning + credits - debits = ending ---
    if all(v is not None for v in [beginning, ending, total_debit, total_credit]):
        expected_ending = round(beginning_signed + total_credit - total_debit, 2)
        actual_ending = ending_signed
        if abs(expected_ending - actual_ending) > 0.01:
            od_note = " (OD account detected — DR balances treated as negative)" if is_od else ""
            findings.append(_finding(
                layer, HIGH,
                f"Balance arithmetic failure{od_note}: Beginning ({beginning_signed:,.2f}) + "
                f"Total Credit ({total_credit:,.2f}) - Total Debit ({total_debit:,.2f}) "
                f"= {expected_ending:,.2f}, but Ending Balance is {actual_ending:,.2f}. "
                f"Difference: {abs(expected_ending - actual_ending):,.2f}. "
                "Transaction amounts or balances have been tampered.",
                {"beginning": beginning_signed, "total_credit": total_credit,
                 "total_debit": total_debit, "expected_ending": expected_ending,
                 "actual_ending": actual_ending,
                 "is_od": is_od,
                 "difference": round(abs(expected_ending - actual_ending), 2)},
            ))
        else:
            od_note = " (OD account — DR balances treated as negative)" if is_od else ""
            findings.append(_finding(
                layer, LOW,
                f"Balance arithmetic verified{od_note}: Beginning + Credits - Debits = Ending Balance.",
                {"beginning": beginning_signed, "ending": actual_ending,
                 "total_debit": total_debit, "total_credit": total_credit,
                 "is_od": is_od,
                 "status": "VERIFIED"},
            ))

    # --- Check running balance line by line ---
    # Parse transaction lines: date, amount+/-, running balance
    txn_amounts: List[Tuple[float, str, float]] = []  # (amount, sign, balance_after)
    for m in _TXN_LINE_RE.finditer(full_text):
        try:
            amount = _parse_amount(m.group(2))
            sign = m.group(3)
            balance = _parse_amount(m.group(4))
            txn_amounts.append((amount, sign, balance))
        except Exception:
            continue

    if len(txn_amounts) >= 2 and beginning is not None:
        running = beginning_signed
        errors = []
        for i, (amount, sign, expected_balance) in enumerate(txn_amounts):
            if sign == "-":
                running = round(running - amount, 2)
            else:
                running = round(running + amount, 2)
            if abs(running - expected_balance) > 0.01:
                errors.append({
                    "txn_index": i + 1,
                    "expected": running,
                    "shown": expected_balance,
                    "diff": round(abs(running - expected_balance), 2),
                })
                running = expected_balance  # reset to shown balance to continue

        if errors:
            # Self-consistency guard: if the macro-level balance check
            # (beginning + credits - debits = ending) already PASSED,
            # running-balance errors are almost certainly a row-extraction
            # artefact, not real tampering.  Downgrade to MEDIUM.
            macro_passed = any(
                f.get("severity") == LOW
                and f.get("layer") == layer
                and "VERIFIED" in str(f.get("detail", {}).get("status", ""))
                for f in findings
            )
            # OD accounts: the line-level regex cannot parse DR-tagged
            # running balances reliably.  When macro already verified AND
            # this is an OD account, demote all the way to LOW.
            if macro_passed and is_od:
                sev = LOW
            elif macro_passed:
                sev = MEDIUM
            else:
                sev = HIGH
            msg = (
                f"Running balance errors in {len(errors)} transaction(s): "
                "the shown balance doesn't match cumulative arithmetic."
            )
            if macro_passed and is_od:
                msg += (
                    " However, the macro balance check passed AND this is an "
                    "OD account where line-level balance extraction is unreliable. "
                    "This is an extraction limitation, not tampering."
                )
            elif macro_passed:
                msg += (
                    " However, the macro balance check (Beginning + Credits "
                    "- Debits = Ending) passed, so this is likely a checker "
                    "extraction issue rather than tampering."
                )
            else:
                msg += " This is a strong indicator of value tampering."
            findings.append(_finding(
                layer, sev, msg,
                {"error_count": len(errors),
                 "first_errors": errors[:5],
                 "macro_verified": macro_passed},
            ))

    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Display names used by the Streamlit apps' bank selectors, mapped to the
# _BANK_PROFILES keys. The user's selection is ground truth for which bank's
# statement this is supposed to be — content/metadata guessing is only a
# fallback for callers that have no selection (e.g. batch comparison).
_BANK_HINT_ALIASES: Dict[str, str] = {
    "affin bank": "affin",
    "agro bank": "agrobank",
    "alliance bank": "alliance",
    "bank muamalat": "muamalat",
    "cimb bank": "cimb",
    "hong leong bank": "hong leong",
    "maybank islamic": "maybank",
    "ocbc bank": "ocbc",
    "public bank (pbb)": "public bank",
    "public bank": "public bank",
    "rhb bank": "rhb",
    "uob bank": "uob",
}


def normalize_bank_hint(name: Optional[str]) -> Optional[str]:
    """Map an app-level bank display name to a _BANK_PROFILES key.

    Returns None when the name is empty or unrecognized (callers then fall
    back to content-based detection).
    """
    if not name:
        return None
    key = str(name).strip().lower()
    if key in _BANK_PROFILES:
        return key
    return _BANK_HINT_ALIASES.get(key)


def analyze_pdf(pdf_bytes: bytes, filename: str = "",
                bank_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Run all 8 detection layers on a PDF.

    ``bank_hint`` is the app-level bank selection (display name or profile
    key). When provided it overrides content-based bank detection for the
    bank-profile and font layers — the user told us which bank's parser is
    about to run, so profile checks must be made against THAT bank, not a
    guess that transaction descriptions can poison (a Maybank statement
    full of 'CIMB CHQ DEP' rows was once profiled against CIMB).

    Returns a dict:
      {
        "filename": str,
        "overall_risk": "LOW" | "MEDIUM" | "HIGH",
        "layer_results": {
            "metadata": [...],
            "fonts": [...],
            "text_layers": [...],
            "visual": [...],
            "cross_validation": [...],
            "bank_profile": [...],
            "structural": [...]
        },
        "all_findings": [...],
        "finding_count": int,
        "high_count": int,
        "medium_count": int,
        "low_count": int,
      }
    """
    all_findings: List[Dict[str, Any]] = []

    layer_results: Dict[str, List[Dict[str, Any]]] = {}

    # --- Early check: encrypted PDF ---
    # Try to detect and decrypt with empty password. If still locked,
    # skip all layers and return a clear "encrypted" result instead of
    # silent failures that look like a clean bill of health.
    is_encrypted = False
    try:
        doc_check = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc_check.is_encrypted:
            # Try empty password (some banks use permission-only encryption)
            if not doc_check.authenticate(""):
                is_encrypted = True
        doc_check.close()
    except Exception:
        pass

    if is_encrypted:
        enc_finding = _finding(
            "encryption", MEDIUM,
            "PDF is password-protected — integrity analysis cannot be performed. "
            "Request an unencrypted copy or the password to enable fraud detection.",
            {"encrypted": True, "status": "ANALYSIS_SKIPPED"},
        )
        all_findings.append(enc_finding)
        layer_results["encryption"] = [enc_finding]
        return {
            "filename": filename,
            "overall_risk": MEDIUM,
            "layer_results": layer_results,
            "all_findings": all_findings,
            "finding_count": 1,
            "high_count": 0,
            "medium_count": 1,
            "low_count": 0,
        }

    hint = normalize_bank_hint(bank_hint)

    for layer_name, layer_fn in [
        ("metadata", _layer_metadata),
        ("fonts", lambda b: _layer_fonts(b, bank_hint=hint)),
        ("text_layers", _layer_text_layers),
        ("visual", _layer_visual),
        ("cross_validation", _layer_cross_validation),
        ("bank_profile", lambda b: _layer_bank_profile(b, bank_hint=hint)),
        ("structural", _layer_structural),
        ("arithmetic", _layer_arithmetic),
    ]:
        try:
            results = layer_fn(pdf_bytes)
        except Exception as e:
            results = [_finding(layer_name, LOW, f"Layer failed: {e}")]
        layer_results[layer_name] = results
        all_findings.extend(results)

    high_count = sum(1 for f in all_findings if f.get("severity") == HIGH)
    medium_count = sum(1 for f in all_findings if f.get("severity") == MEDIUM)
    low_count = sum(1 for f in all_findings if f.get("severity") == LOW)

    overall = _worst_severity(all_findings)

    return {
        "filename": filename,
        "overall_risk": overall,
        "layer_results": layer_results,
        "all_findings": all_findings,
        "finding_count": len(all_findings),
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
    }


def _extract_profile_fingerprint(pdf_bytes: bytes) -> Dict[str, Any]:
    """Extract a compact fingerprint for cross-file comparison."""
    fp: Dict[str, Any] = {}
    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        info = reader.metadata or {}
        fp["creator"] = str(info.get("/Creator", "") or "").strip()
        fp["producer"] = str(info.get("/Producer", "") or "").strip()
    except Exception:
        fp["creator"] = ""
        fp["producer"] = ""

    fp["fonts"] = sorted(_extract_font_names(pdf_bytes))
    fp["bank"] = _detect_bank(pdf_bytes)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        fp["pages"] = len(doc)
        if len(doc) > 0:
            rect = doc[0].rect
            fp["page_size"] = f"{round(rect.width, 1)}x{round(rect.height, 1)}"
        doc.close()
    except Exception:
        fp["pages"] = 0
        fp["page_size"] = ""

    fp["file_size"] = len(pdf_bytes)
    return fp


def compare_batch(results: Dict[str, Dict[str, Any]],
                  pdf_data: Dict[str, bytes]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Cross-file comparison: find outliers in a batch of uploaded PDFs.

    Args:
        results: {filename: analyze_pdf result} for each file
        pdf_data: {filename: raw pdf bytes} for each file

    Returns:
        {filename: [additional findings]} — extra findings to merge into each file's results.
    """
    if len(pdf_data) < 2:
        return {}

    # Extract fingerprints
    fingerprints: Dict[str, Dict[str, Any]] = {}
    for fname, raw in pdf_data.items():
        fingerprints[fname] = _extract_profile_fingerprint(raw)

    # Group files by detected bank
    bank_groups: Dict[Optional[str], List[str]] = {}
    for fname, fp in fingerprints.items():
        bank = fp.get("bank")
        bank_groups.setdefault(bank, []).append(fname)

    extra_findings: Dict[str, List[Dict[str, Any]]] = {f: [] for f in pdf_data}

    # For each bank group with 2+ files, compare profiles
    for bank, filenames in bank_groups.items():
        if bank is None or len(filenames) < 2:
            continue

        # Find dominant creator/producer
        creators = Counter(fingerprints[f]["creator"] for f in filenames)
        producers = Counter(fingerprints[f]["producer"] for f in filenames)
        font_sets = Counter(tuple(fingerprints[f]["fonts"]) for f in filenames)

        dominant_creator = creators.most_common(1)[0][0] if creators else ""
        dominant_producer = producers.most_common(1)[0][0] if producers else ""
        dominant_fonts = font_sets.most_common(1)[0][0] if font_sets else ()

        # If both the outlier AND the dominant profile match KNOWN legitimate
        # profiles for this bank, it's just the bank using two of its own
        # generators (e.g. OCBC's Streamline Pdfgen + omsgen). Not a fraud
        # signal — both are legitimate, just different systems/eras.
        bank_profiles = _BANK_PROFILES.get(bank, [])

        def _matches_known_profile(creator: str, producer: str) -> bool:
            for profile in bank_profiles:
                creator_ok = profile["creator_re"].search(creator) if creator else False
                producer_ok = profile["producer_re"].search(producer) if producer else False
                if creator_ok or producer_ok:
                    return True
            return False

        dominant_is_known = _matches_known_profile(dominant_creator, dominant_producer)

        for fname in filenames:
            fp = fingerprints[fname]
            mismatches = []

            file_is_known = _matches_known_profile(fp["creator"], fp["producer"])
            # Both this file and the batch norm are known-legitimate profiles
            # for the same bank → suppress creator/producer/font mismatch flags.
            both_legitimate = file_is_known and dominant_is_known

            if fp["creator"] != dominant_creator and dominant_creator and not both_legitimate:
                mismatches.append(
                    f"Creator '{fp['creator']}' differs from batch norm '{dominant_creator}'"
                )
            if fp["producer"] != dominant_producer and dominant_producer and not both_legitimate:
                mismatches.append(
                    f"Producer '{fp['producer']}' differs from batch norm '{dominant_producer}'"
                )
            if tuple(fp["fonts"]) != dominant_fonts and dominant_fonts and not both_legitimate:
                mismatches.append(
                    f"Fonts {fp['fonts']} differ from batch norm {list(dominant_fonts)}"
                )

            # File size outlier: if this file's size/page is 3x+ the batch median.
            # Carve-out: thin statements (1–2 pages) from fat generators (Word,
            # Excel) inherit fixed PDF overhead that divides by 1 instead of
            # many, inflating KB/page. Suppress when the file is thin AND its
            # absolute size is modest; real scan-rebuild tampering runs well
            # above 600KB per page in absolute terms.
            sizes_per_page = []
            page_counts = []
            for f2 in filenames:
                fp2 = fingerprints[f2]
                if fp2["pages"] > 0:
                    sizes_per_page.append(fp2["file_size"] / fp2["pages"])
                    page_counts.append(fp2["pages"])
            if sizes_per_page and fp["pages"] > 0:
                median_spp = sorted(sizes_per_page)[len(sizes_per_page) // 2]
                median_pages = sorted(page_counts)[len(page_counts) // 2]
                this_spp = fp["file_size"] / fp["pages"]
                is_thin_outlier = (
                    fp["pages"] <= 2
                    and median_pages >= 4
                    and fp["file_size"] < 600_000
                )
                if median_spp > 0 and this_spp / median_spp > 3 and not is_thin_outlier:
                    mismatches.append(
                        f"File size/page ({round(this_spp/1024)}KB) is "
                        f"{round(this_spp/median_spp, 1)}x the batch median "
                        f"({round(median_spp/1024)}KB)"
                    )

            if mismatches:
                extra_findings[fname].append(_finding(
                    "batch_comparison", HIGH,
                    f"OUTLIER in batch of {len(filenames)} {bank.title()} statements: "
                    f"{'; '.join(mismatches)}. "
                    "When one file in a batch has a different generation profile, "
                    "it was likely recreated or tampered.",
                    {
                        "detected_bank": bank,
                        "batch_size": len(filenames),
                        "this_file": {
                            "creator": fp["creator"],
                            "producer": fp["producer"],
                            "fonts": fp["fonts"],
                        },
                        "batch_norm": {
                            "creator": dominant_creator,
                            "producer": dominant_producer,
                            "fonts": list(dominant_fonts),
                        },
                        "mismatches": mismatches,
                    },
                ))

    return extra_findings
