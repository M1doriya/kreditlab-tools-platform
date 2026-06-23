"""Kredit Lab OCR parity test: OpenIngest (--local) vs saved Tensorlake output.

For each (PDF, reference Tensorlake .md) pair, run OpenIngest with a given OCR
backend and compare. Focus metric = financial-value parity (do the line-item
numbers all survive), since that's what analyze.py actually consumes.
"""
import base64
import re
import sys
from collections import Counter
from pathlib import Path

from tensorlake.applications import run_local_application
from tensorlake_docai.pipeline.api import ParseRequest, ParsedDocument
from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload
from tensorlake_docai.postprocess.formatter import page_to_markdown

OCR_MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini"
REPO = Path(
    "/Users/luqmanulhaqeemmdfauzi/Documents/Project Development Software for Kredit Lab/"
    "Financial Statement Analyzer HTML (Renderer)/repo"
)
OUT = Path("/Users/luqmanulhaqeemmdfauzi/OpenIngest/parity-out")
OUT.mkdir(exist_ok=True)

PAIRS = [
    (
        REPO / "Sample Case/Audit Report_31.12.2024 (1).pdf",
        REPO / "samples/huahub_marketing/tensorlake/Audit Report_31.12.2024 (1).md",
    ),
    (
        REPO / "Sample Case/Audited Account +  Management Account/Report Mangagement Account_31.12.25 (1).pdf",
        REPO / "samples/huahub_marketing/tensorlake/Report Mangagement Account_31.12.25 (1).md",
    ),
]

# Money-like tokens: 1,383,545.42 / 692208.92 / (123.00) etc. Require a decimal or
# a thousands-comma so we capture financial figures, not page numbers / years.
NUM = re.compile(r"\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?\d+\.\d{2}\)?")


def values(text: str) -> Counter:
    out = Counter()
    for m in NUM.findall(text):
        norm = m.replace(",", "").replace("(", "-").replace(")", "")
        try:
            out[round(float(norm), 2)] += 1
        except ValueError:
            pass
    return out


def run_openingest(pdf: Path) -> str:
    req = ParseRequest(
        file_name=pdf.name,
        mime_type="application/pdf",
        file_bytes=base64.b64encode(pdf.read_bytes()).decode(),
        ocr_model=OCR_MODEL,
        chunk_strategy="page",
        table_output_mode="markdown",
        xpage_header_detection=False,  # matches dashboard lean profile; avoids OpenAI dep
    )
    handle = run_local_application(normalize_file_type_and_upload, req.model_dump())
    raw = handle.output()
    if not raw or "document" not in raw:
        raise RuntimeError("No document returned")
    doc = ParsedDocument.model_validate(raw["document"])
    parts = [page_to_markdown(p, req) for p in (doc.pages or [])]
    return "\n\n".join(parts), (doc.parsed_pages_count or len(doc.pages or []))


def main():
    print(f"\n{'='*70}\nOCR backend under test: {OCR_MODEL}\n{'='*70}")
    for pdf, ref_md in PAIRS:
        name = pdf.stem
        print(f"\n### {name}")
        if not pdf.exists():
            print(f"  !! PDF missing: {pdf}"); continue
        if not ref_md.exists():
            print(f"  !! reference .md missing: {ref_md}"); continue
        try:
            got, pages = run_openingest(pdf)
        except Exception as e:
            print(f"  !! OpenIngest failed: {type(e).__name__}: {e}"); continue
        (OUT / f"{name}.{OCR_MODEL}.md").write_text(got, encoding="utf-8")

        ref = ref_md.read_text(encoding="utf-8")
        rv, gv = values(ref), values(got)
        ref_set, got_set = set(rv), set(gv)
        common = ref_set & got_set
        missing = sorted(ref_set - got_set)   # in Tensorlake, lost by OpenIngest
        extra = sorted(got_set - ref_set)     # new in OpenIngest, not in Tensorlake
        recall = len(common) / len(ref_set) if ref_set else 1.0

        print(f"  pages parsed (OpenIngest)   : {pages}")
        print(f"  chars: tensorlake={len(ref):,}  openingest={len(got):,}")
        print(f"  distinct $ values: tensorlake={len(ref_set)}  openingest={len(got_set)}")
        print(f"  VALUE RECALL (tensorlake values found in openingest): "
              f"{len(common)}/{len(ref_set)} = {recall:.1%}")
        if missing:
            print(f"  MISSING ({len(missing)}) e.g.: {missing[:12]}")
        if extra:
            print(f"  EXTRA   ({len(extra)}) e.g.: {extra[:12]}")
    print(f"\nOutputs written to {OUT}/\n")


if __name__ == "__main__":
    main()
