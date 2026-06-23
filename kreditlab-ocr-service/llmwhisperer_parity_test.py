"""Kredit Lab OCR parity test: LLMWhisperer (Unstract) vs saved Tensorlake output.

Backup-OCR evaluation. Mirrors parity_test.py exactly — same golden PDFs, same
reference Tensorlake .md, same financial-value-recall metric (do the line-item
numbers all survive, since that's what analyze.py consumes) — but routes the PDF
through LLMWhisperer's hosted API instead of an OpenIngest backend.

Usage:
    python llmwhisperer_parity_test.py [mode ...]
Default modes tested: form (= "High Quality + Form/Table") and high_quality.

Quota note: Free plan = 100 pages/day. 37+3 pages per mode.
"""
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

# --- key (from OpenIngest/.env) ---------------------------------------------
ENV = Path(__file__).with_name(".env")
for line in ENV.read_text().splitlines():
    if line.startswith("LLMWHISPERER_API_KEY="):
        os.environ.setdefault("LLMWHISPERER_API_KEY", line.split("=", 1)[1].strip())
KEY = os.environ["LLMWHISPERER_API_KEY"]

BASE = "https://llmwhisperer-api.us-central.unstract.com/api/v2"

MODES = sys.argv[1:] or ["form", "high_quality"]

REPO = Path(
    "/Users/luqmanulhaqeemmdfauzi/Documents/Project Development Software for Kredit Lab/"
    "Financial Statement Analyzer HTML (Renderer)/repo"
)
OUT = Path("/Users/luqmanulhaqeemmdfauzi/Documents/OpenIngest/parity-out")
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

# Same money-token regex as parity_test.py: decimals or thousands-commas only.
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


def whisper(pdf: Path, mode: str) -> str:
    """Submit PDF, poll to completion, return layout-preserved text."""
    headers = {"unstract-key": KEY, "Content-Type": "application/octet-stream"}
    params = {
        "mode": mode,
        "output_mode": "layout_preserving",
        "file_name": pdf.name,
    }
    r = requests.post(
        f"{BASE}/whisper", params=params, headers=headers,
        data=pdf.read_bytes(), timeout=120,
    )
    r.raise_for_status()
    whash = r.json()["whisper_hash"]

    # poll status
    for _ in range(120):  # up to ~6 min
        time.sleep(3)
        s = requests.get(
            f"{BASE}/whisper-status", params={"whisper_hash": whash},
            headers={"unstract-key": KEY}, timeout=30,
        ).json()
        status = s.get("status")
        if status == "processed":
            break
        if status in ("error", "unknown"):
            raise RuntimeError(f"status={status}: {s}")
    else:
        raise RuntimeError("timed out waiting for processing")

    # retrieve
    out = requests.get(
        f"{BASE}/whisper-retrieve", params={"whisper_hash": whash},
        headers={"unstract-key": KEY}, timeout=60,
    ).json()
    return out.get("result_text") or out.get("extraction", {}).get("result_text", "")


def main():
    for mode in MODES:
        print(f"\n{'='*70}\nLLMWhisperer mode under test: {mode}\n{'='*70}")
        for pdf, ref_md in PAIRS:
            name = pdf.stem
            print(f"\n### {name}")
            if not pdf.exists():
                print(f"  !! PDF missing: {pdf}"); continue
            if not ref_md.exists():
                print(f"  !! reference .md missing: {ref_md}"); continue
            try:
                got = whisper(pdf, mode)
            except Exception as e:
                print(f"  !! LLMWhisperer failed: {type(e).__name__}: {e}"); continue
            (OUT / f"{name}.llmwhisperer-{mode}.md").write_text(got, encoding="utf-8")

            ref = ref_md.read_text(encoding="utf-8")
            rv, gv = values(ref), values(got)
            ref_set, got_set = set(rv), set(gv)
            common = ref_set & got_set
            missing = sorted(ref_set - got_set)
            extra = sorted(got_set - ref_set)
            recall = len(common) / len(ref_set) if ref_set else 1.0

            print(f"  chars: tensorlake={len(ref):,}  llmwhisperer={len(got):,}")
            print(f"  distinct $ values: tensorlake={len(ref_set)}  llmwhisperer={len(got_set)}")
            print(f"  VALUE RECALL (tensorlake values found in llmwhisperer): "
                  f"{len(common)}/{len(ref_set)} = {recall:.1%}")
            if missing:
                print(f"  MISSING ({len(missing)}) e.g.: {missing[:12]}")
            if extra:
                print(f"  EXTRA   ({len(extra)}) e.g.: {extra[:12]}")
    print(f"\nOutputs written to {OUT}/\n")


if __name__ == "__main__":
    main()
