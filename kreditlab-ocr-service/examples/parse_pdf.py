# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python
"""
Parse a single PDF (or image / DOCX) with Open Ingest.

Usage:
    # Local dry-run, no deploy needed
    python examples/parse_pdf.py --file my.pdf --local

    # Against a deployed workflow
    tl deploy src/workflow.py
    python examples/parse_pdf.py --file my.pdf

    # With VLM enrichment
    python examples/parse_pdf.py --file my.pdf --local \
        --table-merging --table-summarization --figure-summarization \
        --chart-extraction --detect-signature

    # With page classification
    python examples/parse_pdf.py --file my.pdf --local \
        --classify invoice:"Has invoice header + line items" \
        --classify contract:"Legal terms, signature block"

Output: ./debug/document.json plus one ./debug/page_N.md per page.
"""

import argparse
import base64
import shutil
from pathlib import Path

from tensorlake.applications import run_local_application, run_remote_application

from tensorlake_docai.pipeline.api import (
    ClassificationRequest,
    PageClassDefinition,
    ParseRequest,
    ParsedDocument,
)
from tensorlake_docai.pipeline.file_converter import normalize_file_type_and_upload
from tensorlake_docai.postprocess.draw_bboxes import visualize_document_bboxes
from tensorlake_docai.postprocess.formatter import page_to_markdown

MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
    ".txt": "text/plain",
}


def _parse_class_pair(raw: str) -> PageClassDefinition:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(f"--classify expects 'name:description', got: {raw!r}")
    name, _, desc = raw.partition(":")
    name, desc = name.strip(), desc.strip()
    if not name or not desc:
        raise argparse.ArgumentTypeError(
            f"--classify expects non-empty name and description, got: {raw!r}"
        )
    return PageClassDefinition(class_name=name, description=desc)


def build_request(args: argparse.Namespace) -> ParseRequest:
    path = args.file

    if path.startswith("s3://") or path.startswith("http"):
        file_bytes = None
        file_url = path
        file_name = Path(path).name
        mime_type = MIME_BY_EXT.get(Path(path).suffix.lower(), "application/pdf")
    else:
        local = Path(path)
        if not local.exists():
            raise FileNotFoundError(path)
        file_bytes = base64.b64encode(local.read_bytes()).decode()
        file_url = None
        file_name = local.name
        mime_type = MIME_BY_EXT.get(local.suffix.lower(), "application/pdf")

    page_classification_request = None
    if args.classify:
        page_classification_request = ClassificationRequest(
            class_definitions=args.classify,
            classification_type=args.classification_type,
        )

    return ParseRequest(
        file_bytes=file_bytes,
        file_url=file_url,
        file_name=file_name,
        mime_type=mime_type,
        ocr_model=args.ocr_model,
        pages_to_parse=args.pages or None,
        chunk_strategy=args.chunk_strategy,
        table_output_mode=args.table_output_mode,
        detect_signature=args.detect_signature,
        detect_barcode=args.detect_barcode,
        table_merging=args.table_merging,
        table_summarization=args.table_summarization,
        table_summarization_prompt=args.table_summarization_prompt,
        table_cell_grounding=args.table_cell_grounding,
        figure_summarization=args.figure_summarization,
        figure_summarization_prompt=args.figure_summarization_prompt,
        figure_ocr_prompt=args.figure_ocr_prompt,
        figure_grounding=args.figure_grounding,
        chart_extraction=args.chart_extraction,
        key_value_extraction=args.key_value_extraction,
        page_classification_request=page_classification_request,
        xpage_header_detection=args.xpage_header_detection,
        include_images=args.include_images,
        ignore_sections=set(args.ignore_sections) if args.ignore_sections else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    core = parser.add_argument_group("core")
    core.add_argument("--file", required=True, help="Local path, s3:// URL, or https:// URL")
    core.add_argument(
        "--ocr-model",
        default="azure-di",
        choices=["dots-ocr", "azure-di", "textract", "gemini"],
        help="OCR backend (see docs/models.md). Default: azure-di.",
    )
    core.add_argument("--pages", type=int, nargs="*", help="Pages to parse (1-indexed)")
    core.add_argument(
        "--local", action="store_true", help="Run in-process instead of remote deploy"
    )
    core.add_argument("--out", default="debug", help="Output directory")
    core.add_argument(
        "--draw-bboxes",
        action="store_true",
        help="Also write per-page PNGs with fragment bounding boxes overlayed",
    )

    output = parser.add_argument_group("output shape")
    output.add_argument(
        "--chunk-strategy",
        default=None,
        choices=["none", "page", "section", "fragment"],
        help="How to chunk the output document",
    )
    output.add_argument(
        "--table-output-mode",
        default="markdown",
        choices=["markdown", "html", "json"],
        help="Format of table content in the output",
    )
    output.add_argument(
        "--ignore-sections",
        nargs="*",
        default=[],
        help="PageFragmentType values to drop from output (e.g. page_footer figure)",
    )
    output.add_argument(
        "--include-images",
        action="store_true",
        help="Include base64 page/figure images in the output",
    )

    detection = parser.add_argument_group("detection")
    detection.add_argument(
        "--detect-signature",
        action="store_true",
        help="Detect signatures (Textract; requires AWS keys)",
    )
    detection.add_argument(
        "--detect-barcode", action="store_true", help="Detect barcodes in the document"
    )

    tables = parser.add_argument_group("table enrichment")
    tables.add_argument(
        "--table-merging",
        action="store_true",
        help="Stitch tables that span pages or are split by intervening content",
    )
    tables.add_argument(
        "--table-summarization",
        action="store_true",
        help="Describe each table with a VLM",
    )
    tables.add_argument(
        "--table-summarization-prompt",
        default=None,
        help="Override prompt for table summarization",
    )
    tables.add_argument(
        "--table-cell-grounding",
        action="store_true",
        help="Produce per-cell bounding boxes for tables",
    )

    figures = parser.add_argument_group("figure / chart enrichment")
    figures.add_argument(
        "--figure-summarization",
        action="store_true",
        help="Describe each figure with a VLM",
    )
    figures.add_argument(
        "--figure-summarization-prompt",
        default=None,
        help="Override prompt for figure summarization",
    )
    figures.add_argument(
        "--figure-ocr-prompt",
        default=None,
        help="Override prompt for figure OCR (`dots-ocr` only)",
    )
    figures.add_argument(
        "--figure-grounding",
        action="store_true",
        help="Produce per-text-region bounding boxes inside figures",
    )
    figures.add_argument(
        "--chart-extraction",
        action="store_true",
        help="Extract data series (JSON) from charts",
    )

    forms = parser.add_argument_group("forms / key-value")
    forms.add_argument(
        "--key-value-extraction",
        action="store_true",
        help="Extract key-value pairs from form-like regions",
    )

    classify = parser.add_argument_group("page classification")
    classify.add_argument(
        "--classify",
        type=_parse_class_pair,
        action="append",
        default=[],
        metavar="NAME:DESCRIPTION",
        help="Add a page class. Repeat for multiple classes. "
        "Example: --classify invoice:'Has invoice header'",
    )
    classify.add_argument(
        "--classification-type",
        default="multi_label",
        choices=["multi_label", "multi_class"],
        help="Page classification mode (default: multi_label)",
    )

    xpage = parser.add_argument_group("cross-page heuristics")
    xpage.add_argument(
        "--xpage-header-detection",
        action="store_true",
        help="Detect repeating cross-page headers/footers",
    )

    args = parser.parse_args()

    req = build_request(args)

    runner = run_local_application if args.local else run_remote_application
    handle = runner(normalize_file_type_and_upload, req.model_dump())
    print(f"Request ID: {handle.id}")

    raw = handle.output()
    if not raw or "document" not in raw:
        raise RuntimeError("No document returned")

    parsed = ParsedDocument.model_validate(raw["document"])
    print(f"Parsed {parsed.parsed_pages_count} pages with ocr_model={args.ocr_model}")

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    (out_dir / "document.json").write_text(parsed.model_dump_json(indent=2))

    for page in parsed.pages or []:
        (out_dir / f"page_{page.page_number}.md").write_text(page_to_markdown(page, req))

    if args.draw_bboxes:
        is_url = args.file.startswith("s3://") or args.file.startswith("http")
        if is_url:
            print("--draw-bboxes: skipping, remote URLs are not supported (pass a local file)")
        else:
            visualize_document_bboxes(
                parsed,
                file_path=args.file,
                output_prefix=str(out_dir / "bbox_page"),
            )

    if args.table_merging and parsed.merged_tables:
        print(f"Merged tables: {len(parsed.merged_tables)}")
        for mt in parsed.merged_tables:
            print(f"  - {mt.merged_table_id}: pages {mt.start_page}-{mt.end_page}")

    if parsed.page_classes:
        print(f"Page classifications: {len(parsed.page_classes)}")
        for pc in parsed.page_classes:
            print(f"  - {pc.page_class}: pages {pc.page_numbers}")

    print(f"Wrote results to {out_dir}/")


if __name__ == "__main__":
    main()
