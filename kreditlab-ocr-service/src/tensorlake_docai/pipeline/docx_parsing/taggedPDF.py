# SPDX-License-Identifier: Apache-2.0
import os
import re
import subprocess
import zipfile
import xml.etree.ElementTree as ET

import fitz  # PyMuPDF
from tensorlake.applications import RequestError as RequestException


def add_hidden_markers(docx_path, out_path, marker_prefix="MK"):
    """Add white 1‑pt markers to each paragraph across document, headers, footers, notes."""
    # Default namespace
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    ns_alt = {"w": "http://purl.oclc.org/ooxml/wordprocessingml/main"}

    # Detect which namespace to use by checking the main document
    detected_ns = ns  # default
    with zipfile.ZipFile(docx_path) as zf:
        if "word/document.xml" in [zi.filename for zi in zf.infolist()]:
            doc_xml = zf.read("word/document.xml")
            root = ET.fromstring(doc_xml)
            root_ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else None
            if root_ns and "purl.oclc.org/ooxml" in root_ns:
                detected_ns = ns_alt

    def add_to_paragraphs(xml_bytes, start_index, namespace):
        if not xml_bytes:
            return xml_bytes, start_index
        root = ET.fromstring(xml_bytes)
        paragraphs = root.findall(".//w:p", namespace)
        idx = start_index
        for p in paragraphs:
            idx += 1
            marker = f"{marker_prefix}{idx:04d}"
            r = ET.Element(f"{{{namespace['w']}}}r")
            rPr = ET.SubElement(r, f"{{{namespace['w']}}}rPr")
            color = ET.SubElement(rPr, f"{{{namespace['w']}}}color")
            # Use near-white so PDF converters keep text in the text layer
            color.set(f"{{{namespace['w']}}}val", "FDFDFD")
            sz = ET.SubElement(rPr, f"{{{namespace['w']}}}sz")
            sz.set(f"{{{namespace['w']}}}val", "2")  # 1 pt (2 half‑points)
            t = ET.SubElement(r, f"{{{namespace['w']}}}t")
            # Ensure leading space is preserved in DOCX
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            # Leading space helps keep the marker as a separate token/span
            t.text = " " + marker
            p.append(r)
        return ET.tostring(root, encoding="utf-8", xml_declaration=True), idx

    with zipfile.ZipFile(docx_path) as zf:
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
            counter = 0
            for item in zf.infolist():
                data = zf.read(item.filename)
                # Modify paragraphs in main document, headers, footers, notes, and text boxes
                # Note: word/document.xml includes inline text boxes via w:txbxContent
                if (
                    item.filename == "word/document.xml"
                    or re.match(r"word/header\d*\.xml$", item.filename)
                    or re.match(r"word/footer\d*\.xml$", item.filename)
                    or item.filename in ("word/footnotes.xml", "word/endnotes.xml")
                ):
                    data, counter = add_to_paragraphs(data, counter, detected_ns)
                zout.writestr(item, data)


def convert_to_pdf_via_lo(docx_path, pdf_dir):
    """Convert a DOCX to PDF using LibreOffice."""
    try:
        subprocess.check_call(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                docx_path,
                "--outdir",
                pdf_dir,
            ]
        )
        return os.path.join(pdf_dir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")
    except subprocess.CalledProcessError as e:
        raise RequestException(f"Failed in converting DOCX to PDF: {e}")


def build_marker_html_map(docx_path, marker_prefix="MK"):
    """
    Use existing DOCX-to-HTML parser to map each marker to its paragraph HTML
    (with styling like <h1>, <del>, <ins>, and comments preserved).
    Returns dict: { marker -> html_for_paragraph }
    """

    try:
        from tensorlake_docai.pipeline.docx_parsing.docx_parsing import extract_text_with_changes_and_comments  # type: ignore
    except ModuleNotFoundError:
        from tensorlake_docai.pipeline.docx_parsing import extract_text_with_changes_and_comments  # type: ignore

    html = extract_text_with_changes_and_comments(docx_path)

    pattern = re.compile(rf"{marker_prefix}\d{{4}}")

    # Prefer lxml.html if available to robustly find paragraph elements
    try:
        from lxml import html as lhtml  # type: ignore

        root = lhtml.fragment_fromstring(html, create_parent=True)
        marker_to_html = {}
        # Candidate paragraph-like tags
        candidate_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6"}
        for elem in root.iter():
            if elem.tag not in candidate_tags:
                continue
            serialized = lhtml.tostring(elem, encoding="unicode")
            found = set(pattern.findall(serialized))
            if not found:
                continue

            # Check if this paragraph is inside a table
            table_ancestor = None
            parent = elem.getparent()
            while parent is not None:
                if parent.tag == "table":
                    table_ancestor = parent
                    break
                parent = parent.getparent()

            # If inside a table, serialize the entire table instead
            if table_ancestor is not None:
                table_html = lhtml.tostring(table_ancestor, encoding="unicode")
                cleaned = pattern.sub("", table_html)
                for m in found:
                    marker_to_html[m] = cleaned
            else:
                cleaned = pattern.sub("", serialized)
                for m in found:
                    marker_to_html[m] = cleaned

        return marker_to_html
    except Exception:
        # Fallback: regex-based extraction
        marker_to_html = {}

        # First, handle tables: capture entire table if it contains a marker
        table_re = re.compile(r"<table\b[^>]*>.*?%s.*?</table>" % pattern.pattern, re.DOTALL)
        for match in table_re.finditer(html):
            table_html = match.group(0)
            found = set(pattern.findall(table_html))
            cleaned = pattern.sub("", table_html)
            for m in found:
                marker_to_html[m] = cleaned

        # Then handle standalone paragraphs (not in tables)
        para_re = re.compile(r"<(p|h[1-6])\b[^>]*>.*?%s.*?</\1>" % pattern.pattern, re.DOTALL)
        for match in para_re.finditer(html):
            block_html = match.group(0)
            found = set(pattern.findall(block_html))
            # Skip if this marker is already mapped to a table
            if any(m in marker_to_html for m in found):
                continue
            cleaned = pattern.sub("", block_html)
            for m in found:
                marker_to_html[m] = cleaned

        return marker_to_html


def locate_paragraph_bboxes_segmented(pdf_path, marker_prefix="MK"):
    """
    Segmentation-based locator that assigns every text span to the nearest
    following marker across pages. Returns list of dicts per page per marker:
    { page, marker, bbox, text }.
    """
    pattern = re.compile(rf"{marker_prefix}\d{{4}}")
    spans = []

    with fitz.open(pdf_path) as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "") or ""
                        mlist = pattern.findall(text)
                        spans.append(
                            {
                                "page": page_num,
                                "bbox": list(span["bbox"]),
                                "text": text,
                                "contains_marker": bool(mlist),
                                "markers": mlist,
                            }
                        )

    # Identify the index in the flow for each marker occurrence (duplicates allowed)
    marker_occurrences = []  # list of (flow_index, marker_str)
    for idx, sp in enumerate(spans):
        if sp["contains_marker"]:
            for m in sp["markers"]:
                marker_occurrences.append((idx, m))

    if not marker_occurrences:
        return []

    # Sort by flow order just in case
    marker_occurrences.sort(key=lambda t: t[0])

    # Assign spans between prev marker occurrence and current marker occurrence
    results = []
    prev_idx = -1
    for occ_i, (curr_idx, marker) in enumerate(marker_occurrences):
        segment_spans = spans[prev_idx + 1 : curr_idx]
        # Group segment spans by page and union their bboxes, also collect text
        page_to_data = {}
        for sp in segment_spans:
            # Skip empty after removing any incidental marker text
            text_wo_markers = pattern.sub("", sp["text"]).strip()
            if not text_wo_markers:
                continue
            p = sp["page"]
            x0, y0, x1, y1 = sp["bbox"]
            if p not in page_to_data:
                page_to_data[p] = {"bbox": [x0, y0, x1, y1], "text_parts": [text_wo_markers]}
            else:
                bx0, by0, bx1, by1 = page_to_data[p]["bbox"]
                page_to_data[p]["bbox"] = [
                    min(bx0, x0),
                    min(by0, y0),
                    max(bx1, x1),
                    max(by1, y1),
                ]
                page_to_data[p]["text_parts"].append(text_wo_markers)

        for p, data in page_to_data.items():
            results.append(
                {
                    "page": p,
                    "marker": marker,
                    "bbox": data["bbox"],
                    "pdf_text": "".join(data["text_parts"]).strip(),
                }
            )

        prev_idx = curr_idx

    # Handle trailing text after the last marker
    last_idx, last_marker = marker_occurrences[-1]
    trailing_spans = spans[last_idx + 1 :]
    if trailing_spans:
        page_to_data = {}
        for sp in trailing_spans:
            text_wo_markers = pattern.sub("", sp["text"]).strip()
            if not text_wo_markers:
                continue
            p = sp["page"]
            x0, y0, x1, y1 = sp["bbox"]
            if p not in page_to_data:
                page_to_data[p] = {"bbox": [x0, y0, x1, y1], "text_parts": [text_wo_markers]}
            else:
                bx0, by0, bx1, by1 = page_to_data[p]["bbox"]
                page_to_data[p]["bbox"] = [
                    min(bx0, x0),
                    min(by0, y0),
                    max(bx1, x1),
                    max(by1, y1),
                ]
                page_to_data[p]["text_parts"].append(text_wo_markers)
        for p, data in page_to_data.items():
            results.append(
                {
                    "page": p,
                    "marker": last_marker,
                    "bbox": data["bbox"],
                    "pdf_text": "".join(data["text_parts"]).strip(),
                }
            )

    return results


def merge_segmented_bboxes_with_docx_html(pdf_path, docx_path, marker_prefix="MK"):
    """
    Returns list of dicts { page, marker, bbox, html, pdf_text } using segmentation-based
    bboxes and DOCX HTML for styled paragraph text.
    """
    bbox_items = locate_paragraph_bboxes_segmented(pdf_path, marker_prefix=marker_prefix)
    marker_to_html = build_marker_html_map(docx_path, marker_prefix=marker_prefix)

    # Enrich bbox items with HTML from DOCX
    enriched = []
    for item in bbox_items:
        m = item.get("marker")
        html = marker_to_html.get(m, "")
        pdf_text = item.get("pdf_text", "")

        # Filter out bboxes with no actual text content (only HTML tags or empty)
        text_only = re.sub(r"<[^>]+>", "", html).strip()
        if not text_only and not pdf_text:
            continue  # Skip this bbox if it has no actual text content

        # Add HTML to the item
        item["html"] = html
        enriched.append(item)

    # Deduplicate consecutive items with identical table HTML and union their bboxes
    # This handles multiple markers (cells) in the same table
    deduplicated = []

    def is_table(h: str) -> bool:
        return h.strip().startswith("<table")

    i = 0
    while i < len(enriched):
        item = enriched[i]
        curr_html = item["html"]

        # If this is not a table, just add it
        if not is_table(curr_html):
            deduplicated.append(item)
            i += 1
            continue

        # This is a table - collect all consecutive items with the same table HTML
        table_items = [item]
        j = i + 1
        while j < len(enriched) and enriched[j]["html"] == curr_html:
            table_items.append(enriched[j])
            j += 1

        # Group table items by page and union bboxes
        page_to_bbox = {}
        for t_item in table_items:
            page = t_item["page"]
            bbox = t_item["bbox"]
            if page not in page_to_bbox:
                page_to_bbox[page] = bbox[:]
            else:
                # Union the bboxes
                x0, y0, x1, y1 = page_to_bbox[page]
                bx0, by0, bx1, by1 = bbox
                page_to_bbox[page] = [
                    min(x0, bx0),
                    min(y0, by0),
                    max(x1, bx1),
                    max(y1, by1),
                ]

        # Create one item per page for tables
        for page, union_bbox in sorted(page_to_bbox.items()):
            # Concatenate pdf_text from all table items on this page
            page_table_items = [t for t in table_items if t["page"] == page]
            combined_pdf_text = " ".join(t.get("pdf_text", "") for t in page_table_items).strip()

            deduplicated.append(
                {
                    "page": page,
                    "marker": table_items[0]["marker"],  # Use first marker
                    "bbox": union_bbox,
                    "html": curr_html,
                    "pdf_text": combined_pdf_text,
                }
            )

        i = j  # Skip to next unique item

    return deduplicated


def determine_fragment_type_from_html(html: str):
    """
    Determine fragment type based on HTML tags present.
    Returns appropriate PageFragmentType based on content.
    """
    # Import here to avoid circular dependency
    try:
        from tensorlake_docai.pipeline.api import PageFragmentType
    except ImportError:
        from ..api import PageFragmentType

    # Check for table tag
    if "<table" in html:
        return PageFragmentType.TABLE
    # Check for h1 tag (title)
    if "<h1" in html:
        return PageFragmentType.TITLE
    # Check for other heading tags (section headers)
    if any(f"<h{i}" in html for i in range(2, 7)) or "<subtitle" in html:
        return PageFragmentType.SECTION_HEADER
    # Check for tracked changes
    if "<del" in html or "<ins" in html:
        return PageFragmentType.TRACKED_CHANGES
    # Check for comments (span with class="comment" or HTML comments)
    # Any fragment with comment tags should be labeled as COMMENTS
    if 'class="comment"' in html or "<!-- Comment:" in html:
        return PageFragmentType.COMMENTS
    # Default to text
    return PageFragmentType.TEXT


def html_to_markdown(html: str) -> str:
    """
    Convert HTML to markdown.
    """
    from markdownify import markdownify as md

    return md(html, strip=["script", "style"]).strip()


def process_docx_to_structured_pages(docx_bytes, marker_prefix="MK"):
    """
    Process DOCX bytes and return structured page data with bboxes and HTML content.

    Args:
        docx_bytes: Raw bytes of the DOCX file
        marker_prefix: Prefix for hidden markers (default "MK")

    Returns:
        tuple: (page_items_dict, pdf_bytes)
            - page_items_dict: dict mapping page numbers to lists of items with bbox, html, content, and fragment_type
            - pdf_bytes: bytes of the converted PDF
    """
    import tempfile
    from pathlib import Path
    from collections import defaultdict

    # Create temp files
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as original_temp:
        original_temp.write(docx_bytes)
        original_temp.flush()
        original_path = Path(original_temp.name)

    with tempfile.NamedTemporaryFile(suffix="_marked.docx", delete=False) as marked_temp:
        marked_path = Path(marked_temp.name)

    pdf_path = None
    try:
        # Add markers to DOCX
        add_hidden_markers(str(original_path), str(marked_path))

        # Convert to PDF
        pdf_dir = Path(tempfile.gettempdir())
        pdf_path = convert_to_pdf_via_lo(str(marked_path), str(pdf_dir))

        # Extract bboxes with HTML
        items = merge_segmented_bboxes_with_docx_html(pdf_path, str(marked_path), marker_prefix)

        # Import PageFragmentType
        try:
            from tensorlake_docai.pipeline.api import PageFragmentType
        except ImportError:
            from ..api import PageFragmentType

        # Group items by page and enrich with content and fragment_type
        page_items = defaultdict(list)
        for item in items:
            html = item["html"]
            pdf_text = item.get("pdf_text", "")

            # Determine fragment type from original DOCX HTML
            fragment_type = determine_fragment_type_from_html(html)
            item["fragment_type"] = fragment_type

            # For TABLE: keep raw HTML as-is
            if fragment_type == PageFragmentType.TABLE:
                item["content"] = html
            # For TRACKED_CHANGES, COMMENTS: keep semantic tags but remove <p> wrappers
            elif fragment_type in (PageFragmentType.TRACKED_CHANGES, PageFragmentType.COMMENTS):
                item["content"] = re.sub(r"</?p>", "", html)
            # For other types: use PDF text if available, otherwise convert HTML to markdown
            elif pdf_text:
                item["content"] = pdf_text
            else:
                item["content"] = html_to_markdown(html)

            page_items[item["page"]].append(item)

        # Read PDF bytes
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        pdf_size_kb = len(pdf_bytes) / 1024
        print(f" Converted PDF size: {pdf_size_kb:.2f} KB ({len(pdf_bytes)} bytes)")

        return dict(page_items), pdf_bytes

    finally:
        # Cleanup temp files
        if original_path.exists():
            os.unlink(original_path)
        if marked_path.exists():
            os.unlink(marked_path)
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)


def draw_bboxes_on_pdf(pdf_path, marker_info, output_dir):
    from PIL import Image, ImageDraw

    """Render each page, draw rectangles for the markers, and save as PNG."""
    os.makedirs(output_dir, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            # Render page to an image (use zoom=1 for 72 DPI; adjust if needed)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Draw rectangles for paragraphs on this page
            draw = ImageDraw.Draw(img)
            for info in marker_info:
                if info["page"] == page_num:
                    x0, y0, x1, y1 = info["bbox"]
                    draw.rectangle([(x0, y0), (x1, y1)], outline="red", width=2)
                    # Optionally label with pdf_text (preferred for page numbers), html, text, or marker
                    label = (
                        info.get("pdf_text")
                        or info.get("html")
                        or info.get("text", "")
                        or info.get("marker")
                    )
                    if label:
                        # Truncate long labels for readability
                        display_label = label[:50] if len(label) > 50 else label
                        draw.text((x0, y0 - 10), display_label, fill="red")

            img.save(os.path.join(output_dir, f"page-{page_num+1}.png"))
