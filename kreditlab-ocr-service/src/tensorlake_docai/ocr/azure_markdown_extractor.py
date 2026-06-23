# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Azure Document Intelligence OCR to Markdown Extractor
Extracts paragraphs, tables, and figures using Azure Document Intelligence
"""

import os
import ssl
import requests
from typing import List, Dict, Optional, Tuple
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.pipeline.transport import RequestsTransport
from requests.adapters import HTTPAdapter
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    AnalyzeResult,
    DocumentAnalysisFeature,
)
from tensorlake_docai.pipeline.api import PageFragmentType

# Default configuration
USE_HIGH_RESOLUTION_OCR = False
AZURE_TIMEOUT = 600


def _tls12_session():
    """Create a requests Session that enforces TLS >= 1.2."""
    ctx = ssl.create_default_context()
    try:
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    except Exception:
        pass

    class _TLSAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs["ssl_context"] = ctx
            return super().init_poolmanager(*args, **kwargs)

        def proxy_manager_for(self, *args, **kwargs):
            kwargs["ssl_context"] = ctx
            return super().proxy_manager_for(*args, **kwargs)

    s = requests.Session()
    s.mount("https://", _TLSAdapter())
    return s


class AzureMarkdownExtractor:
    """Azure Document Intelligence markdown extractor"""

    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None):

        self.endpoint = endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.key = key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not self.endpoint or not self.key:
            raise Exception("DocAI credentials required.")

        transport = RequestsTransport(
            connection_timeout=30, read_timeout=AZURE_TIMEOUT, session=_tls12_session()
        )
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key),
            transport=transport,
        )
        print("Document Intelligence client initialized")
        self.tables_by_reading_order = {}

    def analyze_document_bytes_direct(
        self, blob: bytes, pages: Optional[str] = None
    ) -> AnalyzeResult:
        """Analyze a document passed as bytes (supports TIFF, PDF, images) in one call."""
        print("🔍 Analyzing document bytes (single call)")
        if pages:
            print(f"📄 Page selection: {pages}")
        try:
            from tensorlake_docai.ocr.azure_retry_utils import robust_azure_analyze_document

            request = AnalyzeDocumentRequest(bytes_source=blob)

            api_kwargs = {
                "timeout": AZURE_TIMEOUT,
                "output_content_format": "markdown",
            }

            if USE_HIGH_RESOLUTION_OCR:
                api_kwargs["features"] = [DocumentAnalysisFeature.OCR_HIGH_RESOLUTION]

            if pages:
                api_kwargs["pages"] = pages

            result = robust_azure_analyze_document(
                client=self.client,
                model_id="prebuilt-layout",
                request=request,
                **api_kwargs,
            )

            actual_pages = len(result.pages) if result.pages else 0
            print(
                f"Document Intelligence analysis (bytes) completed - {actual_pages} pages processed"
            )
            return result
        except Exception as e:
            print(f"Document Intelligence analysis (bytes) failed: {e}")
            if "timeout" in str(e).lower():
                print("💡 Try increasing timeout for large documents")
            raise

    def convert_inches_to_pixels(self, inches: float, dpi: int = 72) -> int:
        """Convert inches to pixels using specified DPI (default 72)"""
        return int(inches * dpi)

    def convert_inches_bbox_to_pixels(self, polygon, dpi: int = 72) -> Optional[Dict]:
        """Convert polygon coordinates from inches to pixel bounding box"""
        if not polygon or len(polygon) < 4:
            return None

        # Extract coordinates in inches
        x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
        y_coords = [polygon[i] for i in range(1, len(polygon), 2)]

        # Convert to pixel coordinates (inches * DPI)
        x_pixels = [int(x * dpi) for x in x_coords]
        y_pixels = [int(y * dpi) for y in y_coords]

        return {"x1": min(x_pixels), "y1": min(y_pixels), "x2": max(x_pixels), "y2": max(y_pixels)}

    def _consolidate_figure_content(self, figure, result: AnalyzeResult) -> str:
        """Consolidate all text content within a figure region"""
        content_parts = []

        # Add figure caption if available
        if hasattr(figure, "caption") and figure.caption:
            content_parts.append(figure.caption.content)

        # Get all paragraphs referenced by this figure
        if hasattr(figure, "elements") and figure.elements and hasattr(result, "paragraphs"):
            for element_ref in figure.elements:
                # Extract paragraph index from reference like "/paragraphs/17"
                if "/paragraphs/" in element_ref:
                    try:
                        para_idx = int(element_ref.split("/paragraphs/")[1])
                        if para_idx < len(result.paragraphs):
                            paragraph = result.paragraphs[para_idx]
                            # Skip if this paragraph is already a caption
                            if (
                                hasattr(figure, "caption")
                                and figure.caption
                                and paragraph.content == figure.caption.content
                            ):
                                continue
                            content_parts.append(paragraph.content)
                    except (ValueError, IndexError):
                        continue

        # Join all content and normalize
        consolidated_content = "\n".join(content_parts)
        return self.normalize_checkboxes(consolidated_content)

    def _extract_table_markdown_and_html(self, table) -> tuple[str, str, str]:
        """Extract table content, markdown, and HTML directly from Azure table object"""
        table_caption = ""
        if hasattr(table, "caption") and table.caption:
            table_caption = self.normalize_checkboxes(table.caption.content)
            table_caption = self.ensure_proper_spacing(table_caption)

        # Get HTML content directly from Azure table object if available
        html_content = ""
        if hasattr(table, "html") and table.html:
            html_content = self.normalize_checkboxes(table.html)

        # Extract cell data and create markdown
        if hasattr(table, "cells") and table.cells:
            rows = {}
            for cell in table.cells:
                if cell.row_index not in rows:
                    rows[cell.row_index] = {}
                rows[cell.row_index][cell.column_index] = cell.content

            if rows:
                max_cols = max(max(row.keys()) for row in rows.values()) + 1
                markdown_lines = []

                if table_caption:
                    markdown_lines.append(table_caption.strip())
                    markdown_lines.append("")

                for row_idx in sorted(rows.keys()):
                    row_cells = [
                        rows[row_idx].get(col_idx, "").replace("|", "\\|")
                        for col_idx in range(max_cols)
                    ]
                    markdown_lines.append("| " + " | ".join(row_cells) + " |")
                    if row_idx == 0:
                        markdown_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

                table_markdown = "\n".join(markdown_lines)
                table_content = "\n".join(
                    [
                        " | ".join(rows[row_idx].get(col, "") for col in range(max_cols))
                        for row_idx in sorted(rows.keys())
                    ]
                )

                # Generate HTML from table structure if not available from Azure
                if not html_content:
                    html_lines = ["<table>"]
                    for row_idx in sorted(rows.keys()):
                        html_lines.append("<tr>")
                        for col_idx in range(max_cols):
                            cell_content = rows[row_idx].get(col_idx, "")
                            tag = "th" if row_idx == 0 else "td"
                            html_lines.append(f"<{tag}>{cell_content}</{tag}>")
                        html_lines.append("</tr>")
                    html_lines.append("</table>")
                    html_content = "\n".join(html_lines)

                full_content = table_caption + table_content if table_caption else table_content
                return (
                    self.normalize_checkboxes(table_markdown),
                    self.normalize_checkboxes(full_content),
                    html_content,
                )

        return table_caption, table_caption, html_content

    def extract_page_layout_from_pdf_result(self, result: AnalyzeResult, page_number: int) -> Dict:
        """Extract layout representation for a specific page from PDF result"""
        if not hasattr(result, "pages") or not result.pages:
            return {"dimensions": [612, 792], "page_fragments": []}  # Default letter size

        # Find the specific page
        target_page = None
        for page in result.pages:
            if page.page_number == page_number:
                target_page = page
                break

        if not target_page:
            return {"dimensions": [612, 792], "page_fragments": []}

        # Get page dimensions in pixels (convert from inches)
        page_width_inches = getattr(target_page, "width", 8.5)
        page_height_inches = getattr(target_page, "height", 11.0)
        page_width_pixels = self.convert_inches_to_pixels(page_width_inches)
        page_height_pixels = self.convert_inches_to_pixels(page_height_inches)

        # Use spans for logical reading order if available
        if hasattr(result, "content") and result.content:
            page_fragments = self._create_fragments_in_span_order_for_page(
                result, page_number, page_width_pixels, page_height_pixels
            )
        else:
            # Fallback: No content available, return empty
            page_fragments = []

        return {
            "dimensions": [page_width_pixels, page_height_pixels],
            "layout": {},
            "page_class": None,
            "page_fragments": page_fragments,
        }

    def _create_fragments_in_span_order_for_page(
        self,
        result: AnalyzeResult,
        page_number: int,
        page_width_pixels: int,
        page_height_pixels: int,
    ) -> List[Dict]:
        """Create fragments in span order for a specific page"""

        # Build section hierarchy map from Azure sections
        hierarchy_map = self.build_section_hierarchy_map(result)

        # First, identify paragraphs that are part of figures or tables to avoid duplicates
        paragraphs_in_figures = set()
        paragraphs_in_tables = set()

        # Collect paragraphs referenced by figures
        if hasattr(result, "figures") and result.figures:
            for figure in result.figures:
                if hasattr(figure, "elements") and figure.elements:
                    for element_ref in figure.elements:
                        if element_ref.startswith("/paragraphs/"):
                            paragraphs_in_figures.add(element_ref)
                # Also check figure caption
                if (
                    hasattr(figure, "caption")
                    and figure.caption
                    and hasattr(figure.caption, "elements")
                ):
                    for element_ref in figure.caption.elements:
                        if element_ref.startswith("/paragraphs/"):
                            paragraphs_in_figures.add(element_ref)

        # Collect paragraphs referenced by tables
        if hasattr(result, "tables") and result.tables:
            for table in result.tables:
                # Check table cells for paragraph references
                if hasattr(table, "cells") and table.cells:
                    for cell in table.cells:
                        if hasattr(cell, "elements") and cell.elements:
                            for element_ref in cell.elements:
                                if element_ref.startswith("/paragraphs/"):
                                    paragraphs_in_tables.add(element_ref)
                # Also check table caption
                if (
                    hasattr(table, "caption")
                    and table.caption
                    and hasattr(table.caption, "elements")
                ):
                    for element_ref in table.caption.elements:
                        if element_ref.startswith("/paragraphs/"):
                            paragraphs_in_tables.add(element_ref)

        # Collect all Azure elements with their spans for this page
        azure_elements = []

        # Add paragraphs for this page (excluding those already in figures/tables)
        if hasattr(result, "paragraphs") and result.paragraphs:
            for i, paragraph in enumerate(result.paragraphs):
                paragraph_ref = f"/paragraphs/{i}"
                if (
                    hasattr(paragraph, "bounding_regions")
                    and paragraph.bounding_regions
                    and hasattr(paragraph.bounding_regions[0], "page_number")
                    and paragraph.bounding_regions[0].page_number == page_number
                    and hasattr(paragraph, "spans")
                    and paragraph.spans
                    and paragraph_ref not in paragraphs_in_figures
                    and paragraph_ref not in paragraphs_in_tables
                ):

                    min_offset = min(span.offset for span in paragraph.spans)
                    azure_elements.append(
                        {
                            "offset": min_offset,
                            "type": "paragraph",
                            "element": paragraph,
                            "paragraph_index": i,  # Store index for hierarchy lookup
                        }
                    )

        # Add tables for this page
        if hasattr(result, "tables") and result.tables:
            for table in result.tables:
                if (
                    hasattr(table, "bounding_regions")
                    and table.bounding_regions
                    and hasattr(table.bounding_regions[0], "page_number")
                    and table.bounding_regions[0].page_number == page_number
                    and hasattr(table, "spans")
                    and table.spans
                ):

                    min_offset = min(span.offset for span in table.spans)
                    azure_elements.append(
                        {
                            "offset": min_offset,
                            "type": "table",
                            "element": table,
                            "ref_id": f"/tables/{result.tables.index(table)}",  # Add ref_id
                        }
                    )

        # Add figures for this page
        if hasattr(result, "figures") and result.figures:
            for figure in result.figures:
                if (
                    hasattr(figure, "bounding_regions")
                    and figure.bounding_regions
                    and hasattr(figure.bounding_regions[0], "page_number")
                    and figure.bounding_regions[0].page_number == page_number
                    and hasattr(figure, "spans")
                    and figure.spans
                ):

                    min_offset = min(span.offset for span in figure.spans)
                    azure_elements.append(
                        {
                            "offset": min_offset,
                            "type": "figure",
                            "element": figure,
                            "ref_id": f"/figures/{result.figures.index(figure)}",  # Add ref_id
                        }
                    )

        # Sort by span offset (natural reading order)
        azure_elements.sort(key=lambda x: x["offset"])

        # Create fragments directly in the correct order
        page_fragments = []
        reading_order = 0

        for i, azure_element in enumerate(azure_elements):
            element_type = azure_element["type"]
            element = azure_element["element"]

            if element_type == "paragraph":
                # Process paragraph
                bbox = self.convert_inches_bbox_to_pixels(element.bounding_regions[0].polygon)
                if bbox:
                    paragraph_index = azure_element.get("paragraph_index", 0)
                    is_header, hierarchy_level = self.is_section_header_paragraph(
                        element, hierarchy_map, paragraph_index
                    )

                    role = getattr(element, "role", None)
                    fragment_type = self.map_role_to_fragment_type(role)

                    paragraph_content = self.normalize_checkboxes(element.content)
                    paragraph_content = self.ensure_proper_spacing(paragraph_content)

                    # Create appropriate content structure based on whether it's a section header
                    if is_header and fragment_type in ["section_header", "title"]:
                        content = {"content": paragraph_content, "level": hierarchy_level}
                    else:
                        content = {"content": paragraph_content}

                    page_fragments.append(
                        {
                            "bbox": bbox,
                            "content": content,
                            "fragment_type": fragment_type,
                            "reading_order": reading_order,
                            "ref_id": f"{page_number}.{reading_order}",
                        }
                    )
                    reading_order += 1

            elif element_type == "table":
                # Process table
                bbox = self.convert_inches_bbox_to_pixels(element.bounding_regions[0].polygon)
                if bbox:
                    if hasattr(element, "caption") and element.caption:
                        # Create separate fragments for each paragraph in the caption
                        if (
                            hasattr(element.caption, "elements")
                            and element.caption.elements
                            and len(element.caption.elements) > 1
                        ):
                            # Multiple paragraphs - create separate fragments for each
                            for i, element_ref in enumerate(element.caption.elements):
                                if element_ref.startswith("/paragraphs/"):
                                    try:
                                        para_idx = int(element_ref.split("/paragraphs/")[1])
                                        if para_idx < len(result.paragraphs):
                                            paragraph = result.paragraphs[para_idx]
                                            para_bbox = self.convert_inches_bbox_to_pixels(
                                                paragraph.bounding_regions[0].polygon
                                            )
                                            if para_bbox:
                                                # First paragraph is table_caption, others are text
                                                fragment_type = (
                                                    "table_caption" if i == 0 else "text"
                                                )
                                                page_fragments.append(
                                                    {
                                                        "bbox": para_bbox,
                                                        "content": {
                                                            "content": self.normalize_checkboxes(
                                                                paragraph.content
                                                            )
                                                        },
                                                        "fragment_type": fragment_type,
                                                        "reading_order": reading_order,
                                                        "ref_id": f"{page_number}.{reading_order}",
                                                    }
                                                )
                                                reading_order += 1
                                    except (ValueError, IndexError):
                                        continue
                        else:
                            # Single paragraph caption
                            page_fragments.append(
                                {
                                    "bbox": self.convert_inches_bbox_to_pixels(
                                        element.caption.bounding_regions[0].polygon
                                    ),
                                    "content": {
                                        "content": self.normalize_checkboxes(
                                            element.caption.content
                                        )
                                    },
                                    "fragment_type": "table_caption",
                                    "reading_order": reading_order,
                                    "ref_id": f"{page_number}.{reading_order}",
                                }
                            )
                            reading_order += 1

                    # Build table content - use HTML directly from Azure table object
                    table_markdown, table_content, html_content_from_table = (
                        self._extract_table_markdown_and_html(element)
                    )

                    # Remove all caption-related text from table content since we create separate fragments
                    if hasattr(element, "caption") and element.caption:
                        # Remove individual paragraph content if multiple paragraphs
                        if hasattr(element.caption, "elements") and element.caption.elements:
                            for element_ref in element.caption.elements:
                                if element_ref.startswith("/paragraphs/"):
                                    try:
                                        para_idx = int(element_ref.split("/paragraphs/")[1])
                                        if para_idx < len(result.paragraphs):
                                            paragraph = result.paragraphs[para_idx]
                                            para_text = paragraph.content
                                            table_content = table_content.replace(
                                                para_text, ""
                                            ).strip()
                                            table_markdown = table_markdown.replace(
                                                para_text, ""
                                            ).strip()
                                    except (ValueError, IndexError):
                                        continue
                        else:
                            # Fallback to removing full caption content
                            caption_text = element.caption.content
                            table_content = table_content.replace(caption_text, "").strip()
                            table_markdown = table_markdown.replace(caption_text, "").strip()

                    content = {
                        "cells": [],
                        "content": self.ensure_proper_spacing(table_content),
                        "html": self.normalize_checkboxes(html_content_from_table),
                        "markdown": self.ensure_proper_spacing(table_markdown),
                        "summary": None,
                    }

                    page_fragments.append(
                        {
                            "bbox": bbox,
                            "content": content,
                            "fragment_type": "table",
                            "reading_order": reading_order,
                            "ref_id": f"{page_number}.{reading_order}",
                        }
                    )

                    page_key = f"{page_number}.{reading_order}"
                    self.tables_by_reading_order[page_key] = element

                    reading_order += 1

            elif element_type == "figure":
                # Process figure
                bbox = self.convert_inches_bbox_to_pixels(element.bounding_regions[0].polygon)
                if bbox:
                    if hasattr(element, "caption") and element.caption:
                        # Create separate fragments for each paragraph in the caption
                        if (
                            hasattr(element.caption, "elements")
                            and element.caption.elements
                            and len(element.caption.elements) > 1
                        ):
                            # Multiple paragraphs - create separate fragments for each
                            for i, element_ref in enumerate(element.caption.elements):
                                if element_ref.startswith("/paragraphs/"):
                                    try:
                                        para_idx = int(element_ref.split("/paragraphs/")[1])
                                        if para_idx < len(result.paragraphs):
                                            paragraph = result.paragraphs[para_idx]
                                            para_bbox = self.convert_inches_bbox_to_pixels(
                                                paragraph.bounding_regions[0].polygon
                                            )
                                            if para_bbox:
                                                # First paragraph is figure_caption, others are text
                                                fragment_type = (
                                                    "figure_caption" if i == 0 else "text"
                                                )
                                                page_fragments.append(
                                                    {
                                                        "bbox": para_bbox,
                                                        "content": {
                                                            "content": self.normalize_checkboxes(
                                                                paragraph.content
                                                            )
                                                        },
                                                        "fragment_type": fragment_type,
                                                        "reading_order": reading_order,
                                                        "ref_id": f"{page_number}.{reading_order}",
                                                    }
                                                )
                                                reading_order += 1
                                    except (ValueError, IndexError):
                                        continue
                        else:
                            # Single paragraph caption
                            page_fragments.append(
                                {
                                    "bbox": self.convert_inches_bbox_to_pixels(
                                        element.caption.bounding_regions[0].polygon
                                    ),
                                    "content": {
                                        "content": self.normalize_checkboxes(
                                            element.caption.content
                                        )
                                    },
                                    "fragment_type": "figure_caption",
                                    "reading_order": reading_order,
                                    "ref_id": f"{page_number}.{reading_order}",
                                }
                            )
                            reading_order += 1

                    figure_content = self._consolidate_figure_content(element, result)
                    # Remove all caption-related text from figure content since we create separate fragments
                    if hasattr(element, "caption") and element.caption:
                        # Remove individual paragraph content if multiple paragraphs
                        if hasattr(element.caption, "elements") and element.caption.elements:
                            for element_ref in element.caption.elements:
                                if element_ref.startswith("/paragraphs/"):
                                    try:
                                        para_idx = int(element_ref.split("/paragraphs/")[1])
                                        if para_idx < len(result.paragraphs):
                                            paragraph = result.paragraphs[para_idx]
                                            para_text = paragraph.content
                                            figure_content = figure_content.replace(
                                                para_text, ""
                                            ).strip()
                                    except (ValueError, IndexError):
                                        continue
                        else:
                            # Fallback to removing full caption content
                            figure_content = figure_content.replace(
                                element.caption.content, ""
                            ).strip()

                    page_fragments.append(
                        {
                            "bbox": bbox,
                            "content": {"content": figure_content},
                            "fragment_type": "figure",
                            "reading_order": reading_order,
                            "ref_id": f"{page_number}.{reading_order}",
                        }
                    )
                    reading_order += 1

        return page_fragments

    def create_layout_json_in_span_order(self, result: AnalyzeResult, page_number: int = 1) -> Dict:
        """Create layout JSON by processing Azure elements in span order"""
        page_width, page_height = self.get_page_dimensions(result)

        if not hasattr(result, "content") or not result.content:
            # Fallback: No content available, return empty
            return {
                "dimensions": [page_width, page_height],
                "layout": {},
                "page_class": None,
                "page_fragments": [],
            }

        # Build section hierarchy map from Azure sections
        hierarchy_map = self.build_section_hierarchy_map(result)

        # First, identify paragraphs that are part of figures or tables to avoid duplicates
        paragraphs_in_figures = set()
        paragraphs_in_tables = set()

        # Collect paragraphs referenced by figures
        if hasattr(result, "figures") and result.figures:
            for figure in result.figures:
                if hasattr(figure, "elements") and figure.elements:
                    for element_ref in figure.elements:
                        if element_ref.startswith("/paragraphs/"):
                            paragraphs_in_figures.add(element_ref)
                # Also check figure caption
                if (
                    hasattr(figure, "caption")
                    and figure.caption
                    and hasattr(figure.caption, "elements")
                ):
                    for element_ref in figure.caption.elements:
                        if element_ref.startswith("/paragraphs/"):
                            paragraphs_in_figures.add(element_ref)

        # Collect paragraphs referenced by tables
        if hasattr(result, "tables") and result.tables:
            for table in result.tables:
                # Check table cells for paragraph references
                if hasattr(table, "cells") and table.cells:
                    for cell in table.cells:
                        if hasattr(cell, "elements") and cell.elements:
                            for element_ref in cell.elements:
                                if element_ref.startswith("/paragraphs/"):
                                    paragraphs_in_tables.add(element_ref)
                # Also check table caption
                if (
                    hasattr(table, "caption")
                    and table.caption
                    and hasattr(table.caption, "elements")
                ):
                    for element_ref in table.caption.elements:
                        if element_ref.startswith("/paragraphs/"):
                            paragraphs_in_tables.add(element_ref)

        # Collect all Azure elements with their spans
        azure_elements = []

        # Add paragraphs (excluding those already in figures/tables)
        if hasattr(result, "paragraphs") and result.paragraphs:
            for i, paragraph in enumerate(result.paragraphs):
                paragraph_ref = f"/paragraphs/{i}"
                if (
                    hasattr(paragraph, "spans")
                    and paragraph.spans
                    and paragraph_ref not in paragraphs_in_figures
                    and paragraph_ref not in paragraphs_in_tables
                ):

                    min_offset = min(span.offset for span in paragraph.spans)
                    azure_elements.append(
                        {
                            "offset": min_offset,
                            "type": "paragraph",
                            "element": paragraph,
                            "paragraph_index": i,  # Store index for hierarchy lookup
                        }
                    )

        # Add tables
        if hasattr(result, "tables") and result.tables:
            for table in result.tables:
                if hasattr(table, "spans") and table.spans:
                    min_offset = min(span.offset for span in table.spans)
                    azure_elements.append({"offset": min_offset, "type": "table", "element": table})

        # Add figures
        if hasattr(result, "figures") and result.figures:
            for figure in result.figures:
                if hasattr(figure, "spans") and figure.spans:
                    min_offset = min(span.offset for span in figure.spans)
                    azure_elements.append(
                        {"offset": min_offset, "type": "figure", "element": figure}
                    )

        # Sort by span offset (natural reading order)
        azure_elements.sort(key=lambda x: x["offset"])

        # Create fragments directly in the correct order
        page_fragments = []
        reading_order = 0

        for i, azure_element in enumerate(azure_elements):
            element_type = azure_element["type"]
            element = azure_element["element"]

            if element_type == "paragraph":
                # Process paragraph
                if hasattr(element, "bounding_regions") and element.bounding_regions:
                    bbox = self.polygon_to_bbox(element.bounding_regions[0].polygon)
                    if bbox:
                        paragraph_index = azure_element.get("paragraph_index", 0)
                        is_header, hierarchy_level = self.is_section_header_paragraph(
                            element, hierarchy_map, paragraph_index
                        )

                        role = getattr(element, "role", None)
                        fragment_type = self.map_role_to_fragment_type(role)

                        paragraph_content = self.normalize_checkboxes(element.content)
                        paragraph_content = self.ensure_proper_spacing(paragraph_content)

                        # Create appropriate content structure based on whether it's a section header
                        if is_header and fragment_type in ["section_header", "title"]:
                            content = {"content": paragraph_content, "level": hierarchy_level}
                        else:
                            content = {"content": paragraph_content}

                        page_fragments.append(
                            {
                                "bbox": bbox,
                                "content": content,
                                "fragment_type": fragment_type,
                                "reading_order": reading_order,
                                "ref_id": f"{page_number}.{reading_order}",
                            }
                        )
                        reading_order += 1

            elif element_type == "table":
                # Process table
                if hasattr(element, "bounding_regions") and element.bounding_regions:
                    bbox = self.polygon_to_bbox(element.bounding_regions[0].polygon)
                    if bbox:
                        if hasattr(element, "caption") and element.caption:
                            # Create separate fragments for each paragraph in the caption
                            if (
                                hasattr(element.caption, "elements")
                                and element.caption.elements
                                and len(element.caption.elements) > 1
                            ):
                                # Multiple paragraphs - create separate fragments for each
                                for i, element_ref in enumerate(element.caption.elements):
                                    if element_ref.startswith("/paragraphs/"):
                                        try:
                                            para_idx = int(element_ref.split("/paragraphs/")[1])
                                            if para_idx < len(result.paragraphs):
                                                paragraph = result.paragraphs[para_idx]
                                                para_bbox = self.polygon_to_bbox(
                                                    paragraph.bounding_regions[0].polygon
                                                )
                                                if para_bbox:
                                                    # First paragraph is table_caption, others are text
                                                    fragment_type = (
                                                        "table_caption" if i == 0 else "text"
                                                    )
                                                    page_fragments.append(
                                                        {
                                                            "bbox": para_bbox,
                                                            "content": {
                                                                "content": self.normalize_checkboxes(
                                                                    paragraph.content
                                                                )
                                                            },
                                                            "fragment_type": fragment_type,
                                                            "reading_order": reading_order,
                                                            "ref_id": f"{page_number}.{reading_order}",
                                                        }
                                                    )
                                                    reading_order += 1
                                        except (ValueError, IndexError):
                                            continue
                            else:
                                # Single paragraph caption
                                page_fragments.append(
                                    {
                                        "bbox": self.polygon_to_bbox(
                                            element.caption.bounding_regions[0].polygon
                                        ),
                                        "content": {
                                            "content": self.normalize_checkboxes(
                                                element.caption.content
                                            )
                                        },
                                        "fragment_type": "table_caption",
                                        "reading_order": reading_order,
                                        "ref_id": f"{page_number}.{reading_order}",
                                    }
                                )
                                reading_order += 1

                        # Build table content - use HTML directly from Azure table object
                        table_markdown, table_content, html_content_from_table = (
                            self._extract_table_markdown_and_html(element)
                        )

                        # Remove all caption-related text from table content since we create separate fragments
                        if hasattr(element, "caption") and element.caption:
                            # Remove individual paragraph content if multiple paragraphs
                            if hasattr(element.caption, "elements") and element.caption.elements:
                                for element_ref in element.caption.elements:
                                    if element_ref.startswith("/paragraphs/"):
                                        try:
                                            para_idx = int(element_ref.split("/paragraphs/")[1])
                                            if para_idx < len(result.paragraphs):
                                                paragraph = result.paragraphs[para_idx]
                                                para_text = paragraph.content
                                                table_content = table_content.replace(
                                                    para_text, ""
                                                ).strip()
                                                table_markdown = table_markdown.replace(
                                                    para_text, ""
                                                ).strip()
                                        except (ValueError, IndexError):
                                            continue
                            else:
                                # Fallback to removing full caption content
                                caption_text = element.caption.content
                                table_content = table_content.replace(caption_text, "").strip()
                                table_markdown = table_markdown.replace(caption_text, "").strip()

                        content = {
                            "cells": [],
                            "content": self.ensure_proper_spacing(table_content),
                            "html": self.normalize_checkboxes(html_content_from_table),
                            "markdown": self.ensure_proper_spacing(table_markdown),
                            "summary": None,
                        }

                        page_fragments.append(
                            {
                                "bbox": bbox,
                                "content": content,
                                "fragment_type": "table",
                                "reading_order": reading_order,
                                "ref_id": f"{page_number}.{reading_order}",
                            }
                        )

                        page_key = f"{page_number}.{reading_order}"
                        self.tables_by_reading_order[page_key] = element

                        reading_order += 1

            elif element_type == "figure":
                # Process figure
                if hasattr(element, "bounding_regions") and element.bounding_regions:
                    bbox = self.polygon_to_bbox(element.bounding_regions[0].polygon)
                    if bbox:
                        if hasattr(element, "caption") and element.caption:
                            # Create separate fragments for each paragraph in the caption
                            if (
                                hasattr(element.caption, "elements")
                                and element.caption.elements
                                and len(element.caption.elements) > 1
                            ):
                                # Multiple paragraphs - create separate fragments for each
                                for i, element_ref in enumerate(element.caption.elements):
                                    if element_ref.startswith("/paragraphs/"):
                                        try:
                                            para_idx = int(element_ref.split("/paragraphs/")[1])
                                            if para_idx < len(result.paragraphs):
                                                paragraph = result.paragraphs[para_idx]
                                                para_bbox = self.polygon_to_bbox(
                                                    paragraph.bounding_regions[0].polygon
                                                )
                                                if para_bbox:
                                                    # First paragraph is figure_caption, others are text
                                                    fragment_type = (
                                                        "figure_caption" if i == 0 else "text"
                                                    )
                                                    page_fragments.append(
                                                        {
                                                            "bbox": para_bbox,
                                                            "content": {
                                                                "content": self.normalize_checkboxes(
                                                                    paragraph.content
                                                                )
                                                            },
                                                            "fragment_type": fragment_type,
                                                            "reading_order": reading_order,
                                                            "ref_id": f"{page_number}.{reading_order}",
                                                        }
                                                    )
                                                    reading_order += 1
                                        except (ValueError, IndexError):
                                            continue
                            else:
                                # Single paragraph caption
                                page_fragments.append(
                                    {
                                        "bbox": self.polygon_to_bbox(
                                            element.caption.bounding_regions[0].polygon
                                        ),
                                        "content": {
                                            "content": self.normalize_checkboxes(
                                                element.caption.content
                                            )
                                        },
                                        "fragment_type": "figure_caption",
                                        "reading_order": reading_order,
                                        "ref_id": f"{page_number}.{reading_order}",
                                    }
                                )
                                reading_order += 1

                        figure_content = self._consolidate_figure_content(element, result)
                        # Remove all caption-related text from figure content since we create separate fragments
                        if hasattr(element, "caption") and element.caption:
                            # Remove individual paragraph content if multiple paragraphs
                            if hasattr(element.caption, "elements") and element.caption.elements:
                                for element_ref in element.caption.elements:
                                    if element_ref.startswith("/paragraphs/"):
                                        try:
                                            para_idx = int(element_ref.split("/paragraphs/")[1])
                                            if para_idx < len(result.paragraphs):
                                                paragraph = result.paragraphs[para_idx]
                                                para_text = paragraph.content
                                                figure_content = figure_content.replace(
                                                    para_text, ""
                                                ).strip()
                                        except (ValueError, IndexError):
                                            continue
                            else:
                                # Fallback to removing full caption content
                                figure_content = figure_content.replace(
                                    element.caption.content, ""
                                ).strip()

                        page_fragments.append(
                            {
                                "bbox": bbox,
                                "content": {"content": figure_content},
                                "fragment_type": "figure",
                                "reading_order": reading_order,
                                "ref_id": f"{page_number}.{reading_order}",
                            }
                        )
                        reading_order += 1

        return {
            "dimensions": [page_width, page_height],
            "layout": {},
            "page_class": None,
            "page_fragments": page_fragments,
        }

    def create_layout_json(self, result: AnalyzeResult, page_number: int = 1) -> Dict:
        """Create layout JSON - uses span order when available"""
        return self.create_layout_json_in_span_order(result, page_number)

    def extract_cell_bboxes_for_tables(
        self,
        page_layout,
        page_width_pixels: int,
        page_height_pixels: int,
        page_width_inches: float = 8.5,
        page_height_inches: float = 11.0,
    ):
        """
        Extract cell-level bounding boxes for all tables in the page.

        Args:
            page_layout: PageLayout object with elements
            page_width_pixels: Page width in pixels
            page_height_pixels: Page height in pixels
            page_width_inches: Page width in inches
            page_height_inches: Page height in inches
        """
        from tensorlake_docai.ocr.azure_cell_bbox_extractor import AzureCellBboxExtractor

        extractor = AzureCellBboxExtractor()

        for element in page_layout.elements:
            if element.fragment_type == PageFragmentType.TABLE:
                page_key = f"{page_layout.page_number}.{element.reading_order}"
                if page_key in self.tables_by_reading_order:
                    azure_table = self.tables_by_reading_order[page_key]

                    # Extract cell bboxes with ref_ids
                    cell_bboxes = extractor.extract_table_cells_with_bboxes(
                        azure_table,
                        page_width_pixels,
                        page_height_pixels,
                        page_width_inches,
                        page_height_inches,
                        page_layout.page_number,
                        element.reading_order,
                    )

                    if cell_bboxes:
                        element.text_bounding_boxes = cell_bboxes

    def polygon_to_bbox(self, polygon) -> Optional[Dict]:
        """Convert polygon to bounding box"""
        if not polygon or len(polygon) < 4:
            return None

        x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
        y_coords = [polygon[i] for i in range(1, len(polygon), 2)]

        return {
            "x1": int(min(x_coords)),
            "y1": int(min(y_coords)),
            "x2": int(max(x_coords)),
            "y2": int(max(y_coords)),
        }

    def get_page_dimensions(self, result: AnalyzeResult) -> tuple[int, int]:
        """Extract page dimensions"""
        if hasattr(result, "pages") and result.pages:
            page = result.pages[0]
            width = getattr(page, "width", 100)
            height = getattr(page, "height", 100)
            return int(width), int(height)
        return 100, 100

    def map_role_to_fragment_type(self, role: Optional[str]) -> str:
        """Map Azure paragraph roles to fragment types"""
        if not role:
            return "text"

        role_mappings = {
            "title": "section_header",
            "sectionHeading": "section_header",
            "pageHeader": "page_header",
            "pageFooter": "page_footer",
            "pageNumber": "page_number",
            "footnote": "page_footer",
        }
        return role_mappings.get(role, "text")

    def normalize_checkboxes(self, content: str) -> str:
        """Replace checkbox characters and text patterns with markdown-style checkboxes"""
        if not content:
            return content

        # Replace checkbox symbols
        content = content.replace("☒", "[x]")
        content = content.replace("☐", "[ ]")

        # Replace Azure text patterns
        content = content.replace(":selected:", "[x]")
        content = content.replace(":unselected:", "[ ]")

        return content

    def build_section_hierarchy_map(self, result: AnalyzeResult) -> Dict[str, int]:
        """
        Build a mapping from paragraph references to their hierarchy levels based on Azure sections.
        Azure uses a flat sections array where sections reference each other via "/sections/X" elements.
        Returns a dictionary mapping "/paragraphs/X" to hierarchy level (0 = top level, 1 = subsection, etc.)
        """
        hierarchy_map = {}

        if not hasattr(result, "sections") or not result.sections:
            print("No sections found in Azure result")
            return hierarchy_map

        # Create a section lookup map for quick access
        sections_by_index = {i: section for i, section in enumerate(result.sections)}

        def process_section_recursive(section_index: int, level: int = 0, visited: set = None):
            """Recursively process sections using flat Azure structure"""
            if visited is None:
                visited = set()

            if section_index in visited:
                return  # Avoid infinite recursion

            visited.add(section_index)

            if section_index not in sections_by_index:
                return

            section = sections_by_index[section_index]

            # Process elements in this section (Azure sections are dicts, not objects)
            elements = section.get("elements", [])
            if elements:
                for element_ref in elements:
                    if element_ref.startswith("/paragraphs/"):
                        # Map this paragraph to its hierarchy level
                        hierarchy_map[element_ref] = level
                    elif element_ref.startswith("/sections/"):
                        # Process referenced section at next level
                        try:
                            ref_section_index = int(element_ref.split("/sections/")[1])
                            process_section_recursive(ref_section_index, level + 1, visited.copy())
                        except (ValueError, IndexError):
                            continue

        # Start with section 0 as the root (assuming it exists and is the main container)
        if len(result.sections) > 0:
            process_section_recursive(0, 0)

        # Check if we need to adjust levels down by 1
        # If the minimum level in the map is 1 (not 0), adjust all levels down
        if hierarchy_map:
            min_level = min(hierarchy_map.values())
            if min_level > 0:
                for paragraph_ref in hierarchy_map:
                    hierarchy_map[paragraph_ref] -= min_level

        return hierarchy_map

    def is_section_header_paragraph(
        self, paragraph, hierarchy_map: Dict[str, int], paragraph_index: int
    ) -> Tuple[bool, int]:
        """
        Determine if a paragraph is a section header and return its hierarchy level.
        Returns (is_header, level) where level is 0 for top-level, 1 for subsection, etc.
        """
        paragraph_ref = f"/paragraphs/{paragraph_index}"

        # Check if this paragraph is referenced in our hierarchy map
        if paragraph_ref in hierarchy_map:
            level = hierarchy_map[paragraph_ref]
            return True, level

        # Fallback: check Azure's role assignment (paragraphs might be objects or dicts)
        paragraph_role = (
            getattr(paragraph, "role", None) or paragraph.get("role", None)
            if hasattr(paragraph, "get")
            else getattr(paragraph, "role", None)
        )
        if paragraph_role in ["title", "sectionHeading"]:
            return True, 0  # Default to top level if not found in sections

        return False, 0

    def ensure_proper_spacing(self, content: str) -> str:
        """Ensure content has proper spacing for markdown rendering"""
        if not content:
            return content

        content = content.strip()
        if content and not content.endswith("\n\n"):
            content += "\n\n"
        return content

    def extract_layout_representation(
        self, result: AnalyzeResult, page_width: int, page_height: int, page_number: int = 1
    ) -> Dict:
        """Extract layout representation - compatibility method for both PDF and image processing"""
        # For PDF processing, use the new page-specific method
        if hasattr(result, "pages") and result.pages and len(result.pages) > 1:
            return self.extract_page_layout_from_pdf_result(result, page_number)
        else:
            # For single page/image processing, use the existing method
            return self.create_layout_json(result, page_number)
