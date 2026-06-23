# SPDX-License-Identifier: Apache-2.0
"""
Citation handler for structured extraction.
Handles adding reference IDs to text, enhancing schemas, and resolving citations.
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tensorlake_docai.extraction.schema_enricher_utils import inline_refs, is_simple_schema

# Set to True to write initial and enriched prompts to debug/citation_prompt.md
DEBUG_CITATION: bool = False


class StructuredExtractionCitationHandler:
    """Handles all citation-related functionality for structured extraction"""

    def __init__(self):
        self.citation_map = {}
        self._debug_enabled = DEBUG_CITATION

    def _write_debug_prompts(self, initial_prompt: Optional[str], enriched_prompt: str) -> None:
        if not self._debug_enabled:
            return
        try:
            debug_dir = Path("citation_debug")
            debug_dir.mkdir(parents=True, exist_ok=True)

            output_path = debug_dir / "citation_prompt.md"
            with output_path.open("a", encoding="utf-8") as f:
                f.write("\n\n---\n")
                f.write("Citation Prompt Snapshot\n\n")
                f.write("### Initial Prompt\n\n")
                f.write("```\n")
                f.write((initial_prompt or "").strip())
                f.write("\n```\n")
                f.write("\n### Citation-Enriched Prompt\n\n")
                f.write("```\n")
                f.write(enriched_prompt.strip())
                f.write("\n```\n")
        except Exception:
            pass

    def snapshot_prompts(self, initial_prompt: Optional[str], enriched_prompt: str) -> None:
        """Public method to snapshot prompts with full context.
        Callers should pass the fully composed user prompts (including schema and text).
        """
        self._write_debug_prompts(initial_prompt=initial_prompt, enriched_prompt=enriched_prompt)

    def enhance_html_with_cell_refs(self, html: str, cell_bboxes: List) -> str:
        """
        Concise, position-aware HTML ref injection:
        - Build a (row_index, column_index) -> ref_id map from bboxes
        - Iterate HTML rows/cells in order and append [REF:...] when a map entry exists
        - If indices are missing, leave the cell unchanged (no heuristic matching)
        """
        if not cell_bboxes or not html:
            return html

        # Build position map
        position_to_ref: dict[tuple[int, int], str] = {}
        for b in cell_bboxes:
            r = b.row_index
            c = b.column_index
            ref = b.ref_id
            if r is None or c is None or not ref:
                continue
            position_to_ref[(int(r), int(c))] = ref

        if not position_to_ref:
            return html

        row_pattern = re.compile(r"<tr[^>]*>([\s\S]*?)</tr>", re.IGNORECASE)
        cell_pattern = re.compile(r"<(td|th)(?:\s[^>]*)?>([\s\S]*?)</\1>", re.IGNORECASE)

        rebuilt_html_parts: list[str] = []
        last_row_end = 0
        row_idx = -1
        for row_match in row_pattern.finditer(html):
            row_idx += 1
            row_start, row_end = row_match.span()
            row_inner = row_match.group(1)
            rebuilt_html_parts.append(html[last_row_end:row_start])

            # Rebuild row with cell refs
            rebuilt_row_parts: list[str] = []
            last_cell_end = 0
            col_idx = -1
            for cell_match in cell_pattern.finditer(row_inner):
                col_idx += 1
                tag = cell_match.group(1)
                inner = cell_match.group(2)
                cell_start, cell_end = cell_match.span()
                rebuilt_row_parts.append(row_inner[last_cell_end:cell_start])

                ref = position_to_ref.get((row_idx, col_idx))
                if ref:
                    rebuilt_row_parts.append(f"<{tag}>{inner} [REF:{ref}]</{tag}>")
                else:
                    rebuilt_row_parts.append(f"<{tag}>{inner}</{tag}>")

                last_cell_end = cell_end
            rebuilt_row_parts.append(row_inner[last_cell_end:])
            rebuilt_html_parts.append(f"<tr>{''.join(rebuilt_row_parts)}</tr>")
            last_row_end = row_end

        rebuilt_html_parts.append(html[last_row_end:])
        return "".join(rebuilt_html_parts)

    # NOTE: Markdown ref injection intentionally omitted for simplicity and reliability.
    # When citations are enabled, we prefer HTML tables with position-aware refs.

    def _bbox_tuple_to_dict(
        self, bbox, page_number: Optional[int] = None
    ) -> Optional[Dict[str, float]]:
        """Convert a bbox tuple (x1, y1, x2, y2) to a dict with numeric values, include page_number."""
        try:
            if not bbox or len(bbox) != 4:
                return None
            x1, y1, x2, y2 = bbox
            bbox_dict: Dict[str, float] = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
            if page_number is not None:
                bbox_dict["page_number"] = int(page_number)
            return bbox_dict
        except Exception:
            return None

    def populate_citation_map(self, page_layouts: List) -> None:
        """Populate internal map of ref_id -> bounding box dict from page layouts."""
        if not page_layouts:
            return
        try:
            for page_layout in page_layouts:
                for element in page_layout.elements or []:
                    # Element-level bbox
                    if element.bbox is not None:
                        bbox_dict = self._bbox_tuple_to_dict(
                            element.bbox,
                            (
                                page_layout.page_number
                                if page_layout.page_number is not None
                                else None
                            ),
                        )
                        if bbox_dict:
                            # Map explicit ref_id if present
                            if element.ref_id:
                                self.citation_map[element.ref_id] = bbox_dict
                            # Also map fallback ref_id of the form page.reading_order
                            fallback_ro = (
                                element.reading_order if element.reading_order is not None else 0
                            )
                            page_no = page_layout.page_number
                            if page_no is not None:
                                fallback_ref = f"{page_no}.{fallback_ro}"
                                self.citation_map.setdefault(fallback_ref, bbox_dict)
                    # Cell-level bboxes for tables
                    for cell in element.text_bounding_boxes or []:
                        ref = cell.ref_id
                        bbox = cell.bbox
                        if ref and bbox:
                            bbox_dict = self._bbox_tuple_to_dict(
                                bbox,
                                (
                                    page_layout.page_number
                                    if page_layout.page_number is not None
                                    else None
                                ),
                            )
                            if bbox_dict:
                                self.citation_map[ref] = bbox_dict
        except Exception:
            # Do not fail citation processing if map population encounters unexpected data
            pass

    def prepare_text_with_citations(
        self, content: str, page_layouts: List, table_output_mode: str = "markdown"
    ) -> Tuple[str, bool]:
        """
        Prepare text content with reference IDs for citation tracking.
        Now uses ref_ids from the layout elements directly.
        Respects the table_output_mode setting (html or markdown).
        Returns: (content_with_refs, has_citations)
        """
        ref_content_parts = []
        has_citations = False  # Track if any citations were added

        self.populate_citation_map(page_layouts)

        for page_layout in page_layouts:
            # Add page marker
            ref_content_parts.append(f"\n--- Page {page_layout.page_number} ---\n")

            # Sort elements by reading order
            sorted_elements = sorted(page_layout.elements, key=lambda x: x.reading_order)

            for element in sorted_elements:
                if not element.ocr_text or not element.ocr_text.strip():
                    continue

                element_text = element.ocr_text.strip()
                # Use the pre-assigned ref_id if present; otherwise generate a deterministic fallback
                # based on page number and reading order to avoid emitting invalid [REF:None]
                fallback_ro = element.reading_order if element.reading_order is not None else 0
                ref_id = element.ref_id or f"{page_layout.page_number}.{fallback_ro}"

                # Handle tables with cell-level references
                if element.fragment_type.value == "table":
                    # For citations, always prefer HTML tables
                    if element.html:
                        # Use HTML table with reference ID
                        ref_content_parts.append(f"[REF:{ref_id}:TABLE]\n")
                        has_citations = True  # Table has ref_id

                        # If we have cell-level bounding boxes, enhance the HTML with cell IDs
                        if element.text_bounding_boxes:
                            # Enhance HTML with cell reference IDs using pre-assigned ref_ids
                            enhanced_html = self.enhance_html_with_cell_refs(
                                element.html, element.text_bounding_boxes
                            )
                            ref_content_parts.append(f"{enhanced_html}\n")

                            # Mark that we have cell citations
                            for cell_bbox in element.text_bounding_boxes:
                                if cell_bbox.ref_id:
                                    has_citations = True
                        else:
                            # No cell-level data, just use the HTML
                            ref_content_parts.append(f"{element.html}\n")
                    else:
                        # If no HTML available, fall back to plain text table content without per-cell refs
                        ref_content_parts.append(f"[REF:{ref_id}:TABLE]\n")
                        if element.markdown:
                            ref_content_parts.append(f"{element.markdown}\n")
                        else:
                            ref_content_parts.append(f"{element_text}\n")
                        has_citations = True
                else:
                    # Regular elements (text, headers, forms, key-value regions, etc.)
                    fragment_type = element.fragment_type.value

                    # Add type tag for headers
                    if fragment_type in ["section_header", "title"]:
                        ref_content_parts.append(f"[REF:{ref_id}:HEADER] {element_text}\n")
                    else:
                        ref_content_parts.append(f"[REF:{ref_id}] {element_text}\n")

                    has_citations = True  # Regular element has ref_id

        return "".join(ref_content_parts), has_citations

    def enhance_schema_for_citations(self, schema: Dict) -> Dict:
        """Add _ref fields to JSON schema for citation tracking"""
        if not schema:
            return schema

        def add_ref_fields(obj: Dict) -> Dict:
            # Treat scalar unions as simple types for _ref injection

            def _is_array_type(t):
                return t == "array" or (isinstance(t, list) and "array" in t)

            def _is_object_type(t):
                return t == "object" or (isinstance(t, list) and "object" in t)

            if _is_object_type(obj.get("type")) and "properties" in obj:
                new_props = {}
                required = set(obj.get("required", []))

                for key, value in obj["properties"].items():
                    # First, recursively process nested objects
                    if _is_object_type(value.get("type")):
                        # Recursively enhance nested objects
                        new_props[key] = add_ref_fields(value)
                    elif _is_array_type(value.get("type")) and _is_object_type(
                        (value.get("items", {}) or {}).get("type")
                    ):
                        # Handle arrays of objects
                        new_props[key] = {**value, "items": add_ref_fields(value["items"])}
                    else:
                        # Regular field - copy it
                        new_props[key] = value

                    # Add ref field for simple types (including Optional via anyOf/oneOf)
                    if is_simple_schema(value):
                        ref_key = f"{key}_ref"
                        new_props[ref_key] = {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Reference ID(s) for {key} - array with one or more reference IDs",
                        }
                        required.add(ref_key)
                    elif _is_array_type(value.get("type")):
                        items = (
                            value.get("items", {}) if isinstance(value.get("items"), dict) else {}
                        )
                        if is_simple_schema(items):
                            ref_key = f"{key}_ref"
                            new_props[ref_key] = {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": f"Reference IDs for items in {key}",
                            }
                            required.add(ref_key)
                    # Note: For nested objects and arrays of objects, we don't add a _ref field at this level
                    # Instead, the _ref fields are added to the leaf properties inside those objects

                obj["properties"] = new_props
                if required:
                    obj["required"] = list(required)
            return obj

        # Work on a copy
        enriched = schema.copy()

        # Enrich top-level object (if any)
        enriched = add_ref_fields(enriched)

        # Enrich all $defs objects referenced via $ref
        if "$defs" in enriched and isinstance(enriched["$defs"], dict):
            new_defs = {}
            for def_name, def_obj in enriched["$defs"].items():
                if isinstance(def_obj, dict):
                    new_defs[def_name] = add_ref_fields(def_obj.copy())
                else:
                    new_defs[def_name] = def_obj
            enriched["$defs"] = new_defs

        return enriched

    def inline_refs_for_response_format(self, schema: Dict) -> Dict:
        """Inline $ref while preserving siblings."""
        return inline_refs(schema, include_title=False)

    def add_citation_instructions(self, prompt: Optional[str]) -> str:
        """Add citation instructions to the prompt"""
        citation_instructions = """
IMPORTANT: Include reference IDs for ALL extracted values at every level, including nested objects.

The document contains reference IDs in the format [REF:page.reading_order] or [REF:page.reading_order.cell_index]:

1. Regular text: [REF:1.5] means page 1, reading order 5
2. Headers/titles: [REF:2.3:HEADER] means page 2, reading order 3, header type
3. Tables: [REF:1.16:TABLE] followed by cells with [REF:1.16.20] for cell 20 in table at page 1, reading order 16

For EVERY field in your response (including fields inside nested objects), add a "_ref" field with an ARRAY of reference IDs where you found the value.

ALWAYS use an array for _ref fields, even if there's only one reference.

Examples:
{
    "company_name": "Acme Corp",
    "company_name_ref": ["2.3"],  // Single reference from page 2, reading order 3
    "parties": {
        "seller": "John Doe",
        "seller_ref": ["1.5"],  // Reference for nested field
        "buyer": "Jane Smith", 
        "buyer_ref": ["1.6"]    // Reference for nested field
    },
    "property": {
        "address": "123 Main St",
        "address_ref": ["3.8", "3.9"],  // Multiple references
        "city": "Austin",
        "city_ref": ["3.10"]
    },
    "account_number": "ENV562016543-00",
    "account_number_ref": ["1.16.20", "1.16.99"],  // Found in cells 20 and 99 of table at page 1, reading order 16
}

Include ALL relevant reference IDs for EVERY extracted value, including values in nested objects.
Only return the reference IDs, not the full bounding box information.
"""
        # Handle None prompt
        if prompt:
            enriched = prompt + "\n\n" + citation_instructions
        else:
            enriched = citation_instructions

        # Write debug snapshot if enabled
        self._write_debug_prompts(prompt, enriched)
        return enriched

    def resolve_citations(
        self,
        data: Dict,
        allowed_pages: Optional[set[int]] = None,
        allowed_ref_ids: Optional[set[str]] = None,
        citation_map: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        """
        Resolve citation references to bounding boxes.
        Filters by allowed_ref_ids (precise) or allowed_pages (fallback).
        """
        if not isinstance(data, dict):
            return data

        result = {}
        ref_map = citation_map if isinstance(citation_map, dict) else self.citation_map

        def _ref_page_num(ref: str) -> Optional[int]:
            try:
                first = str(ref).split(".", 1)[0]
                return int(first)
            except Exception:
                return None

        for key, value in data.items():
            if key.endswith("_ref") and value is not None:
                # Convert only the trailing _ref suffix to _citation
                citation_key = f"{key[:-4]}_citation"

                def map_ref(v):
                    if v is None:
                        return None
                    # Look up bbox mapping first
                    mapped = ref_map.get(v)
                    if not mapped:
                        return None

                    # Apply filtering: prefer ref_id filtering (precise), fall back to page filtering
                    if allowed_ref_ids is not None:
                        # Precise filtering by element ref_id
                        if v not in allowed_ref_ids:
                            return None
                    elif allowed_pages is not None:
                        # Fall back to page-level filtering
                        mapped_page = (
                            mapped.get("page_number") if isinstance(mapped, dict) else None
                        )
                        p = mapped_page if mapped_page is not None else _ref_page_num(v)
                        if p is None or p not in allowed_pages:
                            return None

                    return mapped

                if isinstance(value, list):
                    # Multiple references - map each to bbox if available, filter out None
                    mapped_refs = [map_ref(v) for v in value]
                    result[citation_key] = [ref for ref in mapped_refs if ref is not None]
                else:
                    # Single reference - map to bbox if available
                    mapped_ref = map_ref(value)
                    if mapped_ref is not None:
                        result[citation_key] = mapped_ref
                    else:
                        result[citation_key] = []

                # Also include the original field (without _ref)
                original_key = key[:-4]  # Remove '_ref' suffix
                if original_key in data:
                    result[original_key] = data[original_key]

            elif key.endswith("_ref"):
                # _ref field is None, just include the original field
                original_key = key[:-4]
                if original_key in data:
                    result[original_key] = data[original_key]

            elif isinstance(value, dict):
                # Recursively resolve nested objects
                result[key] = self.resolve_citations(
                    value,
                    allowed_pages=allowed_pages,
                    allowed_ref_ids=allowed_ref_ids,
                    citation_map=ref_map,
                )

            elif isinstance(value, list):
                # Handle lists of objects
                result[key] = [
                    (
                        self.resolve_citations(
                            item,
                            allowed_pages=allowed_pages,
                            allowed_ref_ids=allowed_ref_ids,
                            citation_map=ref_map,
                        )
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
            else:
                # Regular field - copy as-is
                result[key] = value

        return result
