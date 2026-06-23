# SPDX-License-Identifier: Apache-2.0
import re
import json
import asyncio
import inspect
import os
from tensorlake_docai.prompts.prompts import get_table_merging_prompt_messages
from typing import Any, Optional, cast
from tensorlake_docai.pipeline.api import (
    ParsedDocument,
    Page,
    PageFragment,
    PageFragmentType,
    ParsedDocumentRef,
    MergedTable,
    MergeTableActions,
    Table,
    Text,
)
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.pipeline.simple_page_creator import SimplePageCreator, ImageDimensions

from tensorlake.applications import cls, function, RequestError as RequestException, RequestContext

from tensorlake_docai.vlm.workflow_images import table_merging_image
from tensorlake_docai.postprocess.formatter import document_layout_to_document
from tensorlake_docai.pipeline.output_formatter import format_final_output
from tensorlake_docai.extraction.structured_extraction_functions import StructuredExtraction
from tensorlake_docai.vlm.cloud import VLMExtractionTask
from tensorlake_docai.pipeline.routing import (
    ocr_should_go_to_output_formatter,
    ocr_should_go_to_structured_extraction,
    ocr_should_go_to_vlm_extraction,
    dots_ocr_should_go_to_vlm_extraction,
    is_markdown_table,
    markdown_to_html_table,
)
from tensorlake_docai.tables.table_correction import TableGridParser

SECRETS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
]

TABLE_MERGING_SCHEMAS = {
    "merged_summary": json.dumps(
        {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
    ),
    "fast_merge": json.dumps(
        {
            "type": "object",
            "properties": {
                "continuation": {"type": "string", "enum": ["YES", "NO"]},
                "explanation": {"type": "string"},
                "skip_rows": {"type": "integer"},
            },
            "required": ["continuation", "explanation", "skip_rows"],
        }
    ),
    "fast_same_page_merge": json.dumps(
        {
            "type": "object",
            "properties": {
                "should_merge": {"type": "string", "enum": ["YES", "NO"]},
                "explanation": {"type": "string"},
                "skip_rows": {"type": "integer"},
                "corrected_html": {"type": "string", "contentMediaType": "text/html"},
            },
            "required": ["should_merge", "explanation", "skip_rows", "corrected_html"],
        }
    ),
    "align_tables": json.dumps(
        {
            "type": "object",
            "properties": {
                "corrected_html": {"type": "string", "contentMediaType": "text/html"},
                "explanation": {"type": "string"},
            },
            "required": ["corrected_html", "explanation"],
        }
    ),
}

# =============================================================================
# HTML & String Utilities
# =============================================================================


def merge_table_htmls(base_html: str, next_html: str, skip_rows: int = 0) -> str:
    """
    Merge next_html into base_html.
    1. Strip <table>, <thead>, <tbody>, <tfoot> tags from next_html to get raw rows.
    2. Skip 'skip_rows' from next_html.
    3. Insert rows into base_html before the closing of the last structural element or table.
    """
    # Clean base_html of wrapping divs
    base_html = re.sub(r"^<div[^>]*>", "", base_html.strip(), flags=re.IGNORECASE).strip()
    base_html = re.sub(r"</div>$", "", base_html, flags=re.IGNORECASE).strip()

    # 1. Clean next_html to get just rows
    # Remove table tags
    next_content = re.sub(r"<\/?table[^>]*>", "", next_html, flags=re.IGNORECASE)
    # Remove structural tags (thead, tbody, tfoot)
    next_content = re.sub(r"<\/?(thead|tbody|tfoot)[^>]*>", "", next_content, flags=re.IGNORECASE)
    # Remove div tags
    next_content = re.sub(r"<\/?div[^>]*>", "", next_content, flags=re.IGNORECASE)
    next_content = next_content.strip()

    # 2. Skip rows
    if skip_rows > 0:
        next_content = remove_header_rows_regex(next_content, skip_rows)

    if not next_content:
        return base_html

    # 3. Append to base_html
    # Try to find closing structural tag before table close
    match = re.search(
        r"(</(tbody|tfoot|thead)>\s*)?</table>\s*(</div>)?\s*$",
        base_html,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        insertion_index = match.start()
        return (
            base_html[:insertion_index] + "\n" + next_content + "\n" + base_html[insertion_index:]
        )

    # Fallback: just append
    return base_html + "\n" + next_content


def extract_json_from_response(text: str) -> Any:
    cleaned_text = text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```[a-zA-Z]*\n", "", cleaned_text)
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Try to find JSON object
        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}


def slice_table_rows(html: str, start: int = 0, end: int | None = None) -> str:
    """
    Extract a subset of rows from an HTML table, preserving the table structure.
    Useful for reducing context window usage during analysis.
    """
    # Extract opening tag (e.g., <table border="1">) or default
    open_tag_match = re.search(r"(<table[^>]*>)", html, re.IGNORECASE)
    open_tag = open_tag_match.group(1) if open_tag_match else "<table>"

    # Extract all rows
    rows = re.findall(r"<tr[^>]*>.*?</tr>", html, flags=re.IGNORECASE | re.DOTALL)

    # Slice the list of rows
    subset = rows[start:end] if end is not None else rows[start:]

    if not subset:
        # Return the original if no rows found (or empty table) to avoid data loss
        return html if not rows else f"{open_tag}</table>"

    return f"{open_tag}\n" + "\n".join(subset) + "\n</table>"


def remove_header_rows_regex(html_body: str, n: int) -> str:
    """Remove the first n <tr>...</tr> rows from the html body string."""
    if n <= 0:
        return html_body

    current_html = html_body
    for _ in range(n):
        # Remove the first occurrence of a tr tag and its content
        # We use count=1 to remove only the first match found in the string
        current_html = re.sub(
            r"<tr[^>]*>.*?</tr>", "", current_html, count=1, flags=re.IGNORECASE | re.DOTALL
        )

    return current_html


# =============================================================================
# Page Fragment Navigation Utilities
# =============================================================================


def get_last_table(page_fragments: list[PageFragment]) -> PageFragment | None:
    """
    Get the last table fragment from a page.

    Args:
        page_fragments: List of PageFragment objects

    Returns:
        The last table fragment or None if no table found
    """
    for idx in range(len(page_fragments) - 1, -1, -1):
        if page_fragments[idx].fragment_type == PageFragmentType.TABLE:
            return page_fragments[idx]
    return None


def get_first_table(page_fragments: list[PageFragment]) -> PageFragment | None:
    """
    Get the first table fragment from a page.

    Args:
        page_fragments: List of dictionaries with fragment information

    Returns:
        The first table fragment dictionary or None if no table found
    """
    for fragment in page_fragments:
        if fragment.fragment_type == PageFragmentType.TABLE:
            return fragment
    return None


def get_cross_page_tables(
    pages: list[Page], page_index: int
) -> tuple[PageFragment | None, PageFragment | None]:
    """
    Get the last table from current page and first table from next page.

    Args:
        pages: List of page dictionaries
        page_index: Index of the current page

    Returns:
        Tuple of (table_end, table_start) or (None, None) if not found
    """
    if page_index >= len(pages) - 1:
        return None, None

    current_page = pages[page_index]
    next_page = pages[page_index + 1]

    current_page_fragments = current_page.page_fragments or []
    table_end = get_last_table(current_page_fragments)

    next_page_fragments = next_page.page_fragments or []
    table_start = get_first_table(next_page_fragments)

    return table_end, table_start


def get_cross_page_table_candidates(
    pages: list[Page], page_index: int, search_depth: int = 2
) -> list[tuple[PageFragment, PageFragment]]:
    """
    Get candidate pairs of tables for cross-page merging.
    Searches the last few tables of current page and first few tables of next page.
    """
    if page_index >= len(pages) - 1:
        return []

    current_page = pages[page_index]
    next_page = pages[page_index + 1]

    current_page_fragments = current_page.page_fragments or []
    next_page_fragments = next_page.page_fragments or []

    # Tables from current page (searching from end)
    current_tables = []
    for idx in range(len(current_page_fragments) - 1, -1, -1):
        if current_page_fragments[idx].fragment_type == PageFragmentType.TABLE:
            current_tables.append(current_page_fragments[idx])
            if len(current_tables) >= search_depth:
                break

    # Tables from next page (searching from start)
    next_tables = []
    for fragment in next_page_fragments:
        if fragment.fragment_type == PageFragmentType.TABLE:
            next_tables.append(fragment)
            if len(next_tables) >= search_depth:
                break

    candidates = []
    for t1 in current_tables:
        for t2 in next_tables:
            candidates.append((t1, t2))
    return candidates


def get_tables_from_page_start(page: Page, limit: int = 3) -> list[PageFragment]:
    fragments = page.page_fragments or []
    tables = []
    for frag in fragments:
        if frag.fragment_type == PageFragmentType.TABLE:
            tables.append(frag)
            if len(tables) >= limit:
                break
    return tables


def get_context_between(
    pages: list[Page], page_idx: int, table_end: PageFragment, table_start: PageFragment
) -> str:
    """Extract text content between the end table of one page and start table of next."""
    context = []

    # Page N fragments after table_end
    p1_frags = pages[page_idx].page_fragments or []
    idx1 = -1
    # Try identity match
    for i, f in enumerate(p1_frags):
        if f is table_end:
            idx1 = i
            break
    # Try content match if identity failed
    if idx1 == -1:
        t_html = table_end.content.html if isinstance(table_end.content, Table) else None
        if t_html:
            for i, f in enumerate(p1_frags):
                f_html = f.content.html if isinstance(f.content, Table) else None
                if f_html == t_html:
                    idx1 = i
                    break

    if idx1 != -1:
        for f in p1_frags[idx1 + 1 :]:
            txt = getattr(f.content, "content", None)
            if txt:
                context.append(txt)

    # Page N+1 fragments before table_start
    p2_frags = pages[page_idx + 1].page_fragments or []
    idx2 = -1
    for i, f in enumerate(p2_frags):
        if f is table_start:
            idx2 = i
            break
    if idx2 == -1:
        t_html = table_start.content.html if isinstance(table_start.content, Table) else None
        if t_html:
            for i, f in enumerate(p2_frags):
                f_html = f.content.html if isinstance(f.content, Table) else None
                if f_html == t_html:
                    idx2 = i
                    break

    if idx2 != -1:
        for f in p2_frags[:idx2]:
            txt = getattr(f.content, "content", None)
            if txt:
                context.append(txt)

    return "\n".join(context)


def get_table_column_count(html: str) -> int:
    parser = TableGridParser()
    try:
        parser.feed(html)
        if parser.rows:
            # Calculate effective width of first row
            row = parser.rows[0]
            width = 0
            for cell in row:
                width += cell.get("colspan", 1)
            return width
    except Exception:
        pass
    return 0


def infer_cell_type(text: str) -> str:
    text = text.strip()
    if not text:
        return "empty"
    # Remove common currency symbols, percentage, and thousands separators
    clean_text = re.sub(r"[$,%\s]", "", text)
    # Handle parentheses for negative numbers
    if clean_text.startswith("(") and clean_text.endswith(")"):
        clean_text = clean_text[1:-1]

    try:
        float(clean_text)
        return "number"
    except ValueError:
        pass
    return "text"


def get_table_column_types(html: str, rows_to_check: int = 3, from_start: bool = True) -> list[str]:
    parser = TableGridParser()
    try:
        # Slice HTML to get relevant rows
        if from_start:
            sliced = slice_table_rows(html, end=rows_to_check)
        else:
            sliced = slice_table_rows(html, start=-rows_to_check)

        parser.feed(sliced)
        if not parser.rows:
            return []

        # Determine max columns in the sample
        max_cols = 0
        for row in parser.rows:
            width = 0
            for cell in row:
                width += cell.get("colspan", 1)
            max_cols = max(max_cols, width)

        if max_cols == 0:
            return []

        # Collect types per column
        column_types = [set() for _ in range(max_cols)]

        for row in parser.rows:
            col_idx = 0
            for cell in row:
                colspan = cell.get("colspan", 1)
                content = cell.get("content", "")
                # Strip HTML tags from content
                content_text = re.sub(r"<[^>]+>", "", content)
                ctype = infer_cell_type(content_text)

                for i in range(colspan):
                    if col_idx + i < max_cols:
                        if ctype != "empty":
                            column_types[col_idx + i].add(ctype)
                col_idx += colspan

        # Determine dominant type
        final_types = []
        for types in column_types:
            if "text" in types:
                final_types.append("text")
            elif "number" in types:
                final_types.append("number")
            else:
                final_types.append("empty")

        return final_types
    except Exception:
        return []


def _is_vertically_aligned(frag1: PageFragment, frag2: PageFragment) -> bool:
    """
    Check if two tables are vertically aligned (i.e. stacked one above another in the same column).
    This is determined by checking if their horizontal coordinates overlap significantly.
    """
    if not frag1.bbox or not frag2.bbox:
        return True

    b1 = frag1.bbox
    b2 = frag2.bbox

    x1 = max(b1.get("x1", 0), b2.get("x1", 0))
    x2 = min(b1.get("x2", 0), b2.get("x2", 0))

    overlap = max(0, x2 - x1)

    width1 = b1.get("x2", 0) - b1.get("x1", 0)
    width2 = b2.get("x2", 0) - b2.get("x1", 0)

    if width1 <= 0 or width2 <= 0:
        return True

    min_width = min(width1, width2)

    return (overlap / min_width) > 0.5


def are_tables_semantically_aligned(html1: str, html2: str) -> bool:
    # Check last rows of html1 vs first rows of html2
    types1 = get_table_column_types(html1, rows_to_check=5, from_start=False)
    types2 = get_table_column_types(html2, rows_to_check=5, from_start=True)

    if not types1 or not types2:
        return True

    if len(types1) != len(types2):
        return False

    mismatches = 0
    comparisons = 0
    for t1, t2 in zip(types1, types2):
        if t1 != "empty" and t2 != "empty":
            comparisons += 1
            if t1 != t2:
                mismatches += 1

    if comparisons > 0:
        # If significant mismatch
        if mismatches / comparisons > 0.3:  # 30% mismatch threshold
            return False

    return True


def _crop_image_from_bbox(image: Any, bbox: dict[str, float], scale_factor: float) -> Any | None:
    """
    Crop a region from an image based on a bounding box and scale factor.
    """
    if not bbox:
        return None

    x1 = int(bbox["x1"] * scale_factor)
    y1 = int(bbox["y1"] * scale_factor)
    x2 = int(bbox["x2"] * scale_factor)
    y2 = int(bbox["y2"] * scale_factor)

    w, h = image.size
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 > x1 and y2 > y1:
        return image.crop((x1, y1, x2, y2))

    return None


def get_fragment_image_sync(
    fragment: PageFragment, page_number: int, parse_result: ParseResult
) -> Any:
    if not fragment.bbox:
        return None

    # Create a request for just this page to avoid processing full doc
    req = parse_result.request.model_copy()
    req.pages_to_parse = [page_number]

    # Use default scale factor, we will control resolution via ImageDimensions
    spc = SimplePageCreator(scale_factor=1.0)

    # Create a temporary ParseResult
    temp_result = parse_result.model_copy()
    temp_result.request = req

    try:
        # Use 200 DPI for higher resolution, matching table correction process
        generator = spc.get_images_generator(
            temp_result, image_dimensions=ImageDimensions(target_dpi=200)
        )

        for doc_pages in generator:
            if page_number in doc_pages.page_images:
                img = doc_pages.page_images[page_number]

                # Use the scale factor from the generated images
                current_scale = doc_pages.scale_factor

                return _crop_image_from_bbox(img, fragment.bbox, current_scale)
    except Exception as e:
        print(f"Error getting fragment image: {e}", flush=True)

    return None


async def get_fragment_image(
    fragment: PageFragment, page_number: int, parse_result: ParseResult
) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, get_fragment_image_sync, fragment, page_number, parse_result
    )


# =============================================================================
# LLM Interaction
# =============================================================================


async def parse_single_prompt(
    step_key: str = "analysis",
    images: list = None,
    **format_kwargs,
) -> dict[str, Any]:
    """
    Generic prompt parser. Accepts arbitrary keyword arguments used to format
    the selected prompt template. Returns a dict with the step_key mapping
    to the parsed JSON response. Logs all calls for debugging.
    """
    from tensorlake_docai.providers.model_provider_utils import run_clients, _make_gemini_call

    print(
        f"{inspect.currentframe().f_code.co_name}: Calling parse_single_prompt with step_key='{step_key}'",
        flush=True,
    )

    system_prompt, user_template = get_table_merging_prompt_messages(step_key)

    try:
        prompt_text = user_template.format(**format_kwargs)
    except KeyError as e:
        # Fallback to safe formatting: try to inject common names
        print(
            f"{inspect.currentframe().f_code.co_name}: KeyError in template formatting: {e}. Using fallback with common parameter names",
            flush=True,
        )
        prompt_text = user_template.format(
            table_end=format_kwargs.get("table_end", ""),
            table_start=format_kwargs.get("table_start", ""),
            merged_table=format_kwargs.get("merged_table", ""),
            merged_summary=format_kwargs.get("merged_summary", ""),
            next_table=format_kwargs.get("next_table", ""),
            context_between=format_kwargs.get("context_between", ""),
        )

    full_prompt = f"{system_prompt}\n\n{prompt_text}"

    print(f"{inspect.currentframe().f_code.co_name}: Sending request model...", flush=True)

    json_schema = TABLE_MERGING_SCHEMAS.get(step_key)

    try:
        response_text, input_tokens, output_tokens = await run_clients(
            user_prompt=full_prompt,
            images=images or [],
            models=[_make_gemini_call],
            job_type="structured_extraction",
            json_schema=json_schema,
        )
    except Exception as e:
        error_msg = str(e)
        if "length limit was reached" in error_msg or "Model hallucinating spaces" in error_msg:
            print(
                f"{inspect.currentframe().f_code.co_name}: Token limit reached or hallucination detected. Skipping merge.",
                flush=True,
            )
        else:
            print(f"{inspect.currentframe().f_code.co_name}: Error calling model: {e}", flush=True)
        return {step_key: {}, "usage": {"input_tokens": 0, "output_tokens": 0}}

    response = extract_json_from_response(response_text)
    print(
        f"{inspect.currentframe().f_code.co_name}: Successfully parsed response for step_key='{step_key}'",
        flush=True,
    )

    result = {step_key: response}
    result["usage"] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    return result


# =============================================================================
# Same-Page Merging Logic
# =============================================================================


async def _process_single_page_tables(
    page_index: int, page: Page, parse_result: Optional[ParseResult] = None
) -> tuple[int, int, int, list[PageFragment], dict[str, MergedTable]]:
    """
    Process tables on a single page. Returns (merges_count, input_tokens, output_tokens, new_fragments, merged_entries_dict).
    """
    fragments = page.page_fragments or []
    if not fragments:
        return 0, 0, 0, fragments, {}

    new_fragments = []
    merged_entries_dict = {}
    merges = 0
    input_tokens = 0
    output_tokens = 0

    i = 0
    while i < len(fragments):
        frag = fragments[i]

        # If not a table, preserve it
        if frag.fragment_type != PageFragmentType.TABLE:
            new_fragments.append(frag)
            i += 1
            continue

        # Found a table. Check if it should be merged with subsequent tables.
        current_chain_table = frag
        current_chain_ref_ids = [frag.ref_id] if frag.ref_id else []
        i += 1

        if i >= len(fragments):
            new_fragments.append(current_chain_table)
            break

        while i < len(fragments):
            # Collect gap fragments
            gap_frags = []
            j = i
            while j < len(fragments) and fragments[j].fragment_type != PageFragmentType.TABLE:
                gap_frags.append(fragments[j])
                j += 1

            if j >= len(fragments):
                # No more tables to merge with on this page
                new_fragments.append(current_chain_table)
                new_fragments.extend(gap_frags)
                i = len(fragments)
                break

            next_table = fragments[j]

            # Tables in the same column should not be attempted to be merged.
            # Continuous tables in reading order but in different columns can be merged.
            if _is_vertically_aligned(current_chain_table, next_table):
                # Tables are in the same column (vertically aligned), so we treat them as separate.
                # Finalize the current merge chain and start a new one from the next table.
                new_fragments.append(current_chain_table)
                new_fragments.extend(gap_frags)
                i = j
                break

            # Prepare context for LLM
            t1_html = current_chain_table.content.html
            t2_html = next_table.content.html
            gap_text = "\n".join([getattr(f.content, "content", "") for f in gap_frags])

            # Slice tables to reduce token usage
            t1_slice = slice_table_rows(t1_html, start=-20)
            t2_slice = slice_table_rows(t2_html)

            result = {}
            # Call LLM
            try:
                resp = await parse_single_prompt(
                    table1=t1_slice,
                    context_between=gap_text,
                    table2=t2_slice,
                    step_key="fast_same_page_merge",
                )
                result = cast(dict[str, Any], resp.get("fast_same_page_merge", {}))

                step_usage = resp.get("usage", {})
                input_tokens += step_usage.get("input_tokens", 0)
                output_tokens += step_usage.get("output_tokens", 0)

                should_merge = result.get("should_merge") == "YES"
            except Exception as e:
                print(
                    f"{inspect.currentframe().f_code.co_name}: Error checking same-page merge on page {page_index}: {e}",
                    flush=True,
                )
                should_merge = False

            if should_merge:
                print(
                    f"{inspect.currentframe().f_code.co_name}: Merging tables on page {page_index} separated by {len(gap_text)} chars.",
                    flush=True,
                )

                # Perform merge programmatically
                skip_rows = result.get("skip_rows", 0)
                if not isinstance(skip_rows, int):
                    skip_rows = 0

                corrected_html = result.get("corrected_html")
                table_to_merge_html = corrected_html if corrected_html else t2_html

                merged_html = merge_table_htmls(t1_html, table_to_merge_html, skip_rows)

                # Calculate merged bbox
                if next_table.ref_id:
                    current_chain_ref_ids.append(next_table.ref_id)

                merged_bbox = current_chain_table.bbox
                if next_table.bbox:
                    if merged_bbox:
                        merged_bbox = {
                            "x1": min(
                                merged_bbox.get("x1", float("inf")),
                                next_table.bbox.get("x1", float("inf")),
                            ),
                            "y1": min(
                                merged_bbox.get("y1", float("inf")),
                                next_table.bbox.get("y1", float("inf")),
                            ),
                            "x2": max(
                                merged_bbox.get("x2", float("-inf")),
                                next_table.bbox.get("x2", float("-inf")),
                            ),
                            "y2": max(
                                merged_bbox.get("y2", float("-inf")),
                                next_table.bbox.get("y2", float("-inf")),
                            ),
                        }
                    else:
                        merged_bbox = next_table.bbox

                merged_summary = result.get("explanation", "")

                table_content = merged_html
                if (
                    parse_result
                    and getattr(parse_result.request, "table_output_mode", "html") == "markdown"
                ):
                    from markdownify import markdownify

                    table_content = markdownify(merged_html)

                # Create merged fragment
                new_content = Table(
                    content=table_content,
                    cells=[],
                    html=merged_html,
                    summary=merged_summary,
                )
                new_frag = PageFragment(
                    fragment_type=PageFragmentType.TABLE,
                    content=new_content,
                    bbox=merged_bbox,
                    reading_order=current_chain_table.reading_order,
                )

                merge_id = f"same_page_merge_{page.page_number}_{i}"
                new_frag.ref_id = merge_id

                merged_entry = MergedTable(
                    merged_table_id=merge_id,
                    merged_table_html=merged_html,
                    start_page=page.page_number,
                    end_page=page.page_number,
                    pages_merged=1,
                    summary=merged_summary,
                    merge_actions=MergeTableActions(
                        pages=[page.page_number],
                        ref_ids=current_chain_ref_ids.copy(),
                    ),
                )
                merged_entries_dict[merge_id] = merged_entry

                current_chain_table = new_frag
                merges += 1
                i = j + 1
            else:
                # No merge. Commit current table and gaps.
                print(
                    f"{inspect.currentframe().f_code.co_name}: Decided NOT to merge tables on page {page_index}. Reason: {result.get('explanation', 'No explanation or error')}",
                    flush=True,
                )
                new_fragments.append(current_chain_table)
                new_fragments.extend(gap_frags)
                # Next iteration starts at next_table (index j)
                i = j
                break

    return merges, input_tokens, output_tokens, new_fragments, merged_entries_dict


async def _process_same_page_tables_async(
    parsed_document: ParsedDocument,
    parse_result: Optional[ParseResult] = None,
) -> tuple[int, int, list[Page], dict[str, MergedTable]]:
    """
    Async version of process_same_page_tables.
    """
    print(f"{inspect.currentframe().f_code.co_name}: Starting same-page table merge process...")
    pages = parsed_document.pages or []

    tasks = [_process_single_page_tables(i, page, parse_result) for i, page in enumerate(pages)]
    results = await asyncio.gather(*tasks)

    total_merges = sum(r[0] for r in results)
    total_input = sum(r[1] for r in results)
    total_output = sum(r[2] for r in results)

    effective_pages = []
    all_same_page_merges = {}

    for i, (merges, inp, outp, new_frags, merged_entries_dict) in enumerate(results):
        original_page = pages[i]
        # Create a copy of the page with the merged fragments for downstream processing
        effective_page = original_page.model_copy(update={"page_fragments": new_frags})
        effective_pages.append(effective_page)
        all_same_page_merges.update(merged_entries_dict)

    # Note: Token usage is logged but not currently persisted back to ParseResult.usage
    # as ParsedDocument doesn't hold usage info.
    print(
        f"{inspect.currentframe().f_code.co_name}: Same-page merge tokens: Input={total_input}, Output={total_output}",
        flush=True,
    )

    print(
        f"{inspect.currentframe().f_code.co_name}: Same-page merge completed. Total merges: {total_merges}",
        flush=True,
    )
    return total_input, total_output, effective_pages, all_same_page_merges


def process_same_page_tables(
    parsed_document: ParsedDocument,
) -> tuple[int, int, list[Page], dict[str, MergedTable]]:
    """
    Detect and merge fragment tables within each page that are separated by
    non-table content but belong to the same logical table.
    """
    return asyncio.run(_process_same_page_tables_async(parsed_document))


# =============================================================================
# Cross-Page Merging Logic
# =============================================================================
async def _resolve_table_alignment(
    base_html: str,
    target_html: str,
    frag1: PageFragment,
    page1_num: int,
    frag2: PageFragment,
    page2_num: int,
    parse_result: ParseResult | None,
) -> tuple[str, int, int, bool]:
    """
    Checks alignment between base_html and target_html.
    If misaligned, attempts visual alignment using LLM.
    Returns (final_target_html, input_tokens, output_tokens).
    """
    input_tokens = 0
    output_tokens = 0
    was_corrected = False

    if not parse_result or parse_result.request.ocr_model != "dots-ocr":
        return target_html, input_tokens, output_tokens, was_corrected

    try:
        cols1 = get_table_column_count(base_html)
        cols2 = get_table_column_count(target_html)

        misaligned = False
        if cols1 > 0 and cols2 > 0:
            if cols1 != cols2:
                misaligned = True
            elif not are_tables_semantically_aligned(base_html, target_html):
                print(
                    "Semantic mismatch detected between tables. Attempting visual alignment.",
                    flush=True,
                )
                misaligned = True

        if misaligned:
            print(
                f"Table alignment issue detected (Cols: {cols1} vs {cols2}). Attempting visual alignment.",
                flush=True,
            )

            img1 = await get_fragment_image(frag1, page1_num, parse_result)
            img2 = await get_fragment_image(frag2, page2_num, parse_result)

            if img1 and img2:
                align_resp = await parse_single_prompt(
                    table1_html=slice_table_rows(base_html, start=-10),
                    table2_html=target_html,
                    step_key="align_tables",
                    images=[img1, img2],
                )

                usage = align_resp.get("usage", {})
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)

                align_result = cast(dict[str, Any], align_resp.get("align_tables", {}))
                if align_result.get("corrected_html"):
                    print(
                        f"Visual alignment successful: {align_result.get('explanation')}",
                        flush=True,
                    )
                    target_html = align_result["corrected_html"]
                    was_corrected = True

    except Exception as e:
        print(
            f"Error resolving table alignment, the tables will be merged using the original HTML: {e}",
            flush=True,
        )

    return target_html, input_tokens, output_tokens, was_corrected


async def _check_merge_continuation(
    base_html: str,
    next_table_html: str,
    context_between: str,
) -> tuple[bool, dict[str, Any], int, int]:
    """
    Helper to check if two tables should be merged using the fast_merge prompt.
    Returns (should_merge, result_dict, input_tokens, output_tokens).
    """
    t1_slice = slice_table_rows(base_html, start=-20)
    t2_slice = slice_table_rows(next_table_html)

    resp = await parse_single_prompt(
        table_end=t1_slice,
        table_start=t2_slice,
        context_between=context_between,
        step_key="fast_merge",
    )
    result = cast(dict[str, Any], resp.get("fast_merge", {}))

    usage = resp.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    print(
        f"Fast merge explanation: {result.get('continuation')} / {result.get('explanation')}",
        flush=True,
    )

    should_merge = result.get("continuation") == "YES"
    return should_merge, result, input_tokens, output_tokens


async def _merge_next_table(
    base_html: str,
    next_table: PageFragment,
    pages: list[Page],
    context_page_index: int,
    prev_frag: PageFragment,
    prev_page_num: int,
    next_page_num: int,
    parse_result: ParseResult | None,
    current_merge_ref_ids: list[str],
    current_chain_consumed_merges: set[str],
) -> tuple[bool, str, int, int]:

    context_between = get_context_between(pages, context_page_index, prev_frag, next_table)

    should_merge, result, input_tokens, output_tokens = await _check_merge_continuation(
        base_html, next_table.content.html, context_between
    )

    if not should_merge:
        return False, base_html, input_tokens, output_tokens

    skip_rows = result.get("skip_rows", 0)
    if not isinstance(skip_rows, int):
        skip_rows = 0

    table_to_merge_html = next_table.content.html

    table_to_merge_html, align_in, align_out, was_corrected = await _resolve_table_alignment(
        base_html,
        table_to_merge_html,
        prev_frag,
        prev_page_num,
        next_table,
        next_page_num,
        parse_result,
    )

    if was_corrected:
        skip_rows = 0

    input_tokens += align_in
    output_tokens += align_out

    merged_html = merge_table_htmls(base_html, table_to_merge_html, skip_rows)

    if merged_html == base_html:
        return False, base_html, input_tokens, output_tokens

    if next_table.ref_id:
        current_merge_ref_ids.append(next_table.ref_id)
        if next_table.ref_id.startswith("same_page_merge_"):
            current_chain_consumed_merges.add(next_table.ref_id)

    return True, merged_html, input_tokens, output_tokens


async def _process_cross_page_tables_async(
    parsed_document: ParsedDocument,
    max_pages_before_summary: int = 2,
    max_total_merge_pages: int = -1,
    pages: list[Page] | None = None,
    parse_result: ParseResult | None = None,
) -> tuple[int, int, dict[str, MergedTable], set[str]]:

    pages = pages or parsed_document.pages or []
    if not pages:
        return 0, 0, {}, set()

    cross_page_merges: dict[str, MergedTable] = {}
    consumed_same_page_merges: set[str] = set()
    consumed_pages: set[int] = set()  # Page numbers consumed by a cross-page merge
    consumed_ref_ids: set[str] = set()  # Ref IDs of fragments consumed by a cross-page merge
    page_index = 0
    search_depth = 2

    total_input_tokens = 0
    total_output_tokens = 0

    while page_index < len(pages) - 1:
        # Skip pages already consumed by a previous merge
        if page_index in consumed_pages:
            page_index += 1
            continue

        # Get candidates for cross-page merging
        candidates = get_cross_page_table_candidates(pages, page_index)

        # Filter out candidates that have already been merged
        candidates = [
            (t1, t2)
            for t1, t2 in candidates
            if (not t1.ref_id or t1.ref_id not in consumed_ref_ids)
            and (not t2.ref_id or t2.ref_id not in consumed_ref_ids)
        ]

        if not candidates:
            page_index += 1
            continue

        chain_started = False
        merged_html = None
        current_merge_ref_ids = []
        current_chain_consumed_merges = set()
        pages_merged = 1
        summary_text = ""
        last_merged_frag = None
        last_merged_page = -1

        # Try candidates for the FIRST merge step
        for table_end, table_start in candidates:
            temp_ref_ids = []
            temp_consumed = set()

            success, merged_html, in_tok, out_tok = await _merge_next_table(
                table_end.content.html,
                table_start,
                pages,
                page_index,
                table_end,
                pages[page_index].page_number,
                pages[page_index + 1].page_number,
                parse_result,
                temp_ref_ids,
                temp_consumed,
            )

            total_input_tokens += in_tok
            total_output_tokens += out_tok

            if success:
                chain_started = True
                current_merge_ref_ids = temp_ref_ids
                current_chain_consumed_merges = temp_consumed

                # Add start table ref_id (table_end)
                if table_end.ref_id:
                    current_merge_ref_ids.insert(0, table_end.ref_id)
                    if table_end.ref_id.startswith("same_page_merge_"):
                        current_chain_consumed_merges.add(table_end.ref_id)

                last_merged_frag = table_start
                last_merged_page = pages[page_index + 1].page_number
                pages_merged = 2
                print(f"Merge started at page {page_index} with candidate pair.", flush=True)
                break

        if not chain_started:
            page_index += 1
            continue

        # Extend chain loop
        current_index = page_index + 1
        last_page_has_remaining_tables = False

        while current_index < len(pages) - 1:
            if max_total_merge_pages != -1 and pages_merged > max_total_merge_pages:
                break

            # Check candidates from previous page (current_index)
            prev_page_frags = pages[current_index].page_fragments or []
            last_frag_idx = -1
            for idx, frag in enumerate(prev_page_frags):
                if frag is last_merged_frag:
                    last_frag_idx = idx
                    break

            should_yield = False
            if last_frag_idx != -1:
                tables_after = [
                    f
                    for f in prev_page_frags[last_frag_idx + 1 :]
                    if f.fragment_type == PageFragmentType.TABLE
                ]
                if tables_after:
                    candidate_unmerged = tables_after[-1]
                    if (
                        not candidate_unmerged.ref_id
                        or candidate_unmerged.ref_id not in consumed_ref_ids
                    ):
                        next_page_candidates_check = get_tables_from_page_start(
                            pages[current_index + 1], limit=search_depth
                        )
                        for next_table in next_page_candidates_check:
                            context = get_context_between(
                                pages, current_index, candidate_unmerged, next_table
                            )
                            should_merge, _, in_tok, out_tok = await _check_merge_continuation(
                                candidate_unmerged.content.html, next_table.content.html, context
                            )
                            total_input_tokens += in_tok
                            total_output_tokens += out_tok
                            if should_merge:
                                should_yield = True
                                break

            if should_yield:
                last_page_has_remaining_tables = True
                break

            next_page_candidates = get_tables_from_page_start(
                pages[current_index + 1], limit=search_depth
            )
            extended = False

            for next_table in next_page_candidates:
                success, new_merged_html, in_tok, out_tok = await _merge_next_table(
                    merged_html,
                    next_table,
                    pages,
                    current_index,
                    last_merged_frag,
                    last_merged_page,
                    pages[current_index + 1].page_number,
                    parse_result,
                    current_merge_ref_ids,
                    current_chain_consumed_merges,
                )

                total_input_tokens += in_tok
                total_output_tokens += out_tok

                if success:
                    merged_html = new_merged_html

                    last_merged_frag = next_table
                    last_merged_page = pages[current_index + 1].page_number
                    pages_merged += 1
                    extended = True

                    # Summary generation
                    if pages_merged >= max_pages_before_summary:
                        summary_resp = await parse_single_prompt(
                            merged_table=merged_html,
                            step_key="merged_summary",
                        )
                        summary_obj = cast(dict[str, Any], summary_resp.get("merged_summary", {}))
                        usage = summary_resp.get("usage", {})
                        total_input_tokens += usage.get("input_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)
                        summary_text = cast(Optional[str], summary_obj.get("summary")) or ""

                    break

            if not extended:
                break
            current_index += 1

        # Record metadata for this merged table if any merging happened
        if pages_merged > 1 and merged_html:
            print(
                f"{inspect.currentframe().f_code.co_name}: Recording merged table entry for pages {page_index} to {page_index + pages_merged - 1}",  # noqa E501
                flush=True,
            )

            consumed_same_page_merges.update(current_chain_consumed_merges)

            start_page_idx = page_index
            end_page_idx = page_index + pages_merged - 1

            start_page_num = pages[start_page_idx].page_number
            end_page_num = pages[end_page_idx].page_number

            merge_id = f"cross_page_merge_{start_page_num}_{end_page_num}"

            page_nums = [pages[i].page_number for i in range(start_page_idx, end_page_idx + 1)]

            merged_entry = MergedTable(
                merged_table_id=merge_id,
                merged_table_html=merged_html,
                start_page=start_page_num,
                end_page=end_page_num,
                pages_merged=pages_merged,
                summary=summary_text,
                merge_actions=MergeTableActions(
                    pages=page_nums,
                    ref_ids=current_merge_ref_ids,
                    target_columns=None,
                ),
            )

            cross_page_merges[merge_id] = merged_entry
            consumed_ref_ids.update(current_merge_ref_ids)

            # Mark these pages as consumed so we skip them in the outer loop
            consumed_range_end = end_page_idx + 1
            if last_page_has_remaining_tables:
                consumed_range_end -= 1
            consumed_pages.update(range(start_page_idx, consumed_range_end))

        # Advance to the next unconsumed page
        if pages_merged > 1 and merged_html:
            if last_page_has_remaining_tables:
                page_index = page_index + pages_merged - 1
            else:
                page_index = page_index + pages_merged
        else:
            page_index += 1

    print(
        f"{inspect.currentframe().f_code.co_name}: Cross-page merge tokens: Input={total_input_tokens}, Output={total_output_tokens}",
        flush=True,
    )
    return (
        total_input_tokens,
        total_output_tokens,
        cross_page_merges,
        consumed_same_page_merges,
    )


def _preprocess_table_correction(table_html: str) -> str:
    """
    Preprocess table HTML to wrap orphaned text outside of <tr> tags into new rows.
    This helps table correction models to better understand the table structure.
    """

    # Calculate colspan from base_html for wrapping orphaned text
    colspan = 0
    base_rows = re.findall(r"<tr[^>]*>.*?</tr>", table_html, re.IGNORECASE | re.DOTALL)

    if base_rows:
        sample_row = base_rows[-1]
        # Find all opening cell tags (td or th)
        cells = re.findall(r"<(td|th)([^>]*)>", sample_row, re.IGNORECASE)

        for _, attributes in cells:
            # Search for colspan="n" within the attributes of the tag
            match = re.search(r'colspan\s*=\s*["\']?(\d+)["\']?', attributes, re.IGNORECASE)
            if match:
                colspan += int(match.group(1))
            else:
                colspan += 1

    # Default back to 1 if no cells were found to avoid a 0-width result
    colspan = max(colspan, 1)

    clearning_tag_pattern = r"<\/?(div|table|thead|tbody|tfoot)[^>]*>"

    # Check for orphaned text at the beginning of next_content
    # This happens if there was text inside the div but outside the table in next_html
    match_row = re.search(r"<tr[^>]*>", table_html, re.IGNORECASE)
    if match_row:
        pre_text = table_html[: match_row.start()].strip()
        if pre_text:
            pre_text = re.sub(clearning_tag_pattern, "", pre_text, flags=re.IGNORECASE)
            wrapped_text = (
                f'<tr><td colspan="{colspan}">{pre_text}</td></tr>'
                if pre_text and pre_text != "\n"
                else ""
            )
            table_html = "<table>" + wrapped_text + "\n" + table_html[match_row.start() :]

    # Check for orphaned text at the end of next_content
    matches = list(re.finditer(r"</tr>", table_html, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        post_text = table_html[last_match.end() :].strip()
        if post_text:
            post_text = re.sub(clearning_tag_pattern, "", post_text, flags=re.IGNORECASE)
            wrapped_text = (
                f'<tr><td colspan="{colspan}">{post_text}</td></tr>'
                if post_text and post_text != "\n"
                else ""
            )
            table_html = table_html[: last_match.end()] + "\n" + wrapped_text + "</table>"

    return table_html


async def _process_table_correction_async(
    parsed_document: ParsedDocument, parse_result: ParseResult
) -> tuple[int, int]:
    from markdownify import markdownify
    from tensorlake_docai.tables.table_correction import run_table_correction_process

    if not parse_result:
        return 0, 0

    table_output_mode = getattr(parse_result.request, "table_output_mode", "html")

    # Identify pages that need correction to avoid processing all pages
    pages_to_process = set()
    for page in parsed_document.pages:
        for frag in page.page_fragments:
            if (
                frag.fragment_type == PageFragmentType.TABLE
                and isinstance(frag.content, Table)
                and frag.content.html
                and not frag.content.table_checked
            ):
                pages_to_process.add(page.page_number)

    if not pages_to_process:
        return 0, 0

    # Create a request copy with only the pages that need correction
    req_copy = parse_result.request.model_copy()
    req_copy.pages_to_parse = sorted(list(pages_to_process))
    parse_result_subset = parse_result.model_copy()
    parse_result_subset.request = req_copy

    scale_factor = 1.0
    if parse_result.document_layout:
        scale_factor = parse_result.document_layout.scale_factor

    spc = SimplePageCreator(scale_factor=scale_factor)
    total_inp = 0
    total_outp = 0

    pages_by_num = {p.page_number: p for p in parsed_document.pages}

    async def process_batch(doc_pages):
        current_scale = doc_pages.scale_factor
        batch_tasks = []
        batch_fragments_to_update = []

        for page_num, image in doc_pages.page_images.items():
            if page_num in pages_by_num:
                page = pages_by_num[page_num]

                for frag in page.page_fragments:
                    if (
                        frag.fragment_type == PageFragmentType.TABLE
                        and isinstance(frag.content, Table)
                        and frag.content.html
                        and not frag.content.table_checked  # Do not re-process already checked tables
                    ):
                        if not frag.bbox:
                            continue

                        cropped = _crop_image_from_bbox(image, frag.bbox, current_scale)

                        if frag.content.html and frag.content.html.startswith("<div>"):
                            revised_content = _preprocess_table_correction(frag.content.html)
                            frag.content.html = revised_content
                            frag.content.markdown = markdownify(revised_content)
                            if table_output_mode == "markdown":
                                frag.content.content = frag.content.markdown
                            else:
                                frag.content.content = revised_content
                            frag.content.table_checked = True
                        else:
                            revised_content = frag.content.html

                        if cropped:
                            try:
                                batch_tasks.append(
                                    run_table_correction_process(revised_content, cropped)
                                )
                                batch_fragments_to_update.append(frag)
                            except Exception as e:
                                print(
                                    f"Error processing table correction for fragment {frag.ref_id}: {e}, skipping it.",
                                    flush=True,
                                )

        b_inp = 0
        b_outp = 0
        if batch_tasks:
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for frag, res in zip(batch_fragments_to_update, results):
                if isinstance(res, Exception):
                    print(f"Table correction failed for fragment {frag.ref_id}: {res}", flush=True)
                else:
                    res_data, inp, outp = res
                    b_inp += inp
                    b_outp += outp

                    if res_data and "corrected_html" in res_data and res_data["corrected_html"]:
                        corrected_html = res_data["corrected_html"]
                        frag.content.html = corrected_html
                        frag.content.markdown = markdownify(corrected_html)

                        print(f"Table correction processed fragment {frag.ref_id}", flush=True)
                    else:
                        print(
                            f"Table correction returned empty result for fragment {frag.ref_id}",
                            flush=True,
                        )

                if table_output_mode == "markdown":
                    frag.content.content = frag.content.markdown
                else:
                    frag.content.content = frag.content.html

                frag.content.table_checked = True  # Prevent re-processing

        return b_inp, b_outp

    tasks = [
        process_batch(doc_pages)
        for doc_pages in spc.get_images_generator(
            parse_result_subset, image_dimensions=ImageDimensions(target_dpi=200)
        )
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            print(f"Batch processing error: {res}", flush=True)
        else:
            inp, outp = res
            total_inp += inp
            total_outp += outp

    print(
        f"{inspect.currentframe().f_code.co_name}: Table correction completed. Total input tokens: {total_inp}, output tokens: {total_outp}",
        flush=True,
    )

    return total_inp, total_outp


async def _orchestrate_table_merging_async(
    parsed_document: ParsedDocument,
    parse_result: Optional[ParseResult] = None,
    ctx: Optional[RequestContext] = None,
) -> tuple[int, int, list[MergedTable]]:
    total_input = 0
    total_output = 0

    # Estimation of work
    tables_to_correct = 0
    same_page_checks = 0
    cross_page_checks = 0

    prev_page_had_table = False

    for page in parsed_document.pages:
        tables_on_current_page = 0
        current_page_has_table = False

        for frag in page.page_fragments:
            is_potential_table = False

            if frag.fragment_type == PageFragmentType.TABLE:
                is_potential_table = True
                if (
                    parse_result
                    and isinstance(frag.content, Table)
                    and frag.content.html
                    and not getattr(frag.content, "table_checked", False)
                ):
                    tables_to_correct += 1
            elif (
                parse_result
                and parse_result.request.ocr_model
                == "dots-ocr"  # gated behind the model check — the body is expensive
                and frag.fragment_type == PageFragmentType.TEXT
                and isinstance(frag.content, Text)
                and frag.content.content
                and is_markdown_table(frag.content.content)
            ):
                is_potential_table = True
                print(f"Fixing table on page {page.page_number} identified as text", flush=True)
                html_content = markdown_to_html_table(frag.content.content)
                frag.fragment_type = PageFragmentType.TABLE

                table_content = html_content
                if (
                    parse_result
                    and getattr(parse_result.request, "table_output_mode", "html") == "markdown"
                ):
                    table_content = frag.content.content

                frag.content = Table(
                    content=table_content,
                    cells=[],
                    html=html_content,
                    summary="",
                    markdown=frag.content.content,
                    table_checked=False,
                )

                if parse_result:
                    tables_to_correct += 1

            if is_potential_table:
                tables_on_current_page += 1
                current_page_has_table = True

        if tables_on_current_page > 1:
            same_page_checks += tables_on_current_page - 1

        if prev_page_had_table and current_page_has_table:
            cross_page_checks += 1

        prev_page_had_table = current_page_has_table

    est_seconds = (tables_to_correct * 5) + ((same_page_checks + cross_page_checks) * 3)
    msg = f"Table Processing Estimate: {tables_to_correct} corrections, {same_page_checks} same-page checks, {cross_page_checks} cross-page checks. Approx time: {est_seconds}s"
    print(f"{inspect.currentframe().f_code.co_name}: {msg}", flush=True)
    if ctx:
        ctx.progress.update(current=0, total=100, message=msg)

    # 0. Table correction
    if parse_result:
        inp, outp = await _process_table_correction_async(parsed_document, parse_result)
        total_input += inp
        total_output += outp

    # 1. Process same-page merges
    inp, outp, effective_pages, same_page_merges = await _process_same_page_tables_async(
        parsed_document, parse_result
    )
    total_input += inp
    total_output += outp

    # 2. Process cross-page merges using the result of same-page merges
    inp, outp, cross_page_merges, consumed_ids = await _process_cross_page_tables_async(
        parsed_document, pages=effective_pages, max_total_merge_pages=-1, parse_result=parse_result
    )
    total_input += inp
    total_output += outp

    # 3. Reconcile merges: Filter out same-page merges that were consumed by cross-page merges
    final_same_page_merges = {
        mid: merge for mid, merge in same_page_merges.items() if mid not in consumed_ids
    }

    all_merged_tables = list(final_same_page_merges.values()) + list(cross_page_merges.values())

    all_merged_tables.sort(key=lambda x: (x.start_page, x.end_page))

    return total_input, total_output, all_merged_tables


def orchestrate_table_merging(
    parsed_document: ParsedDocument,
    parse_result: Optional[ParseResult] = None,
    ctx: Optional[RequestContext] = None,
) -> tuple[int, int, list[MergedTable]]:
    """
    Orchestrate the full table merging process:
    0. Table correction (if parse_result is provided)
    1. Same-page merging
    2. Cross-page merging
    3. Reconciliation

    Returns:
        tuple of (input_tokens, output_tokens, merged_tables_list)
    """
    return asyncio.run(_orchestrate_table_merging_async(parsed_document, parse_result, ctx))


@cls()
class TableMerging:
    @function(
        image=table_merging_image,
        timeout=30 * 60,  # 30 minutes
        cpu=2,
        memory=8,
        # output_encoder = "json"
        # The function is not using /tmp disk space, just reserve a small amount
        ephemeral_disk=2,
        secrets=SECRETS,
        min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
    )
    def run(self, result: ParseResult) -> ParsedDocumentRef:
        print(
            f"{inspect.currentframe().f_code.co_name}: Running table merging run method ...",
            flush=True,
        )
        ctx = RequestContext.get()

        try:
            # Create a temporary ParsedDocument for table merging
            pages = document_layout_to_document(
                result.document_layout.pages,
                result.document_layout.scale_factor,
                result.request.ignore_sections,
            )

            # We need a ParsedDocument to pass to merging functions
            temp_doc = ParsedDocument(pages=pages, chunks=[])

            inp, outp, merged_tables = orchestrate_table_merging(temp_doc, result, ctx=ctx)

            # Update usage
            if result.usage:
                result.usage.ocr_input_tokens_used = (result.usage.ocr_input_tokens_used or 0) + inp
                result.usage.ocr_output_tokens_used = (
                    result.usage.ocr_output_tokens_used or 0
                ) + outp

            # Store in result.document_layout.merged_tables
            if result.document_layout:
                result.document_layout.merged_tables = merged_tables

            # Propagate corrected tables back to document_layout
            for page in temp_doc.pages:
                # Find corresponding page layout
                page_layout = next(
                    (p for p in result.document_layout.pages if p.page_number == page.page_number),
                    None,
                )
                if not page_layout:
                    continue

                for frag in page.page_fragments:
                    if isinstance(frag.content, Table):
                        # Only update if it was checked/corrected.
                        if getattr(frag.content, "table_checked", False):
                            # Find corresponding element in page_layout
                            element = next(
                                (
                                    e
                                    for e in page_layout.elements
                                    if e.reading_order == frag.reading_order
                                ),
                                None,
                            )
                            if element:
                                element.fragment_type = (
                                    PageFragmentType.TABLE
                                )  # Ensure that it is a table in the page layout
                                # Update element with corrected content
                                if frag.content.html:
                                    element.html = frag.content.html
                                if frag.content.markdown:
                                    element.markdown = frag.content.markdown
                                if frag.content.content:
                                    element.ocr_text = frag.content.content

            print(
                f"{inspect.currentframe().f_code.co_name}: Table merging run function complete. Found {len(merged_tables)} merged tables.",
                flush=True,
            )

        except Exception as e:
            raise RequestException(
                f"{inspect.currentframe().f_code.co_name}: Error running table merging run function: {e}"
            )

        # Node-by-node routing decisions
        if ocr_should_go_to_output_formatter(result.request):
            print("🔀 TableMerging → OutputFormatter")
            return format_final_output(result)
        elif ocr_should_go_to_structured_extraction(result.request, result):
            print("🔀 TableMerging → StructuredExtraction")
            return StructuredExtraction().run.future(result)
        # Special handling for VLM extraction to support dots-ocr
        elif ocr_should_go_to_vlm_extraction(result.request, result) or (
            result.request.ocr_model == "dots-ocr"
            and dots_ocr_should_go_to_vlm_extraction(result.request, result)
        ):
            print("🔀 TableMerging → VLMExtractionTask")
            return VLMExtractionTask().run.future(result)
        else:
            print("🔀 TableMerging → OutputFormatter")
            return format_final_output(result)
