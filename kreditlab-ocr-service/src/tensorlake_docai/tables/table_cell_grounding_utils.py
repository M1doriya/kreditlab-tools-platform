# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from typing import List, Optional
from PIL import Image

from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.models.layout_objects import PageLayoutElement, TextBoundingBox
from tensorlake_docai.providers.model_provider_utils import run_clients
from tensorlake_docai.tables.table_correction import run_table_correction_process
from tensorlake_docai.pipeline.routing import is_markdown_table, markdown_to_html_table
from tensorlake_docai.models.intermediate_objects import ParseResult

TABLE_GROUNDING_PROMPT = """
Identify the bounding box for each table cell in the image.
The cells are defined using the td and th tags in the following HTML snippet by their `ref_id` attribute.
The bounding box (bbox) should be returned as an array of integers in the format [ymin, xmin, ymax, xmax] normalized to the dimensions of the image (0-{x_coordinates} for x and 0-{y_coordinates} for y).
The bounding box should tightly enclose the content of the cell related to the `ref_id`, but should not include table borders or extra padding.
The bounding boxes will be used to ground the text content of each cell in the original document image, so accuracy is important, the bboxes need to be aligned to the cell content as much as possible.

To identify the bounding boxes for each cell, identify first the four corners of the table first, then interpolate the internal cells.

HTML:
{html_content}

Return a JSON object with a "cells" dictionary where keys are `ref_id` and values are `bbox_2d` [ymin, xmin, ymax, xmax].
"""

GROUNDING_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "cells": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "[ymin, xmin, ymax, xmax]",
                },
            }
        },
        "required": ["cells"],
    }
)


async def _make_gemini_call_fast(*args, **kwargs):
    from tensorlake_docai.providers.model_provider_utils import _make_gemini_call
    from google.genai import types

    return await _make_gemini_call(
        *args,
        **kwargs,
        model_name="gemini-3-flash-preview",
        config_overrides={
            "temperature": 0.0,
            "max_output_tokens": 65536,
            "thinking_config": {"include_thoughts": False, "thinking_level": "low"},
            "media_resolution": types.MediaResolution.MEDIA_RESOLUTION_LOW,
        },
    )


async def run_element_table_cell_grounding_and_modify_page_elements(
    cropped_images: List[Image.Image],
    page_elements: List[PageLayoutElement],
    element_types: Optional[List[PageFragmentType]] = None,
    element_types_to_check: Optional[List[PageFragmentType]] = None,
    parse_result: Optional[ParseResult] = None,
) -> tuple[int, int]:
    """
    Run table extraction for specified element types and modify page elements in place.

    Returns:
        tuple[int, int]: (input_tokens, output_tokens) used for extraction
    """
    table_output_mode = "html"
    if parse_result and parse_result.request:
        table_output_mode = getattr(parse_result.request, "table_output_mode", "html")

    from markdownify import markdownify as md
    from bs4 import BeautifulSoup

    if element_types is None:
        element_types = [PageFragmentType.TABLE]

    total_input_tokens = 0
    total_output_tokens = 0

    tasks = []
    elements = []
    images = []

    for cropped_image, page_element in zip(cropped_images, page_elements):
        if (
            page_element.fragment_type == PageFragmentType.TEXT
            and page_element.ocr_text
            and is_markdown_table(page_element.ocr_text)
        ):
            print(f"Fixing table identified as text: {page_element.ref_id}", flush=True)
            html_content = markdown_to_html_table(page_element.ocr_text)
            page_element.fragment_type = PageFragmentType.TABLE
            page_element.html = html_content
            if not page_element.markdown:
                page_element.markdown = page_element.ocr_text

        if page_element.fragment_type in PageFragmentType.TABLE:
            tasks.append(
                run_table_correction_process(
                    page_element.html,
                    cropped_image,
                )
            )
            elements.append(page_element)
            images.append(cropped_image)

    if not tasks:
        print("No tables detected to be processed.", flush=True)
        return total_input_tokens, total_output_tokens

    results = await asyncio.gather(*tasks, return_exceptions=True)

    grounding_tasks = []
    grounding_elements = []
    grounding_images = []

    for page_element, result, cropped_image in zip(elements, results, images):
        html_content = None

        if isinstance(result, Exception):
            print(f"Table cell grounding failed for element {page_element.ref_id}: {result}")
            html_content = page_element.html
        else:
            correction_result, corr_in, corr_out = result
            total_input_tokens += corr_in
            total_output_tokens += corr_out
            if correction_result and "corrected_html" in correction_result:
                html_content = correction_result["corrected_html"]
            else:
                html_content = page_element.html

        if html_content:
            print(
                f"Table cell grounding preparation for element {page_element.ref_id}.", flush=True
            )

            try:
                soup = BeautifulSoup(html_content, "html.parser")
                rows = soup.find_all("tr")

                base_ref_id = page_element.ref_id or "unknown"
                cell_counter = 0

                for row in rows:
                    cells = row.find_all(["td", "th"])
                    for cell in cells:
                        ref_id = f"{base_ref_id}.{cell_counter}"
                        cell["ref_id"] = ref_id
                        cell_counter += 1

                html_with_refs = str(soup)

                # Update element with corrected HTML (now with ref_ids)
                page_element.html = html_with_refs
                page_element.markdown = md(html_with_refs)

                w_attr = 1000
                h_attr = 1000

                # Prepare grounding task
                prompt = TABLE_GROUNDING_PROMPT.format(
                    x_coordinates=w_attr, y_coordinates=h_attr, html_content=html_with_refs
                )
                grounding_tasks.append(
                    run_clients(
                        user_prompt=prompt,
                        images=[cropped_image],
                        models=[_make_gemini_call_fast],
                        job_type="structured_extraction",
                        json_schema=GROUNDING_SCHEMA,
                        timeout=600,
                    )
                )
                grounding_elements.append(page_element)
                grounding_images.append(cropped_image)

            except Exception as e:
                print(f"Error processing table HTML for grounding preparation: {e}")

        if table_output_mode == "markdown":
            page_element.ocr_text = page_element.markdown
        else:
            page_element.ocr_text = page_element.html

    if not grounding_tasks:
        return total_input_tokens, total_output_tokens

    grounding_results = await asyncio.gather(*grounding_tasks, return_exceptions=True)

    for page_element, result, cropped_image in zip(
        grounding_elements, grounding_results, grounding_images
    ):
        print(f"Table cell grounding result for element {page_element.ref_id}.", flush=True)

        if isinstance(result, Exception):
            print(f"Table grounding failed for element {page_element.ref_id}: {result}")
            continue

        response_text, g_in, g_out = result
        total_input_tokens += g_in
        total_output_tokens += g_out

        w_attr = 1000
        h_attr = 1000

        try:
            data = json.loads(response_text)
            cells_data = data.get("cells", {})
            ref_bbox_map = cells_data

            if not ref_bbox_map:
                print(
                    f"Warning: No bounding boxes returned for table {page_element.ref_id}",
                    flush=True,
                )

            # Re-parse HTML to populate text_bounding_boxes
            soup = BeautifulSoup(page_element.html, "html.parser")
            rows = soup.find_all("tr")

            bboxes = []

            px1, py1, px2, py2 = page_element.bbox
            pw = px2 - px1
            ph = py2 - py1

            missing_refs = 0
            total_cells = 0

            for r_idx, row in enumerate(rows):
                cells = row.find_all(["td", "th"])
                for c_idx, cell in enumerate(cells):
                    total_cells += 1
                    ref_id = cell.get("ref_id")
                    if ref_id:
                        if ref_id in ref_bbox_map:
                            bbox_2d = ref_bbox_map[ref_id]
                            ymin, xmin, ymax, xmax = bbox_2d

                            abs_x1 = (xmin / w_attr) * pw + px1
                            abs_y1 = (ymin / h_attr) * ph + py1
                            abs_x2 = (xmax / w_attr) * pw + px1
                            abs_y2 = (ymax / h_attr) * ph + py1

                            bbox_obj = TextBoundingBox(
                                bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                                text=cell.get_text(strip=True),
                                ref_id=ref_id,
                                row_index=r_idx,
                                column_index=c_idx,
                            )
                            bboxes.append(bbox_obj)
                        else:
                            missing_refs += 1

            if missing_refs > 0:
                print(
                    f"Warning: {missing_refs}/{total_cells} cells missing bounding boxes for table {page_element.ref_id}",
                    flush=True,
                )

            page_element.text_bounding_boxes = bboxes
            page_element.table_checked = True

        except Exception as e:
            print(f"Error processing grounding result: {e}")

    return total_input_tokens, total_output_tokens
