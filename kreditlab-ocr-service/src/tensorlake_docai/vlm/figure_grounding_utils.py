# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from typing import List, Optional
from PIL import Image

from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.models.layout_objects import PageLayoutElement, TextBoundingBox
from tensorlake_docai.providers.model_provider_utils import run_clients, _make_gemini_call
from tensorlake_docai.models.intermediate_objects import ParseResult

FIGURE_GROUNDING_PROMPT = """
Extract all lines of text from the image.
For each line, provide its content and its bounding box (bbox).
The bounding box (bbox) should be returned as an array of integers in the format [ymin, xmin, ymax, xmax] normalized to the dimensions of the image (0-1000 for both x and y).
The bounding box should tightly enclose the text line.

Return a JSON object with a "lines" array. Each object in the array should have "text" and "bbox_2d" [ymin, xmin, ymax, xmax].
"""

GROUNDING_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "bbox_2d": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "[ymin, xmin, ymax, xmax]",
                        },
                    },
                    "required": ["text", "bbox_2d"],
                },
            }
        },
        "required": ["lines"],
    }
)


async def _make_gemini_call_fast(*args, **kwargs):
    from google.genai import types

    return await _make_gemini_call(
        *args,
        **kwargs,
        model_name="gemini-3-flash-preview",
        config_overrides={
            "temperature": 0.0,
            "max_output_tokens": 8192,
            "media_resolution": types.MediaResolution.MEDIA_RESOLUTION_LOW,
        },
    )


async def run_element_figure_grounding_and_modify_page_elements(
    cropped_images: List[Image.Image],
    page_elements: List[PageLayoutElement],
    element_types: Optional[List[PageFragmentType]] = None,
    element_types_to_check: Optional[List[PageFragmentType]] = None,
    parse_result: Optional[ParseResult] = None,
) -> tuple[int, int]:
    """
    Run figure grounding for specified element types and modify page elements in place.

    Returns:
        tuple[int, int]: (input_tokens, output_tokens) used for grounding
    """
    if element_types is None:
        element_types = [PageFragmentType.FIGURE]

    total_input_tokens = 0
    total_output_tokens = 0

    tasks = []
    elements_to_process = []

    for cropped_image, page_element in zip(cropped_images, page_elements):
        if page_element.fragment_type in element_types:
            prompt = FIGURE_GROUNDING_PROMPT
            tasks.append(
                run_clients(
                    user_prompt=prompt,
                    images=[cropped_image],
                    models=[_make_gemini_call_fast],
                    job_type="structured_extraction",
                    json_schema=GROUNDING_SCHEMA,
                    timeout=600,
                )
            )
            elements_to_process.append(page_element)

    if not tasks:
        return 0, 0

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for page_element, result in zip(elements_to_process, results):
        if isinstance(result, Exception):
            print(f"Figure grounding failed for element {page_element.ref_id}: {result}")
            continue

        response_text, g_in, g_out = result
        total_input_tokens += g_in
        total_output_tokens += g_out

        try:
            data = json.loads(response_text)
            lines_data = data.get("lines", [])

            if not lines_data:
                print(f"Warning: No text lines returned for figure {page_element.ref_id}")
                continue

            bboxes = []
            px1, py1, px2, py2 = page_element.bbox
            pw = px2 - px1
            ph = py2 - py1

            for i, line_info in enumerate(lines_data):
                text = line_info.get("text")
                bbox_2d = line_info.get("bbox_2d")

                if text and bbox_2d:
                    ymin, xmin, ymax, xmax = bbox_2d

                    # The VLM gives coordinates normalized to 1000x1000 on the cropped image.
                    # We need to convert them to absolute coordinates on the full page.
                    abs_x1 = (xmin / 1000) * pw + px1
                    abs_y1 = (ymin / 1000) * ph + py1
                    abs_x2 = (xmax / 1000) * pw + px1
                    abs_y2 = (ymax / 1000) * ph + py1

                    ref_id = f"{page_element.ref_id}.{i}"

                    bbox_obj = TextBoundingBox(
                        bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                        text=text,
                        ref_id=ref_id,
                    )
                    bboxes.append(bbox_obj)

            page_element.text_bounding_boxes = bboxes

        except Exception as e:
            print(f"Error processing figure grounding result for {page_element.ref_id}: {e}")

    return total_input_tokens, total_output_tokens
