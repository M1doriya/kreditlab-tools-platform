# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import re
from PIL import Image
from typing import List, Optional
from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.models.layout_objects import PageLayoutElement
from tensorlake_docai.prompts.prompts import get_chart_prompt_messages
from tensorlake_docai.models.intermediate_objects import ParseResult


async def run_element_chart_extraction_and_modify_page_elements(
    cropped_images: List[Image.Image],
    page_elements: List[PageLayoutElement],
    element_types: Optional[List[PageFragmentType]] = None,
    element_types_to_check: Optional[List[PageFragmentType]] = None,
    parse_result: Optional[ParseResult] = None,
) -> tuple[int, int]:
    """
    Run chart extraction for specified element types and modify page elements in place.

    Returns:
        tuple[int, int]: (input_tokens, output_tokens) used for extraction
    """
    from tensorlake_docai.providers.model_provider_utils import run_clients, _make_gemini_call

    async def _make_gemini_call_charts(
        user_prompt,
        images,
        page_image=None,
        json_schema=None,
        job_type=None,
        timeout: Optional[int] = None,
        pdf_bytes=None,
    ):
        return await _make_gemini_call(
            user_prompt,
            images,
            page_image,
            json_schema,
            job_type,
            pdf_bytes=pdf_bytes,
            model_name="gemini-3-flash-preview",
        )

    if element_types is None:
        element_types = [PageFragmentType.CHART]
    if element_types_to_check is None:
        element_types_to_check = [
            PageFragmentType.FIGURE,
            PageFragmentType.TABLE,
        ]

    if not cropped_images or not page_elements:
        return 0, 0

    total_input_tokens = 0
    total_output_tokens = 0

    # Detect charts for all potential elements (charts, figures, tables)
    detection_tasks = []
    candidate_indices = []

    for i, (cropped_image, page_element) in enumerate(zip(cropped_images, page_elements)):
        if (
            page_element.fragment_type in element_types
            or page_element.fragment_type in element_types_to_check
        ):
            candidate_indices.append(i)
            prompt_messages = get_chart_prompt_messages("detection")
            _, user_prompt = prompt_messages

            detection_schema = json.dumps(
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bbox": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": "[ymin, xmin, ymax, xmax] in 0-1000 scale",
                            },
                            "risk_table_bbox": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": "[ymin, xmin, ymax, xmax] in 0-1000 scale for the risk table (optional)",
                            },
                            "series_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of series names found in the legend or risk table (optional)",
                            },
                        },
                        "required": ["bbox"],
                    },
                }
            )

            detection_tasks.append(
                run_clients(
                    user_prompt=user_prompt,
                    images=[cropped_image],
                    page_image=None,
                    models=[_make_gemini_call_charts],
                    job_type="structured_extraction",
                    json_schema=detection_schema,
                )
            )

    if not detection_tasks:
        return total_input_tokens, total_output_tokens

    detection_results_with_tokens = await asyncio.gather(*detection_tasks)

    # Prepare extraction tasks for detected charts
    extraction_tasks = []
    extraction_map = []  # Maps extraction task index to metadata
    bboxes_map = {}

    for i, (result, input_tokens, output_tokens) in enumerate(detection_results_with_tokens):
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        original_idx = candidate_indices[i]
        page_element = page_elements[original_idx]
        cropped_image = cropped_images[original_idx]
        width, height = cropped_image.size

        detected_items = []
        try:
            bboxes_data = json.loads(result)
            if isinstance(bboxes_data, list):
                for item in bboxes_data:
                    if "bbox" in item:
                        detected_items.append(item)
        except Exception:
            print(f"Chart detection parsing failed for element index {original_idx}")

        if not detected_items:
            # No charts detected in this element
            continue

        bboxes_map[original_idx] = detected_items

        # Charts detected - update fragment type
        page_element.fragment_type = PageFragmentType.CHART

        def crop_bbox(bbox):
            ymin, xmin, ymax, xmax = bbox
            left = int(xmin * width / 1000)
            top = int(ymin * height / 1000)
            right = int(xmax * width / 1000)
            bottom = int(ymax * height / 1000)

            if right > left and bottom > top:
                return cropped_image.crop((left, top, right, bottom))
            else:
                return cropped_image

        chart_extraction_messages = get_chart_prompt_messages("extraction")
        _, chart_user_prompt = chart_extraction_messages

        risk_table_messages = get_chart_prompt_messages("risk_table_extraction")
        _, risk_table_user_prompt = risk_table_messages

        for sub_idx, item in enumerate(detected_items):
            # 1. Process the Chart
            chart_bbox = item["bbox"]
            chart_img = crop_bbox(chart_bbox)

            series_names_hint = ""
            if "series_names" in item and item["series_names"]:
                series_names_hint = f"\n\nThe following series names were detected in the chart area: {item['series_names']}. Use these names to ensure consistency."

            extraction_tasks.append(
                run_clients(
                    user_prompt=chart_user_prompt + series_names_hint,
                    images=[chart_img],
                    page_image=None,
                    models=[_make_gemini_call_charts],
                    job_type="structured_extraction",
                    json_schema=None,
                )
            )
            extraction_map.append(
                {"original_idx": original_idx, "sub_idx": sub_idx, "type": "chart"}
            )

            # 2. Process the Risk Table (if present)
            if "risk_table_bbox" in item:
                risk_table_bbox = item["risk_table_bbox"]
                table_img = crop_bbox(risk_table_bbox)

                extraction_tasks.append(
                    run_clients(
                        user_prompt=risk_table_user_prompt + series_names_hint,
                        images=[table_img],
                        page_image=None,
                        models=[_make_gemini_call_charts],
                        job_type="structured_extraction",
                        json_schema=None,
                    )
                )
                extraction_map.append(
                    {
                        "original_idx": original_idx,
                        "sub_idx": sub_idx,
                        "type": "risk_table",
                    }
                )

    if not extraction_tasks:
        return total_input_tokens, total_output_tokens

    extraction_results_with_tokens = await asyncio.gather(*extraction_tasks)

    # Temporary storage to merge chart and risk table results
    # Structure: temp_results[original_idx][sub_idx] = {'chart': ..., 'risk_table': ...}
    temp_results = {}

    for i, (result, input_tokens, output_tokens) in enumerate(extraction_results_with_tokens):
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        meta = extraction_map[i]
        original_idx = meta["original_idx"]
        sub_idx = meta["sub_idx"]
        task_type = meta["type"]

        if original_idx not in temp_results:
            temp_results[original_idx] = {}
        if sub_idx not in temp_results[original_idx]:
            temp_results[original_idx][sub_idx] = {}

        try:
            # Clean markdown
            clean_result = result.strip()
            if clean_result.startswith("```"):
                clean_result = re.sub(r"^```[a-zA-Z]*\s*", "", clean_result)
                clean_result = re.sub(r"\s*```$", "", clean_result).strip()

            data = None
            try:
                data = json.loads(clean_result)
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        print(
                            f"Nested JSON decoding failed for element index {original_idx}, sub-index {sub_idx}"
                        )
            except json.JSONDecodeError:
                # Handle case where output is wrapped in quotes but invalid as a JSON string
                if clean_result.startswith('"'):
                    inner = clean_result.strip('"')
                    try:
                        data = json.loads(inner)
                    except json.JSONDecodeError:
                        try:
                            data = json.loads(inner.replace('\\"', '"'))
                        except json.JSONDecodeError:
                            print(
                                f"Quoted JSON decoding failed for element index {original_idx}, sub-index {sub_idx}"
                            )

                # Handle truncated JSON (missing closing brackets)
                if data is None and clean_result.startswith("["):
                    try:
                        data = json.loads(clean_result.rstrip().rstrip(",") + "]")
                    except json.JSONDecodeError:
                        try:
                            data = json.loads(clean_result.rstrip().rstrip(",") + "}]")
                        except json.JSONDecodeError:
                            print(
                                f"Truncated array JSON decoding failed for element index {original_idx}, sub-index {sub_idx}"
                            )
                elif data is None and clean_result.startswith("{"):
                    try:
                        data = json.loads(clean_result.rstrip().rstrip(",") + "}")
                    except json.JSONDecodeError:
                        print

            if data is None:
                continue

            if task_type == "chart":
                # Chart extraction returns a list of charts (usually 1 for the cropped image)
                if isinstance(data, list) and len(data) > 0:
                    temp_results[original_idx][sub_idx]["chart"] = data[0]
                elif isinstance(data, dict):
                    temp_results[original_idx][sub_idx]["chart"] = data
            elif task_type == "risk_table":
                # Risk table extraction returns a single object
                temp_results[original_idx][sub_idx]["risk_table"] = data
        except Exception:
            print(
                f"Chart extraction parsing failed for element index {original_idx}, sub-index {sub_idx}"
            )

    for original_idx, sub_charts_map in temp_results.items():
        final_charts = []
        # Sort by sub_idx to maintain order
        for sub_idx in sorted(sub_charts_map.keys()):
            entry = sub_charts_map[sub_idx]
            chart_obj = entry.get("chart", {})
            risk_table = entry.get("risk_table")

            # Restore bbox from detection
            if original_idx in bboxes_map and sub_idx < len(bboxes_map[original_idx]):
                chart_obj["bbox"] = bboxes_map[original_idx][sub_idx]["bbox"]

            # Merge risk table if present
            if risk_table:
                chart_obj["risk_table"] = risk_table

            final_charts.append(chart_obj)

        page_elements[original_idx].llm_summary = json.dumps(final_charts)

    return total_input_tokens, total_output_tokens
