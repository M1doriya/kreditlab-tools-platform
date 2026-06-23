# SPDX-License-Identifier: Apache-2.0
"""
Form extraction utilities.
"""

import asyncio
import json
from typing import List, Tuple, Optional
from PIL import Image

from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.models.layout_objects import PageLayoutElement
from tensorlake_docai.prompts.prompts import get_form_prompt_messages
from tensorlake_docai.models.intermediate_objects import ParseResult


def convert_form_json_to_markdown(content: str) -> str:
    """Convert form JSON content to Markdown."""
    try:
        if isinstance(content, str):
            data = json.loads(content)
        else:
            data = content

        md_lines = []

        if isinstance(data, dict):
            for k, v in data.items():
                md_lines.append(f"**{k}**: {v}")

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Specific schema from form_extraction_utils
                    if "field_name" in item and "value" in item:
                        box_id = item.get("box_id", "")
                        name = item.get("field_name", "")
                        field_type = item.get("type", "")
                        val = item.get("value", "")

                        line = f"**{name}**"
                        if box_id:
                            line = f"[{box_id}] {line}"
                        if field_type:
                            line = f"{line} ({field_type})"
                        line = f"{line}: {val}"
                        md_lines.append(line)
                    else:
                        # Generic fallback
                        k = (
                            item.get("key")
                            or item.get("field")
                            or item.get("label")
                            or item.get("question")
                        )
                        v = (
                            item.get("value")
                            or item.get("text")
                            or item.get("answer")
                            or item.get("response")
                        )

                        if k is not None and v is not None:
                            md_lines.append(f"**{k}**: {v}")
                        elif len(item) == 1:
                            k, v = list(item.items())[0]
                            md_lines.append(f"**{k}**: {v}")
                        else:
                            for sub_k, sub_v in item.items():
                                md_lines.append(f"**{sub_k}**: {sub_v}")
                else:
                    md_lines.append(str(item))

        if md_lines:
            return "\n\n".join(md_lines)

        return content if isinstance(content, str) else json.dumps(content)

    except (json.JSONDecodeError, TypeError):
        return content if isinstance(content, str) else str(content)


async def run_element_form_extraction_and_modify_page_elements(
    cropped_images: List[Image.Image],
    page_elements: List[PageLayoutElement],
    element_types: List[PageFragmentType],
    element_types_to_check: List[PageFragmentType],
    parse_result: Optional[ParseResult] = None,
) -> Tuple[int, int]:

    total_input_tokens = 0
    total_output_tokens = 0

    tasks = []

    for img, element in zip(cropped_images, page_elements):
        tasks.append(_process_single_element(img, element, element_types, element_types_to_check))

    results = await asyncio.gather(*tasks)

    for input_tokens, output_tokens in results:
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    return total_input_tokens, total_output_tokens


async def _process_single_element(
    image: Image.Image,
    element: PageLayoutElement,
    target_types: List[PageFragmentType],
    check_types: List[PageFragmentType],
) -> Tuple[int, int]:
    from tensorlake_docai.providers.model_provider_utils import run_clients, _make_gemini_call

    input_tokens = 0
    output_tokens = 0

    is_form = False

    # If element is already a target type (FORM), proceed to extraction
    if element.fragment_type in target_types:
        is_form = True
    # If element is in check types (FIGURE), run detection
    elif element.fragment_type in check_types:
        _, user_prompt = get_form_prompt_messages("detection")
        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {"is_form": {"type": "boolean"}},
                "required": ["is_form"],
            }
        )

        try:
            response_text, in_tok, out_tok = await run_clients(
                user_prompt=user_prompt,
                images=[image],
                models=[_make_gemini_call],
                job_type="structured_extraction",
                json_schema=json_schema,
            )
            input_tokens += in_tok
            output_tokens += out_tok

            result = json.loads(response_text)
            if result.get("is_form", False):
                is_form = True
                # Update fragment type to FORM
                element.fragment_type = PageFragmentType.FORM
        except Exception as e:
            print(f"Form detection failed: {e}", flush=True)

    if is_form:
        _, user_prompt = get_form_prompt_messages("extraction")

        extraction_schema = json.dumps(
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "box_id": {
                            "type": "string",
                            "description": "The box id of the form field, if present. Empty string if not available.",
                        },
                        "field_name": {
                            "type": "string",
                            "description": "The name or context of the form field",
                        },
                        "type": {
                            "type": "string",
                            "description": "The type of the form field, e.g. text, checkbox, radio button",
                        },
                        "value": {"type": "string", "description": "The value of the form field"},
                    },
                    "required": ["field_name", "type", "value"],
                },
            }
        )

        try:
            response_text, in_tok, out_tok = await run_clients(
                user_prompt=user_prompt,
                images=[image],
                models=[_make_gemini_call],
                job_type="structured_extraction",
                json_schema=extraction_schema,
            )
            input_tokens += in_tok
            output_tokens += out_tok

            # Convert JSON to markdown for element.markdown
            markdown_response_text = convert_form_json_to_markdown(response_text)

            # Update element content
            element.ocr_text = response_text
            element.html = ""
            element.markdown = markdown_response_text

        except Exception as e:
            print(f"Form extraction failed: {e}", flush=True)

    return input_tokens, output_tokens
