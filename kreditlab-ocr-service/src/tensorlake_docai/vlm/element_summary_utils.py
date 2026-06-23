# SPDX-License-Identifier: Apache-2.0
import asyncio
import uuid
import os
import re
from pathlib import Path
from typing import List


from PIL import Image

from tensorlake_docai.models.layout_objects import PageLayoutElement
from tensorlake_docai.pipeline.api import PageFragmentType


def summarization_result_cleanup(result):
    return re.sub(r"^```[a-zA-Z]*\n|```$", "", result.strip())


def crop_elements(
    page_layout,
    padding,
    page_image,
    element_types,
    scale_factor=1.0,
    save_images=False,
    output_dir="cropped_images",
    filename_prefix=None,
):
    from PIL import Image
    import numpy as np

    cropped_images, page_elements = [], []

    # Create output directory if saving images
    if save_images:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if filename_prefix is None:
            filename_prefix = f"page_{uuid.uuid4().hex[:8]}"

    element_counter = {}  # To track count of each element type for naming

    for page_element in page_layout.elements:
        if page_element.fragment_type in element_types:
            x1, y1, x2, y2 = map(int, page_element.bbox)  # Convert to integers

            # Apply scale factor to convert from PDF coordinates to image coordinates
            x1 = int(x1 * scale_factor)
            y1 = int(y1 * scale_factor)
            x2 = int(x2 * scale_factor)
            y2 = int(y2 * scale_factor)

            img_arr = np.array(page_image)
            # Apply padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_arr.shape[1], x2 + padding)
            y2 = min(img_arr.shape[0], y2 + padding)
            cropped_img = img_arr[y1:y2, x1:x2]

            if cropped_img.size == 0:
                print(
                    f"Skipping element with empty crop region "
                    f"(bbox={page_element.bbox}, scaled_region=({x1},{y1},{x2},{y2}), "
                    f"image_shape={img_arr.shape})"
                )
                continue

            cropped_image = Image.fromarray(cropped_img)
            cropped_images.append(cropped_image)
            page_elements.append(page_element)

            # Save image if requested
            if save_images:
                element_type = page_element.fragment_type.name.lower()
                if element_type not in element_counter:
                    element_counter[element_type] = 0
                element_counter[element_type] += 1

                filename = (
                    f"{filename_prefix}_{element_type}_{element_counter[element_type]:02d}.png"
                )
                filepath = os.path.join(output_dir, filename)

                try:
                    cropped_image.save(filepath, format="PNG")
                    print(
                        f"Saved cropped {element_type} image: {filepath} (scale_factor: {scale_factor})"
                    )
                except Exception as e:
                    print(f"Error saving cropped image {filepath}: {e}")

    return cropped_images, page_elements


def add_white_padding(image, padding_pixels=100):
    from PIL import Image
    import numpy as np

    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    height, width = img_array.shape[:2]
    new_height = height + 2 * padding_pixels
    new_width = width + 2 * padding_pixels

    if len(img_array.shape) == 3:
        padded_array = np.full(
            (new_height, new_width, img_array.shape[2]), 255, dtype=img_array.dtype
        )
    else:
        padded_array = np.full((new_height, new_width), 255, dtype=img_array.dtype)

    padded_array[
        padding_pixels : padding_pixels + height, padding_pixels : padding_pixels + width
    ] = img_array

    return Image.fromarray(padded_array)


async def run_element_summary_and_modify_page_elements(
    cropped_images: List[Image.Image],
    page_elements: List[PageLayoutElement],
    page_image: Image.Image,
    user_prompt: str,
    element_types: List[PageFragmentType],
) -> tuple[int, int]:
    """
    Run element summary for specified element types and modify page elements in place.

    Returns:
        tuple[int, int]: (input_tokens, output_tokens) used for summarization
    """

    from tensorlake_docai.providers.model_provider_utils import (
        run_clients,
        _make_gemini_call,
        _make_oai_call,
    )

    if not cropped_images or not page_elements:
        return 0, 0

    # Handle default user prompts
    if user_prompt is None or user_prompt.strip() == "":
        from tensorlake_docai.prompts.prompts import get_element_summary_prompt

        has_page_image = page_image is not None
        user_prompt = get_element_summary_prompt(element_types, has_page_image)

    # print("input prompt for summarization: ", user_prompt)

    tasks = []
    elements_to_process = []

    models = [_make_gemini_call, _make_oai_call]
    # random.shuffle(models)

    for crop_img, page_element in zip(cropped_images, page_elements):
        if page_element.fragment_type in element_types:
            elements_to_process.append(page_element)
            tasks.append(
                run_clients(
                    user_prompt=user_prompt,
                    images=[crop_img],
                    page_image=page_image,
                    models=models,
                    job_type="element_summary",
                    json_schema=None,
                )
            )

    print(f"Awaiting element summary tasks of len {len(tasks)}")
    results_with_tokens = await asyncio.gather(*tasks)

    # Extract results and aggregate token usage
    total_input_tokens = 0
    total_output_tokens = 0

    for i, (result, input_tokens, output_tokens) in enumerate(results_with_tokens):
        result = summarization_result_cleanup(result)
        # print(f"Element summary result: {result}")
        elements_to_process[i].llm_summary = result
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    print(f"Element summary tokens - Input: {total_input_tokens}, Output: {total_output_tokens}")
    return total_input_tokens, total_output_tokens
