# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for the GPU-based `dots-ocr` pipeline.
"""

import json
import asyncio
import re
from typing import List, Dict, Any, Optional, NamedTuple, Literal, Iterator
from abc import ABC, abstractmethod

from tensorlake.applications import RequestError as RequestException
from tensorlake.applications import RequestContext
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import PageLayoutElement, PageLayout
from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.pipeline.simple_page_creator import SimplePageCreator, ImageDimensions
from tensorlake_docai.postprocess.output_cleaner import OutputCleaner

# OCR Constants
PADDING = 4
OCR_LAYOUT_MIN_PIXELS = 3136  # Standard minimum
OCR_LAYOUT_MAX_PIXELS = 11289600  # 14400 tokens      # Standard maximum from dots.mocr

# Fragment groupings
TABLE_TYPES = {
    PageFragmentType.TABLE,
    PageFragmentType.FIGURE,
}

FORMULA_TYPES = {
    PageFragmentType.FORMULA,
}

TEXT_TYPES = {
    PageFragmentType.SECTION_HEADER,
    PageFragmentType.TEXT,
    PageFragmentType.PAGE_FOOTER,
    PageFragmentType.PAGE_NUMBER,
}


def map_dotsocr_category_to_fragment_type(category: str) -> PageFragmentType:
    """Map DotsOCR category to PageFragmentType enum"""
    mapping = {
        "Caption": PageFragmentType.TEXT,
        "Footnote": PageFragmentType.PAGE_FOOTER,
        "Formula": PageFragmentType.FORMULA,
        "List-item": PageFragmentType.TEXT,
        "Page-footer": PageFragmentType.PAGE_FOOTER,
        "Page-header": PageFragmentType.PAGE_HEADER,
        "Picture": PageFragmentType.FIGURE,
        "Section-header": PageFragmentType.SECTION_HEADER,
        "Table": PageFragmentType.TABLE,
        "Text": PageFragmentType.TEXT,
        "Title": PageFragmentType.TITLE,
    }
    return mapping.get(category, PageFragmentType.TEXT)


def parse_dotsocr_full_page_output(output_text: str, scale_factor: float) -> List[dict]:
    """Parse DotsOCR JSON output and return list of layout elements with proper coordinates."""
    try:
        # Clean up the output text
        cleaned_output = output_text.strip()

        # Remove common prefixes that DotsOCR adds before JSON
        prefixes_to_remove = [
            "- The structure of the JSON object: ",
            "The structure of the JSON object: ",
            "Here is the JSON output: ",
            "Format: ",
            "```json",
            "- ",
        ]

        for prefix in prefixes_to_remove:
            if cleaned_output.startswith(prefix):
                cleaned_output = cleaned_output[len(prefix) :].strip()
                break

        # Remove markdown code block endings
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3].strip()

        # Handle escape sequences
        if cleaned_output.startswith("\\n"):
            cleaned_output = cleaned_output[2:]

        # Match dots.mocr behavior:
        # - Try strict JSON parse first (preserves valid model output, especially long tables/HTML)
        # - Only fall back to OutputCleaner when JSON parsing fails
        layout_data: Any
        try:
            layout_data = json.loads(cleaned_output)
        except Exception:
            layout_data = None

        if not isinstance(layout_data, list):
            # Fallback to output cleaner (best-effort recovery from malformed JSON)
            cleaner = OutputCleaner()
            layout_data = cleaner.clean_model_output(cleaned_output)

        if not isinstance(layout_data, list) or not layout_data:
            print(
                "Warning: No valid layout data found (JSON parse + OutputCleaner fallback failed)"
            )
            return []

        # Process each layout element; assume bboxes are already in image coordinates
        processed_elements = []
        for i, element in enumerate(layout_data):
            if not isinstance(element, dict) or "bbox" not in element:
                continue

            bbox = element["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            # Validate bbox
            if float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]):
                print(f"Warning: Invalid bbox for element {i}: {bbox}")
                continue

            category = element.get("category", "Text")
            text_content = element.get("text", "")

            processed_element = {
                "bbox": [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ],
                "category": category,
                "text": text_content,
                "reading_order": i,
            }
            processed_elements.append(processed_element)
            print(f"  Element {i}: {category} at {processed_element['bbox']}")

        print(f"✓ Processed {len(processed_elements)} layout elements from DotsOCR output")
        return processed_elements

    except Exception as e:
        print(f"Error parsing DotsOCR full page output: {e}")
        import traceback

        traceback.print_exc()
        return []


def create_page_elements_from_dotsocr_output(
    dotsocr_elements: List[dict], page_number: int = 1
) -> List[PageLayoutElement]:
    """Convert DotsOCR output elements to PageLayoutElement objects."""
    try:
        from markdownify import markdownify as md
    except ImportError:
        # Fallback if markdownify is not available
        def md(text):
            return text

    page_elements = []

    for element in dotsocr_elements:
        try:
            bbox = element["bbox"]
            category = element["category"]
            # Handle missing text field (from layout_only elements without matched content)
            text_content = element.get("text", "")
            reading_order = element["reading_order"]

            # Map DotsOCR category to PageFragmentType
            fragment_type = map_dotsocr_category_to_fragment_type(category)

            # Process text content based on category
            if category == "Formula":
                # Formula text is already in LaTeX format from DotsOCR
                processed_text = text_content
                markdown_content = md(processed_text) if text_content else ""
            elif category == "Table":
                # Table text is in HTML format from DotsOCR
                processed_text = text_content
                markdown_content = md(processed_text) if text_content else ""
            elif category == "Picture":
                # Picture elements typically don't have text content
                processed_text = text_content if text_content else ""
                markdown_content = ""
            else:
                # Other elements (Text, Title, etc.) are in Markdown format
                processed_text = text_content
                markdown_content = processed_text

            # Determine hierarchy level for headers based on leading hashtags
            hierarchy_level = None
            if fragment_type in {PageFragmentType.SECTION_HEADER, PageFragmentType.TITLE}:
                if processed_text:  # Only check if there's text
                    stripped = processed_text.lstrip()
                    match = re.match(r"^(#+)\s", stripped)
                    hierarchy_level = (len(match.group(1)) - 1) if match else 2

            page_element = PageLayoutElement(
                bbox=[
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ],
                fragment_type=fragment_type,
                score=1.0,  # Default confidence score
                reading_order=reading_order,
                ref_id=f"{page_number}.{reading_order}",
                ocr_text=processed_text,
                markdown=markdown_content,
                html=text_content if (category == "Table" and text_content) else None,
                hierarchy_level=hierarchy_level,
                image_base64=None,  # Will be set later in _process_figures_with_ovis if needed
            )

            page_elements.append(page_element)

        except Exception as e:
            print(f"Error creating PageLayoutElement from DotsOCR element: {e}")
            continue

    return page_elements


def scale_bbox_to_original_coordinates(
    dotsocr_elements: List[dict], image_size: tuple[int, int], pdf_size: tuple[int, int]
):
    """
    Scale bounding boxes from page image coordinates back to original PDF coordinates.

    The coordinate transformation chain is:
    1. Original PDF coordinates (what we want for debug plotting)
    2. Scaled page image coordinates (from SimplePageCreator with scale_factor)
    3. Resized DotsOCR input coordinates (from smart_resize)

    We've already converted from step 3 -> 2 in parse_dotsocr_full_page_output.
    Now we need to convert from step 2 -> 1 using the scale_factor.
    """
    print("Scaling bounding boxes back to original PDF coordinates using sizes")

    # Scale coordinates back to original PDF coordinates using size ratios
    img_w, img_h = image_size
    pdf_w, pdf_h = pdf_size
    sx = (pdf_w / img_w) if img_w else 1.0
    sy = (pdf_h / img_h) if img_h else 1.0

    for element in dotsocr_elements:
        if "bbox" not in element:
            continue

        bbox = element["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        original_bbox = [
            int(float(bbox[0]) * sx),
            int(float(bbox[1]) * sy),
            int(float(bbox[2]) * sx),
            int(float(bbox[3]) * sy),
        ]

        element["bbox"] = original_bbox


def detect_consecutive_repetition(
    text: str, window_size: int = 2000, repeat_threshold: int = 20, coverage_threshold: float = 0.7
) -> bool:
    """
    Detect consecutive repetitive patterns in generated text using a streaming window.
    Finds the longest repeating pattern and counts consecutive occurrences.

    Args:
        text: Current generated text
        window_size: Size of the sliding window to check (default: 2000 characters)
        repeat_threshold: Number of consecutive repeats to trigger detection (default: 20)
        coverage_threshold: Fraction of window covered by pattern to trigger detection (default: 0.7 = 70%)

    Returns:
        True if repetition detected, False otherwise
    """
    if len(text) < window_size:
        return False

    # Use last 2000 characters as streaming window
    window = text[-window_size:]

    # Helper function to check if a string contains only HTML tags (no real content)
    import re

    def has_real_content(s: str) -> bool:
        # Remove all HTML tags
        without_tags = re.sub(r"<[^>]+>", "", s)
        # Check if remaining content has real text (not just whitespace)
        return bool(without_tags.strip())

    # Try to find the longest repeating pattern
    # Start with minimum pattern size and work up
    min_pattern_length = 5  # Minimum pattern size to consider
    max_pattern_length = min(
        500, len(window) // 2
    )  # Don't check patterns longer than half the window

    best_pattern = None
    best_count = 0

    # Search for repeating patterns at different positions (not just the end)
    for pattern_length in range(min_pattern_length, max_pattern_length + 1):
        # Try patterns from different positions in the last part of the window
        # Check last 200 chars for potential pattern candidates
        search_start = max(0, len(window) - 200)

        for pattern_start in range(search_start, len(window) - pattern_length + 1):
            pattern = window[pattern_start : pattern_start + pattern_length]

            # Count how many times this pattern repeats consecutively backwards from this position
            count = 0
            pos = pattern_start

            while pos >= 0:
                # Check if pattern matches at this position
                if window[pos : pos + pattern_length] == pattern:
                    count += 1
                    pos -= pattern_length
                else:
                    # Pattern doesn't match - check if there's real content that breaks it
                    break_segment = window[pos : pos + pattern_length]
                    if has_real_content(break_segment):
                        # Real content found, reset count (this is good, means generation has variety)
                        count = 0
                    break

            # Update best pattern if this one has more consecutive repeats
            if count > best_count:
                best_count = count
                best_pattern = pattern

    # Check if we found a repetitive pattern exceeding threshold (either count OR coverage)
    if best_pattern and best_count > 0:
        coverage = (len(best_pattern) * best_count) / window_size

        if best_count >= repeat_threshold or coverage >= coverage_threshold:
            pattern_type = "text" if has_real_content(best_pattern) else "HTML"
            print(f"   ⚠️ Repetitive {pattern_type} pattern detected!")
            print(
                f"   Pattern length: {len(best_pattern)} chars, repeats: {best_count} times, coverage: {coverage:.1%}"
            )

            # Show which criterion triggered
            triggers = []
            if best_count >= repeat_threshold:
                triggers.append(f"count>={repeat_threshold}")
            if coverage >= coverage_threshold:
                triggers.append(f"coverage>={coverage_threshold:.0%}")
            print(f"   Triggered by: {' and '.join(triggers)}")

            print(f"   Pattern sample: {best_pattern[:100]}...")
            return True

    return False


def create_masked_image(img, elements: list) -> tuple:
    """
    Create a masked image by whiting out regions with good content.
    Mask elements that:
    1. Have good content (successfully extracted) - don't need to retry, OR
    2. Are Pictures (will be re-extracted by Ovis) - handled separately
    Don't mask elements with failed/empty text - we want to retry those!

    Args:
        img: Original PIL Image
        elements: List of elements to mask

    Returns:
        Tuple of (masked_img, masked_count)
    """
    from PIL import ImageDraw

    masked_img = img.copy()
    draw = ImageDraw.Draw(masked_img)
    masked_count = 0

    for element in elements:
        bbox = element.get("bbox", [])
        text = element.get("text", "")
        category = element.get("category", "")

        # Mask elements that:
        # 1. Have good content (successfully extracted), OR
        # 2. Are Pictures (will be re-extracted by Ovis)
        should_mask = False
        if len(bbox) == 4:
            # Mask if has good content
            if text and len(text.strip()) > 10:
                should_mask = True
            # Also mask Pictures (marked for Ovis re-extraction)
            elif category == "Picture":
                should_mask = True

        if should_mask:
            # Normalize bbox — model can produce inverted coordinates (x1<x0 or y1<y0)
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
            draw.rectangle([(min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))], fill="white")
            masked_count += 1

    return masked_img, masked_count


def merge_partial_outputs(all_elements: list) -> list:
    """
    Merge multiple partial outputs into one clean list.
    Removes duplicates (by bbox IoU) and keeps best content.

    Args:
        all_elements: List of all elements from multiple iterations

    Returns:
        Merged list of unique elements with best content
    """

    def bbox_iou(bbox1, bbox2):
        """Calculate Intersection over Union for two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    # Deduplicate elements by bbox IoU, keeping the one with most content
    result = []

    for element in all_elements:
        bbox = element.get("bbox")
        text = element.get("text", "")

        if not bbox or len(bbox) != 4:
            continue

        # Check if this element overlaps significantly with any existing element
        found_duplicate = False
        for i, existing in enumerate(result):
            existing_bbox = existing.get("bbox")
            if not existing_bbox or len(existing_bbox) != 4:
                continue

            iou = bbox_iou(bbox, existing_bbox)
            if iou > 0.7:  # High overlap = duplicate
                # Keep the element with more content
                existing_text = existing.get("text", "")
                if len(text) > len(existing_text):
                    result[i] = element
                found_duplicate = True
                break

        if not found_duplicate:
            result.append(element)

    # Re-index reading order
    for i, elem in enumerate(result):
        elem["reading_order"] = i

    return result


class BatchProcessor(ABC):

    def __init__(
        self,
        scale_factor: Optional[float] = None,
        batch_size: Optional[int] = None,
        memory_gb: Optional[float] = None,
    ):
        # Always provide a default batch_size (20) even when using memory_gb
        # The memory_gb parameter will control dynamic batching, but we need a max batch size for min() comparisons
        default_batch_size = 25
        self.page_creator = SimplePageCreator(
            scale_factor=scale_factor or 200 / 72,
            batch_size=batch_size if batch_size is not None else default_batch_size,
            memory_gb=memory_gb,
        )

    @abstractmethod
    def process_batch(
        self, page_images_dict: Dict[int, Any], batch_number: int
    ) -> List[PageLayout]:
        pass

    def should_preserve_existing_pages(self) -> bool:
        return False

    class ProcessingBatch(NamedTuple):
        page_numbers: List[int]
        payload_kind: Literal["images", "pdf"]
        payload: Any  # list[PIL.Image] for images; bytes for pdf
        original_sizes: Dict[int, tuple[int, int]]  # Original PDF/image sizes per page

    def get_processing_batches(
        self, parse_result: ParseResult, image_dimensions: ImageDimensions = ImageDimensions()
    ) -> Iterator["BatchProcessor.ProcessingBatch"]:
        for doc_pages in self.page_creator.get_images_generator(parse_result, image_dimensions):
            page_numbers = list(doc_pages.page_images.keys())
            images = [doc_pages.page_images[p] for p in page_numbers]
            yield BatchProcessor.ProcessingBatch(
                page_numbers=page_numbers,
                payload_kind="images",
                payload=images,
                original_sizes=doc_pages.original_sizes,
            )

    def run_batch_processing(
        self,
        ctx: RequestContext,
        parse_result: ParseResult,
        image_dimensions: ImageDimensions = ImageDimensions(),
    ) -> ParseResult:
        if not self.should_preserve_existing_pages():
            parse_result.document_layout.pages = []

        batch_generator = self.get_processing_batches(parse_result, image_dimensions)
        all_results = []

        async def _run_async_batches():
            max_concurrency = 8
            semaphore = asyncio.Semaphore(max_concurrency)
            running_tasks = set()

            async def _run_single_batch(page_batch: "BatchProcessor.ProcessingBatch", batch_num):
                max_retries = 3
                base_delay = 1.0

                async with semaphore:
                    # Use injected batch payload
                    processing_batch = page_batch

                    for retry_count in range(max_retries + 1):
                        print(
                            f" Sending batch {batch_num} to model ({len(page_batch.page_numbers)} pages)"
                            + (f" - retry {retry_count}" if retry_count > 0 else "")
                        )

                        try:
                            # process_batch is now async; call directly
                            result = await self.process_batch(
                                processing_batch,
                                batch_num,
                            )
                            return batch_num, result, None
                        except Exception as e:
                            error_str = str(e).lower()
                            is_rate_limit = "429" in error_str or "rate limit" in error_str

                            if retry_count < max_retries:
                                if is_rate_limit:
                                    # Exponential backoff with jitter for rate limits
                                    delay = base_delay * (2**retry_count) + (0.1 * retry_count)
                                    print(
                                        f" Batch {batch_num} rate limited, retrying in {delay:.1f}s (attempt {retry_count + 1}/{max_retries})"
                                    )
                                else:
                                    # Linear backoff for other errors
                                    delay = base_delay * (retry_count + 1)
                                    print(
                                        f" Batch {batch_num} failed, retrying in {delay:.1f}s (attempt {retry_count + 1}/{max_retries}): {e}"
                                    )

                                await asyncio.sleep(delay)
                            else:
                                print(f" Batch {batch_num} failed after {max_retries} retries: {e}")
                                return batch_num, None, e

            # Incremental scheduling (sliding window)
            scheduled_batches = 0
            completed = 0
            total = 0
            batch_results_map = {}
            pages_completed = 0
            total_pages = 0

            # Helper to schedule up to max_concurrency tasks
            def schedule_up_to_limit():
                nonlocal scheduled_batches, total, total_pages
                while len(running_tasks) < max_concurrency:
                    try:
                        page_batch = next(batch_iter)
                    except StopIteration:
                        break
                    if not page_batch.page_numbers:
                        continue
                    scheduled_batches += 1
                    total = scheduled_batches
                    total_pages += len(page_batch.page_numbers)
                    running_tasks.add(
                        asyncio.create_task(_run_single_batch(page_batch, scheduled_batches))
                    )

            # Create an iterator from the generator and prefill the window
            batch_iter = iter(batch_generator)
            schedule_up_to_limit()

            # Process tasks; when one finishes, schedule the next
            while running_tasks:
                done, running_tasks = await asyncio.wait(
                    running_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    batch_num, batch_results, err = task.result()
                    if err:
                        raise RequestException(f"error processing batch {batch_num}: {str(err)}")
                    batch_results_map[batch_num] = batch_results
                    completed += 1

                    # Print each page and update progress per page
                    for page_result in batch_results:
                        pages_completed += 1
                        print(f"  Page {pages_completed}/{total_pages} processed")
                        if ctx:
                            ctx.progress.update(
                                current=pages_completed,
                                total=total_pages,
                                message=f"Processed {pages_completed}/{total_pages} pages",
                            )

                    print(
                        f"batch {batch_num} done: {len(batch_results)} pages ({completed}/{total})"
                    )

                # Top back up to the concurrency window
                schedule_up_to_limit()

            for bn in sorted(batch_results_map.keys()):
                all_results.extend(batch_results_map[bn])

            return scheduled_batches

        # Run the async scheduler/collector in a fresh loop
        asyncio.run(_run_async_batches())

        # Add results to parse_result (OCR tasks always create new pages)
        parse_result.document_layout.pages.extend(all_results)
        return parse_result


def create_page_layout_from_dotsocr_elements(
    page_num: int,
    dotsocr_elements: List[dict],
    image_size: tuple[int, int],
    pdf_size: tuple[int, int],
) -> PageLayout:
    """
    Create a complete PageLayout from DotsOCR elements.

    Args:
        page_num: Page number
        dotsocr_elements: Raw DotsOCR output elements
        image_size: Size of the page image (width, height)
        pdf_size: Size of the PDF page (width, height)

    Returns:
        Complete PageLayout object
    """
    # Parse DotsOCR output to get layout elements
    parsed_elements = parse_dotsocr_full_page_output(
        dotsocr_elements if isinstance(dotsocr_elements, str) else "", 1.0
    )

    if not parsed_elements:
        print(f"Warning: No layout elements found for page {page_num}")
        # Return empty page layout
        original_pdf_width = int(pdf_size[0])
        original_pdf_height = int(pdf_size[1])
        return PageLayout(
            elements=[],
            shape=(original_pdf_width, original_pdf_height),
            page_number=page_num,
            page_dimensions={"width": original_pdf_width, "height": original_pdf_height},
        )

    # Scale bounding boxes from image coordinates to original PDF coordinates
    scale_bbox_to_original_coordinates(parsed_elements, image_size, pdf_size)

    page_elements = create_page_elements_from_dotsocr_output(parsed_elements, page_num)

    if not page_elements:
        print(f"Warning: No valid PageLayoutElement objects created for page {page_num}")

    # Calculate original PDF dimensions
    original_pdf_width = int(pdf_size[0])
    original_pdf_height = int(pdf_size[1])

    # Sort elements by reading order
    page_elements.sort(key=lambda x: x.reading_order)

    # Create new page layout with required fields
    page_layout = PageLayout(
        elements=page_elements,
        shape=(original_pdf_width, original_pdf_height),
        page_number=page_num,
        page_dimensions={"width": original_pdf_width, "height": original_pdf_height},
    )

    return page_layout


def collect_figure_inputs_from_page(
    page_layout,
    original_image,
    scale_factor,
    page_num,
    keep_all_figures: bool = False,
    figure_prompt: str = "",
    form_prompt: str = "",
    form_coverage_threshold: float = 0.6,
):
    """
    Collect figure inputs from a single page for batch processing.
    Based on process_figure_regions logic but only collects, doesn't process.

    Args:
        page_layout: PageLayout containing elements
        original_image: Original page image
        scale_factor: Scale factor for coordinate conversion
        page_num: Page number
        keep_all_figures: When True, include all figures regardless of size
        figure_prompt: Prompt to use for small figure text extraction
        form_prompt: Prompt to use for large figures (likely forms). If empty, uses figure_prompt for all.
        form_coverage_threshold: Coverage threshold above which to use form_prompt (default: 0.6 = 60%)
    """
    import numpy as np
    from PIL import Image

    # Calculate total image area in original PDF coordinates
    original_pdf_width = int(original_image.width / scale_factor)
    original_pdf_height = int(original_image.height / scale_factor)
    total_area = original_pdf_width * original_pdf_height
    # If summarization explicitly requested, keep all figures; otherwise keep >=1%
    min_area_threshold = 0 if keep_all_figures else (total_area * 0.01)

    # Find figure elements that are large enough
    large_figures = []
    for element in page_layout.elements:
        if element.fragment_type == PageFragmentType.FIGURE:
            bbox = element.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height

            if area >= min_area_threshold:
                large_figures.append(element)

    if not large_figures:
        return [], []

    # Prepare batch inputs for figure OCR
    batch_inputs = []
    batch_metadata = []

    for i, element in enumerate(large_figures):
        try:
            # Convert bbox from PDF coordinates back to image coordinates
            pdf_bbox = element.bbox
            img_bbox = [
                int(pdf_bbox[0] * scale_factor),
                int(pdf_bbox[1] * scale_factor),
                int(pdf_bbox[2] * scale_factor),
                int(pdf_bbox[3] * scale_factor),
            ]

            # Crop the figure region from original image
            img_arr = np.array(original_image)
            x1, y1, x2, y2 = img_bbox

            # Add padding and ensure bounds
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(img_arr.shape[1], x2 + PADDING)
            y2 = min(img_arr.shape[0], y2 + PADDING)

            if x1 >= x2 or y1 >= y2:
                continue

            cropped_img = Image.fromarray(img_arr[y1:y2, x1:x2])

            # Decide which prompt to use based on figure coverage
            element_area = (pdf_bbox[2] - pdf_bbox[0]) * (pdf_bbox[3] - pdf_bbox[1])
            coverage = element_area / total_area if total_area > 0 else 0

            # Use form prompt for large figures if provided, otherwise use figure prompt
            if form_prompt and coverage > form_coverage_threshold:
                selected_prompt = form_prompt
                print(f"  Page {page_num} Figure {i}: Using FORM prompt (coverage: {coverage:.1%})")
            else:
                selected_prompt = figure_prompt

            # Prepare prompt with selected prompt
            prompt = f"<|img|><|imgpad|><|endofimg|>{selected_prompt}" if selected_prompt else ""
            batch_input = {"prompt": prompt, "multi_modal_data": {"image": cropped_img}}

            batch_inputs.append(batch_input)

            # Store metadata to update the correct element later
            batch_metadata.append({"element": element, "page_num": page_num, "figure_idx": i})

        except Exception as e:
            print(f"  Page {page_num} Figure {i}: Error preparing for OCR: {e}")
            continue

    return batch_inputs, batch_metadata


def _process_single_figure_batch(batch_inputs, batch_metadata, inference_function):
    """
    Process a single batch of figures (helper function for process_all_figures_batch).

    Args:
        batch_inputs: List of prepared figure inputs for this batch
        batch_metadata: List of metadata for each figure in this batch
        inference_function: Function (sync or async) that takes batch inputs and returns list of text results
    """
    import asyncio
    import inspect

    # Run inference using provided function (handle both sync and async)
    if inspect.iscoroutinefunction(inference_function):
        results_texts = asyncio.run(inference_function(batch_inputs))
    else:
        results_texts = inference_function(batch_inputs)

    if results_texts is None:
        raise RequestException("Figure inference failed for batch processing.")

    # Process results and update elements
    for metadata, result_text in zip(batch_metadata, results_texts):
        element = metadata["element"]
        page_num = metadata["page_num"]
        figure_idx = metadata["figure_idx"]

        output_text = result_text.strip()

        if output_text:
            # Update the PageLayoutElement with extracted text
            element.ocr_text = output_text
            element.markdown = output_text
            print(
                f"  Page {page_num} Figure {figure_idx}: Extracted text ({len(output_text)} chars)"
            )
        else:
            print(f"  Page {page_num} Figure {figure_idx}: No text extracted")

    print(f"✓ Completed batch figure OCR processing for {len(batch_inputs)} figures")


def process_all_figures_batch(
    all_figure_inputs, all_figure_metadata, inference_function, subbatch_size=None
):
    """
    Process all collected figures from all pages in one batch or sub-batches.

    Args:
        all_figure_inputs: List of prepared figure inputs
        all_figure_metadata: List of metadata for each figure
        inference_function: Function that takes batch inputs and returns list of text results
        subbatch_size: Optional size for sub-batching to manage memory. If None, processes all at once.
    """
    try:
        # If no sub-batching requested, process all at once (backward compatibility)
        if subbatch_size is None:
            return _process_single_figure_batch(
                all_figure_inputs, all_figure_metadata, inference_function
            )

        # Process in sub-batches to manage memory
        total_figures = len(all_figure_inputs)
        if total_figures == 0:
            print("No figures to process")
            return

        print(f"Processing {total_figures} figures in sub-batches of {subbatch_size}")

        from itertools import batched

        processed_count = 0
        for subbatch_inputs, subbatch_metadata in zip(
            batched(all_figure_inputs, subbatch_size), batched(all_figure_metadata, subbatch_size)
        ):
            batch_start = processed_count + 1
            batch_end = processed_count + len(subbatch_inputs)
            print(f"  Processing sub-batch: figures {batch_start}-{batch_end} of {total_figures}")

            # Process this sub-batch
            _process_single_figure_batch(
                list(subbatch_inputs), list(subbatch_metadata), inference_function
            )
            processed_count += len(subbatch_inputs)

        print(f"✓ Completed all sub-batch figure processing: {processed_count} figures total")

    except Exception as e:
        print(f"Error during batch figure OCR: {e}")
        import traceback

        traceback.print_exc()
        raise RequestException(
            message=f"Error during batch figure OCR: {str(e)}. Please check the figure content and try again."
        )
