# SPDX-License-Identifier: Apache-2.0
# Document Understanding Tasks constains
# after migration it will has these 4 pieces in this function
# 1. Signature detection: Textract
# 2. Figure and Table summarization: Gemini
# 3. Page classification: Gemini
# 4. Structure extraction when skip_ocr is True: Gemini

import asyncio
import json
import time
import os
from tensorlake_docai.pipeline.api import (
    # "ANTHROPIC_API_KEY",
    PageFragmentType,
)
from tensorlake_docai.models.intermediate_objects import ParseResult

from tensorlake_docai.vlm.chart_extraction_utils import (
    run_element_chart_extraction_and_modify_page_elements,
)
from tensorlake_docai.vlm.element_summary_utils import (
    crop_elements,
    run_element_summary_and_modify_page_elements,
)
from tensorlake.applications import function, Retries, cls, RequestContext
from tensorlake.applications import RequestError as RequestException
from tensorlake_docai.vlm.workflow_images import vlm_extraction_image
from tensorlake_docai.pipeline.routing import (
    create_classification_choice_and_prompt,
    skip_ocr_requests,
    vlm_extraction_should_go_to_structured_extraction,
)
from tensorlake_docai.extraction.chunking_functions import ChunkingStrategy
from tensorlake_docai.pipeline.simple_page_creator import ImageDimensions
from tensorlake_docai.pipeline.output_formatter import format_final_output
from tensorlake_docai.extraction.structured_extraction_functions import StructuredExtraction
from tensorlake_docai.extraction.form_extraction_utils import (
    run_element_form_extraction_and_modify_page_elements,
)
from tensorlake_docai.tables.table_cell_grounding_utils import (
    run_element_table_cell_grounding_and_modify_page_elements,
)
from tensorlake_docai.vlm.figure_grounding_utils import (
    run_element_figure_grounding_and_modify_page_elements,
)
from tensorlake_docai.ocr.utils import BatchProcessor

PADDING = 4
TABLE_WHITE_PADDING = 100

SECRETS = [
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "USE_AZURE_OPENAI",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_MODEL_DEPLOYMENT_NAME",
]

MEMORY_IN_GB = 8

TEXT_FRAGMENT_TYPES = [
    PageFragmentType.TITLE,
    PageFragmentType.SECTION_HEADER,
    PageFragmentType.TEXT,
    PageFragmentType.LIST_ITEM,
    PageFragmentType.TABLE_CAPTION,
    PageFragmentType.FIGURE_CAPTION,
    PageFragmentType.FORMULA_CAPTION,
    PageFragmentType.PAGE_FOOTER,
    PageFragmentType.PAGE_HEADER,
    PageFragmentType.PAGE_NUMBER,
]

TABLE_FRAGMENT_TYPES = [
    PageFragmentType.TABLE,
    PageFragmentType.DOCUMENT_INDEX,
]

FORM_FRAGMENT_TYPES = [
    PageFragmentType.FORM,
    PageFragmentType.KEY_VALUE_REGION,
]

FIGURE_FRAGMENT_TYPES = [
    PageFragmentType.FIGURE,
    PageFragmentType.FORMULA,
]


def clean_page_classification_result(raw_result):
    """
    Clean up page classification result to extract page class and reason.

    Handles various formats:
    - Direct class name: "form125"
    - JSON string: '{"page_class": "form125"}'
    - JSON with reason: '{"page_classes": ["form140"], "reason": "..."}'
    - Markdown wrapped JSON: '```json\n{"page_class": "form125"}\n```'
    - JSON array: '["form125", "form126"]' in multi_label classification

    Args:
        raw_result: Raw result from the model

    Returns:
        tuple: (page_class, reason, confidence) where reason and confidence can be None
    """
    if not raw_result:
        return "unknown", None, None

    # Convert to string if not already
    result_str = str(raw_result).strip()

    # Remove markdown code blocks if present
    if result_str.startswith("```"):
        # Find the content between code blocks
        lines = result_str.split("\n")
        content_lines = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                content_lines.append(line)

        result_str = "\n".join(content_lines).strip()

    # Try to parse as JSON
    try:
        parsed = json.loads(result_str)
        if isinstance(parsed, dict):
            reason = parsed.get("reason", None)
            confidence = parsed.get("confidence", None)
            # Clamp confidence to [0.0, 1.0] if present
            if confidence is not None:
                try:
                    confidence = max(0.0, min(1.0, float(confidence)))
                except (TypeError, ValueError):
                    confidence = None
            # Multi-label: return the list if present
            if "page_classes" in parsed:
                return parsed["page_classes"], reason, confidence
            # Multi-class: return the single class if present
            elif "page_class" in parsed:
                return parsed["page_class"], reason, confidence
            else:
                return "unknown", reason, confidence
        else:
            # If it's not a dict, use the parsed value directly
            return str(parsed), None, None
    except (json.JSONDecodeError, ValueError):
        # If JSON parsing fails, return the cleaned string
        return result_str, None, None


def signature_detection(page_image, scale_factor=1.0):
    """
    Detect signatures in a page image using Textract.

    Args:
        page_image: PIL Image object of the page (potentially scaled)
        scale_factor: The scaling factor applied to the image

    Returns:
        list: List of signature elements (PageLayoutElement objects) with coordinates in original document space
    """
    from textractor import Textractor
    from textractor.data.constants import TextractFeatures
    from tensorlake_docai.models.layout_objects import PageLayoutElement
    import time
    import random

    try:
        # Retry logic for rate limiting
        max_retries = 3
        document = None

        for attempt in range(max_retries):
            try:
                extractor = Textractor(region_name="us-west-1")
                document = extractor.analyze_document(
                    file_source=page_image,
                    features=[TextractFeatures.SIGNATURES],
                    save_image=False,
                )
                break  # Success, exit retry loop
            except Exception as e:
                if "ProvisionedThroughputExceededException" in str(e) and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    print(f"Rate limit exceeded, retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Re-raise if not rate limit or max retries reached
                    print(f"Error during Textract signature detection: {e}")
                    raise RequestException(
                        message="Error during signature detection, Please contact Tensorlake support"
                    )

        signatures = document.signatures
        signature_elements = []

        if len(signatures) > 0:
            page_width, page_height = page_image.size
            for signature in signatures:
                # Convert normalized coordinates to absolute coordinates in scaled space
                x, y, w, h = (
                    signature.bbox.x,
                    signature.bbox.y,
                    signature.bbox.width,
                    signature.bbox.height,
                )
                abs_x1 = int(x * page_width)
                abs_y1 = int(y * page_height)
                abs_x2 = int((x + w) * page_width)
                abs_y2 = int((y + h) * page_height)

                # Convert scaled coordinates back to original document coordinates
                original_x1 = int(abs_x1 / scale_factor)
                original_y1 = int(abs_y1 / scale_factor)
                original_x2 = int(abs_x2 / scale_factor)
                original_y2 = int(abs_y2 / scale_factor)

                signature_element = PageLayoutElement(
                    bbox=(original_x1, original_y1, original_x2, original_y2),
                    fragment_type=PageFragmentType.SIGNATURE,
                    score=1.0,
                    reading_order=-1,
                    ocr_text="Signature detected",
                )
                signature_elements.append(signature_element)
        else:
            # Add a "No signature detected" element
            signature_element = PageLayoutElement(
                bbox=(0, 0, 0, 0),
                fragment_type=PageFragmentType.SIGNATURE,
                score=0.0,
                reading_order=-1,
                ocr_text="No signature detected",
            )
            signature_elements.append(signature_element)

        return signature_elements

    except Exception as e:
        print(f"Error during signature detection: {e}")
        # Return a "No signature detected" element instead of raising exception
        # This allows parallel processing to continue even if one page fails
        signature_element = PageLayoutElement(
            bbox=(0, 0, 0, 0),
            fragment_type=PageFragmentType.SIGNATURE,
            score=0.0,
            reading_order=-1,
            ocr_text="No signature detected",
        )
        return [signature_element]


@cls()
class VLMExtractionTask(BatchProcessor):
    def __init__(self):
        # Don't pass batch_size=None explicitly - let BatchProcessor use default
        BatchProcessor.__init__(
            self, memory_gb=MEMORY_IN_GB
        )  # Let SimplePageCreator calculate batch size dynamically

    async def _process_page_extraction(
        self,
        page_layout,
        page_image,
        element_types,
        element_types_to_check,
        extraction_func,
        scale_factor,
        parse_result,
    ):
        cropped_images, elements = crop_elements(
            page_layout=page_layout,
            padding=PADDING,
            page_image=page_image,
            element_types=element_types + element_types_to_check,
            scale_factor=scale_factor,
            save_images=parse_result.request.debug,
        )

        input_tokens, output_tokens = await extraction_func(
            cropped_images=cropped_images,
            page_elements=elements,
            element_types=element_types,
            element_types_to_check=element_types_to_check,
            parse_result=parse_result,
        )

        # Explicitly cleanup cropped images to free memory
        for img in cropped_images:
            try:
                img.close()
            except Exception:
                pass
        del cropped_images

        return input_tokens, output_tokens

    async def _run_extraction_common(
        self,
        parse_result,
        page_images_dict,
        element_types,
        element_types_to_check,
        extraction_func,
        task_name,
        scale_factor=1.0,
    ):
        start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0

        tasks = []

        for page_num, page_image in page_images_dict.items():
            # Find the corresponding page layout
            page_layout = None
            for layout in parse_result.document_layout.pages:
                if layout.page_number == page_num:
                    page_layout = layout
                    break

            if not page_layout:
                continue

            tasks.append(
                self._process_page_extraction(
                    page_layout,
                    page_image,
                    element_types,
                    element_types_to_check,
                    extraction_func,
                    scale_factor,
                    parse_result,
                )
            )

        if tasks:
            results = await asyncio.gather(*tasks)
            for in_tok, out_tok in results:
                total_input_tokens += in_tok
                total_output_tokens += out_tok

        end_time = time.time()
        print(f"Time taken for {task_name}: {end_time - start_time} seconds")
        return total_input_tokens, total_output_tokens

    async def _run_chart_extraction(
        self,
        parse_result,
        page_images_dict,
        element_types,
        element_types_to_check,
        scale_factor=1.0,
    ):
        input_tokens, output_tokens = await self._run_extraction_common(
            parse_result,
            page_images_dict,
            element_types,
            element_types_to_check,
            run_element_chart_extraction_and_modify_page_elements,
            "chart extraction",
            scale_factor,
        )
        # Track chart extraction token usage
        self.chart_extraction_input_tokens += input_tokens
        self.chart_extraction_output_tokens += output_tokens

    async def _run_table_cell_grounding(
        self,
        parse_result,
        page_images_dict,
        element_types,
        element_types_to_check,
        scale_factor=1.0,
    ):
        input_tokens, output_tokens = await self._run_extraction_common(
            parse_result,
            page_images_dict,
            element_types,
            element_types_to_check,
            run_element_table_cell_grounding_and_modify_page_elements,
            "table cell grounding",
            scale_factor,
        )
        # Track table extraction token usage
        self.table_extraction_input_tokens += input_tokens
        self.table_extraction_output_tokens += output_tokens

    async def _run_figure_grounding(
        self, parse_result, page_images_dict, element_types, scale_factor=1.0
    ):
        input_tokens, output_tokens = await self._run_extraction_common(
            parse_result,
            page_images_dict,
            element_types,
            [],  # no element types to check, only figures
            run_element_figure_grounding_and_modify_page_elements,
            "figure grounding",
            scale_factor,
        )
        # Track figure grounding token usage
        if not hasattr(self, "figure_grounding_input_tokens"):
            self.figure_grounding_input_tokens = 0
            self.figure_grounding_output_tokens = 0
        self.figure_grounding_input_tokens += input_tokens
        self.figure_grounding_output_tokens += output_tokens

    async def _run_form_extraction(
        self,
        parse_result,
        page_images_dict,
        element_types,
        element_types_to_check,
        scale_factor=1.0,
    ):
        input_tokens, output_tokens = await self._run_extraction_common(
            parse_result,
            page_images_dict,
            element_types,
            element_types_to_check,
            run_element_form_extraction_and_modify_page_elements,
            "form extraction",
            scale_factor,
        )
        # Track form extraction token usage
        self.form_extraction_input_tokens += input_tokens
        self.form_extraction_output_tokens += output_tokens

    async def _summarize_single_page(
        self,
        page_layout,
        page_image,
        element_types,
        user_prompt,
        save_images,
        include_full_page_image,
        scale_factor,
    ):
        """
        Process summarization for a single page.
        Returns tuple of (input_tokens, output_tokens).
        """
        cropped_images, elements = crop_elements(
            page_layout=page_layout,
            padding=PADDING,
            page_image=page_image,
            element_types=element_types,
            scale_factor=scale_factor,
            save_images=save_images,
        )

        input_tokens, output_tokens = await run_element_summary_and_modify_page_elements(
            cropped_images=cropped_images,
            page_elements=elements,
            page_image=page_image if include_full_page_image else None,
            user_prompt=user_prompt,
            element_types=element_types,
        )

        # Explicitly cleanup cropped images to free memory
        for img in cropped_images:
            try:
                img.close()
            except Exception:
                pass
        del cropped_images

        return input_tokens, output_tokens

    async def _run_element_summarization(
        self,
        parse_result,
        page_images_dict,
        element_types,
        user_prompt,
        element_name,
        scale_factor=1.0,
    ):
        start_time = time.time()

        # Collect all tasks for concurrent processing
        summarization_tasks = []

        for page_num, page_image in page_images_dict.items():
            # Find the corresponding page layout
            page_layout = None
            for layout in parse_result.document_layout.pages:
                if layout.page_number == page_num:
                    page_layout = layout
                    break

            if not page_layout:
                continue

            # Create async task for this page
            summarization_tasks.append(
                self._summarize_single_page(
                    page_layout=page_layout,
                    page_image=page_image,
                    element_types=element_types,
                    user_prompt=user_prompt,
                    save_images=parse_result.request.debug,
                    include_full_page_image=parse_result.request.include_full_page_image,
                    scale_factor=scale_factor,
                )
            )

        # Process all pages concurrently in within a batch
        if summarization_tasks:
            print(
                f"Processing {len(summarization_tasks)} pages concurrently for {element_name} summarization..."
            )
            results_with_tokens = await asyncio.gather(*summarization_tasks)

            # Aggregate token usage
            for input_tokens, output_tokens in results_with_tokens:
                self.summarization_input_tokens += input_tokens
                self.summarization_output_tokens += output_tokens

        end_time = time.time()
        print(f"Time taken for {element_name} summary: {end_time - start_time} seconds")

    async def _prepare_and_run_extraction(
        self, extraction_request, document_pages, page_images_dict, pdf_bytes=None
    ):
        """
        Prepare and run structured extraction for a single extraction request.

        Args:
            extraction_request: StructuredExtractionRequest object
            document_pages: List of page layouts to process for this extraction request
            page_images_dict: Dictionary of page images
            pdf_bytes: Optional PDF bytes for direct PDF processing

        Returns:
            Dict mapping page numbers to extraction results
        """
        from tensorlake_docai.providers.model_provider_utils import _make_gemini_call, run_clients
        import io
        import fitz  # PyMuPDF

        outputs_by_page = {}

        # We pass the schema as a string, because the underlying model providers expect a string
        json_schema = extraction_request.json_schema

        chunk_strategy = extraction_request.chunking_strategy

        # pages to process for this extraction request
        target_pages = document_pages

        page_images_for_extraction = []
        page_numbers_for_extraction = []
        pdf_bytes_for_extraction = []

        # Handle different chunking strategies
        # for skip_ocr, we only need per page chunking or None chunking
        if chunk_strategy == ChunkingStrategy.PAGE.value:
            # Process each page individually
            # For PAGE chunking, use rendered images instead of extracting PDFs
            # (extracting single pages as PDFs creates bloated files with all fonts/resources)
            for page in target_pages:
                page_image = page_images_dict.get(page.page_number)
                if page_image:
                    page_images_for_extraction.append([page_image])
                    page_numbers_for_extraction.append([page.page_number])
                    pdf_bytes_for_extraction.append(None)
        else:
            # Process all pages together
            if pdf_bytes:
                # For PDF, extract subset of pages if needed
                pdf_doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
                target_page_numbers = [page.page_number for page in target_pages]

                if len(target_page_numbers) == pdf_doc.page_count:
                    # All pages - use original PDF
                    subset_pdf_bytes = pdf_bytes
                else:
                    # Extract subset of pages
                    subset_pdf = fitz.open()
                    for page_num in sorted(target_page_numbers):
                        subset_pdf.insert_pdf(pdf_doc, from_page=page_num - 1, to_page=page_num - 1)
                    subset_pdf_bytes = subset_pdf.tobytes()
                    subset_pdf.close()

                pdf_doc.close()
                page_images_for_extraction.append([])
                page_numbers_for_extraction.append(target_page_numbers)
                pdf_bytes_for_extraction.append(subset_pdf_bytes)
            else:
                # Use rendered images
                all_images = [
                    page_images_dict.get(page.page_number)
                    for page in target_pages
                    if page_images_dict.get(page.page_number)
                ]
                actual_page_numbers = [
                    page.page_number
                    for page in target_pages
                    if page_images_dict.get(page.page_number)
                ]
                if all_images:
                    page_images_for_extraction.append(all_images)
                    page_numbers_for_extraction.append(actual_page_numbers)
                    pdf_bytes_for_extraction.append(None)

        # Add a check to ensure there are pages to process
        if not page_images_for_extraction:
            print("No pages found for VLM extraction, skipping.")
            return outputs_by_page

        # Run extraction for each set of images
        structured_extraction_models = [
            _make_gemini_call,
            # _make_oai_call
        ]

        for images, page_numbers, pdf_bytes_chunk in zip(
            page_images_for_extraction, page_numbers_for_extraction, pdf_bytes_for_extraction
        ):
            user_prompt = (
                extraction_request.prompt
                or "Extract information from this document according to the provided JSON schema."
            )

            try:
                # Run the extraction using the model provider utils
                result, input_tokens, output_tokens = await run_clients(
                    user_prompt=user_prompt,
                    images=images,
                    models=structured_extraction_models,
                    json_schema=json_schema,
                    job_type="structured_extraction",
                    pdf_bytes=pdf_bytes_chunk,
                    # timeout=30,  # Use configured timeout from environment variables
                )

                # Track VLM structured extraction tokens
                self.structured_extraction_input_tokens += input_tokens
                self.structured_extraction_output_tokens += output_tokens
                print(f"VLM Extraction tokens - Input: {input_tokens}, Output: {output_tokens}")

                # Store results
                if chunk_strategy == ChunkingStrategy.PAGE.value:
                    # For page-level extraction, store result for single page
                    outputs_by_page[page_numbers[0]] = result
                else:  # Default to combined extraction (DOCUMENT, CLASS, or None)
                    # For combined extraction, store result with a disambiguated key:
                    # ((page1, page2, ...), chunk_idx=0) to avoid confusion with (page, chunk_idx)
                    outputs_by_page[(tuple(page_numbers), 0)] = result

            except Exception as e:
                # Internal logging - detailed for debugging
                print(f"Error during structured extraction: {e}")
                # User-facing error - no internal details
                raise RequestException(
                    message="Data extraction failed. Please try again or contact Tensorlake support with the trace ID of the job."
                )

        return outputs_by_page

    def should_preserve_existing_pages(self) -> bool:
        """
        VLM tasks modify existing page layouts in place rather than creating new ones.
        """
        return True

    def get_processing_batches(
        self, parse_result: ParseResult, image_dimensions: ImageDimensions = ImageDimensions()
    ):
        """
        Override to keep using locally rendered images for VLM tasks.
        """
        for doc_pages in self.page_creator.get_images_generator(parse_result, image_dimensions):
            page_numbers = list(doc_pages.page_images.keys())
            payload = {
                "page_images": dict(doc_pages.page_images),
                "scale_factor": doc_pages.scale_factor,
            }
            yield BatchProcessor.ProcessingBatch(
                page_numbers=page_numbers,
                payload_kind="images",
                payload=payload,
                original_sizes=doc_pages.original_sizes,
            )

    async def process_batch(self, processing_batch, batch_number):
        # Get context for progress tracking
        ctx = RequestContext.get()

        payload = processing_batch.payload or {}
        page_images_dict = payload.get("page_images", {})
        scale_factor = payload.get("scale_factor", 1.0)

        # Store the current request for processing
        current_request = self._current_parse_result.request

        # Process table summarization for this batch
        if current_request.table_summarization:
            await self._run_element_summarization(
                self._current_parse_result,
                page_images_dict,
                [PageFragmentType.TABLE],
                current_request.table_summarization_prompt,
                "table",
                scale_factor,
            )

        # Process figure summarization for this batch
        if current_request.figure_summarization:
            await self._run_element_summarization(
                self._current_parse_result,
                page_images_dict,
                [PageFragmentType.FIGURE],
                current_request.figure_summarization_prompt,
                "figure",
                scale_factor,
            )

        if current_request.chart_extraction:
            await self._run_chart_extraction(
                self._current_parse_result,
                page_images_dict,
                [PageFragmentType.CHART],
                [PageFragmentType.FIGURE, PageFragmentType.TABLE],
                scale_factor,
            )

        if current_request.key_value_extraction:
            await self._run_form_extraction(
                self._current_parse_result,
                page_images_dict,
                FORM_FRAGMENT_TYPES,
                [PageFragmentType.FIGURE, PageFragmentType.TABLE],
                scale_factor,
            )

        if current_request.table_cell_grounding:
            await self._run_table_cell_grounding(
                self._current_parse_result,
                page_images_dict,
                TABLE_FRAGMENT_TYPES + [PageFragmentType.TEXT],
                [],
                scale_factor,
            )

        if current_request.figure_grounding:
            await self._run_figure_grounding(
                self._current_parse_result,
                page_images_dict,
                [PageFragmentType.FIGURE],
                scale_factor,
            )

        # Process signature detection for this batch
        if current_request.detect_signature:
            self._process_signature_detection_batch(page_images_dict, scale_factor, batch_number)

        # Process page classification for this batch
        if current_request.page_classification_request:
            await self._process_page_classification_batch(page_images_dict, batch_number)

        # Process structured extraction for this batch if skip_ocr is enabled
        await self._process_structured_extraction_batch(page_images_dict, batch_number)

        # Cleanup page images to free memory immediately
        for page_num, img in list(page_images_dict.items()):
            try:
                img.close()
            except Exception:
                pass
        page_images_dict.clear()

        # Force garbage collection to free memory
        import gc

        gc.collect()

        ctx.progress.update(current=batch_number, total=batch_number + 1)
        print(f"Batch {batch_number} completed: {len(page_images_dict)} pages processed")
        # Return empty list since we modify pages in-place (should_preserve_existing_pages=True)
        return []

    def _process_signature_detection_batch(self, page_images_dict, scale_factor, batch_number):
        """Process signature detection for a batch of pages."""
        print(
            f"Running signature detection for batch {batch_number}: {len(page_images_dict)} pages"
        )

        signature_start_time = time.time()

        # Collect valid page layouts and images for parallel processing
        valid_items = []
        for page_num, page_image in page_images_dict.items():
            # Find the corresponding page layout
            for layout in self._current_parse_result.document_layout.pages:
                if layout.page_number == page_num:
                    valid_items.append((layout, page_image))
                    break

        if not valid_items:
            return

        # Run signature detection in parallel
        from concurrent.futures import ThreadPoolExecutor

        page_layouts, page_images = zip(*valid_items)
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(
                signature_detection, page_images, [scale_factor] * len(page_images)
            )

        # Add results to page layouts
        for page_layout, signature_elements in zip(page_layouts, results):
            page_layout.elements.extend(signature_elements)
            print(
                f"Batch {batch_number}: Added {len(signature_elements)} signature elements to page {page_layout.page_number}"
            )

        signature_end_time = time.time()
        print(
            f"Batch {batch_number} signature detection time: {signature_end_time - signature_start_time:.2f} seconds"
        )

    async def _process_page_classification_batch(self, page_images_dict, batch_number):
        """Process page classification for a batch of pages."""
        print(
            f"Running page classification for batch {batch_number}: {len(page_images_dict)} pages"
        )

        page_classification_start_time = time.time()

        # Get classification configuration
        page_classification_schema, _, page_classification_prompt = (
            create_classification_choice_and_prompt(
                self._current_parse_result.request.page_classification_request.class_definitions,
                self._current_parse_result.request.page_classification_request.classification_type,
            )
        )

        from tensorlake_docai.providers.model_provider_utils import (
            _make_gemini_call,
            _make_oai_call,
            run_clients,
        )

        page_classification_models = [_make_gemini_call, _make_oai_call]

        page_classification_tasks = []
        pages_for_classification = []

        for page_num, page_image in page_images_dict.items():
            # Find the corresponding page layout
            page_layout = None
            for layout in self._current_parse_result.document_layout.pages:
                if layout.page_number == page_num:
                    page_layout = layout
                    break

            if page_layout:
                pages_for_classification.append(page_layout)
                page_classification_tasks.append(
                    run_clients(
                        user_prompt=page_classification_prompt,
                        images=[page_image],
                        models=page_classification_models,
                        json_schema=page_classification_schema,
                        job_type="page_classification",
                    )
                )

        print(
            f"Batch {batch_number}: Awaiting {len(page_classification_tasks)} page classification tasks..."
        )
        results_with_tokens = await asyncio.gather(*page_classification_tasks)

        # Extract results and track tokens
        raw_results = []
        for result, input_tokens, output_tokens in results_with_tokens:
            raw_results.append(result)
            self.structured_extraction_input_tokens += input_tokens
            self.structured_extraction_output_tokens += output_tokens

        for page_layout, raw_result in zip(pages_for_classification, raw_results):
            page_class, reason, confidence = clean_page_classification_result(raw_result)
            page_layout.page_class = page_class
            page_layout.classification_reason = reason
            page_layout.classification_confidence = confidence
            # print(f"Batch {batch_number}: Page {page_layout.page_number} classified as: {page_class}")

        page_classification_end_time = time.time()
        print(
            f"Batch {batch_number} page classification time: {page_classification_end_time - page_classification_start_time:.2f} seconds"
        )

    async def _process_structured_extraction_batch(self, page_images_dict, batch_number):
        """Process structured extraction for a batch of pages if skip_ocr is enabled."""
        structured_extraction_requests = skip_ocr_requests(self._current_parse_result.request)

        if not structured_extraction_requests:
            return

        # Split requests by chunking strategy: run PAGE-chunked now with other page-independent tasks, defer None-chunked tasks
        page_chunked_requests = []
        none_chunked_requests = []
        for extraction_request in structured_extraction_requests:
            if extraction_request.chunking_strategy == ChunkingStrategy.PAGE.value:
                page_chunked_requests.append(extraction_request)
            else:
                # Defer None-chunked requests until after all batches complete
                none_chunked_requests.append(extraction_request)

        if not page_chunked_requests:
            # Nothing to do in this batch for structured extraction
            return

        print(
            f"Running structured extraction for batch {batch_number}: {len(page_chunked_requests)} PAGE-chunked requests"
        )

        if not self._current_parse_result.structured_outputs_by_page:
            self._current_parse_result.structured_outputs_by_page = {}

        # Process only PAGE-chunked extraction requests for this batch
        for extraction_request in page_chunked_requests:
            print(f"Batch {batch_number}: Processing extraction request ")
            # print(
            #     f"Batch {batch_number}: Processing extraction request for schema: {extraction_request.schema_name}"
            # )

            # Filter pages for this batch that match the extraction criteria
            pages_to_process = []
            if extraction_request.page_classes:
                # Filter pages in this batch that match the specified page classes
                for page_num in page_images_dict.keys():
                    # Find the corresponding page layout
                    page_layout = None
                    for layout in self._current_parse_result.document_layout.pages:
                        if layout.page_number == page_num:
                            page_layout = layout
                            break

                    if not page_layout:
                        continue

                    # Check if page matches extraction criteria
                    if isinstance(page_layout.page_class, list):
                        if any(
                            page_class in extraction_request.page_classes
                            for page_class in page_layout.page_class
                        ):
                            pages_to_process.append(page_layout)
                    else:
                        if page_layout.page_class in extraction_request.page_classes:
                            pages_to_process.append(page_layout)
            else:
                # If no page_classes are specified, use all pages in this batch
                pages_to_process = [
                    page_layout
                    for page_layout in self._current_parse_result.document_layout.pages
                    if page_layout.page_number in page_images_dict
                ]

            if not pages_to_process:
                print(f"Batch {batch_number}: No pages found")
                continue

            # Run extraction for this batch
            # Pass PDF bytes if input is PDF for direct processing
            pdf_bytes = None
            if self._current_parse_result.request.mime_type == "application/pdf":
                pdf_bytes = self._current_parse_result.request.file_bytes

            outputs_by_page = await self._prepare_and_run_extraction(
                extraction_request, pages_to_process, page_images_dict, pdf_bytes=pdf_bytes
            )

            # Store results
            schema_name = extraction_request.schema_name
            for page_key, result in outputs_by_page.items():
                try:
                    if isinstance(result, str):
                        json_result = result
                    else:
                        json_result = json.dumps(result)
                except Exception as e:
                    raise RequestException(
                        message=f"Error getting valid JSON from structured extraction: {e}"
                    )

                if page_key not in self._current_parse_result.structured_outputs_by_page:
                    self._current_parse_result.structured_outputs_by_page[page_key] = {}
                self._current_parse_result.structured_outputs_by_page[page_key][
                    schema_name
                ] = json_result

    async def _run_deferred_structured_extraction(self, parse_result: ParseResult):
        """
        Run structured extraction for skip_ocr requests with None chunking AFTER all batches complete,
        so page_class info across the whole document is available. To avoid OOM, we re-generate
        the small set of page images (<=5) needed for extraction instead of storing them according to the page number subset.
        """
        structured_extraction_requests = skip_ocr_requests(parse_result.request)

        if not structured_extraction_requests:
            return

        # Filter only None-chunked requests: treat anything not 'page' as none-chunked
        none_chunked_requests = [
            req
            for req in structured_extraction_requests
            if req.chunking_strategy != ChunkingStrategy.PAGE.value
        ]

        if not none_chunked_requests:
            return

        if not parse_result.structured_outputs_by_page:
            parse_result.structured_outputs_by_page = {}

        print(
            f"\n=== Running deferred structured extraction for {len(none_chunked_requests)} None-chunked requests ==="
        )

        # Helper to regenerate images for a specific set of page numbers (keeps memory minimal)
        def regenerate_images_for_pages(target_page_numbers):
            # Use SimplePageCreator's pages_to_parse to regenerate only needed pages
            original_pages_to_parse = getattr(parse_result.request, "pages_to_parse", None)
            try:
                parse_result.request.pages_to_parse = list(sorted(set(target_page_numbers)))
                regenerated = {}
                for page_batch in self.page_creator.get_images_generator(parse_result):
                    # Because pages_to_parse is limited (<=5), a single yield should cover it
                    for pn, im in page_batch.page_images.items():
                        regenerated[pn] = im
                    break
                return regenerated
            finally:
                # Restore original pages_to_parse to avoid side effects
                parse_result.request.pages_to_parse = original_pages_to_parse

        # Process each deferred request
        for extraction_request in none_chunked_requests:
            print(f"Deferred SE: Processing schema {extraction_request.schema_name}")

            # Build page_class -> [pages] map across entire document
            page_class_to_pages = {}
            if extraction_request.page_classes:
                for page_layout in parse_result.document_layout.pages:
                    classes = page_layout.page_class
                    if isinstance(classes, list):
                        for cls_name in classes:
                            if cls_name in extraction_request.page_classes:
                                page_class_to_pages.setdefault(cls_name, []).append(page_layout)
                    else:
                        if classes in extraction_request.page_classes:
                            page_class_to_pages.setdefault(classes, []).append(page_layout)
            else:
                # No page_classes: all pages in one bucket
                page_class_to_pages["__all__"] = parse_result.document_layout.pages

            # Iterate per class bucket
            for cls_name, pages in page_class_to_pages.items():
                if not pages:
                    continue
                if cls_name != "__all__" and len(pages) > 5:
                    # print(f"Warning: Skipping page class '{cls_name}' with {len(pages)} pages (>5) for None-chunk SE")
                    print(
                        f"Warning: Skipping extraction with {len(pages)} pages (>5) for None-chunk SE"
                    )
                    continue
                if cls_name == "__all__" and len(pages) > 5:
                    # For no page_classes case, enforce <=5 here (early error was handled earlier, but keep guard)
                    print(
                        f"Warning: Skipping None-chunk SE for all pages because page count is {len(pages)} (>5)"
                    )
                    continue

                # Run extraction with None chunking across the selected pages in this class bucket
                # For PDFs, use direct PDF processing without regenerating images
                target_page_numbers = [pl.page_number for pl in pages]

                if parse_result.request.mime_type == "application/pdf":
                    # Use PDF directly - no need to regenerate images
                    pdf_bytes = parse_result.request.file_bytes
                    page_images_dict = {}  # Empty dict since we're using PDF
                else:
                    # For non-PDF, regenerate images as before
                    pdf_bytes = None
                    regenerated_images = regenerate_images_for_pages(target_page_numbers)
                    page_images_dict = {
                        pn: regenerated_images.get(pn)
                        for pn in target_page_numbers
                        if regenerated_images.get(pn) is not None
                    }

                    if not page_images_dict:
                        print("Deferred SE: No images regenerated for selected pages")
                        continue

                outputs_by_page = await self._prepare_and_run_extraction(
                    extraction_request,
                    pages,
                    page_images_dict,
                    pdf_bytes=pdf_bytes,
                )

            # Store results
            schema_name = extraction_request.schema_name
            for page_key, result in outputs_by_page.items():
                try:
                    if isinstance(result, str):
                        json_result = result
                    else:
                        json_result = json.dumps(result)
                except Exception as e:
                    raise RequestException(
                        message=f"Error getting valid JSON from structured extraction: {e}"
                    )

                if page_key not in parse_result.structured_outputs_by_page:
                    parse_result.structured_outputs_by_page[page_key] = {}
                parse_result.structured_outputs_by_page[page_key][schema_name] = json_result

    async def _classify_pages_with_text(self, parse_result: ParseResult):
        """
        Perform page classification using only text content.
        For text-based input files (text/plain, text/csv, etc.), no OCR is performed.
        Text content is already extracted during file conversion.
        """
        print(
            f"Running text-only page classification for {len(parse_result.document_layout.pages)} pages"
        )

        page_classification_start_time = time.time()

        # Get classification configuration
        page_classification_schema, _, page_classification_prompt = (
            create_classification_choice_and_prompt(
                parse_result.request.page_classification_request.class_definitions,
                parse_result.request.page_classification_request.classification_type,
            )
        )

        from tensorlake_docai.providers.model_provider_utils import (
            _make_gemini_call,
            _make_oai_call,
            run_clients,
        )

        page_classification_models = [_make_gemini_call, _make_oai_call]

        page_classification_tasks = []
        pages_for_classification = []

        for page_layout in parse_result.document_layout.pages:
            # For text files, element.ocr_text contains the raw text content (no OCR performed)
            page_text = "\n".join(
                element.ocr_text
                for element in page_layout.elements
                if element.ocr_text and element.ocr_text.strip()
            )

            # Create text-only prompt with page content
            text_classification_prompt = (
                f"{page_classification_prompt}\n\nPage content:\n{page_text}"
            )

            pages_for_classification.append(page_layout)
            page_classification_tasks.append(
                run_clients(
                    user_prompt=text_classification_prompt,
                    images=[],  # No images for text-only classification
                    models=page_classification_models,
                    json_schema=page_classification_schema,
                    job_type="page_classification",
                )
            )

        print(f"Awaiting {len(page_classification_tasks)} text-only page classification tasks...")
        results_with_tokens = await asyncio.gather(*page_classification_tasks)

        # Extract results and track tokens
        raw_results = []
        for result, input_tokens, output_tokens in results_with_tokens:
            raw_results.append(result)
            self.structured_extraction_input_tokens += input_tokens
            self.structured_extraction_output_tokens += output_tokens

        for page_layout, raw_result in zip(pages_for_classification, raw_results):
            page_class, reason, confidence = clean_page_classification_result(raw_result)
            page_layout.page_class = page_class
            page_layout.classification_reason = reason
            page_layout.classification_confidence = confidence

        page_classification_end_time = time.time()
        print(
            f"Text-only page classification time: {page_classification_end_time - page_classification_start_time:.2f} seconds"
        )

    @function(
        image=vlm_extraction_image,
        timeout=30 * 60,  # 30 minutes
        cpu=2,
        memory=MEMORY_IN_GB,
        # The function is not using /tmp disk space, just reserve a small amount
        ephemeral_disk=2,
        secrets=SECRETS,
        retries=Retries(max_retries=2),
        max_containers=200,
        min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
    )
    def run(self, parse_result: ParseResult) -> ParseResult:
        # Initialize token tracking at the start of each run
        self.summarization_input_tokens = 0
        self.summarization_output_tokens = 0
        self.structured_extraction_input_tokens = 0
        self.structured_extraction_output_tokens = 0
        self.chart_extraction_input_tokens = 0
        self.chart_extraction_output_tokens = 0
        self.table_extraction_input_tokens = 0
        self.table_extraction_output_tokens = 0
        self.form_extraction_input_tokens = 0
        self.form_extraction_output_tokens = 0
        self.figure_grounding_input_tokens = 0
        self.figure_grounding_output_tokens = 0

        # Handle text-only inputs (text/plain, text/csv, text/html, etc.)
        # NO OCR NEEDED: Text content is already extracted in file_convertor.py
        # Pages already contain text in element.ocr_text field (despite the name, no OCR is performed)
        if (
            parse_result.request.mime_type.startswith("text/")
            and parse_result.request.page_classification_request
        ):
            print("🔧 Running text-only page classification (no OCR, no images)...")
            asyncio.run(self._classify_pages_with_text(parse_result))

            # Update token usage
            if parse_result.usage:
                parse_result.usage.extraction_input_tokens_used = (
                    self.structured_extraction_input_tokens
                )
                parse_result.usage.extraction_output_tokens_used = (
                    self.structured_extraction_output_tokens
                )
            else:
                from tensorlake_docai.pipeline.api import Usage

                parse_result.usage = Usage(
                    pages_parsed=0,
                    extraction_input_tokens_used=self.structured_extraction_input_tokens,
                    extraction_output_tokens_used=self.structured_extraction_output_tokens,
                    summarization_input_tokens_used=0,
                    summarization_output_tokens_used=0,
                )

            # For text-only inputs, return early after classification (skip all image-based VLM tasks)
            # Node-by-node routing decisions
            if vlm_extraction_should_go_to_structured_extraction(parse_result.request):
                print("🔀 VLM_EXTRACTION (text-only) → StructuredExtraction")
                return StructuredExtraction().run.future(parse_result)
            else:
                print("🔀 VLM_EXTRACTION (text-only) → OutputFormatter")
                return format_final_output(parse_result)

        # Determine if any task requires page images based on skip_ocr flag and vlm related tasks
        skip_ocr_reqs = skip_ocr_requests(parse_result.request)
        skip_ocr = bool(skip_ocr_reqs)

        needs_images = (
            parse_result.request.table_summarization
            or parse_result.request.figure_summarization
            or parse_result.request.chart_extraction
            or parse_result.request.table_cell_grounding
            or parse_result.request.key_value_extraction
            or parse_result.request.figure_grounding
            or parse_result.request.detect_signature
            or parse_result.request.page_classification_request
            or skip_ocr
        )

        if needs_images:
            print("🔧 Using batch processing for VLM tasks...")

            # Store current parse result for batch processing methods to access
            self._current_parse_result = parse_result

            # Initialize page layouts for skip_ocr case before batch processing
            if skip_ocr and not parse_result.document_layout.pages:
                print("🔧 Creating page layouts for VLM task since skip_ocr=True...")
                from tensorlake_docai.models.layout_objects import PageLayout

                # Create basic page layouts for each page using existing total_pages
                total_pages = parse_result.document_layout.total_pages
                for page_num in range(1, total_pages + 1):
                    page_layout = PageLayout(
                        page_number=page_num,
                        elements=[],  # No OCR elements, VLM will work directly with images
                        shape=(1, 1),  # Will be updated during batch processing
                        page_dimensions={"width": 1, "height": 1},
                    )
                    parse_result.document_layout.pages.append(page_layout)

                print(
                    f"✅ Created {len(parse_result.document_layout.pages)} page layouts for skip_ocr."
                )

            # Validation for structured extraction with skip_ocr when chunking is None
            if skip_ocr:
                # Check if any skip_ocr request has None chunking
                for extraction_request in skip_ocr_reqs:
                    # If chunking is None:
                    # - If page_classes specified: handle per-class later (no early error here)
                    # - If no page_classes: enforce global <=5 pages rule early
                    if (
                        not extraction_request.chunking_strategy
                        and not extraction_request.page_classes
                    ):
                        if parse_result.document_layout.total_pages > 5:
                            raise RequestException(
                                message=f"Structured extraction with skip_ocr=True and no chunking strategy is only supported for documents with 5 or fewer pages. "
                                f"This document has {parse_result.document_layout.total_pages} pages. "
                                f"Please specify a chunking_strategy (e.g., 'page') to process this document."
                            )

            # Run batch processing - this will process each batch through process_batch method
            ctx: RequestContext = RequestContext.get()
            image_dimensions = ImageDimensions(target_dpi=200, upgrade_image_dpi=False)
            parse_result = self.run_batch_processing(ctx, parse_result, image_dimensions)

            print("✅ Completed VLM batch processing")

            # After all batches, run deferred structured extraction (None chunking) once,
            # so page_class from all pages is available.
            if skip_ocr:
                asyncio.run(self._run_deferred_structured_extraction(parse_result))

            # Clean up current parse result reference
            if hasattr(self, "_current_parse_result"):
                delattr(self, "_current_parse_result")

        # Update usage with both summarization and structured extraction token information
        if parse_result.usage:
            # Set summarization token usage (summarization only happens here)
            parse_result.usage.summarization_input_tokens_used = (
                self.summarization_input_tokens
                + self.chart_extraction_input_tokens
                + self.form_extraction_input_tokens
                + self.table_extraction_input_tokens
                + self.figure_grounding_input_tokens
            )
            parse_result.usage.summarization_output_tokens_used = (
                self.summarization_output_tokens
                + self.chart_extraction_output_tokens
                + self.form_extraction_output_tokens
                + self.table_extraction_output_tokens
                + self.figure_grounding_output_tokens
            )

            # Set structured extraction token usage (structured extraction starts here)
            parse_result.usage.extraction_input_tokens_used = (
                self.structured_extraction_input_tokens
            )
            parse_result.usage.extraction_output_tokens_used = (
                self.structured_extraction_output_tokens
            )
        else:
            # Create new usage object with both token types
            from tensorlake_docai.pipeline.api import Usage

            parse_result.usage = Usage(
                pages_parsed=0,  # Will be set by other workflow components
                extraction_input_tokens_used=self.structured_extraction_input_tokens,
                extraction_output_tokens_used=self.structured_extraction_output_tokens,
                summarization_input_tokens_used=self.summarization_input_tokens
                + self.chart_extraction_input_tokens
                + self.form_extraction_input_tokens
                + self.table_extraction_input_tokens
                + self.figure_grounding_input_tokens,
                summarization_output_tokens_used=self.summarization_output_tokens
                + self.chart_extraction_output_tokens
                + self.form_extraction_output_tokens
                + self.table_extraction_output_tokens
                + self.figure_grounding_output_tokens,
            )

        print(
            f"Total summarization tokens used - Input: {self.summarization_input_tokens}, Output: {self.summarization_output_tokens}"
        )
        print(
            f"Total structured extraction tokens used - Input: {self.structured_extraction_input_tokens}, Output: {self.structured_extraction_output_tokens}"
        )

        # Node-by-node routing decisions
        if vlm_extraction_should_go_to_structured_extraction(parse_result.request):
            print("🔀 VLM_EXTRACTION → StructuredExtraction")
            return StructuredExtraction().run.future(parse_result)

        else:
            # Default fallback
            print("🔀 VLM_EXTRACTION → OutputFormatter (default fallback)")
            return format_final_output(parse_result)
