# SPDX-License-Identifier: Apache-2.0
"""
Local-GPU Figure OCR task using the Ovis VLM for classification and extraction.

- Two-stage pipeline: classification -> type-specific extraction
- Figure types: BARCODE, CHART, DIAGRAM, FORM, TABLE, OTHER
- Barcode detection and decoding (pyzbar, zxing)
- Base64 image encoding option
- Async batch processing for efficiency
- Runs as a dedicated GPU function alongside DotsOCRTask
"""

import os
from typing import List, Dict
from PIL import Image

from tensorlake.applications import function, Retries, cls
from tensorlake.applications import RequestError as RequestException
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.vlm.workflow_images import ocr_gpu_cuda_image
from tensorlake_docai.pipeline.output_formatter import format_final_output
from tensorlake_docai.prompts.dotsocr_prompts import (
    FIGURE_CLASSIFICATION_PROMPT,
    FIGURE_CHART_EXTRACTION_PROMPT,
    FIGURE_DIAGRAM_EXTRACTION_PROMPT,
    FIGURE_FORM_EXTRACTION_PROMPT,
    FIGURE_TABLE_EXTRACTION_PROMPT,
    FIGURE_CAPTION_PROMPT,
)
from tensorlake_docai.pipeline.api import PageFragmentType
from tensorlake_docai.pipeline.routing import (
    pil_image_to_base64,
    should_route_to_table_merging,
    dots_ocr_should_go_to_output_formatter,
    dots_ocr_should_go_to_vlm_extraction,
    dots_ocr_should_go_to_structured_extraction,
)
from tensorlake_docai.ocr.utils import BatchProcessor
from tensorlake_docai.tables.table_merging import TableMerging
from tensorlake_docai.vlm.cloud import VLMExtractionTask
from tensorlake_docai.extraction.structured_extraction_functions import StructuredExtraction

SECRETS: list[str] = []

# GPU-specific config for Ovis
OVIS_GPU_MEMORY_UTILIZATION = 0.85
OVIS_MEMORY_IN_GB = int(os.getenv("OVIS_MEMORY_IN_GB", "24"))  # Ovis needs less memory than DotsOCR
GPU_MODELS = ["H100", "A100-80GB"]


@cls()
class OvisFigureOCRTask(BatchProcessor):
    """
    Tensorlake function for GPU-based figure OCR using the Ovis model.
    Runs on a separate GPU container from DotsOCR for better resource utilization.

    Two-stage pipeline:
    1. Classify figures (BARCODE, CHART, DIAGRAM, FORM, TABLE, OTHER)
    2. Extract content with type-specific prompts
    """

    def __init__(self):
        # Initialize BatchProcessor parent class
        # Don't pass batch_size=None explicitly - let BatchProcessor handle it
        BatchProcessor.__init__(self, memory_gb=OVIS_MEMORY_IN_GB)

        self.model_dir = None
        self.llm = None
        self.tokenizer = None

        # Sampling parameters for different tasks
        self.classification_sampling_params = None
        self.figure_sampling_params = None
        self.diagram_sampling_params = None
        self.caption_sampling_params = None

        # Prompt template (Ovis2.5 manual format - apply_chat_template causes vLLM multimodal issues)
        # Adding empty <think></think> tags to disable thinking mode
        self.figure_ocr_prompt_template = "<|im_start|>user\n\n<image>\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def should_preserve_existing_pages(self) -> bool:
        """
        Override BatchProcessor method to preserve existing pages from DotsOCR.
        We're updating figure elements in place, not creating new pages.
        """
        return True

    async def process_batch(self, processing_batch, batch_number):
        """
        Process a batch of page images to extract and OCR figures.
        Implementation of abstract method from BatchProcessor.

        Args:
            processing_batch: Batch containing page images and metadata
            batch_number: Current batch number

        Returns:
            List of processed layouts (not used, figures are updated in place)
        """

        # Extract images from processing batch
        page_numbers = processing_batch.page_numbers
        page_images = processing_batch.payload  # List of PIL images
        original_sizes = processing_batch.original_sizes  # Original PDF/image sizes

        print(f"\n=== Processing figure batch {batch_number}: {len(page_images)} pages ===")

        # Build images_by_page dict
        images_by_page = {pn: img for pn, img in zip(page_numbers, page_images)}

        # Get original page sizes (PDF dimensions)
        pdf_sizes = {
            pn: original_sizes.get(pn, (img.width, img.height))
            for pn, img in images_by_page.items()
        }

        # Pre-build layout lookup dict for O(1) access instead of O(n) search
        layouts_by_page = {
            page_layout.page_number: page_layout
            for page_layout in self._parse_result.document_layout.pages
        }

        # Find layouts for these pages and collect figures
        figures_to_process = []
        figure_elements = []
        figure_metadata = []

        for page_num in page_numbers:
            if page_num not in images_by_page:
                continue

            # O(1) lookup instead of O(n) search
            layout = layouts_by_page.get(page_num)
            if not layout:
                continue

            page_image = images_by_page[page_num]
            # Derive PDF → image scaling from actual sizes
            pdf_width, pdf_height = pdf_sizes.get(page_num, (page_image.width, page_image.height))
            sx = page_image.width / pdf_width if pdf_width else 1.0
            sy = page_image.height / pdf_height if pdf_height else 1.0

            figure_count = 0

            for element in layout.elements:
                if element.fragment_type == PageFragmentType.FIGURE:
                    figure_count += 1
                    bbox = element.bbox  # (x1_pdf, y1_pdf, x2_pdf, y2_pdf)

                    cropped_img = self._crop_with_padding(page_image, bbox, sx, sy, padding=0)
                    if cropped_img:
                        figures_to_process.append(cropped_img)
                        figure_elements.append(element)
                        figure_metadata.append(
                            {
                                "page": page_num,
                                "figure": figure_count,
                                "bbox": bbox,
                                "sx": sx,
                                "sy": sy,
                            }
                        )

        if not figures_to_process:
            print(f"No figures found in batch {batch_number}")
            return []

        print(f"Collected {len(figures_to_process)} figures in batch {batch_number} for Ovis OCR")

        try:
            # Process figures using sync vLLM engine
            self._process_figures_batch(
                figures_to_process,
                figure_elements,
                figure_metadata,
                images_by_page,
                batch_number,
            )
        except Exception as e:
            import traceback

            print(
                f"⚠️ Warning: Figure OCR failed for batch {batch_number}, continuing with page OCR results: {e}"
            )
            traceback.print_exc()

            req = self._parse_result.request
            include_image_in_output = (
                req.include_images if hasattr(req, "include_images") else False
            )
            if include_image_in_output:
                try:
                    for element, cropped_img in zip(figure_elements, figures_to_process):
                        element.image_base64 = pil_image_to_base64(cropped_img)
                    print(
                        f"Encoded {len(figure_elements)} figure images as base64 (figure OCR failed but page OCR output preserved)"
                    )
                except Exception as img_err:
                    print(f"⚠️ Warning: Failed to encode figure images after OCR failure: {img_err}")

        return []

    def _initialize_model(self):
        """Initialize the Ovis model for GPU processing."""
        if self.llm is not None:
            # Already initialized
            return

        print("🔄 Initializing Ovis model for GPU figure OCR...")

        import gc
        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(
                f"GPU memory before model load: {free_mem / 1024**3:.2f} GiB free / {total_mem / 1024**3:.2f} GiB total"
            )

        # Download model if needed
        self.model_dir = self._download_model()

        from vllm import SamplingParams, LLM
        from vllm.sampling_params import StructuredOutputsParams
        from transformers import AutoTokenizer

        print(f"Initializing Ovis from: {self.model_dir}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)

        # Initialize sync vLLM engine (Tensorlake recreates event loops per request,
        # which kills AsyncLLMEngine)
        self.llm = LLM(
            model=self.model_dir,
            max_model_len=8192,
            dtype="bfloat16",
            max_num_batched_tokens=16384,
            max_num_seqs=128,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            limit_mm_per_prompt={"image": 1, "video": 0},
            gpu_memory_utilization=OVIS_GPU_MEMORY_UTILIZATION,
        )

        # Sampling params for classification with structured outputs (vLLM 0.13.0+)
        structured_outputs_params = StructuredOutputsParams(
            choice=["BARCODE", "CHART", "DIAGRAM", "FORM", "TABLE", "OTHER"]
        )
        self.classification_sampling_params = SamplingParams(
            temperature=0.0, max_tokens=20, structured_outputs=structured_outputs_params
        )

        # Regular sampling params for general figure OCR (CHART, FORM, TABLE)
        self.figure_sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=0, max_tokens=4096, repetition_penalty=1.1
        )

        # Stricter sampling params for DIAGRAM to prevent repetition hallucination
        self.diagram_sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=0, max_tokens=2048, repetition_penalty=1.2
        )

        # Compact sampling params for OTHER (captions/simple figures)
        self.caption_sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=0, max_tokens=512, repetition_penalty=1.1
        )

        print("✅ Ovis GPU model initialized successfully")

    def _download_model(self):
        """Download the Ovis model from Hugging Face Hub if needed."""
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id="AIDC-AI/Ovis2.5-9B")

    def _classify_figures(self, cropped_images: List[Image.Image]) -> List[tuple]:
        """
        Classify all figures into types: BARCODE, CHART, DIAGRAM, FORM, TABLE, OTHER
        Uses structured outputs with choice constraint for guaranteed valid output.

        Args:
            cropped_images: List of cropped figure images

        Returns:
            List of tuples: (index, figure_type, input_tokens, output_tokens)
        """

        print(f"📋 Step 1: Classifying {len(cropped_images)} figures with guided decoding...")

        from vllm import TextPrompt

        # Batch all classification requests in a single generate() call
        formatted_prompt = self.figure_ocr_prompt_template.replace(
            "{user_prompt}", FIGURE_CLASSIFICATION_PROMPT
        )
        prompts = [
            TextPrompt(prompt=formatted_prompt, multi_modal_data={"image": img})
            for img in cropped_images
        ]

        batch_outputs = self.llm.generate(
            prompts, sampling_params=self.classification_sampling_params
        )

        results = []
        for idx, output in enumerate(batch_outputs):
            result_text = output.outputs[0].text if output.outputs else ""
            in_tokens = len(output.prompt_token_ids)
            out_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
            figure_type = result_text.strip()
            results.append((idx, figure_type, in_tokens, out_tokens))

        return results

    @staticmethod
    def _crop_with_padding(image, bbox, sx, sy, padding=0):
        """
        Crop image using PDF bbox coordinates with optional padding.

        Args:
            image: PIL Image to crop from
            bbox: Bounding box in PDF coordinates (x1, y1, x2, y2)
            sx: X-axis scaling factor (image_width / pdf_width)
            sy: Y-axis scaling factor (image_height / pdf_height)
            padding: Number of pixels to add as padding on all sides

        Returns:
            Cropped PIL Image or None if invalid crop
        """
        x1 = int(bbox[0] * sx)
        y1 = int(bbox[1] * sy)
        x2 = int(bbox[2] * sx)
        y2 = int(bbox[3] * sy)

        # Apply padding
        if padding > 0:
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)

        if x2 > x1 and y2 > y1:
            return image.crop((x1, y1, x2, y2))
        return None

    def _process_figures_batch(
        self,
        figures_to_process: List[Image.Image],
        figure_elements: List,
        figure_metadata: List[Dict],
        images_by_page: Dict[int, Image.Image],
        batch_number: int = 0,
    ):
        """
        Process a batch of cropped figure images using Ovis for OCR.
        Updates the figure elements in place with OCR text.

        Args:
            figures_to_process: List of cropped figure images
            figure_elements: List of corresponding PageElement objects
            figure_metadata: List of metadata dicts (page, figure, bbox, sx, sy)
            images_by_page: Dict mapping page numbers to PIL images (for barcode re-cropping)
            batch_number: Current batch number for logging
        """
        import time

        if not figures_to_process:
            return

        print(f"Processing {len(figures_to_process)} figures in batch {batch_number}")
        # Step 1: Classify all figures in this batch
        t0 = time.time()
        classifications = self._classify_figures(figures_to_process)

        # Extract classification results
        figure_types = {}  # idx -> figure_type
        total_input_tokens = 0
        total_output_tokens = 0

        for idx, figure_type, in_tok, out_tok in classifications:
            figure_types[idx] = figure_type
            total_input_tokens += in_tok
            total_output_tokens += out_tok

        print(f"Classification results: {dict((k, v) for k, v in sorted(figure_types.items()))}")

        # Step 2: Process all figures concurrently with type-specific prompts
        prompt_map = {
            "CHART": FIGURE_CHART_EXTRACTION_PROMPT,
            "DIAGRAM": FIGURE_DIAGRAM_EXTRACTION_PROMPT,
            "FORM": FIGURE_FORM_EXTRACTION_PROMPT,
            "TABLE": FIGURE_TABLE_EXTRACTION_PROMPT,
            "OTHER": FIGURE_CAPTION_PROMPT,
        }

        params_map = {
            "DIAGRAM": self.diagram_sampling_params,  # Stricter for diagrams
            "OTHER": self.caption_sampling_params,  # Compact for captions
        }

        # Batch extraction: group figures by (prompt, sampling_params) for efficient batched generate()
        from vllm import TextPrompt
        from collections import defaultdict

        print(f"🚀 Processing all {len(figures_to_process)} figures in batches by type...")
        figure_results = []

        # Separate barcodes (no inference needed) from figures needing extraction
        groups = defaultdict(list)  # (prompt_text, figure_type) -> [(idx, image)]
        for idx in range(len(figures_to_process)):
            figure_type = figure_types[idx]
            if figure_type == "BARCODE":
                figure_results.append((idx, "FIGURE_TYPE: BARCODE", 0, 0))
                continue
            prompt = prompt_map.get(figure_type, FIGURE_CAPTION_PROMPT)
            groups[(prompt, figure_type)].append((idx, figures_to_process[idx]))

        # Process each group as a batch
        for (prompt, figure_type), items in groups.items():
            sampling_params = params_map.get(figure_type, self.figure_sampling_params)
            formatted_prompt = self.figure_ocr_prompt_template.replace("{user_prompt}", prompt)
            prompts = [
                TextPrompt(prompt=formatted_prompt, multi_modal_data={"image": img})
                for _, img in items
            ]
            batch_outputs = self.llm.generate(prompts, sampling_params=sampling_params)

            for (idx, _), output in zip(items, batch_outputs):
                text = output.outputs[0].text if output.outputs else ""
                in_tok = len(output.prompt_token_ids)
                out_tok = len(output.outputs[0].token_ids) if output.outputs else 0
                figure_results.append(
                    (idx, f"FIGURE_TYPE: {figure_type}\nCONTENT:\n{text}", in_tok, out_tok)
                )

        # Collect results in order
        results = [None] * len(figures_to_process)
        for idx, text, in_tok, out_tok in figure_results:
            results[idx] = text
            total_input_tokens += in_tok
            total_output_tokens += out_tok

        dt = time.time() - t0
        print(f"⏱️ Ovis figure OCR batch {batch_number} completed in {dt:.2f}s")
        print(f"Token usage - Input: {total_input_tokens}, Output: {total_output_tokens}")

        # Get request parameters once before loop (not for each figure)
        req = self._parse_result.request
        include_image_in_output = req.include_images if hasattr(req, "include_images") else False
        detect_barcode = req.detect_barcode if hasattr(req, "detect_barcode") else False

        # Merge results back into elements
        import re

        for idx, (element, text, cropped_img, meta) in enumerate(
            zip(figure_elements, results, figures_to_process, figure_metadata)
        ):
            # Parse FIGURE_TYPE from structured output if present
            is_barcode = False
            is_table = False
            is_form = False
            figure_type = None

            if "FIGURE_TYPE:" in text:
                for line in text.split("\n"):
                    if line.startswith("FIGURE_TYPE:"):
                        figure_type = line.replace("FIGURE_TYPE:", "").strip()
                        is_barcode = figure_type == "BARCODE"
                        is_table = figure_type == "TABLE"
                        is_form = figure_type == "FORM"
                        break

            # TABLE and FORM prompts can both return HTML-like structured content.
            uses_html = is_table or is_form

            # Extract clean content based on type
            clean_text = text
            if uses_html:
                matches = re.findall(r"<table.*?</table>", text, re.DOTALL | re.IGNORECASE)
                clean_text = "\n".join(matches) if matches else text

            # Store content in element
            if uses_html:
                from markdownify import markdownify as md

                markdown_text = md(clean_text, heading_style="ATX")

                element.ocr_text = clean_text
                element.html = clean_text
                element.markdown = (
                    f"{element.markdown}\n\n{markdown_text}" if element.markdown else markdown_text
                )
                element.fragment_type = (
                    PageFragmentType.TABLE if is_table else PageFragmentType.FORM
                )
                print(
                    f"   ✓ Figure {idx} → {figure_type} (Ovis classification, converted to markdown)"
                )
            else:
                element.ocr_text = clean_text
                element.markdown = (
                    f"{element.markdown}\n\n{clean_text}" if element.markdown else clean_text
                )

            if detect_barcode and is_barcode:
                print(f"🔍 Figure {idx}: Ovis text = '{text[:100]}...'", flush=True)
                # Change fragment type to BARCODE
                element.fragment_type = PageFragmentType.BARCODE

                try:
                    from pyzbar.pyzbar import decode

                    # Re-crop with padding for better barcode detection
                    padded_img = self._crop_with_padding(
                        images_by_page[meta["page"]],
                        meta["bbox"],
                        meta["sx"],
                        meta["sy"],
                        padding=10,
                    )

                    # Read barcode(s) from the image
                    results_pyzbar = decode(padded_img)
                    print(f"Decoded barcode: {results_pyzbar}")

                    if results_pyzbar:
                        decoded_text = f"{results_pyzbar[0].type}: {results_pyzbar[0].data.decode('utf-8', errors='replace')}"
                        print(f"✅ Decoded barcode: {decoded_text}")

                        # Update element with decoded barcode
                        element.ocr_text = decoded_text
                        element.markdown = decoded_text
                    else:
                        print("⚠️ Warning: Pyzbar barcode decoding failed, trying zxing")
                        # Try zxing on the same barcode image
                        try:
                            import zxing

                            reader = zxing.BarCodeReader()
                            result = reader.decode(padded_img)
                            print(f"Decoded barcode with zxing: {result}")
                            if result:
                                decoded_text = f"{result.format}: {result.parsed}"
                                print(f"✅ Decoded barcode with zxing: {decoded_text}")
                                element.ocr_text = decoded_text
                                element.markdown = decoded_text
                            else:
                                print("⚠️ Warning: ZXing barcode decoding failed")
                        except Exception as zxing_err:
                            print(f"⚠️ Warning: ZXing import/decode failed: {zxing_err}")

                except Exception as barcode_err:
                    print(f"⚠️ Warning: Barcode decoding failed: {barcode_err}")

            # Encode image as base64 if requested
            if include_image_in_output:
                try:
                    element.image_base64 = pil_image_to_base64(cropped_img)
                    print(f"Encoded figure image as base64 ({len(element.image_base64)} bytes)")
                except Exception as img_err:
                    print(f"⚠️ Warning: Failed to encode figure image: {img_err}")

        # Clean up large collections to free memory
        del figures_to_process, figure_elements, figure_metadata, images_by_page
        print(f"✅ Figure batch {batch_number} processing completed")

    @function(
        image=ocr_gpu_cuda_image,
        timeout=30 * 60,
        cpu=2,
        memory=OVIS_MEMORY_IN_GB,
        ephemeral_disk=40,
        gpu=GPU_MODELS,
        secrets=SECRETS,
        retries=Retries(max_retries=2),
        min_containers=1,
        max_containers=1,
    )
    def run(self, parse_result: ParseResult) -> ParseResult:
        """
        Core figure OCR logic using batch processing.

        Processes all figures in the document using Ovis model:
        1. Use BatchProcessor to load page images in batches (memory efficient)
        2. For each batch, find figures on those pages
        3. Classify figures by type
        4. Extract content with type-specific prompts
        5. Decode barcodes if requested
        6. Encode images as base64 if requested

        Args:
            parse_result: ParseResult with document layouts containing figures

        Returns:
            ParseResult with updated figure elements
        """
        import torch

        if not torch.cuda.is_available():
            raise RequestException(
                message=(
                    "OvisFigureOCRTask requires a CUDA-equipped worker, but no GPU "
                    "is available. Pick a cloud OCR backend (azure-di, textract, gemini) "
                    "or run on a GPU host."
                )
            )

        print("🔍 Starting Ovis figure OCR processing...")

        # Initialize model
        self._initialize_model()

        # Check if there are any figures to process
        has_figures = any(
            element.fragment_type == PageFragmentType.FIGURE
            for page in parse_result.document_layout.pages
            for element in page.elements
        )

        if not has_figures:
            print("No figures found in document, skipping figure OCR")
            return parse_result

        # Store parse_result for access in process_batch
        from tensorlake.applications import RequestContext

        ctx: RequestContext = RequestContext.get()
        self._parse_result = parse_result

        from tensorlake_docai.pipeline.simple_page_creator import ImageDimensions

        image_dimensions = ImageDimensions(
            min_pixels=3136,
            max_pixels=11289600,
            target_dpi=200,
            upgrade_image_dpi=True,
        )

        # Use BatchProcessor to handle images in batches
        # This will call process_batch for each batch of pages
        parse_result = self.run_batch_processing(ctx, parse_result, image_dimensions)

        print("✅ Figure OCR processing completed")

        # Node-by-node routing decisions
        if should_route_to_table_merging(parse_result.request, parse_result):
            print("🔀 OvisFigureOCRTask → TableMerging")
            return TableMerging().run.future(parse_result)

        elif dots_ocr_should_go_to_output_formatter(parse_result.request, parse_result):
            print("🔀 OvisFigureOCRTask → OutputFormatter")
            return format_final_output(parse_result)

        elif dots_ocr_should_go_to_vlm_extraction(parse_result.request, parse_result):
            print("🔀 OvisFigureOCRTask → VLMExtractionTask")
            return VLMExtractionTask().run.future(parse_result)

        elif dots_ocr_should_go_to_structured_extraction(parse_result.request, parse_result):
            print("🔀 OvisFigureOCRTask → StructuredExtraction")
            return StructuredExtraction().run.future(parse_result)

        else:
            print("🔀 OvisFigureOCRTask → OutputFormatter")
            return format_final_output(parse_result)
