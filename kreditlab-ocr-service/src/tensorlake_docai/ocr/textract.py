# SPDX-License-Identifier: Apache-2.0
import time
import os
from typing import List, Optional
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.vlm.workflow_images import simple_page_creator_image
from tensorlake_docai.pipeline.api import PageFragmentType, Usage
from tensorlake_docai.models.layout_objects import PageLayoutElement, PageLayout
from tensorlake.applications import Retries, cls, function
from tensorlake.applications import RequestError as RequestException

from tensorlake_docai.postprocess.header_correction import correct_document_headers

from tensorlake_docai.pipeline.routing import (
    route_after_ocr,
    handle_processing_error,
    FILE_TYPE_MAPPING,
    pil_image_to_base64,
)

SECRETS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "S3_BUCKET_NAME",
    "OPENAI_API_KEY",
    "USE_AZURE_OPENAI",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_MODEL_DEPLOYMENT_NAME",
]


@cls()
class FullPageTextractTask:  # type: ignore
    """AWS Textract task class that uses TextractMarkdownExtractor for both layout and markdown"""

    def __init__(self):
        super().__init__()
        from tensorlake_docai.ocr.textract_markdown_extractor import TextractMarkdownExtractor

        self.extractor: Optional[TextractMarkdownExtractor] = None
        self._initialize_extractor()

    @function(
        image=simple_page_creator_image,
        timeout=30 * 60,
        cpu=4,
        memory=12,
        ephemeral_disk=2,
        secrets=SECRETS,
        retries=Retries(max_retries=2),
        min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
    )
    def run(self, parse_result: ParseResult) -> ParseResult:
        """Main orchestration method - uses image-based multiprocessing for PDFs and direct processing for images"""

        if parse_result.request.mime_type.startswith("text/"):
            return parse_result

        print("[FTT] Running FullPageTextractTask using AWS Textract")
        start_time = time.time()

        extension = FILE_TYPE_MAPPING.get(parse_result.request.mime_type, None)
        if extension not in ["pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp"]:
            raise RequestException(
                message="unsupported file type: " + parse_result.request.mime_type
            )

        if extension == "pdf":
            print("[FTT] Detected PDF file - using pdf processing with Textract")
            textract_result = self._process_pdf_direct(parse_result, start_time)
        else:
            print("[FTT] Detected image file - using image processing with Textract")
            textract_result = self._process_image_direct(parse_result, start_time)

        # Optional: header correction
        if parse_result.request.xpage_header_detection:
            try:
                textract_result = correct_document_headers(
                    textract_result, api_key=os.getenv("OPENAI_API_KEY")
                )
            except RequestException:
                raise
            except Exception as e:
                print(f"[FTT] Header correction skipped: {e}")

        # textract_result is the same object as parse_result (mutated in place).
        return route_after_ocr(textract_result, log_prefix="FULL_PAGE_TEXTRACT")

    def _process_pdf_direct(self, parse_result: ParseResult, start_time: float) -> ParseResult:
        """Process PDF with AWS Textract using image-based multiprocessing"""
        try:
            file_bytes = parse_result.request.file_bytes
            if isinstance(file_bytes, str):
                file_bytes = file_bytes.encode("utf-8")
            elif file_bytes is None:
                raise RequestException(
                    message="No file data provided. Please ensure a valid file is uploaded."
                )
            # Extractor is initialized in constructor

            requested_pages = parse_result.request.pages_to_parse
            # Preserve original total pages for UI when user selects a subset
            original_total_pages = None
            try:
                import io as _io
                from pypdf import PdfReader as _PdfReader  # type: ignore

                original_total_pages = len(_PdfReader(_io.BytesIO(file_bytes)).pages)
            except Exception:
                original_total_pages = None
            if requested_pages:
                print(f"[FTT] Processing PDF with Textract (S3 async) pages {requested_pages}")
            else:
                print("[FTT] Processing PDF with Textract (S3 async) all pages")
            textract_results = self.extractor.process_pdf_via_s3_bytes(
                file_bytes, pages_to_extract=requested_pages
            )
            total_pages = len(textract_results)
            print(f"[FTT] Textract processed {total_pages} pages")

            # Set usage information
            parse_result.usage = Usage(
                pages_parsed=total_pages,
                extraction_input_tokens_used=0,
                extraction_output_tokens_used=0,
                summarization_input_tokens_used=0,
                summarization_output_tokens_used=0,
            )

            # Create DocumentLayout if it doesn't exist
            if not parse_result.document_layout:
                from tensorlake_docai.models.layout_objects import DocumentLayout

                parse_result.document_layout = DocumentLayout(
                    pages=[],
                    scale_factor=1.0,  # PDF uses 72 DPI scale
                    total_pages=(original_total_pages or total_pages),
                )
            else:
                parse_result.document_layout.scale_factor = 1.0
                parse_result.document_layout.total_pages = original_total_pages or total_pages

            # Build pages with memory-efficient image handling
            if total_pages > 0:
                processed_tps = set()

                # Helper to consolidate processing logic
                def process_page(page_num, response, img=None):
                    layout_representation = self.extractor.extract_page_layout_from_textract_result(
                        response, page_num
                    )
                    page_width, page_height = layout_representation.get("dimensions", [612, 792])
                    page_layout = PageLayout(
                        page_number=page_num,
                        elements=[],
                        shape=(page_height, page_width),
                        page_dimensions={"width": page_width, "height": page_height},
                    )
                    page_elements = self._create_page_elements_from_layout_fragments(
                        layout_representation.get("page_fragments", []), img, page_num
                    )
                    page_layout.elements = page_elements
                    parse_result.document_layout.pages.append(page_layout)

                if parse_result.request.include_images:
                    print("[FTT] Using image generator for figure extraction")
                    from tensorlake_docai.pipeline.simple_page_creator import SimplePageCreator

                    spc = SimplePageCreator(scale_factor=1)

                    for batch in spc.get_images_generator(parse_result):
                        for page_num, img in batch.page_images.items():
                            if page_num in textract_results:
                                process_page(page_num, textract_results[page_num], img)
                                processed_tps.add(page_num)
                        # Clear batch images to reduce memory pressure
                        batch.page_images.clear()

                # Process any remaining pages (those without images or if images weren't requested)
                for page_num, response in textract_results.items():
                    if page_num not in processed_tps:
                        process_page(page_num, response, None)

                # Ensure pages are sorted by page number
                parse_result.document_layout.pages.sort(key=lambda x: x.page_number)

            print(f"[FTT] PDF processing completed successfully - {total_pages} pages processed")

        except Exception as e:
            user_message = handle_processing_error(e, "PDF analysis with Textract", "document")
            self._handle_error(e, user_message)

        end_time = time.time()
        print(f"[FTT] FullPageTextractTask (PDF) completed in {end_time - start_time:.2f} seconds")
        return parse_result

    def _process_image_direct(self, parse_result: ParseResult, start_time: float) -> ParseResult:
        """Process standalone image using AWS Textract"""
        try:
            file_bytes = parse_result.request.file_bytes
            if isinstance(file_bytes, str):
                file_bytes = file_bytes.encode("utf-8")
            elif file_bytes is None:
                raise RequestException(
                    message="No file data provided. Please ensure a valid file is uploaded."
                )

            file_name = parse_result.request.file_name or "unknown_image"
            print(f"[FTT] Processing image with Textract: {file_name}")

            # Extractor is initialized in constructor

            # Process image - returns single layout data dict
            layout_data = self.extractor.process_image(file_bytes)
            page_fragments = layout_data.get("page_fragments", [])

            print(f"[FTT] Textract processed 1 page with {len(page_fragments)} fragments")

            # Set usage information (single page)
            parse_result.usage = Usage(
                pages_parsed=1,
                extraction_input_tokens_used=0,
                extraction_output_tokens_used=0,
                summarization_input_tokens_used=0,
                summarization_output_tokens_used=0,
            )

            # Create DocumentLayout if it doesn't exist
            if not parse_result.document_layout:
                from tensorlake_docai.models.layout_objects import DocumentLayout

                parse_result.document_layout = DocumentLayout(
                    pages=[], scale_factor=1.0, total_pages=1  # Image uses 72 DPI scale
                )
            else:
                parse_result.document_layout.scale_factor = 1.0
                parse_result.document_layout.total_pages = 1

            # Create single page layout
            page_width, page_height = layout_data.get("dimensions", [612, 792])
            page_layout = PageLayout(
                page_number=1,
                elements=[],
                shape=(page_height, page_width),
                page_dimensions={"width": page_width, "height": page_height},
            )

            # Generate page image if we need to include it in output (using generator to avoid OOM)
            page_image = None
            if parse_result.request.include_images:
                print("[FTT] Generating page image for figure extraction")
                from tensorlake_docai.pipeline.simple_page_creator import SimplePageCreator

                spc = SimplePageCreator(scale_factor=1)
                for batch in spc.get_images_generator(parse_result):
                    page_image = batch.page_images.get(1)
                    # Clear batch images to reduce memory pressure
                    batch.page_images.clear()
                    break  # Only need first page for single image processing

            # Create page elements from fragments (with or without image)
            page_elements = self._create_page_elements_from_layout_fragments(
                page_fragments, page_image, 1
            )
            page_layout.elements = page_elements
            parse_result.document_layout.pages.append(page_layout)

            print(
                f"[FTT] Image processing completed successfully - 1 page with {len(page_elements)} elements processed"
            )

        except Exception as e:
            user_message = handle_processing_error(
                e, "Single image analysis with Textract", "image"
            )
            self._handle_error(e, user_message)

        end_time = time.time()
        print(
            f"[FTT] FullPageTextractTask (Image) completed in {end_time - start_time:.2f} seconds"
        )
        return parse_result

    def _initialize_extractor(self):
        """Initialize the Textract extractor with credentials from environment"""
        if self.extractor is None:
            import os
            from tensorlake_docai.ocr.textract_markdown_extractor import TextractMarkdownExtractor

            region = os.getenv("AWS_REGION", "us-east-1")

            try:
                self.extractor = TextractMarkdownExtractor(region_name=region)
            except Exception as e:
                print(f"[FTT] Failed to initialize Textract extractor: {e}")
                raise RequestException(
                    message="Service temporarily unavailable. Please contact Tensorlake support with the trace ID of the job."
                )

    def _create_page_elements_from_layout_fragments(
        self, page_fragments: List[dict], page_image=None, page_number: int = 1
    ) -> List[PageLayoutElement]:
        """Create PageLayoutElement objects from Textract layout fragments"""
        page_elements = []

        for i, fragment in enumerate(page_fragments):
            # Extract data from fragment
            bbox = fragment.get("bbox", {})
            content = fragment.get("content", {})
            fragment_type_str = fragment.get("fragment_type", "text")
            reading_order = fragment.get("reading_order", 0)

            # Extract the actual text content
            ocr_text = content.get("content", "")

            # Skip fragments with no meaningful content
            if not ocr_text or not ocr_text.strip():
                continue

            # Convert string fragment type to PageFragmentType enum
            fragment_type = self._map_string_to_fragment_type(fragment_type_str)

            # Extract hierarchy level for section headers
            hierarchy_level = None
            if (
                fragment_type in [PageFragmentType.SECTION_HEADER, PageFragmentType.TITLE]
                and "level" in content
            ):
                hierarchy_level = content.get("level", 0)

            # Extract confidence score
            confidence = fragment.get("confidence", 1.0)

            # Crop and encode image for FIGURE elements if page_image is provided
            image_base64 = None
            if fragment_type == PageFragmentType.FIGURE and page_image is not None:
                try:
                    x1, y1, x2, y2 = (
                        bbox.get("x1", 0),
                        bbox.get("y1", 0),
                        bbox.get("x2", 0),
                        bbox.get("y2", 0),
                    )
                    # Ensure coordinates are within image bounds
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(page_image.width, int(x2))
                    y2 = min(page_image.height, int(y2))

                    if x2 > x1 and y2 > y1:
                        image_crop = page_image.crop((x1, y1, x2, y2))
                        image_base64 = pil_image_to_base64(image_crop)
                        print(f"[FTT] Encoded figure image, bbox ({x1},{y1},{x2},{y2})")
                except Exception as e:
                    print(f"[FTT] Failed to crop/encode figure image: {e}")

            # Create PageLayoutElement directly
            page_element = PageLayoutElement(
                bbox=(bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)),
                fragment_type=fragment_type,
                score=confidence,  # Use Textract confidence
                reading_order=reading_order,
                ref_id=f"{page_number}.{reading_order}",
                ocr_text=ocr_text.strip(),
                markdown=self._get_markdown_for_element(fragment_type, content, ocr_text.strip()),
                html=content.get("html"),
                hierarchy_level=hierarchy_level,
                image_base64=image_base64,
            )
            page_elements.append(page_element)

        return page_elements

    def _get_markdown_for_element(
        self, fragment_type: PageFragmentType, content: dict, default_text: str
    ) -> str:
        """Get appropriate markdown content based on fragment type"""
        # For headers, use plain text so downstream pipeline can format based on hierarchy
        if fragment_type in [
            PageFragmentType.SECTION_HEADER,
            PageFragmentType.TITLE,
            PageFragmentType.PAGE_HEADER,
            PageFragmentType.PAGE_FOOTER,
        ]:
            return default_text

        # For other content types, use markdown if available
        return content.get("markdown", default_text)

    def _map_string_to_fragment_type(self, fragment_type_str: str) -> PageFragmentType:
        """Map string fragment type to PageFragmentType enum"""
        mapping = {
            "section_header": PageFragmentType.SECTION_HEADER,
            "title": PageFragmentType.TITLE,
            "text": PageFragmentType.TEXT,
            "table": PageFragmentType.TABLE,
            "figure": PageFragmentType.FIGURE,
            "formula": PageFragmentType.FORMULA,
            "form": PageFragmentType.FORM,
            "key_value_region": PageFragmentType.KEY_VALUE_REGION,
            "document_index": PageFragmentType.DOCUMENT_INDEX,
            "list_item": PageFragmentType.LIST_ITEM,
            "table_caption": PageFragmentType.TABLE_CAPTION,
            "figure_caption": PageFragmentType.FIGURE_CAPTION,
            "formula_caption": PageFragmentType.FORMULA_CAPTION,
            "page_footer": PageFragmentType.PAGE_FOOTER,
            "page_header": PageFragmentType.PAGE_HEADER,
            "page_number": PageFragmentType.PAGE_NUMBER,
            "signature": PageFragmentType.SIGNATURE,
        }
        return mapping.get(fragment_type_str, PageFragmentType.TEXT)

    def _handle_error(self, exception: Exception, user_error_message: str = None):
        """App-specific error handling with optional custom user message"""
        import traceback

        stack_trace = traceback.format_exc()

        error_message = f"TextractTask failed: {str(exception)}"
        print(f"❌ {error_message}")
        print(f"📋 Full stack trace:\n{stack_trace}")
        print(f"🔍 Exception type: {type(exception).__name__}")
        print(f"🔍 Exception args: {exception.args}")

        if not user_error_message:
            user_error_message = "Document processing failed. Please try again or contact Tensorlake support with the trace ID of the job."

        raise RequestException(message=user_error_message)
