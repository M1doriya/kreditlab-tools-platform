# SPDX-License-Identifier: Apache-2.0
import os
from typing import Dict, List, Optional, Tuple

from tensorlake_docai.pipeline.api import PageFragmentType, Usage
from tensorlake_docai.postprocess.header_correction import correct_document_headers
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import PageLayout, PageLayoutElement
from tensorlake_docai.pipeline.simple_page_creator import SimplePageCreator
from tensorlake.applications import function, Retries, cls, RequestContext
from tensorlake.applications import RequestError as RequestException
from tensorlake_docai.pipeline.routing import (
    route_after_ocr,
    FILE_TYPE_MAPPING,
    pil_image_to_base64,
)
from tensorlake_docai.vlm.workflow_images import simple_page_creator_image
from tensorlake_docai.providers.error_utils import extract_provider_error_message

SECRETS = [
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
    "AZURE_DOCUMENT_INTELLIGENCE_KEY",
    "OPENAI_API_KEY",
    "USE_AZURE_OPENAI",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_MODEL_DEPLOYMENT_NAME",
]

MEMORY_IN_GB = 8


@cls()
class FullPageAzureTask:
    def __init__(self) -> None:
        from tensorlake_docai.ocr.azure_markdown_extractor import AzureMarkdownExtractor

        self.extractor: Optional[AzureMarkdownExtractor] = None
        self._init_extractor()

    @function(
        image=simple_page_creator_image,
        timeout=30 * 60,
        cpu=2,
        memory=MEMORY_IN_GB,
        ephemeral_disk=2,
        secrets=SECRETS,
        retries=Retries(max_retries=2),
        min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
    )
    def run(self, parse_result: ParseResult) -> ParseResult:
        if parse_result.request.mime_type.startswith("text/"):
            return parse_result

        (
            doc_bytes,
            page_dims,
            bbox_from_images,
            pages_param,
            input_scale_factor,
            preloaded_images,
        ) = self._prepare_document(parse_result)

        try:
            # PDF inputs: chunk into 100-page slices and process serially
            CHUNK_SIZE = 100
            if not bbox_from_images:
                import io
                from pypdf import PdfReader

                try:
                    total_pages = len(PdfReader(io.BytesIO(doc_bytes)).pages)
                except Exception as e:
                    print(f"DEBUG: Failed to read PDF for page counting: {e}")
                    raise RequestException(
                        message="Unable to open the PDF document. Please ensure the file is a valid, non-corrupted PDF. Error: "
                        + str(e)
                    )

                requested_pages = list(range(1, total_pages + 1))
                if parse_result.request.pages_to_parse:
                    requested_pages = parse_result.request.pages_to_parse
                print(f"[FAT] Total pages: {total_pages}, Requested pages: {requested_pages}")

                ctx: RequestContext = RequestContext.get()
                chunks = [
                    requested_pages[i : i + CHUNK_SIZE]
                    for i in range(0, len(requested_pages), CHUNK_SIZE)
                ]
                completed_pages = 0
                total_to_process = len(requested_pages)
                for chunk in chunks:
                    print(f"[FAT] Processing chunk {chunk}")
                    result = self.extractor.analyze_document_bytes_direct(
                        doc_bytes, pages=(self._format_pages_param(chunk) if chunk else None)
                    )
                    self._build_layout(
                        parse_result,
                        result,
                        page_dims,
                        False,
                        input_scale_factor,
                        ctx,
                        completed_pages,
                        total_to_process,
                        doc_bytes if parse_result.request.include_images else None,
                    )
                    completed_pages += len(chunk)
                # update total pages since we processed multiple chunks
                if parse_result.document_layout:
                    parse_result.document_layout.total_pages = (
                        total_pages or parse_result.document_layout.total_pages
                    )

            # Image inputs: process in one call
            else:
                page_images = preloaded_images
                if page_images:
                    print(f"[FAT] Reusing preloaded images for {len(page_images)} pages")

                ctx: RequestContext = RequestContext.get()
                result = self.extractor.analyze_document_bytes_direct(doc_bytes, pages=pages_param)
                self._build_layout(
                    parse_result,
                    result,
                    page_dims,
                    bbox_from_images,
                    input_scale_factor,
                    ctx,
                    0,
                    len(page_dims),
                    page_images,
                )
        except RequestException:
            raise
        except Exception as e:
            print(f"[FAT] Azure OCR failed: {e}")
            message = extract_provider_error_message(e)
            raise RequestException(message=message)
        if parse_result.request.xpage_header_detection:
            try:
                parse_result = correct_document_headers(
                    parse_result, api_key=os.getenv("OPENAI_API_KEY")
                )
            except RequestException:
                raise
            except Exception as e:
                print(f"[FAT] Header correction skipped: {e}")

        return route_after_ocr(parse_result, log_prefix="FULL_PAGE_AZURE")

    def _prepare_document(
        self, parse_result: ParseResult
    ) -> Tuple[bytes, Dict[int, Tuple[int, int]], bool, Optional[str], float, Optional[dict]]:
        req = parse_result.request
        extension = FILE_TYPE_MAPPING.get(req.mime_type, None)
        if extension not in ["pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp"]:
            raise RequestException(message="unsupported file type: " + req.mime_type)

        if extension == "pdf":
            file_bytes = (
                req.file_bytes.encode("utf-8")
                if isinstance(req.file_bytes, str)
                else (req.file_bytes or b"")
            )
            parse_result.usage = Usage(
                pages_parsed=0,
                extraction_input_tokens_used=0,
                extraction_output_tokens_used=0,
                summarization_input_tokens_used=0,
                summarization_output_tokens_used=0,
            )
            return (
                file_bytes,
                {},
                False,
                self._format_pages_param(req.pages_to_parse),
                1.0,
                None,
            )

        # Non-PDF: send original bytes (images/TIFF)
        # Only load images if needed for figure extraction
        page_images_dict = None
        page_dims = {}

        if req.include_images:
            # Load once for both dimensions and figure extraction
            spc = SimplePageCreator(scale_factor=1, memory_gb=MEMORY_IN_GB)
            pages = spc.get_images(parse_result)
            page_images_dict = pages.page_images
            page_dims = {p: (img.width, img.height) for p, img in page_images_dict.items()}
        else:
            # Only need dimensions - load to get dims then discard
            spc = SimplePageCreator(scale_factor=1)
            pages = spc.get_images(parse_result)
            page_dims = {p: (img.width, img.height) for p, img in pages.page_images.items()}

        file_bytes = (
            req.file_bytes.encode("utf-8")
            if isinstance(req.file_bytes, str)
            else (req.file_bytes or b"")
        )
        parse_result.usage = Usage(
            pages_parsed=len(page_dims),
            extraction_input_tokens_used=0,
            extraction_output_tokens_used=0,
            summarization_input_tokens_used=0,
            summarization_output_tokens_used=0,
        )
        return (
            file_bytes,
            page_dims,
            True,
            self._format_pages_param(req.pages_to_parse),
            1.0,
            page_images_dict,
        )

    def _build_layout(
        self,
        parse_result: ParseResult,
        azure_result,
        page_dims: Dict[int, Tuple[int, int]],
        bbox_from_images: bool,
        input_scale_factor: float,
        ctx: RequestContext,
        pages_completed_so_far: int,
        total_pages: int,
        doc_bytes_or_images=None,
    ) -> None:
        from tensorlake_docai.models.layout_objects import DocumentLayout

        if not parse_result.document_layout:
            parse_result.document_layout = DocumentLayout(
                pages=[],
                scale_factor=float(input_scale_factor or 1.0),
                total_pages=len(getattr(azure_result, "pages", []) or []),
            )
        else:
            parse_result.document_layout.scale_factor = float(input_scale_factor or 1.0)
            parse_result.document_layout.total_pages = len(getattr(azure_result, "pages", []) or [])

        pages = (
            sorted((azure_result.pages or []), key=lambda p: p.page_number)
            if hasattr(azure_result, "pages")
            else []
        )
        for idx, page in enumerate(pages):
            azure_page_no = page.page_number
            page_no = azure_page_no

            # Update progress for each page
            current_page = pages_completed_so_far + idx + 1
            ctx.progress.update(
                current=current_page,
                total=total_pages,
                message=f"Processing page {current_page}/{total_pages}",
            )
            if bbox_from_images:
                # For image inputs, no 72-DPI conversion
                target_w, target_h = page_dims[page_no]
                layout_repr = self.extractor.extract_layout_representation(
                    azure_result, target_w, target_h, azure_page_no
                )
                sx, sy = 1.0, 1.0
            else:
                # For PDFs, convert inches to pixels (72 DPI)
                layout_repr = self.extractor.extract_page_layout_from_pdf_result(
                    azure_result, azure_page_no
                )
                dims = layout_repr.get("dimensions", [612, 792])
                src_w, src_h = dims[0], dims[1]
                target_w, target_h, sx, sy = src_w, src_h, 1.0, 1.0

            page_layout = PageLayout(
                page_number=page_no,
                elements=[],
                shape=(int(target_h), int(target_w)),
                page_dimensions={"width": int(target_w), "height": int(target_h)},
            )
            scaled_frags: List[dict] = []
            for frag in layout_repr.get("page_fragments", []):
                bb = frag.get("bbox", {}) or {}
                if bb:
                    bb = {
                        "x1": int(bb.get("x1", 0) * sx),
                        "y1": int(bb.get("y1", 0) * sy),
                        "x2": int(bb.get("x2", 0) * sx),
                        "y2": int(bb.get("y2", 0) * sy),
                    }
                scaled_frags.append(
                    {
                        "bbox": bb,
                        "content": frag.get("content", {}),
                        "fragment_type": frag.get("fragment_type", "text"),
                        "reading_order": frag.get("reading_order", 0),
                    }
                )

            # Extract page image or PDF bytes for figure cropping
            page_image = (
                doc_bytes_or_images.get(page_no) if isinstance(doc_bytes_or_images, dict) else None
            )
            pdf_bytes = doc_bytes_or_images if isinstance(doc_bytes_or_images, bytes) else None
            # For PDF cropping, pass azure_page_no (page index in the PDF we're cropping from)
            page_layout.elements = self._to_elements(
                scaled_frags, page_no, page_image, pdf_bytes, azure_page_no if pdf_bytes else None
            )

            if hasattr(azure_result, "pages") and azure_result.pages:
                for ap in azure_result.pages:
                    # Use Azure page numbering here
                    if ap.page_number == azure_page_no:
                        pw_in, ph_in = getattr(ap, "width", 8.5), getattr(ap, "height", 11.0)
                        self.extractor.extract_cell_bboxes_for_tables(page_layout, target_w, target_h, pw_in, ph_in)  # type: ignore
                        break

            parse_result.document_layout.pages.append(page_layout)

    def _to_elements(
        self, frags: List[dict], page_no: int, page_image=None, pdf_bytes=None, azure_page_no=None
    ) -> List[PageLayoutElement]:
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
        out: List[PageLayoutElement] = []
        for frag in frags:
            content = frag.get("content", {})
            text = (content.get("content", "") or "").strip()
            if not text:
                continue
            bbox = frag.get("bbox", {})
            fragment_type = mapping.get(frag.get("fragment_type", "text"), PageFragmentType.TEXT)

            # Extract image for FIGURE elements
            image_base64 = None
            if fragment_type == PageFragmentType.FIGURE:
                if page_image is not None:
                    # Image input: crop from PIL image
                    image_base64 = self._crop_from_image(page_image, bbox, page_no)
                elif pdf_bytes is not None and azure_page_no is not None:
                    # PDF input: crop directly from PDF using PyMuPDF
                    # Use azure_page_no for the actual page index in the PDF
                    image_base64 = self._crop_from_pdf(pdf_bytes, azure_page_no, bbox, page_no)

            out.append(
                PageLayoutElement(
                    bbox=(
                        bbox.get("x1", 0),
                        bbox.get("y1", 0),
                        bbox.get("x2", 0),
                        bbox.get("y2", 0),
                    ),
                    fragment_type=fragment_type,
                    score=1.0,
                    reading_order=frag.get("reading_order", 0),
                    ref_id=f"{page_no}.{frag.get('reading_order', 0)}",
                    ocr_text=text,
                    markdown=content.get("markdown", text),
                    html=content.get("html"),
                    hierarchy_level=(
                        content.get("level")
                        if frag.get("fragment_type") in ("section_header", "title")
                        else None
                    ),
                    image_base64=image_base64,
                )
            )
        return out

    def _crop_from_image(self, page_image, bbox: dict, page_no: int) -> Optional[str]:
        """Crop a region from a PIL image and return base64-encoded bytes."""
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
                print(
                    f"[FAT] Encoded figure image from PIL for page {page_no}, bbox ({x1},{y1},{x2},{y2})"
                )
                return image_base64
        except Exception as e:
            print(f"[FAT] Failed to crop/encode figure from image: {e}")
        return None

    def _crop_from_pdf(
        self, pdf_bytes: bytes, azure_page_no: int, bbox: dict, original_page_no: int
    ) -> Optional[str]:
        """Crop a region from PDF using PyMuPDF partial rasterization and return base64-encoded bytes.

        Args:
            pdf_bytes: PDF document bytes
            azure_page_no: Azure's page number (1-indexed) - used to access the page in the PDF
            bbox: Bounding box dict with x1, y1, x2, y2
            original_page_no: Original document page number (for logging only)
        """
        try:
            import fitz  # PyMuPDF

            x1, y1, x2, y2 = (
                bbox.get("x1", 0),
                bbox.get("y1", 0),
                bbox.get("x2", 0),
                bbox.get("y2", 0),
            )

            # Ensure valid bbox
            if x2 <= x1 or y2 <= y1:
                return None

            # Open PDF and get page (azure_page_no is 1-indexed, fitz uses 0-indexed)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                if azure_page_no < 1 or azure_page_no > len(doc):
                    print(
                        f"[FAT] Invalid Azure page number {azure_page_no} for PDF cropping (original page {original_page_no})"
                    )
                    return None

                page = doc[azure_page_no - 1]

                # Create rect for the bbox - PyMuPDF uses (x0, y0, x1, y1) format
                clip_rect = fitz.Rect(x1, y1, x2, y2)

                # Render only the clipped region (partial rasterization)
                pix = page.get_pixmap(
                    clip=clip_rect
                )  # use default dpi of 72 is enough for image preview

                # Convert pixmap to PIL Image
                from PIL import Image

                # cover all possible modes
                modes = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}
                mode = modes.get(pix.n, "RGB")
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

                # Convert to base64
                image_base64 = pil_image_to_base64(img)
                print(
                    f"[FAT] Encoded figure image from PDF for page {original_page_no} (Azure page {azure_page_no}), bbox ({x1},{y1},{x2},{y2})"
                )
                return image_base64
            finally:
                doc.close()
        except Exception as e:
            print(f"[FAT] Failed to crop/encode figure from PDF page {original_page_no}: {e}")
        return None

    def _init_extractor(self) -> None:
        if self.extractor is not None:
            return
        from tensorlake_docai.ocr.azure_markdown_extractor import AzureMarkdownExtractor

        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        if not endpoint or not key:
            raise RequestException(
                message="Service temporarily unavailable due to configuration error. Please contact support with the trace ID of the job."
            )
        self.extractor = AzureMarkdownExtractor(endpoint=endpoint, key=key)

    def _format_pages_param(self, pages: Optional[List[int]]) -> Optional[str]:
        """Compress page numbers for Azure, e.g., [1, 2, 5] → '1-2,5'."""
        if not pages:
            return None
        s = sorted(set(pages))
        groups: List[List[int]] = []
        cur = [s[0]]
        for i in range(1, len(s)):
            if s[i] == s[i - 1] + 1:
                cur.append(s[i])
            else:
                groups.append(cur)
                cur = [s[i]]
        groups.append(cur)
        return ",".join([f"{g[0]}-{g[-1]}" if len(g) > 1 else f"{g[0]}" for g in groups])
