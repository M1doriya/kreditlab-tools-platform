# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict
from typing import Any, Optional, List
from itertools import batched
import time
from tensorlake_docai.pipeline.routing import FILE_TYPE_MAPPING
from tensorlake_docai.ocr import image_preprocessing_utils
from PIL import Image

from pydantic import BaseModel

# PDF standard: 1 inch = 72 points
PDF_POINTS_PER_INCH = 72


def consecutive_numbers(pages: List[int]) -> List[List[int]]:
    from itertools import groupby
    from operator import itemgetter

    batches = []
    for _, g in groupby(enumerate(pages), lambda ix: ix[0] - ix[1]):
        batches.append(list(map(itemgetter(1), g)))
    return batches


class DocumentPages:
    def __init__(
        self,
        total_pages: int,
        page_images: OrderedDict[int, Any],
        scale_factor: float,
        original_sizes: OrderedDict[int, tuple[int, int]] | None = None,
    ):
        self.total_pages = total_pages
        self.page_images = page_images
        self.scale_factor = scale_factor
        self.original_sizes = original_sizes or OrderedDict()


def correct_skew(image: Any) -> Any:
    import numpy as np
    from PIL import Image
    from jdeskew.estimator import get_angle
    from jdeskew.utility import rotate

    arr = np.array(image)
    angle = get_angle(arr)
    rotated = rotate(arr, angle)
    return Image.fromarray(rotated)


class ImageDimensions(BaseModel):
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    target_dpi: int = PDF_POINTS_PER_INCH
    upgrade_image_dpi: bool = False


class SimplePageCreator:
    """Create page images from PDFs, images, or TIFFs, with batching."""

    def __init__(self, scale_factor: float = 1.0, batch_size: int = 20, memory_gb: float = None):
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        if memory_gb is None:
            import psutil

            memory_gb = psutil.virtual_memory().total / 1_000_000_000
        self.max_ram_bytes = int(memory_gb * 1_000_000_000 * 0.1)

    def _iter_batches_image(self, data: bytes, image_dimensions: ImageDimensions):
        import io
        from PIL import Image

        # Match dots.mocr: if RGBA, composite onto white before RGB conversion
        img = Image.open(io.BytesIO(data))
        img = image_preprocessing_utils.to_rgb(img)
        original_size = img.size
        scale_factor = 1.0
        if image_dimensions.upgrade_image_dpi:
            # Upgrade DPI via a PDF passthrough; returns a new PIL image
            img = image_preprocessing_utils.get_image_by_fitz_doc(
                img, target_dpi=image_dimensions.target_dpi
            )
            scale_factor = original_size[0] / float(img.size[0])
        # Optional resize based on min/max pixels (preserve aspect ratio correctly)
        if image_dimensions.min_pixels and image_dimensions.max_pixels:
            # smart_resize signature: (height, width) -> (new_height, new_width)
            target_height, target_width = image_preprocessing_utils.smart_resize(
                img.height,
                img.width,
                min_pixels=image_dimensions.min_pixels,
                max_pixels=image_dimensions.max_pixels,
            )
            if (target_width, target_height) != (img.width, img.height):
                # PIL expects (width, height)
                img = img.resize((target_width, target_height))
                # Recompute total scale w.r.t. original input size
                scale_factor = original_size[0] / float(img.size[0])
        batch = OrderedDict()
        original_sizes = OrderedDict()
        # Single image is always page 1
        batch[1] = img
        original_sizes[1] = original_size
        yield DocumentPages(
            total_pages=1,
            page_images=batch,
            scale_factor=scale_factor,
            original_sizes=original_sizes,
        )

    def _iter_batches_tiff(self, data: bytes, wanted: List[int], image_dimensions: ImageDimensions):
        import io
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            total = getattr(img, "n_frames", 1)
            pages = wanted or list(range(1, total + 1))
            # Calculate dynamic batch size based on TIFF dimensions
            tiff_width, tiff_height = img.size
            bytes_per_image = tiff_width * tiff_height * 3  # RGB
            dynamic_batch_size = max(1, int(self.max_ram_bytes / bytes_per_image))
            batch_size = min(dynamic_batch_size, self.batch_size)
            for page_batch in batched(pages, batch_size):
                batch = OrderedDict()
                original_sizes = OrderedDict()
                for p in page_batch:
                    if 1 <= p <= total:
                        img.seek(p - 1)
                        # Use to_rgb for proper color mode handling (same as regular images)
                        converted = image_preprocessing_utils.to_rgb(img.copy())
                        original_size = converted.size
                        scale_factor = 1.0

                        # Upgrade DPI for better OCR (same as regular images)
                        if image_dimensions.upgrade_image_dpi:
                            converted = image_preprocessing_utils.get_image_by_fitz_doc(
                                converted, target_dpi=image_dimensions.target_dpi
                            )
                            scale_factor = original_size[0] / float(converted.size[0])

                        # Optional resize for TIFF frames (preserve aspect ratio correctly)
                        if image_dimensions.min_pixels and image_dimensions.max_pixels:
                            target_height, target_width = image_preprocessing_utils.smart_resize(
                                converted.height,
                                converted.width,
                                min_pixels=image_dimensions.min_pixels,
                                max_pixels=image_dimensions.max_pixels,
                            )
                            if (target_width, target_height) != (converted.width, converted.height):
                                converted = converted.resize((target_width, target_height))
                                scale_factor = original_size[0] / float(converted.size[0])

                        batch[p] = converted
                        original_sizes[p] = original_size
                yield DocumentPages(
                    total_pages=total,
                    page_images=batch,
                    scale_factor=scale_factor,
                    original_sizes=original_sizes,
                )

    def _iter_batches_pdf(self, data: bytes, wanted: List[int], image_dimensions: ImageDimensions):
        from tensorlake.applications import RequestError as RequestException
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        import fitz

        MAX_PIXELS = 11_289_600
        FALLBACK_DPI = PDF_POINTS_PER_INCH
        RENDER_TIMEOUT_SECONDS = 240

        try:
            doc = fitz.open(stream=data, filetype="pdf")
            total = doc.page_count
            if total == 0:
                raise RequestException(message="PDF has no pages.")
            first_page = doc.load_page(0)
            rect = first_page.rect
            pdf_width_inches = float(rect.width) / PDF_POINTS_PER_INCH
            pdf_height_inches = float(rect.height) / PDF_POINTS_PER_INCH
        except Exception as e:
            raise RequestException(
                message="Unable to open the PDF document. Please ensure the file is a valid, non-corrupted PDF. Error: "
                + str(e)
            )

        dpi = image_dimensions.target_dpi or int(
            PDF_POINTS_PER_INCH * max(1.0, float(self.scale_factor))
        )
        image_width = int(pdf_width_inches * dpi)
        image_height = int(pdf_height_inches * dpi)
        bytes_per_image = image_width * image_height * 3  # RGB
        if bytes_per_image <= 0:
            raise RequestException(message="PDF page dimensions are invalid or too small.")
        calculated_batch_size = max(1, int(self.max_ram_bytes / bytes_per_image))

        pages = wanted or list(range(1, total + 1))

        try:
            for page_batch in batched(pages, calculated_batch_size):
                consecutive_page_batches = consecutive_numbers(page_batch)
                for page_range in consecutive_page_batches:
                    batch = OrderedDict()
                    original_sizes = OrderedDict()
                    page_scale_factors = OrderedDict()

                    for idx in page_range:
                        if not (1 <= idx <= total):
                            continue
                        page = doc.load_page(idx - 1)

                        # Adaptive DPI: reduce if rendering would exceed pixel limit
                        page_dpi = dpi
                        target_width = int((page.rect.width / PDF_POINTS_PER_INCH) * page_dpi)
                        target_height = int((page.rect.height / PDF_POINTS_PER_INCH) * page_dpi)
                        total_pixels = target_width * target_height

                        if total_pixels > MAX_PIXELS:
                            page_dpi = FALLBACK_DPI
                            target_width = int((page.rect.width / PDF_POINTS_PER_INCH) * page_dpi)
                            target_height = int((page.rect.height / PDF_POINTS_PER_INCH) * page_dpi)
                            print(
                                f"[SPC] Page {idx}: Adaptive DPI - reducing from {dpi} to {page_dpi}dpi "
                                f"({total_pixels / 1_000_000:.1f}MP -> {(target_width * target_height) / 1_000_000:.1f}MP)"
                            )

                        mat = fitz.Matrix(
                            page_dpi / PDF_POINTS_PER_INCH, page_dpi / PDF_POINTS_PER_INCH
                        )

                        # Render with timeout - each page gets its own executor for isolation
                        render_start = time.time()
                        pix = None
                        executor = ThreadPoolExecutor(max_workers=1)
                        try:
                            future = executor.submit(page.get_pixmap, matrix=mat, alpha=False)
                            try:
                                pix = future.result(timeout=RENDER_TIMEOUT_SECONDS)
                                render_time = time.time() - render_start
                                if render_time > 10:
                                    print(
                                        f"[SPC] Page {idx}: Rendered in {render_time:.1f}s ({target_width}x{target_height})"
                                    )
                                executor.shutdown(wait=True)
                            except FuturesTimeoutError:
                                print(
                                    f"[SPC] Page {idx}: Rendering timeout after {time.time() - render_start:.1f}s - skipping (background thread continues)"
                                )
                                executor.shutdown(wait=False)  # Don't wait for stuck thread
                                continue
                        except Exception as e:
                            print(
                                f"[SPC] Page {idx}: Rendering error after {time.time() - render_start:.1f}s: {e}"
                            )
                            executor.shutdown(wait=False)
                            continue

                        if pix is None:
                            continue

                        try:
                            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        except Exception as e:
                            print(f"[SPC] Page {idx}: Image conversion error: {e}")
                            continue

                        # Store original PDF dimensions in points
                        original_sizes[idx] = (round(page.rect.width), round(page.rect.height))
                        page_scale_factors[idx] = float(page_dpi) / PDF_POINTS_PER_INCH

                        # Optional resize based on min/max pixels (preserve aspect ratio)
                        if image_dimensions.min_pixels and image_dimensions.max_pixels:
                            target_height, target_width = image_preprocessing_utils.smart_resize(
                                im.height,
                                im.width,
                                min_pixels=image_dimensions.min_pixels,
                                max_pixels=image_dimensions.max_pixels,
                            )
                            if (target_width, target_height) != (im.width, im.height):
                                im = im.resize((target_width, target_height))

                        batch[idx] = im

                    # Use first-page DPI as default scale factor; per-page factors in page_scale_factors
                    scale_factor = float(dpi) / PDF_POINTS_PER_INCH
                    if batch:
                        yield DocumentPages(
                            total_pages=total,
                            page_images=batch,
                            scale_factor=scale_factor,
                            original_sizes=original_sizes,
                        )
        finally:
            doc.close()

    def get_images_generator(
        self, parse_result, image_dimensions: ImageDimensions = ImageDimensions()
    ):
        from tensorlake.applications import RequestError as RequestException

        data = parse_result.request.file_bytes
        wanted = sorted(set(parse_result.request.pages_to_parse or []))
        file_type = FILE_TYPE_MAPPING.get(parse_result.request.mime_type, None)
        if file_type is None:
            raise RequestException(
                message="couldn't determine file extension for mime type: "
                + parse_result.request.mime_type
            )
        try:
            apply_skew = parse_result.request.skew_correction
            iterator = None
            if file_type in ["jpg", "jpeg", "png"]:
                iterator = self._iter_batches_image(data, image_dimensions)
            elif file_type in ["tif", "tiff"]:
                iterator = self._iter_batches_tiff(data, wanted, image_dimensions)
            elif file_type == "pdf":
                iterator = self._iter_batches_pdf(data, wanted, image_dimensions)
            else:
                raise RequestException(message=f"Unsupported file type: {file_type}")

            for doc_pages in iterator:
                if apply_skew and doc_pages.page_images:
                    for k, v in list(doc_pages.page_images.items()):
                        doc_pages.page_images[k] = correct_skew(v)
                yield doc_pages
        except Exception as e:
            raise RequestException(message=f"failed to create pages: {e}")

    def get_images(self, parse_result) -> DocumentPages:
        all_images: OrderedDict[int, Any] = OrderedDict()
        total_pages = 0
        result_scale: float | None = None
        t0 = time.time()
        for batch in self.get_images_generator(parse_result):
            if total_pages == 0:
                total_pages = batch.total_pages
                result_scale = batch.scale_factor
            all_images.update(batch.page_images)
        print(
            f"[SPC] total: {len(all_images)} pages in {time.time()-t0:.2f}s, scale_factor={result_scale}"
        )
        # For consolidated result, original sizes are not aggregated; callers should use the generator per batch
        return DocumentPages(
            total_pages=total_pages, page_images=all_images, scale_factor=float(result_scale or 1.0)
        )
