# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
AWS Textract Markdown Extractor using textractor
"""

import os
import sys
import io
from typing import List, Dict, Optional, Tuple

from textractor import Textractor
from textractor.data.constants import TextractFeatures
from PIL import Image
from tensorlake_docai.ocr.textract_retry_utils import (
    robust_textract_analyze_document,
    TIMEOUT_IMAGE,
)
import uuid

REGION_NAME = "us-east-1"
DPI = 200

TEXTRACT_S3_BUCKET = os.getenv("S3_BUCKET_NAME")

# Textract features to extract
TEXTRACT_FEATURES = [TextractFeatures.LAYOUT, TextractFeatures.TABLES]


def _create_fragment_from_layout(
    layout,
    reading_order: int,
    image_width: int,
    image_height: int,
    preserve_original_resolution: bool = False,
) -> Optional[Dict]:
    """Create a fragment from a textractor layout element"""
    try:
        # Map layout type to fragment type
        layout_type = layout.layout_type
        fragment_type_mapping = {
            "LAYOUT_TITLE": "title",
            "LAYOUT_HEADER": "page_header",
            "LAYOUT_FOOTER": "page_footer",
            "LAYOUT_SECTION_HEADER": "section_header",
            "LAYOUT_PAGE_NUMBER": "page_number",
            "LAYOUT_LIST": "text",
            "LAYOUT_FIGURE": "figure",
            "LAYOUT_TABLE": "table",
            "LAYOUT_KEY_VALUE": "key_value_region",
            "LAYOUT_TEXT": "text",
        }
        fragment_type = fragment_type_mapping.get(layout_type, "text")

        # Extract bbox using actual image dimensions, then convert to 72 DPI for UI
        bbox = layout.bbox

        x1_high_dpi = int(bbox.x * image_width)
        y1_high_dpi = int(bbox.y * image_height)
        x2_high_dpi = int((bbox.x + bbox.width) * image_width)
        y2_high_dpi = int((bbox.y + bbox.height) * image_height)

        # Apply DPI scaling only for PDF-to-image conversion, preserve original for direct images
        if preserve_original_resolution:
            # Use original resolution for direct image processing (for bbox visualization)
            bbox_dict = {"x1": x1_high_dpi, "y1": y1_high_dpi, "x2": x2_high_dpi, "y2": y2_high_dpi}
        else:
            # Convert to 72 DPI for PDF-to-image UI compatibility
            dpi_scale_factor = 72.0 / DPI
            bbox_dict = {
                "x1": int(x1_high_dpi * dpi_scale_factor),
                "y1": int(y1_high_dpi * dpi_scale_factor),
                "x2": int(x2_high_dpi * dpi_scale_factor),
                "y2": int(y2_high_dpi * dpi_scale_factor),
            }

        # Get content - use plain text instead of markdown for headers to avoid double formatting
        try:
            if fragment_type in ["title", "section_header", "page_header", "page_footer"]:
                # For headers, use plain text and let downstream pipeline handle markdown formatting
                content_text = getattr(layout, "text", "") or f"[{layout_type} content]"
            else:
                # For other content types, use markdown if available
                content_text = layout.to_markdown()
        except Exception:
            content_text = getattr(layout, "text", "") or f"[{layout_type} content]"

        fragment = {
            "bbox": bbox_dict,
            "content": {"content": content_text},
            "fragment_type": fragment_type,
            "reading_order": reading_order,
            "confidence": 1.0,
        }

        # Add special content for tables
        if fragment_type == "table":
            try:
                fragment["content"]["html"] = layout.to_html()
                fragment["content"]["markdown"] = layout.to_markdown()
            except Exception:
                pass

        # Add level for headers - improved hierarchy detection
        if fragment_type in ["title", "section_header"]:
            # Try to determine hierarchy level from text content or layout properties
            level = 0 if fragment_type == "title" else 1

            # Check if we can extract level from text patterns (e.g., "1.", "1.1", "a.", etc.)
            text_content = content_text.strip()
            if fragment_type == "section_header":
                # Look for numbering patterns to determine hierarchy
                import re

                # Check for patterns like "1.", "1.1.", "1.1.1.", etc.
                number_match = re.match(r"^(\d+(?:\.\d+)*)", text_content)
                if number_match:
                    dots_count = number_match.group(1).count(".")
                    level = min(dots_count + 1, 3)  # Cap at level 3 for markdown compatibility
                # Check for patterns like "A.", "a.", etc.
                elif re.match(r"^[A-Za-z]\.", text_content):
                    level = 2

            fragment["content"]["level"] = level

        return fragment

    except Exception as e:
        print(f"[TME] ❌ Error creating fragment: {e}")
        return None


class TextractMarkdownExtractor:
    """AWS Textract markdown extractor using textractor"""

    def __init__(self, region_name: str = REGION_NAME):
        """Initialize with textractor"""
        self.region_name = region_name
        try:
            self.extractor = Textractor(region_name=region_name)
            print(f"[TME] Textractor initialized for region: {region_name}")
        except Exception as e:
            # Internal logging keeps details, but don't expose service name in exception message
            print(f"[TME] Failed to initialize Textractor: {e}")
            raise Exception(f"Failed to initialize document analysis service: {e}")

    def process_image(self, image_input) -> Dict:
        """Process a single image and return layout information

        Args:
            image_input: Can be either:
                - str: Path to image file
                - bytes: Raw image bytes
                - PIL.Image: PIL Image object
        """
        try:
            import io

            # Handle different input types
            if isinstance(image_input, str):
                # File path
                image = Image.open(image_input)
            elif isinstance(image_input, bytes):
                # Raw bytes
                image = Image.open(io.BytesIO(image_input))
            elif hasattr(image_input, "mode"):
                # PIL Image object
                image = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # For uploaded images, preserve original format and quality when possible
            # Only apply JPG compression if image has transparency or is very large
            if image.mode in ("RGBA", "LA", "P"):
                # Convert transparency modes to RGB for Textract compatibility
                image = image.convert("RGB")
                # Apply light compression for converted images
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="JPEG", quality=90, optimize=True)
                img_buffer.seek(0)
                compressed_image = Image.open(img_buffer)
                print("[TME] 🔄 Converted transparent image to RGB for Textract")
            else:
                # Use original image directly for RGB/Grayscale images
                compressed_image = image
                print("[TME] ✅ Using original image format for Textract")

            # Get actual image dimensions for proper bbox conversion
            image_width, image_height = compressed_image.size

            # Analyze with textractor using retry mechanism
            document = robust_textract_analyze_document(
                extractor=self.extractor,
                file_source=compressed_image,
                features=TEXTRACT_FEATURES,
                timeout=TIMEOUT_IMAGE,
                save_image=False,
            )

            # For direct image processing, preserve original resolution for bbox visualization
            layout_data = {"dimensions": [image_width, image_height], "page_fragments": []}

            if document.pages:
                page = document.pages[0]
                reading_order = 0

                for layout in page.layouts:
                    fragment = _create_fragment_from_layout(
                        layout,
                        reading_order,
                        image_width,
                        image_height,
                        preserve_original_resolution=True,
                    )
                    if fragment:
                        layout_data["page_fragments"].append(fragment)
                        reading_order += 1

            return layout_data

        except Exception as e:
            print(f"[TME] ❌ Error processing image: {e}")
            raise

    def process_image_bytes(self, image_bytes: bytes) -> Dict:
        """Convenience method to process raw image bytes"""
        return self.process_image(image_bytes)

    def extract_page_layout_from_textract_result(
        self, textract_document: Dict, page_number: int = 1
    ) -> Dict:
        """Extract layout from textract document for compatibility"""
        if isinstance(textract_document, dict) and "page_fragments" in textract_document:
            return textract_document
        return {
            "dimensions": [612, 792],
            "page_fragments": textract_document.get("page_fragments", []),
        }

    def process_pdf_via_s3_bytes(
        self, pdf_bytes: bytes, pages_to_extract: Optional[List[int]] = None
    ) -> Dict[int, Dict]:
        """Process multi-page PDF via Textract async API (S3).

        Returns a mapping of 1-based page_no (in the processed/trimmed PDF) -> layout_data dict
        with dimensions in 72 DPI and page_fragments.
        """
        try:
            import boto3

            # if page selection is provided, trim the PDF to the selected pages
            if pages_to_extract:
                try:
                    from pypdf import PdfReader, PdfWriter

                    reader = PdfReader(io.BytesIO(pdf_bytes))  # type: ignore
                    writer = PdfWriter()
                    for p in sorted(pages_to_extract):
                        if 1 <= p <= len(reader.pages):
                            writer.add_page(reader.pages[p - 1])
                    out_buf = io.BytesIO()
                    writer.write(out_buf)
                    pdf_bytes = out_buf.getvalue()
                except Exception as e:
                    print(f"⚠️ Failed to trim PDF pages, proceeding with full PDF: {e}")

            # Determine per-page dimensions (points == 72 DPI pixels)
            page_dims: Dict[int, Tuple[int, int]] = {}
            try:
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(pdf_bytes))  # type: ignore
                for i, page in enumerate(reader.pages, start=1):
                    try:
                        w = int(float(page.cropbox.width))
                        h = int(float(page.cropbox.height))
                    except Exception:
                        w, h = 612, 792
                    page_dims[i] = (w, h)
            except Exception as e:
                print(f"⚠️ Could not read PDF dimensions, defaulting to 612x792: {e}")

            # Upload to S3, explicitly specify the region
            s3 = boto3.client("s3", region_name=self.region_name)
            bucket = TEXTRACT_S3_BUCKET
            if not bucket:
                raise Exception(
                    "S3_BUCKET_NAME environment variable is not set. Please configure the S3 bucket for Textract processing."
                )
            key = f"tmp/textract-inputs/{uuid.uuid4()}.pdf"
            s3.put_object(Bucket=bucket, Key=key, Body=pdf_bytes, ContentType="application/pdf")
            print(f"[TME] PDF uploaded to S3: {key}")

            # Use textractor's async API to obtain a Document with layouts
            s3_uri = f"s3://{bucket}/{key}"
            document = None

            try:
                # Textractor automatically handle pooling and status checking, so we are returning the error directly
                document = self.extractor.start_document_analysis(
                    s3_uri,
                    features=TEXTRACT_FEATURES,
                    save_image=False,
                )
            except Exception as e:
                print(f"[TME] ⚠️ OCR API failed: {e}")
                raise

            if document is None:
                print("[TME] ❌ No OCR returned")
                raise

            # Build results by iterating textractor page layouts
            results: Dict[int, Dict] = {}
            # Map trimmed page index to original page number for consistency with process_pdf
            original_page_numbers = sorted(pages_to_extract) if pages_to_extract else None
            page_index = 0
            for page in getattr(document, "pages", []):
                page_index += 1
                # Use original page number if pages were extracted, otherwise use trimmed index
                if original_page_numbers and page_index <= len(original_page_numbers):
                    result_page_num = original_page_numbers[page_index - 1]
                else:
                    result_page_num = page_index
                dims = page_dims.get(page_index, (612, 792))
                layout_data = {"dimensions": [dims[0], dims[1]], "page_fragments": []}
                reading_order = 0
                for layout in getattr(page, "layouts", []):
                    frag = _create_fragment_from_layout(
                        layout, reading_order, dims[0], dims[1], preserve_original_resolution=True
                    )
                    if frag:
                        layout_data["page_fragments"].append(frag)
                        reading_order += 1
                results[result_page_num] = layout_data

            # Cleanup upload
            try:
                s3.delete_object(Bucket=bucket, Key=key)
                print(f"[TME] S3 object deleted: {key}")
            except Exception as e:
                print(f"⚠️ Failed to delete temp S3 object: {e}")

            return results
        except Exception as e:
            print(f"❌ Failed to process PDF via S3 Textract async: {e}")
            raise


def main():
    """Main function for standalone usage"""
    if len(sys.argv) < 2:
        print("Usage: python textract_markdown_extractor.py <file_path>")
        print("Supports: PDF files, image files (JPG, PNG, etc.)")
        print("Images are automatically compressed to JPG quality 85 for optimal processing")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"❌ [TME] File not found: {file_path}")
        sys.exit(1)

    try:
        extractor = TextractMarkdownExtractor()

        if file_path.lower().endswith(".pdf"):
            print("Processing PDF file...")
            results = extractor.process_pdf(file_path)
            total_fragments = sum(
                len(layout_data.get("page_fragments", [])) for layout_data in results.values()
            )
            print(f"\nTotal fragments processed: {total_fragments}")
        else:
            print("Processing image file...")
            layout_data = extractor.process_image(file_path)
            print(f"\nProcessed {len(layout_data.get('page_fragments', []))} fragments")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
