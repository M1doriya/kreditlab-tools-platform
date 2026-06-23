# SPDX-License-Identifier: Apache-2.0
from PIL import Image, ImageDraw, ImageFont
from typing import List
from pathlib import Path
from io import BytesIO
from tensorlake_docai.pipeline.api import ParsedDocument


def _select_page_image(
    images: List[Image.Image],
    page,
    copy_image: bool = False,
) -> Image.Image:

    preferred_index = max(0, min(page.page_number - 1, len(images) - 1))
    img = images[preferred_index]
    return img.copy() if copy_image else img


def convert_file_to_images(
    file_path: str = None, file_bytes: bytes = None, dpi: int = 72, use_cropbox: bool = False
) -> List[Image.Image]:
    """Convert PDF to images or load image files directly."""
    from pdf2image import convert_from_path, convert_from_bytes

    try:
        # Determine if it's an image or PDF
        is_image = False
        if file_path:
            path = Path(file_path)
            if path.exists():
                is_image = path.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tiff",
                    ".tif",
                    ".bmp",
                    ".gif",
                    ".webp",
                ]
        elif file_bytes:
            # Try to detect if it's an image by attempting to load with PIL first
            try:
                test_image = Image.open(BytesIO(file_bytes))
                test_image.verify()  # Verify it's a valid image
                is_image = True
            except Exception:
                is_image = False

        if is_image:
            # Handle direct image loading (support multi-frame images like TIFF)
            print("Loading image file directly for bbox visualization...")

            def _load_frames(img: Image.Image) -> List[Image.Image]:
                frames: List[Image.Image] = []
                n = getattr(img, "n_frames", 1)
                try:
                    for i in range(n):
                        try:
                            img.seek(i)
                        except EOFError:
                            break
                        frames.append(img.copy())
                finally:
                    try:
                        img.close()
                    except Exception:
                        pass
                return frames if frames else [img]

            if file_bytes:
                image = Image.open(BytesIO(file_bytes))
                return _load_frames(image)
            elif file_path and Path(file_path).exists():
                image = Image.open(file_path)
                return _load_frames(image)
        else:
            # Handle PDF conversion
            print("Converting PDF to images for bbox visualization...")
            if file_bytes:
                images = convert_from_bytes(file_bytes, dpi=dpi, use_cropbox=use_cropbox)
            elif file_path and Path(file_path).exists():
                images = convert_from_path(file_path, dpi=dpi, use_cropbox=use_cropbox)
            else:
                print("Warning: Cannot convert file - no valid file path or bytes provided")
                return []
            return images

    except Exception as e:
        print(f"Error loading file: {e}")
        return []


def draw_bboxes_on_image(
    image: Image.Image, page_fragments: List[dict], page_dimensions: List[int] = None
) -> Image.Image:
    """Draw bounding boxes on image with different colors for different fragment types."""
    if not page_fragments:
        return image

    # Create a copy to avoid modifying the original
    img_with_bboxes = image.copy()
    draw = ImageDraw.Draw(img_with_bboxes)

    # Color mapping for different fragment types
    color_map = {
        "title": "#FF0000",  # Red
        "text": "#00FF00",  # Green
        "table": "#0000FF",  # Blue
        "figure": "#FF00FF",  # Magenta
        "section_header": "#FFFF00",  # Yellow
        "form": "#00FFFF",  # Cyan
        "signature": "#FFA500",  # Orange
    }
    default_color = "#CCCCCC"  # Light gray for unknown types

    try:
        # Use a small font for labels
        font = ImageFont.load_default()
    except Exception:
        font = None

    for fragment in page_fragments:
        bbox = fragment.get("bbox")
        if not bbox:
            continue

        # Extract coordinates (bbox coordinates should match image resolution)
        # Note: document.json uses x1,y1,x2,y2 format
        x0 = bbox.get("x1", 0)
        y0 = bbox.get("y1", 0)
        x1 = bbox.get("x2", 0)
        y1 = bbox.get("y2", 0)

        fragment_type = fragment.get("fragment_type", "unknown")
        color = color_map.get(fragment_type.lower(), default_color)

        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        # Add label with fragment type
        if font:
            label = fragment_type.upper()
            # Position label slightly above the bbox
            label_y = max(0, y0 - 15)
            draw.text((x0, label_y), label, fill=color, font=font)

    return img_with_bboxes


def visualize_document_bboxes(
    parsed_document: ParsedDocument,
    file_path: str = None,
    file_bytes: bytes = None,
    output_prefix: str = "bbox_page",
) -> None:
    """Create bbox visualization images for all pages in the document."""

    if not parsed_document.pages:
        print("No pages found in parsed document")
        return

    print("Loading file for bbox visualization...")
    # For PDFs, use 72 DPI to match common document layouts
    # For images, original resolution is preserved automatically
    images = convert_file_to_images(file_path, file_bytes, dpi=72, use_cropbox=True)

    if not images:
        print("Could not load file for visualization")
        return

    print(f"Creating bbox visualizations for {len(images)} pages...")

    for i, page in enumerate(parsed_document.pages):
        if not images:
            break
        page_image = _select_page_image(images, page, copy_image=False)
        page_fragments = []

        # Convert page fragments to dict format for drawing function
        if page.page_fragments:
            for fragment in page.page_fragments:
                if fragment.bbox:
                    page_fragments.append(
                        {"bbox": fragment.bbox, "fragment_type": fragment.fragment_type}
                    )

        # Draw bboxes on the image
        img_with_bboxes = draw_bboxes_on_image(page_image, page_fragments, page.dimensions)

        # Save the visualization
        output_filename = f"{output_prefix}_{page.page_number}.png"
        img_with_bboxes.save(output_filename)
        print(f"Saved bbox visualization: {output_filename}")


def _has_citations(data: dict) -> bool:
    """Check if a data dict has any citation fields."""
    if not isinstance(data, dict):
        return False

    for key, value in data.items():
        if key.endswith("_citation"):
            return True
        elif isinstance(value, dict):
            if _has_citations(value):
                return True
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and _has_citations(item):
                    return True
    return False


def visualize_citation_bboxes(
    parsed_document: ParsedDocument,
    file_path: str = None,
    file_bytes: bytes = None,
    output_prefix: str = "citation_bbox_page",
) -> None:
    """Create citation bbox visualization images for structured extraction results."""

    if not parsed_document.pages:
        print("No pages found in parsed document")
        return

    # Check if we have structured data with citations
    has_citations = False
    structured_pages = []

    # Check if structured_data exists at document level
    if parsed_document.structured_data:
        structured_pages = parsed_document.structured_data
        # Check for citations in any page
        for page_data in structured_pages:
            if page_data.data:
                data_dict = (
                    page_data.data
                    if isinstance(page_data.data, dict)
                    else (
                        page_data.data.model_dump() if hasattr(page_data.data, "model_dump") else {}
                    )
                )
                if _has_citations(data_dict):
                    has_citations = True
                    break

    if not has_citations:
        print("No citations found in structured extraction results")
        return

    print("Loading file for citation bbox visualization...")
    images = convert_file_to_images(file_path, file_bytes, dpi=72, use_cropbox=True)

    if not images:
        print("Could not load file for visualization")
        return

    print(f"Creating citation bbox visualizations for {len(images)} pages...")

    # Build map of refs to field names and bboxes
    ref_to_info = {}

    def extract_citations(data, field_path="", page_num=1):
        """Recursively extract citation info from structured data."""
        if not isinstance(data, dict):
            return

        for key, value in data.items():
            if key.endswith("_citation") and value:
                field_name = key[:-9]  # Remove '_citation' suffix
                current_path = f"{field_path}.{field_name}" if field_path else field_name

                # Handle both list and single citations
                citations = value if isinstance(value, list) else [value]
                for citation in citations:
                    if isinstance(citation, dict) and "x1" in citation:
                        # It's a bbox dict; prefer explicit page_number from citation if available
                        page_for_bbox = (
                            int(citation.get("page_number", page_num))
                            if citation.get("page_number") is not None
                            else page_num
                        )
                        ref_key = f"{page_for_bbox}:{citation['x1']},{citation['y1']},{citation['x2']},{citation['y2']}"
                        if ref_key not in ref_to_info:
                            ref_to_info[ref_key] = {
                                "bbox": citation,
                                "fields": [],
                                "page": page_for_bbox,
                            }
                        ref_to_info[ref_key]["fields"].append(current_path)
                    elif isinstance(citation, str):
                        # It's a ref string like "1.3" - skip for now
                        pass
            elif isinstance(value, dict):
                # Recursively process nested objects
                current_path = f"{field_path}.{key}" if field_path else key
                extract_citations(value, current_path, page_num)
            elif isinstance(value, list):
                # Process lists
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        extract_citations(item, f"{field_path}.{key}[{i}]", page_num)

    # Extract citations from structured data pages
    for page_data in structured_pages:
        if hasattr(page_data, "page_numbers") and page_data.page_numbers:
            page_num = page_data.page_numbers[0] if page_data.page_numbers else 1
        else:
            page_num = 1

        if hasattr(page_data, "data") and page_data.data:
            data_dict = (
                page_data.data
                if isinstance(page_data.data, dict)
                else page_data.data.model_dump() if hasattr(page_data.data, "model_dump") else {}
            )
            schema_name = page_data.schema_name if hasattr(page_data, "schema_name") else ""
            extract_citations(data_dict, schema_name, page_num)

    # Draw citations on each page
    for i, page in enumerate(parsed_document.pages):
        if not images:
            break
        page_image = _select_page_image(images, page, copy_image=True)
        draw = ImageDraw.Draw(page_image)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        drawn_count = 0

        # Draw all citations for this page
        for ref_key, info in ref_to_info.items():
            if info["page"] != page.page_number:
                continue

            bbox = info["bbox"]
            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))

            if x2 > x1 and y2 > y1:
                # Draw red rectangle
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

                # Draw field names label
                label = ", ".join(info["fields"])
                if font and label:
                    # Position label above bbox with white background
                    label_y = max(0, y1 - 15)
                    try:
                        text_bbox = draw.textbbox((x1, label_y), label, font=font)
                        draw.rectangle(text_bbox, fill=(255, 255, 255))
                        draw.text((x1, label_y), label, fill=(255, 0, 0), font=font)
                    except Exception:
                        # Fallback for older PIL versions
                        draw.text((x1, label_y), label, fill=(255, 0, 0), font=font)

                drawn_count += 1

        # Save the visualization
        output_filename = f"{output_prefix}_{page.page_number}.png"
        page_image.save(output_filename)
        print(
            f"Saved citation bbox visualization: {output_filename} (drew {drawn_count} citations)"
        )
