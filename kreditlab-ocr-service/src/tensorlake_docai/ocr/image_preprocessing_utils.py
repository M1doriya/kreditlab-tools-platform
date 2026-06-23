# SPDX-License-Identifier: Apache-2.0
"""
Image preprocessing utilities to match dots.mocr demo behavior.
This replicates the exact preprocessing pipeline used in the live demo.
"""

import math
from io import BytesIO
from PIL import Image
from typing import Tuple


def to_rgb(image: Image.Image) -> Image.Image:
    """
    Match dots.mocr RGBA handling: composite onto a white background before converting to RGB.
    This avoids dark/black backgrounds when converting transparent PNGs.
    """
    if image.mode == "RGBA":
        white_background = Image.new("RGB", image.size, (255, 255, 255))
        white_background.paste(image, mask=image.split()[3])  # alpha as mask
        return white_background
    return image.convert("RGB")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 11289600,
) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
        if h_bar * w_bar > max_pixels:  # max_pixels first to control the token length
            beta = math.sqrt((h_bar * w_bar) / max_pixels)
            h_bar = max(factor, floor_by_factor(h_bar / beta, factor))
            w_bar = max(factor, floor_by_factor(w_bar / beta, factor))
    return h_bar, w_bar


def fitz_doc_to_image(page, target_dpi: int = 200) -> "Image.Image":
    """Convert fitz.Document page to image with target DPI."""
    import fitz
    from PIL import Image

    mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
    pm = page.get_pixmap(matrix=mat, alpha=False)

    if (
        pm.width > 4500 or pm.height > 4500
    ):  # This is slighly different from dots.mocr demo, which uses 4500.# TODO needs tuning
        mat = fitz.Matrix(72 / 72, 72 / 72)  # use fitz default dpi # TODO needs tuning
        pm = page.get_pixmap(matrix=mat, alpha=False)

    image = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
    return image


def get_image_by_fitz_doc(image: Image.Image, target_dpi: int = 200) -> Image.Image:
    """
    Get image through fitz, to get target dpi image, mainly for higher image quality.
    This replicates the DPI preprocessing from dots.mocr demo.
    """
    import fitz

    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image")

    # Convert PIL Image to bytes
    data_bytes = BytesIO()
    image.save(data_bytes, format="PNG")
    data_bytes.seek(0)

    # Convert to PDF and back to image at target DPI
    pdf_bytes = fitz.open(stream=data_bytes.getvalue()).convert_to_pdf()
    doc = fitz.open("pdf", pdf_bytes)
    page = doc[0]
    image_fitz = fitz_doc_to_image(page, target_dpi=target_dpi)

    doc.close()
    return image_fitz


def preprocess_image_for_dotsocr(
    image: Image.Image,
    min_pixels: int = 3136,  # Demo setting
    max_pixels: int = 11289600,
    target_dpi: int = 200,
    enable_dpi_preprocessing: bool = True,
) -> Image.Image:
    """
    Complete preprocessing pipeline to match dots.mocr demo behavior.

    Args:
        image: Input PIL Image
        min_pixels: Minimum pixels (demo uses 800,000)
        max_pixels: Maximum pixels
        target_dpi: Target DPI for preprocessing
        enable_dpi_preprocessing: Whether to apply DPI conversion (demo default: True)

    Returns:
        Preprocessed PIL Image ready for DotsOCR inference
    """
    print(f"Input image size: {image.width}x{image.height}")

    # Step 1: DPI preprocessing (if enabled, matches demo's fitz_preprocess=True)
    if enable_dpi_preprocessing:
        processed_image = get_image_by_fitz_doc(image, target_dpi=target_dpi)
        print(f"After DPI preprocessing: {processed_image.width}x{processed_image.height}")
    else:
        processed_image = image

    # Step 2: Smart resize with demo's min_pixels setting
    input_height, input_width = smart_resize(
        processed_image.height,
        processed_image.width,
        factor=28,  # IMAGE_FACTOR from dots.mocr
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Step 3: Resize to final dimensions
    final_image = processed_image.resize((input_width, input_height), resample=Image.LANCZOS)
    print(f"Final image size: {final_image.width}x{final_image.height}")
    print(f"Final pixels: {final_image.width * final_image.height:,}")

    return final_image
