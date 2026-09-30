## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  1 -> Week 1 - Image Preprocessing File

"""Prepare uploaded images for consistent grid-based mosaic reconstruction.

Process:
    * Convert supported inputs to RGB
    * Center-crop incompatible aspect ratios
    * Resize images to the fixed pipeline resolution
"""

## ********** Import Required Libraries **********
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


## Function 1 -> Prepare an input image for fixed-size grid processing
def prepare_image(image: Any, output_size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Center-crop and resize an input image to the pipeline's fixed RGB resolution.

    Args:
        * image (Any) -> PIL image or NumPy image array supplied by the user
        * output_size (tuple) -> Target width and height for consistent processing

    Returns:
        * np.ndarray -> RGB uint8 image with the requested dimensions
    """
    ## Convert supported inputs into a PIL image and normalize color channels.
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    pil_image = pil_image.convert("RGB")

    ## Center-crop to the target aspect ratio before resizing.
    target_width, target_height = output_size
    source_width, source_height = pil_image.size
    target_ratio = target_width / target_height
    source_ratio = source_width / source_height
    if source_ratio > target_ratio:
        crop_width = int(source_height * target_ratio)
        left = (source_width - crop_width) // 2
        crop_box = (left, 0, left + crop_width, source_height)
    else:
        crop_height = int(source_width / target_ratio)
        top = (source_height - crop_height) // 2
        crop_box = (0, top, source_width, top + crop_height)

	## Resize with a high-quality filter and return uint8 RGB pixels.
    cropped = pil_image.crop(crop_box)
    resized = cropped.resize(output_size, Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype = np.uint8)
