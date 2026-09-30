## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  1 -> Week 1 - Image Batch Validation File

"""Main codefile for validating the *mosaic pipeline* with the user's local sample images;
as well as the resulting metrics (i.e.; MSE, SSIM).

Process:
    * Discover PNG and JPEG inputs
    * Run every grid size and tile family
    * Save preprocessed, segmented, and mosaic outputs

"""

## ********** Import Required Libraries **********
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mosaic_pipeline import create_mosaic


## ********** Define Input and Output Locations **********
INPUT_DIRECTORY = Path(__file__).parent / "sample_images"
OUTPUT_DIRECTORY = Path(__file__).parent / "outputs"


## Function 1 -> Process all supported sample images
def process_png_files() -> None:
    """Process every supported raster input with all required grid sizes and tile sets

    Raises:
        * 'FileNotFoundError' -> If no supported image filetype exists in the adjacent *sample_images* folder
    """
    ## Collect PNG and JPEG files because Pillow supports both assignment inputs.
    image_paths = sorted(
        path for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in INPUT_DIRECTORY.glob(pattern)
    )
    if not image_paths:
        raise FileNotFoundError("Add PNG or JPEG files to sample_images before running this script.")

    ## Create the output directory for reproducible image artifacts.
    OUTPUT_DIRECTORY.mkdir(exist_ok = True)

    ## Validate every image across assignment grid sizes and tile families.
    for image_path in image_paths:

        ## Load the image and convert it to RGB for consistent processing
        image = np.asarray(Image.open(image_path).convert("RGB"))

        ## Iterate across every grid size and tile family to generate outputs
        for grid_size in (16, 32, 64):

            ## Iterate across every tile family to generate outputs
            for tile_set_name in ("Solid", "Diagonal", "Checkerboard"):

                ## Create the mosaic with the current image, grid size, and tile set
                prepared, segmented, mosaic = create_mosaic(image, grid_size, tile_set_name)
                stem = image_path.stem + "_" + str(grid_size) + "_" + tile_set_name.lower()

                ## Save the preprocessed, segmented, and mosaic outputs to the output directory    
                Image.fromarray(prepared).save(OUTPUT_DIRECTORY / (stem + "_preprocessed.png"))
                Image.fromarray(segmented).save(OUTPUT_DIRECTORY / (stem + "_segmented.png"))
                Image.fromarray(mosaic).save(OUTPUT_DIRECTORY / (stem + "_mosaic.png"))

                ## Print the resulting shape of the generated mosaic
                print(stem + ": " + str(mosaic.shape))


## ********** Run Batch Validation **********
if __name__ == "__main__":
    process_png_files()
