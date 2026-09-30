## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  1 -> Week 1 - Mosaic Pipeline File

"""Combine every assignment stage into one executable mosaic pipeline.

Process:
    * Prepare the uploaded image
    * Analyze and segment its grid cells
    * Map cells to predefined tiles
    * Print metrics and performance results
"""

## ********** Import Required Libraries **********
from __future__ import annotations

from time import perf_counter

import numpy as np

from benchmarking import benchmark_grid_operations, print_benchmark_results
from grid_analysis import cell_means_vectorized, segmented_image
from metrics import compare_images
from mosaic_mapping import reconstruct_mosaic
from preprocessing import prepare_image
from tile_library import create_tile_set


## Function 1 -> Execute the complete image mosaic workflow
def create_mosaic(image: np.ndarray, grid_size: int, tile_set_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare an image, segment it, reconstruct its mosaic, and print metrics.

    Args:
        * image (np.ndarray) -> User-provided image array
        * grid_size (int) -> Number of rows and columns in the image grid
        * tile_set_name (str) -> Predefined tile family selected by the user

    Returns:
        * tuple -> Preprocessed, segmented, and reconstructed RGB images
    """
    ## Start timing the complete image reconstruction workflow.
    processing_start = perf_counter()

    ## Normalize the uploaded image to the fixed pipeline resolution.
    prepared = prepare_image(image)
    tile_size = prepared.shape[0] // grid_size

    ## Analyze grid cells and construct the segmented visualization.
    cell_means = cell_means_vectorized(prepared, grid_size)
    segmented = segmented_image(prepared, cell_means)

    ## Generate tiles matching each cell and reconstruct the mosaic.
    tiles = create_tile_set(tile_size, tile_set_name)
    mosaic = reconstruct_mosaic(cell_means, tiles)

    ## Calculate reconstruction quality and processing time.
    metric_values = compare_images(prepared, mosaic)
    benchmark_results = benchmark_grid_operations(prepared)
    processing_seconds = perf_counter() - processing_start
    print("\nMetrics:")
    print("MSE:", format(metric_values["MSE"], ".2f"))
    print("SSIM:", format(metric_values["SSIM"], ".4f"))
    print("Processing time:", format(processing_seconds, ".6f"), "seconds\n")
    print_benchmark_results(benchmark_results)
    return prepared, segmented, mosaic
