## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  1 -> Week 1 - Grid Analysis File

"""Analyze image cells with vectorized and loop-based implementations.

Process:
    * Validate grid compatibility
    * Calculate representative RGB values
    * Expand cell means for segmented visualization
"""

## ********** Import Required Libraries **********
from __future__ import annotations

import numpy as np


## Function 1 -> Validate image dimensions against the requested grid
def validate_grid_size(image: np.ndarray, grid_size: int) -> None:
    """Ensure the image dimensions divide evenly into the requested grid.

    Args:
        * image (np.ndarray) -> RGB image whose dimensions are being checked
        * grid_size (int) -> Number of rows and columns in the grid

    Raises:
        * ValueError -> If the grid cannot divide both image dimensions evenly
    """
    ## Reject unsupported grid dimensions before reshaping the image.
    height, width = image.shape[:2]
    if height % grid_size or width % grid_size:
        raise ValueError("Grid size must divide both image dimensions evenly.")


## Function 2 -> Calculate cell means using vectorized NumPy operations
def cell_means_vectorized(image: np.ndarray, grid_size: int) -> np.ndarray:
    """Calculate every cell's RGB mean with NumPy reshaping and reduction.

    Args:
        * image (np.ndarray) -> RGB image whose cells are being analyzed
        * grid_size (int) -> Number of rows and columns in the grid

    Returns:
        * np.ndarray -> Grid of representative RGB values

    """
    ## Validate dimensions before creating the four-dimensional cell view.
    validate_grid_size(image, grid_size)
    height, width = image.shape[:2]
    cell_height = height // grid_size
    cell_width = width // grid_size

    ## Group pixels by grid row, grid column, cell row, and cell column.
    cells = image.reshape(grid_size, cell_height, grid_size, cell_width, 3)
    cells = cells.transpose(0, 2, 1, 3, 4)

    ## Reduce each cell to one representative RGB color.
    return cells.mean(axis=(2, 3), dtype = np.float32)


## Function 3 -> Calculate cell means using explicit Python loops
def cell_means_loop(image: np.ndarray, grid_size: int) -> np.ndarray:
    """Calculate cell means with explicit loops for performance comparison.

    Args:
        * image (np.ndarray) -> RGB image whose cells are being analyzed
        * grid_size (int) -> Number of rows and columns in the grid

    Returns:
        * np.ndarray -> Grid of representative RGB values
    """
    ## Validate dimensions before iterating over grid coordinates.
    validate_grid_size(image, grid_size)
    height, width = image.shape[:2]
    cell_height = height // grid_size
    cell_width = width // grid_size
    means = np.empty((grid_size, grid_size, 3), dtype = np.float32)

    ## Compute each cell mean explicitly to benchmark against vectorization.
    for row in range(grid_size):

        ## Iterate across every *column* in the current row for extracting and averaging *cell pixels*
        for column in range(grid_size):
            top = row * cell_height
            left = column * cell_width
            cell = image[top : top + cell_height, left : left + cell_width]
            means[row, column] = cell.mean(axis=(0, 1))

    return means


## Function 4 -> Reconstruct a block-segmented image from cell means
def segmented_image(image: np.ndarray, cell_means: np.ndarray) -> np.ndarray:
    """Expand cell colors back into a block-segmented image.

    Args:
        * image (np.ndarray) -> Original RGB image for dimension reference
        * cell_means (np.ndarray) -> Grid of representative RGB values

    Returns:
        * np.ndarray -> RGB uint8 image with one uniform color per cell
    """
    ## Determine each grid cell's pixel dimensions.
    height, width = image.shape[:2]
    grid_height, grid_width = cell_means.shape[:2]
    cell_height = height // grid_height
    cell_width = width // grid_width

    ## Repeat each mean color over its original cell area.
    segmented = np.repeat(np.repeat(cell_means, cell_height, axis=0), cell_width, axis=1)
    return np.clip(segmented, 0, 255).astype(np.uint8)
