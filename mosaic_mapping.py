## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  1 -> Week 1 - Mosaic Mapping File

"""Map representative image colors to predefined tiles and rebuild the mosaic.

Process:
    * Calculate distances between cells and tile representatives
    * Select the nearest tile for each cell
    * Reassemble selected tiles into one image
"""

## ********** Import Required Libraries **********

## Import 'annotations' module for fwd-reference type hints (used in the Lab for type annotations)
from __future__ import annotations

import numpy as np

from tile_library import tile_representative_colors


## Function 1 -> Classify every image cell by nearest tile color
def classify_cells(cell_means: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    """Assign each grid cell to its nearest predefined tile color.

    Args:
        * cell_means (np.ndarray) -> Representative RGB value for every image cell
        * tiles (np.ndarray) -> Predefined RGB tile images

    Returns:
        * np.ndarray -> Tile index assigned to every grid cell
    """
    ## Compute squared RGB distances between every cell and tile representative.
    representatives = tile_representative_colors(tiles)
    distances = ((cell_means[..., None, :] - representatives) ** 2).sum(axis=-1)

    ## Select the closest tile independently for every grid cell.
    return distances.argmin(axis = -1)


## Function 2 -> Reconstruct the image from classified tile indices
def reconstruct_mosaic(cell_means: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    """Replace every grid cell with its classified tile using array indexing.

    Args:
        * cell_means (np.ndarray) -> Representative RGB value for every image cell
        * tiles (np.ndarray) -> Predefined RGB tile images

    Returns:
        * np.ndarray -> RGB uint8 mosaic image assembled from selected tiles
    """
    ## Classify cells and gather their corresponding tile images.
    tile_indices = classify_cells(cell_means, tiles)
    selected_tiles = tiles[tile_indices]
    grid_height, grid_width, tile_height, tile_width, channels = selected_tiles.shape

    ## Reassemble tiled cells into one contiguous image without pixel loops.
    mosaic = selected_tiles.transpose(0, 2, 1, 3, 4)
    return mosaic.reshape(grid_height * tile_height, grid_width * tile_width, channels)
