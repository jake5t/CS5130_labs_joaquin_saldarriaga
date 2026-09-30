## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  1 -> Week 1 - Tile Library File

"""Create predefined procedural image tiles for mosaic reconstruction.

Process:
    * Build normalized tile coordinates
    * Select the requested tile family
    * Apply predefined colors to each pattern

"""

## ********** Import Required Libraries **********
from __future__ import annotations

import numpy as np


## Function 1 -> Create a predefined RGB tile set
def create_tile_set(tile_size: int, tile_set_name: str) -> np.ndarray:
    """Create a predefined RGB tile set with distinct visual structures.

    Args:
        * tile_size (int) -> Width and height of every generated tile
        * tile_set_name (str) -> Requested Solid, Diagonal, or Checkerboard family

    Returns:
        * np.ndarray -> Stack of RGB uint8 tile images
    """
    ## Build normalized coordinates shared by every generated tile.
    coordinates = np.linspace(0, 1, tile_size, dtype = np.float32)
    x, y = np.meshgrid(coordinates, coordinates)
    tiles = []

    ## Select the requested predefined tile family.
    if tile_set_name == "Solid":
        patterns = [np.ones_like(x), x, y, (x + y) / 2]

    ## If the tile set is Diagonal, create four diagonal patterns with different orientations.
    elif tile_set_name == "Diagonal":
        patterns = [x, y, np.abs(x - y), (x + y) / 2]

    ## If the tile set is *Checkerboard*, create four checkerboard patterns with different colors.
    elif tile_set_name == "Checkerboard":
        checks = ((np.floor(x * 8) + np.floor(y * 8)) % 2).astype(np.float32)
        patterns = [checks, 1 - checks, x * checks, y * (1 - checks)]

    ## Otherwise, if the tile set is *unknown*, raise an error.
    else:
        raise ValueError("Unknown tile set: " + tile_set_name)

    ## Convert each grayscale pattern into a colored RGB tile.
    colors = np.array([[220, 70, 55], [45, 125, 210], [45, 165, 105], [235, 180, 45]], dtype = np.float32)

    ## Iterate across each 'pattern, color' pair for tile generation and clipping to *uint8*
    for pattern, color in zip(patterns, colors):
        tile = pattern[..., None] * color
        tiles.append(np.clip(tile, 0, 255).astype(np.uint8))

    return np.stack(tiles)


## Function 2 -> Calculate representative RGB colors for generated tiles
def tile_representative_colors(tiles: np.ndarray) -> np.ndarray:
    """Calculate the average RGB color representing every tile.

    Args:
        * tiles (np.ndarray) -> Stack of RGB uint8 tile images

    Returns:
        * np.ndarray -> One representative RGB color per tile
    """
    ## Average tile pixels to support nearest-color tile selection.
    return tiles.mean(axis = (1, 2), dtype = np.float32)
