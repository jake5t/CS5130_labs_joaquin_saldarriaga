## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  1 -> Week 1 - Similarity Metrics File

"""Measure similarity between the preprocessed source and mosaic images.

Metrics:
    * Mean Squared Error, where lower values indicate smaller pixel error
    * Structural Similarity, where values closer to one indicate stronger similarity
"""

## ********** Import Required Libraries **********
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


## Function 1 -> Calculate Mean Squared Error between two RGB images
def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return average squared pixel error between two RGB images.

    Args:
        * original (np.ndarray) -> The original RGB image
        * reconstructed (np.ndarray) -> The reconstructed RGB image
        
    Returns:
        * float -> Mean squared pixel difference
    """

    ## Convert to float before subtracting to prevent uint8 arithmetic overflow.
    difference = original.astype(np.float32) - reconstructed.astype(np.float32)

    ## Return the *mean* of the 'squared pixel' differences across all channels
    return float(np.mean(difference ** 2))


## Function 2 -> Calculate Structural Similarity between two RGB images
def structural_similarity_index(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return multichannel structural similarity between two RGB images.

    Args:
        * original (np.ndarray) -> The original RGB image
        * reconstructed (np.ndarray) -> The reconstructed RGB image

    Returns:
        * float -> Structural similarity score in the metric's normalized range
    
    """

    ## Compute SSIM over RGB channels using the current scikit-image API.
    return float(structural_similarity(original, reconstructed, channel_axis = -1, data_range = 255))


## Function 3 -> Calculate both assignment metrics for one reconstruction
def compare_images(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """Return the required image similarity metrics.

    Args:
        * original (np.ndarray) -> The original RGB image
        * reconstructed (np.ndarray) -> The reconstructed RGB image
        
    Returns:
        * dict -> MSE and SSIM values indexed by metric name
    """

    ## Calculate and return *both* assignment metrics for one reconstruction.
    return {
        "MSE": mean_squared_error(original, reconstructed),
        "SSIM": structural_similarity_index(original, reconstructed),
    }
