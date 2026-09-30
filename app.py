## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  1 -> Week 1 - Gradio Application

"""Main code file that runs the Gradio interface for interactive image mosaic reconstruction.

Interface:
    * Accept an uploaded image
    * Select a grid size
    * Select a predefined tile family
    * Display preprocessing, segmentation, and reconstruction outputs

"""

## ********** Import Required Libraries **********
from __future__ import annotations

import gradio as gr

from mosaic_pipeline import create_mosaic


## Function 1 -> Build the interactive Gradio interface
def build_interface() -> gr.Interface:
    """Build the assignment interface with image, grid, and tile controls.

    Returns:
        * gr.Interface -> Configured interface connected to the mosaic pipeline
    """
    ## Define the user inputs required by the mosaic pipeline.
    inputs = [
        gr.Image(type="numpy", label = "Input image"),
        gr.Dropdown([16, 32, 64], value=32, label = "Grid size"),
        gr.Dropdown(["Solid", "Diagonal", "Checkerboard"], value="Diagonal", label = "Tile set"),
    ]

    ## Define the preprocessed, segmented, and mosaic outputs.
    outputs = [
        gr.Image(label = "Preprocessed image"),
        gr.Image(label = "Segmented image"),
        gr.Image(label = "Mosaic reconstruction"),
    ]

    ## Create the interactive Gradio application with the pipeline callback.
    return gr.Interface(
        fn = create_mosaic,
        inputs = inputs,
        outputs = outputs,
        title = "Interactive Image Mosaic Generator",
        description = "Upload an image, choose a grid, and reconstruct it with predefined image tiles.",
    )


## ********** Launch Gradio Application **********
if __name__ == "__main__":
    build_interface().launch()
