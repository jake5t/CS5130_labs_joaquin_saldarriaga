# Interactive Image Mosaic Generator
## By - Joaquin Saldarriaga (Northeastern University)

This project reconstructs an input image with small procedural image tiles. It follows the assignment stages without training a machine-learning model.

## Files

- `preprocessing.py`: center-crops, converts to RGB, and resizes every input to `512x512`.
- `grid_analysis.py`: computes cell colors with vectorized NumPy operations and provides the loop baseline.
- `tile_library.py`: creates the predefined `Solid`, `Diagonal`, and `Checkerboard` tile sets.
- `mosaic_mapping.py`: maps each cell to its nearest tile color and reassembles the mosaic.
- `metrics.py`: calculates MSE and RGB SSIM against the preprocessed source.
- `benchmarking.py`: compares vectorized and loop timings at `16x16`, `32x32`, and `64x64`.
- `mosaic_pipeline.py`: imports the modules and exposes the complete reconstruction function.
- `app.py`: builds and launches the Gradio interface.
- `test_pipeline.py`: processes PNG files from `sample_images/` and saves outputs.
- `performance_report.md`: concise report structure for submission results.

## Install and launch

Run these commands from this folder:

```powershell
python -m pip install -r requirements.txt
gradio app.py
```

The interface accepts an image, grid size, and tile set. It displays the preprocessed image, segmented image, mosaic reconstruction, MSE, SSIM, and benchmark timings.

## Test your five images

1. Copy the five `.png`, `.jpg`, or `.jpeg` files into `sample_images/`.
2. Run:

```powershell
python test_pipeline.py
```

3. Find each image's preprocessed, segmented, and mosaic outputs in `outputs/`.

Square and non-square images are both supported. JPEG is fully compatible: Pillow decodes it before the same RGB preprocessing used for PNG. The preprocessing stage center-crops the longer dimension before resizing, preserving the central composition while producing dimensions compatible with every requested grid.

## Scope clarification

This assignment does not define a trainable ML model, dataset split, or learned parameters. Therefore, the valid workflow is algorithm testing and validation: run the same deterministic pipeline on your five images, inspect outputs, compare MSE/SSIM, and report timing. Training would be outside the stated scope.
