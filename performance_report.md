# Interactive Image Mosaic Generator

By: Joaquin Saldarriaga  
CS5130 – Lab #1; Lab Report

## Introduction

The main task for this assignment was to create an Interactive Image Mosaic Generator; using Gradio as a visual interface for representing a few sample images to eventually display the preprocessed image (original), segmented (pixelized) image, and eventually a mosaic representation of the initial image.

This document presents the approach defined for this task, the similarity metrics used for the evaluation of the resulting mosaics, the performance procedure, the interpretation of the results and, eventually, the conclusions obtained from the implementation.

The implementation was tested using four personal JPEG images. Since the images had different compositions and aspect ratios, each image was center-cropped and resized before applying the grid. In this way, both square and non-square images could be processed consistently.

## Development

### Method for the approach

The defined pipeline center-cropped each RGB input and resized it to a dimension of ‘512x512’. Afterwards, the image was represented as a fixed grid; each cell being summarized by its mean RGB color using vectorized NumPy reshaping. Eventually, the nearest representative tile was selected by squared RGB distance, and indexed tile arrays were reassembled into the final mosaic.

The resulting segmented image expanded each ‘cell mean’ back across its respective area. The final reconstruction used one of three predefined tile families: ‘Solid’, ‘Diagonal’, or ‘Checkerboard’. This allowed the same image to be represented through different tile patterns while maintaining the same general pipeline.

### Similarity metrics for the evaluation

To evaluate the resulting mosaic representation, two similarity metrics were used. The first one was Mean Squared Error (MSE), which calculates the average squared difference between the pixels of the preprocessed image and the pixels of the resulting mosaic. Therefore, lower MSE values represent smaller pixel-level differences.

The second metric was the Structural Similarity Index (SSIM). This metric evaluates how structurally similar the two images are, where values closer to ‘1.0’ indicate stronger similarity. The metrics were calculated against the resized ‘512x512’ image because this was the image that was actually divided into the grid.

### Performance procedure

The performance comparison was developed by measuring the execution time of the vectorized NumPy implementation and the equivalent loop-based implementation. The comparison was performed using ‘16x16’, ‘32x32’, and ‘64x64’ grid sizes.

The following table presents one representative benchmark obtained during the testing process:

| Grid size | Vectorized time | Loop time | Speedup |
| --- | ---: | ---: | ---: |
| ‘16x16’ | 0.005080 seconds | 0.007807 seconds | 1.54x |
| ‘32x32’ | 0.005938 seconds | 0.016864 seconds | 2.84x |
| ‘64x64’ | 0.007165 seconds | 0.046622 seconds | 6.51x |

The results show that the vectorized implementation remained relatively stable in its execution time. On the other hand, the loop-based implementation became progressively slower as the number of grid cells increased. Eventually, the ‘64x64’ comparison showed that the vectorized approach was more than six times faster than the loop-based approach.

### Testing procedure and resulting observations

The four sample images were processed using all three grid sizes and all three tile families. Therefore, the complete test produced ‘36’ reconstructions. Every resulting output had dimensions of ‘512x512x3’, which confirmed that the square and non-square input images were successfully normalized before reconstruction.

Across the tested combinations, the MSE values ranged from ‘2608.09’ to ‘17977.33’, while the SSIM values ranged from ‘0.0146’ to ‘0.4505’. The lowest MSE was obtained for sample image 1 using the ‘32x32’ grid and the ‘Diagonal’ tile set. The highest SSIM was obtained for sample image 2 using the ‘16x16’ grid and the ‘Solid’ tile set.

These results indicate that the tile family and the visual content of the image influenced the similarity metrics more strongly than the grid size alone. The ‘Solid’ tile family generally preserved broader color structures, while the ‘Diagonal’ and ‘Checkerboard’ tile families created more stylized representations that were not always as structurally similar to the original image.

### Gradio demonstration

The Gradio application was launched successfully at ‘http://127.0.0.1:7860’. The interface allowed an image to be uploaded, a grid size to be selected, and a tile family to be chosen. Afterwards, the interface displayed the preprocessed image, the segmented image, and the final mosaic representation.

The MSE, SSIM, processing time, and vectorized-versus-loop benchmark results were printed in the running Python terminal. This provided both a visual result through Gradio and a numerical result through the terminal output.

## Conclusions

The developed solution correctly implemented the required image-to-mosaic workflow. The application successfully handled square and non-square JPEG inputs, divided each image into the requested grid, replaced each cell with a representative tile, and displayed the resulting images through Gradio.

The performance results also demonstrated the benefit of using vectorized NumPy operations instead of nested Python loops. This difference became particularly clear with the ‘64x64’ grid, where the vectorized implementation achieved a speedup greater than six times in the representative benchmark.

Finally, this implementation represents a deterministic image-processing pipeline rather than a trainable machine-learning model. Therefore, a training, train-test split, and learned validation workflow were not included, since those elements were outside the scope of the assignment. The reproducible testing process was performed with ‘python test_pipeline.py’, and the generated preprocessed, segmented, and mosaic images were saved in the ‘outputs’ folder.
