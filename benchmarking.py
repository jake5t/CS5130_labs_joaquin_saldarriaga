## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  1 -> Week 1 - Performance Benchmarking File

"""Compare vectorized and loop-based grid analysis performance.

Process:
    * Run both implementations for each requested grid size
    * Measure elapsed execution time
    * Print the resulting timings and speedups
"""

## ********** Import Required Libraries **********
from __future__ import annotations

from time import perf_counter

import numpy as np

from grid_analysis import cell_means_loop, cell_means_vectorized


## Function 1 -> Measure vectorized and loop-based grid analysis
def benchmark_grid_operations(image: np.ndarray, grid_sizes: tuple[int, ...] = (16, 32, 64)) -> list[dict[str, float | int]]:
    """Measure vectorized and loop-based grid analysis across requested resolutions.

    Returns:
        * list -> Timing dictionaries for every requested grid size
    """
    ## Store one result row for each requested grid size.
    results = []
    for grid_size in grid_sizes:
        ## Time vectorized analysis once for a directly comparable measurement.
        vector_start = perf_counter()
        cell_means_vectorized(image, grid_size)
        vectorized_seconds = perf_counter() - vector_start

        ## Time loop-based analysis once for the required comparison.
        loop_start = perf_counter()
        cell_means_loop(image, grid_size)
        loop_seconds = perf_counter() - loop_start

        ## Record grid dimensions, timings, and relative speedup.
        results.append({
            "grid_size": grid_size,
            "vectorized_seconds": vectorized_seconds,
            "loop_seconds": loop_seconds,
            "speedup": loop_seconds / vectorized_seconds if vectorized_seconds else float("inf"),
        })
    return results


## Function 2 -> Print benchmark results as individual lines
def print_benchmark_results(results: list[dict[str, float | int]]) -> None:
    """Print benchmark results as individual lines.

    Args:
        * results (list) -> Timing dictionaries returned by the benchmark function
    """
    ## Print a readable performance table header.
    print("Grid | Vectorized (s) | Loops (s) | Speedup")
    print("--- | ---: | ---: | ---:")

    ## Format each measured grid size with stable decimal precision.
    for result in results:
        grid_label = str(result["grid_size"]) + "x" + str(result["grid_size"])
        vectorized_label = format(result["vectorized_seconds"], ".6f")
        loop_label = format(result["loop_seconds"], ".6f")
        speedup_label = format(result["speedup"], ".2f") + "x"
        print(grid_label, "|", vectorized_label, "|", loop_label, "|", speedup_label)
