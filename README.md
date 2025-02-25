# Optimized Fully Fused Differentiable SSIM

This repository provides an optimized version of [fused-ssim](https://github.com/rahul-goel/fused-ssim), achieving better performance while maintaining the same interface. Below are performance comparisons measured using `tests/test.py`.

## Performance Comparison (measured on RTX 4090)

| Implementation            | Forward Time (ms) | Backward Time (ms) | Inference Time (ms) |
|---------------------------|------------------|------------------|------------------|
| **Optimized Fused-SSIM**  | **2.49**         | **2.68**         | **1.43**         |
| **Original Fused-SSIM**   | 3.66             | 3.52             | 2.59             |

## Optimizations

### 1. Fused Computations
- The original implementation computes multiple statistics (mean, variance, covariance) in separate passes, requiring multiple memory accesses.
- **Optimization:** The computations are fused into a **single pass**, reducing redundant memory accesses by using shared memory for intermediate results.

### 2. Exploiting Symmetry in Gaussian Convolution
- The Gaussian filter is symmetric (e.g., `G_00 = G_10`), but the original implementation does not take advantage of this.
- **Optimization:** Pairs symmetric elements to **halve the number of multiplications** from **11 to 6**.

### 3. Constant Memory for Gaussian Coefficients
- The original implementation uses `#define` macros for Gaussian coefficients, increasing register pressure.
- **Optimization:** Stores coefficients in **CUDA constant memory (`__constant__ float cGauss[11]`)** for:
  - Faster memory access.
  - Reduced register pressure.
  - Improved scalability across GPUs.

### 4. Smaller Block Size
- The original implementation uses **32x32** thread blocks.
- **Optimization:** Uses **16x16** blocks, which improves GPU **occupancy** by reducing resource usage per block.

### 5. Efficient Shared Memory Usage
- The original implementation loads `img1` and `img2` into separate shared memory arrays.
- **Optimization:**
  - Uses a **3D shared memory array** (`sTile[SHARED_Y][SHARED_X][2]`) to **load both images simultaneously**, reducing global memory accesses.
  - Stores intermediate sums in a **unified shared memory array**, improving **data locality** and minimizing memory fragmentation.

### 6. Reduced Computational Redundancy
- The original implementation performs **separate convolution operations** for each statistic.
- **Optimization:** Uses a **single convolution pass** for all statistics, reducing redundant computations.

##
Special thanks to [Florian Hahlbohm](https://github.com/fhahlbohm) for helping me verifying that my optimization don't break anything.

## Radiance Fields Dev Discord
[![](https://dcbadge.limes.pink/api/server/https://discord.gg/TbxJST2BbC)](https://discord.gg/TbxJST2BbC)

## 📖 Citation

If you use this optimized fused-SSIM implementation for your research, please cite both the original paper and this implementation:

```bibtex
@inproceedings{optimized-fused-ssim,
    author = {Janusch Patas},
    title = {Optimized Fused-SSIM},
    year = {2025},
    url = {https://github.com/MrNeRF/optimized-fused-ssim},
}
@inproceedings{taming3dgs,
    author = {Mallick, Saswat Subhajyoti and Goel, Rahul and Kerbl, Bernhard and Steinberger, Markus and Carrasco, Francisco Vicente and De La Torre, Fernando},
    title = {Taming 3DGS: High-Quality Radiance Fields with Limited Resources},
    year = {2024},
    url = {https://doi.org/10.1145/3680528.3687694},
    doi = {10.1145/3680528.3687694},
    booktitle = {SIGGRAPH Asia 2024 Conference Papers},
    series = {SA '24}
}
