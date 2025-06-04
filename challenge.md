# Challenge Explanation

For this assignment, I implemented a full 3D Gaussian Splatting (3DGS) rendering pipeline on the GPU using CUDA. The main challenge was to efficiently map the splatting and compositing process to the GPU, while minimizing memory transfers and maximizing parallelism.

## Optimization 1: GPU-Based Tile Counting and Prefix Sum

To improve performance, I offloaded both tile counting and the prefix sum (exclusive scan) operations entirely to the GPU using CUB. This removed unnecessary host-device synchronization and enabled the pipeline to efficiently handle large scenes with many Gaussians. All per-Gaussian and per-tile computations now run in parallel on the device, with only the final image transferred back to the host.

### Performance Results

- **Before optimization :** 752.37 ms/iteration (average)
- **After optimization :** 695.28 ms/iteration (average)
- **Gain** : 7.5% per iterations 
