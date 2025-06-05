# Challenge Explanation

For this assignment, I implemented a full 3D Gaussian Splatting (3DGS) rendering pipeline on the GPU using CUDA. The main challenge was to efficiently map the splatting and compositing process to the GPU, while minimizing memory transfers and maximizing parallelism.

## Optimization 1: GPU-Based Tile Counting and Prefix Sum

To improve performance, I offloaded both tile counting and the prefix sum (exclusive scan) operations entirely to the GPU using CUB. This removed unnecessary host-device synchronization and enabled the pipeline to efficiently handle large scenes with many Gaussians. All per-Gaussian and per-tile computations now run in parallel on the device, with only the final image transferred back to the host.

### Performance Results

- **Before optimization** : 752.37 ms/iteration (average)
- **After optimization** :  695.28 ms/iteration (average)
- **Gain** : 7.5% per iterations 


## Optimization 2: CUB Radix sort : `end_bit`

### First version
This does not really improove performance but reduce the number of computations by only looking at the 40 bits "in the middle". To found that i tried different `start_bit` and `end_bit` value to reduce the number of digits to look at.

### Performance Results

Same than before or really near form previous.
- **Before optimization :** 695.28 ms/iteration (average)
- **After optimization :** 689.29 ms/iteration (average)
- **Gain** : ~1% per iterations

### Second version
Fir the second version I changed how keys were created and then using radix sort. 

### Performance Results

Same than before or really near form previous.
- **Before optimization :** 695.28 ms/iteration (average)
- **After optimization :** 672.91 ms/iteration (average)
- **Gain** : ~3% per iterations
