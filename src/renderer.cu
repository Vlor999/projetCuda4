
#include "renderer.h"

#include "renderer_common.h"
#include "3dgs_common.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"

#include "cub/cub.cuh"

#include "float.h"

__device__ inline uint64_t constructKey(uint32_t tile_idx, float depth)
{
	uint64_t key = tile_idx;
	key <<= 32;
	key |= *((uint32_t*)&depth);
	return key;
}

__device__ inline void deconstructKey(const uint64_t key, uint32_t& tile_idx, float& depth)
{
	tile_idx = key >> 32;
    uint32_t lower_key = key & 0xFFFFFFFF;
    depth = *((float*) &lower_key);
}

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
}

void Renderer::setup(const DatasetGPU data, uint2 resolution)
{
    // Allocate buffers for splats, and tiles

    // You are also allowed to bring "data" into a different form, if you think it helps rendering performance
    // In this case, ignore the "data" provided during "run" and instead use your own dataset buffers
    // Important: Make sure to not change the original contents of the Dataset!
}

void Renderer::run(const DatasetGPU data, FrameInfo& frame, cudaSurfaceObject_t img_out_surf, bool warmup)
{
    /*
        TODO: Implement the full 3DGS rendering pipeline, splatting the Gaussian point cloud for a given camera view (frame).
              We added a step-by-step guide for how to implement a standard tile-based renderer, as proposed by the original 3DGS.
              If your initial implementation works, feel free to diverge from the provided framework if you feel that it is hindering performance
        
        Do NOT re-use rendering information between runs. You have to perform the complete evaluation each run!
    */

    // 1 - Preprocess: Similar to "prepare" in reference + additional calculation of number of tiles touched by each Gaussian

    // 2 - Prefix Sum over the number of tiles touched by each Gaussian to retrieve overall N sort entries and Gaussian offsets
	uint32_t n_sort_entries = 0;

    // An example of how to properly allocate dynamically sized memory during "warmup". Feel free to add/change this to your requirements
    if (warmup && n_sort_entries > _buffer_size)
    {
        if (_buffer_size > 0)
        {
            // Make sure to free buffers if you are resizing them
            CUDA_CHECK_THROW(cudaFree(_d_unsorted_keys));
            CUDA_CHECK_THROW(cudaFree(_d_unsorted_idcs));
            CUDA_CHECK_THROW(cudaFree(_d_sorted_keys));
            CUDA_CHECK_THROW(cudaFree(_d_sorted_idcs));

            CUDA_CHECK_THROW(cudaFree(_d_temp_storage_sort));            
            _d_temp_storage_sort = nullptr;
            _temp_storage_sort_size_bytes = 0;
        }

        // Overallocate at least 2x memory (+ round up to the next power of two)
        _buffer_size = 2 * (1U << (int) ceil(log2(n_sort_entries)));
        std::cout << "Allocating Buffer of size " << _buffer_size << std::endl;

        CUDA_CHECK_THROW(cudaMalloc(&_d_unsorted_keys, sizeof(uint64_t) * _buffer_size));
        CUDA_CHECK_THROW(cudaMalloc(&_d_unsorted_idcs, sizeof(uint32_t) * _buffer_size));
        CUDA_CHECK_THROW(cudaMalloc(&_d_sorted_keys, sizeof(uint64_t) * _buffer_size));
        CUDA_CHECK_THROW(cudaMalloc(&_d_sorted_idcs, sizeof(uint32_t) * _buffer_size));

        CUDA_CHECK_THROW(cub::DeviceRadixSort::SortPairs(_d_temp_storage_sort, _temp_storage_sort_size_bytes, 
                                                         _d_unsorted_keys, _d_sorted_keys, _d_unsorted_idcs, _d_sorted_idcs, _buffer_size));
        CUDA_CHECK_THROW(cudaMalloc(&_d_temp_storage_sort, _temp_storage_sort_size_bytes));
    }

    // 3 - Duplicate: Populate the sort entries (keys + values) with the (<tile, depth> + gaussian index) values for all possible combinations

    // 4 - Global sort: Perform a large device-wide sort over all possible combinations. Feel free to use other sorting algorithms (e.g. partitioned sorts)
    CUDA_CHECK_THROW(cub::DeviceRadixSort::SortPairs(_d_temp_storage_sort, _temp_storage_sort_size_bytes,
                                                     _d_unsorted_keys, _d_sorted_keys, _d_unsorted_idcs, _d_sorted_idcs, _buffer_size));

    // 5 - Identify Tile Ranges: After sorting, identify the range (from, to) of each Tile's entries

    // 6 - Render: Render each pixel by iterating over all Gaussians of its tile. You should be able to simply copy your implementation from Assignment 3
}