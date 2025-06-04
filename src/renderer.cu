#include "renderer.h"

#include "renderer_common.h"
#include "3dgs_common.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"

#include "cub/cub.cuh"

#include <cmath>
#include <vector>
#include "float.h"

constexpr float EXTENT = 3.33f;

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

// 1. Preprocess: Compute 2D means, covariances, colors, and tile bounds for each Gaussian
__global__ void preprocess_gaussians(
    uint32_t n_gaussians,
    FrameInfo frame,
    const float3* __restrict__ pos,
    const float3* __restrict__ scale,
    const float4* __restrict__ rot,
    const float* __restrict__ shs,
    const float* __restrict__ opacity,
    float4* __restrict__ invCov2D_opacities,
    float2* __restrict__ means2D,
    float3* __restrict__ colors,
    uint2* __restrict__ rect_min,
    uint2* __restrict__ rect_max,
    float* __restrict__ depths,
    uint32_t* __restrict__ valid
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_gaussians) return;

    const float3 mean3D = pos[idx];
    glm::vec3 mean3D_vs = frame.view_matrix * glm::vec4(mean3D.x, mean3D.y, mean3D.z, 1.0f);

    if (mean3D_vs.z < frame.near) {
        valid[idx] = 0;
        depths[idx] = FLT_MAX;
        return;
    }

    const glm::mat3 cov3D = computeCov3D(scale[idx], rot[idx]);
    const glm::mat2 cov2D = computeCov2D(mean3D_vs, cov3D, frame.view_matrix, frame.focal, frame.resolution);

    const glm::mat3x3 proj_matrix = buildProjMat(frame.focal, frame.resolution);
    const glm::vec2 mean2D = glm::vec2(proj_matrix * (mean3D_vs / mean3D_vs.z)) - 0.5f;

    // Compute circular radius, dependent on the largest eigenvalue
    const float min_lambda = 0.01f;
    const float mid = 0.5f * (cov2D[0][0] + cov2D[1][1]);
    const float lambda = mid + sqrtf(fmaxf(min_lambda, mid * mid - glm::determinant(cov2D)));
    const float circle_extent_radius = EXTENT * sqrtf(lambda);
    const float2 circle_extent_square_dims = make_float2(circle_extent_radius);

    // Determine square dimensions in pixel-coordinates (left upper and right lower pixel), bounded by screen resolution
    uint2 rmin, rmax;
    getRect(make_float2(mean2D.x, mean2D.y), circle_extent_square_dims, frame.resolution, rmin, rmax);
    int pixel_count = (rmax.x - rmin.x) * (rmax.y - rmin.y);

    if (pixel_count == 0) {
        valid[idx] = 0;
        depths[idx] = FLT_MAX;
        return;
    }

    valid[idx] = 1;
    depths[idx] = mean3D_vs.z;
    rect_min[idx] = rmin;
    rect_max[idx] = rmax;

    const glm::mat2 inv_cov2D = glm::inverse(cov2D);
    invCov2D_opacities[idx] = make_float4(inv_cov2D[0][0], inv_cov2D[0][1], inv_cov2D[1][1], opacity[idx]);
    means2D[idx] = make_float2(mean2D.x, mean2D.y);

    const glm::vec3 viewdir = glm::normalize(glm::vec3(mean3D.x, mean3D.y, mean3D.z) - frame.cam_pos);
    const glm::vec3 color = computeColorFromSH(idx, 3, viewdir, shs);
    colors[idx] = make_float3(color.x, color.y, color.z);
}

// 2. Compute tile counts per Gaussian (on device)
__global__ void compute_tile_counts(
    uint32_t n_gaussians,
    const uint2* __restrict__ rect_min,
    const uint2* __restrict__ rect_max,
    const uint32_t* __restrict__ valid,
    uint32_t* __restrict__ tile_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_gaussians) return;
    if (!valid[i]) {
        tile_counts[i] = 0;
        return;
    }
    uint32_t tile_min_x = rect_min[i].x / TILE_SIZE;
    uint32_t tile_max_x = (rect_max[i].x - 1) / TILE_SIZE;
    uint32_t tile_min_y = rect_min[i].y / TILE_SIZE;
    uint32_t tile_max_y = (rect_max[i].y - 1) / TILE_SIZE;
    tile_counts[i] = (tile_max_x - tile_min_x + 1) * (tile_max_y - tile_min_y + 1);
}

// 3. Duplicate: For each Gaussian, register it to all tiles it covers
__global__ void duplicate_gaussians_to_tiles(
    uint32_t n_gaussians,
    const uint2* __restrict__ rect_min,
    const uint2* __restrict__ rect_max,
    const float* __restrict__ depths,
    const uint32_t* __restrict__ valid,
    uint32_t tiles_x,
    uint32_t tiles_y,
    const uint32_t* __restrict__ offsets,
    uint64_t* __restrict__ unsorted_keys,
    uint32_t* __restrict__ unsorted_idcs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_gaussians || !valid[idx]) return;

    uint2 rmin = rect_min[idx];
    uint2 rmax = rect_max[idx];
    float depth = depths[idx];

    uint32_t tile_min_x = rmin.x / TILE_SIZE;
    uint32_t tile_max_x = (rmax.x - 1) / TILE_SIZE;
    uint32_t tile_min_y = rmin.y / TILE_SIZE;
    uint32_t tile_max_y = (rmax.y - 1) / TILE_SIZE;

    uint32_t offset = offsets[idx];
    for (uint32_t ty = tile_min_y; ty <= tile_max_y; ++ty) {
        for (uint32_t tx = tile_min_x; tx <= tile_max_x; ++tx) {
            uint32_t tile_idx = ty * tiles_x + tx;
            unsorted_keys[offset] = constructKey(tile_idx, depth);
            unsorted_idcs[offset] = idx;
            ++offset;
        }
    }
}

// 5. Identify tile ranges in sorted list
__global__ void identify_tile_ranges(
    uint32_t n_sort_entries,
    const uint64_t* __restrict__ sorted_keys,
    uint32_t* __restrict__ tile_range_starts,
    uint32_t* __restrict__ tile_range_ends,
    uint32_t n_tiles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sort_entries) return;

    uint32_t tile_idx, tile_idx_prev;
    float depth;
    deconstructKey(sorted_keys[idx], tile_idx, depth);

    if (idx == 0) {
        tile_range_starts[tile_idx] = 0;
    } else {
        deconstructKey(sorted_keys[idx - 1], tile_idx_prev, depth);
        if (tile_idx != tile_idx_prev) {
            tile_range_starts[tile_idx] = idx;
            tile_range_ends[tile_idx_prev] = idx;
        }
    }
    if (idx == n_sort_entries - 1) {
        tile_range_ends[tile_idx] = n_sort_entries;
    }
}

// 6. Render: Each pixel only considers Gaussians in its tile
__global__ void render_tiles(
    FrameInfo frame,
    uint32_t n_tiles_x,
    uint32_t n_tiles_y,
    const uint64_t* __restrict__ sorted_keys,
    const uint32_t* __restrict__ sorted_idcs,
    const uint32_t* __restrict__ tile_range_starts,
    const uint32_t* __restrict__ tile_range_ends,
    const float4* __restrict__ invCov2D_opacities,
    const float2* __restrict__ means2D,
    const float3* __restrict__ colors,
    cudaSurfaceObject_t img_out_surf
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= frame.resolution.x || y >= frame.resolution.y) return;

    uint32_t tile_x = x / TILE_SIZE;
    uint32_t tile_y = y / TILE_SIZE;
    uint32_t tile_idx = tile_y * n_tiles_x + tile_x;

    uint32_t start = tile_range_starts[tile_idx];
    uint32_t end = tile_range_ends[tile_idx];

    float T = 1.0f;
    float3 out_color = make_float3(0.0f);

    for (uint32_t i = start; i < end; ++i) {
        uint32_t gaussian_idx = sorted_idcs[i];
        const float4 invCov2D_opacity = invCov2D_opacities[gaussian_idx];
        const float3 inv_cov2D = make_float3(invCov2D_opacity);
        const float opacity = invCov2D_opacity.w;
        const float2 mean2D = means2D[gaussian_idx];

        float2 d = make_float2(x - mean2D.x, y - mean2D.y);
        float power = -0.5f * ((inv_cov2D.x * d.x * d.x + inv_cov2D.z * d.y * d.y) + 2.0f * inv_cov2D.y * d.x * d.y);
        if (power > 0.0f) continue;

        float alpha = min(0.99f, opacity * expf(power));
        if (alpha < ALPHA_THRESHOLD) continue;

        float3 color = colors[gaussian_idx];
        out_color += color * T * alpha;
        T *= (1.0f - alpha);

        if (T < T_THRESHOLD) break;
    }

    surf2Dwrite(color_to_uchar4(out_color), img_out_surf, x * sizeof(uchar4), y);
}

// --- Renderer class ---

Renderer::Renderer()
    : _buffer_size(0), _d_unsorted_keys(nullptr), _d_unsorted_idcs(nullptr),
      _d_sorted_keys(nullptr), _d_sorted_idcs(nullptr), _d_temp_storage_sort(nullptr),
      _temp_storage_sort_size_bytes(0), _d_tile_range_starts(nullptr), _d_tile_range_ends(nullptr),
      _d_tile_counts(nullptr)
{
}

Renderer::~Renderer()
{
    if (_buffer_size > 0) {
        cudaFree(_d_unsorted_keys);
        cudaFree(_d_unsorted_idcs);
        cudaFree(_d_sorted_keys);
        cudaFree(_d_sorted_idcs);
        cudaFree(_d_temp_storage_sort);
        cudaFree(_d_tile_range_starts);
        cudaFree(_d_tile_range_ends);
        cudaFree(_d_tile_counts);
    }
}

void Renderer::setup(const DatasetGPU data, uint2 resolution)
{
    int n_gaussians = data.n_gaussians;
    CUDA_CHECK_THROW(cudaMalloc(&(_d_splats.invCov2D_opacities), sizeof(float4) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&(_d_splats.means2D), sizeof(float2) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&(_d_splats.colors), sizeof(float3) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&_d_rect_min, sizeof(uint2) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&_d_rect_max, sizeof(uint2) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&_d_depths, sizeof(float) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&_d_valid, sizeof(uint32_t) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&_d_tile_counts, sizeof(uint32_t) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&_d_tile_offsets, sizeof(uint32_t) * (n_gaussians + 1)));
}

void Renderer::run(const DatasetGPU data, FrameInfo& frame, cudaSurfaceObject_t img_out_surf, bool warmup)
{
    uint32_t n_gaussians = data.n_gaussians;
    uint32_t tiles_x = (frame.resolution.x + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t tiles_y = (frame.resolution.y + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t n_tiles = tiles_x * tiles_y;

    constexpr uint32_t BLOCK_SIZE = 256;
    preprocess_gaussians<<<(n_gaussians + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        n_gaussians, frame, data.d_pos, data.d_scale, data.d_rot, (float*)data.d_shs, data.d_opacity,
        _d_splats.invCov2D_opacities, _d_splats.means2D, _d_splats.colors,
        _d_rect_min, _d_rect_max, _d_depths, _d_valid
    );

    // --- 2. Compute tile counts and prefix sum on device ---
    compute_tile_counts<<<(n_gaussians + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        n_gaussians, _d_rect_min, _d_rect_max, _d_valid, _d_tile_counts);

    // Prefix sum (exclusive scan) for offsets
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, _d_tile_counts, _d_tile_offsets, n_gaussians + 1);
    CUDA_CHECK_THROW(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, _d_tile_counts, _d_tile_offsets, n_gaussians + 1);

    // Get total number of <tile, gaussian> pairs (n_sort_entries)
    uint32_t n_sort_entries = 0;
    CUDA_CHECK_THROW(cudaMemcpy(&n_sort_entries, _d_tile_offsets + n_gaussians, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // --- Buffer allocation ---
    if ((warmup && n_sort_entries > _buffer_size) || _buffer_size == 0) {
        if (_buffer_size > 0) {
            cudaFree(_d_unsorted_keys);
            cudaFree(_d_unsorted_idcs);
            cudaFree(_d_sorted_keys);
            cudaFree(_d_sorted_idcs);
            cudaFree(_d_temp_storage_sort);
            cudaFree(_d_tile_range_starts);
            cudaFree(_d_tile_range_ends);
        }
        _buffer_size = 2 * (1U << (int)ceil(log2(n_sort_entries + 1)));
        CUDA_CHECK_THROW(cudaMalloc(&_d_unsorted_keys, sizeof(uint64_t) * _buffer_size));
        CUDA_CHECK_THROW(cudaMalloc(&_d_unsorted_idcs, sizeof(uint32_t) * _buffer_size));
        CUDA_CHECK_THROW(cudaMalloc(&_d_sorted_keys, sizeof(uint64_t) * _buffer_size));
        CUDA_CHECK_THROW(cudaMalloc(&_d_sorted_idcs, sizeof(uint32_t) * _buffer_size));
        CUDA_CHECK_THROW(cudaMalloc(&_d_tile_range_starts, sizeof(uint32_t) * n_tiles));
        CUDA_CHECK_THROW(cudaMalloc(&_d_tile_range_ends, sizeof(uint32_t) * n_tiles));
        _temp_storage_sort_size_bytes = 0;
        cub::DeviceRadixSort::SortPairs(_d_temp_storage_sort, _temp_storage_sort_size_bytes,
            _d_unsorted_keys, _d_sorted_keys, _d_unsorted_idcs, _d_sorted_idcs, _buffer_size);
        CUDA_CHECK_THROW(cudaMalloc(&_d_temp_storage_sort, _temp_storage_sort_size_bytes));
    }

    // --- 3. Duplicate ---
    duplicate_gaussians_to_tiles<<<(n_gaussians + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        n_gaussians, _d_rect_min, _d_rect_max, _d_depths, _d_valid,
        tiles_x, tiles_y, _d_tile_offsets, _d_unsorted_keys, _d_unsorted_idcs
    );

    // --- 4. Global sort ---
    cub::DeviceRadixSort::SortPairs(_d_temp_storage_sort, _temp_storage_sort_size_bytes,
        _d_unsorted_keys, _d_sorted_keys, _d_unsorted_idcs, _d_sorted_idcs, n_sort_entries);

    // --- 5. Identify tile ranges ---
    cudaMemset(_d_tile_range_starts, 0xFF, sizeof(uint32_t) * n_tiles);
    cudaMemset(_d_tile_range_ends, 0xFF, sizeof(uint32_t) * n_tiles);
    identify_tile_ranges<<<(n_sort_entries + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        n_sort_entries, _d_sorted_keys, _d_tile_range_starts, _d_tile_range_ends, n_tiles
    );

    // --- 6. Render ---
    constexpr uint32_t BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 image_grid_size_2D((frame.resolution.x + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, (frame.resolution.y + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D);
    render_tiles<<<image_grid_size_2D, block_size_2D>>>(
        frame, tiles_x, tiles_y, _d_sorted_keys, _d_sorted_idcs, _d_tile_range_starts, _d_tile_range_ends,
        _d_splats.invCov2D_opacities, _d_splats.means2D, _d_splats.colors, img_out_surf
    );

    cudaFree(d_temp_storage);
}