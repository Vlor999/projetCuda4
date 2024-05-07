
#include "renderer_reference.h"

#include "renderer_common.h"
#include "3dgs_common.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"

#include "cub/cub.cuh"

#include "float.h"

__global__ void prepare(const uint32_t n_gaussians,
                        const FrameInfo frame,
                        uint32_t *__restrict__ idcs,    
                        float *__restrict__ depths,

                        const float3 *__restrict__ pos,
                        const float3 *__restrict__ scale,
                        const float4 *__restrict__ rot,
                        const float *__restrict__ shs,
                        const float *__restrict__ opacity,
                        
                        float4 *__restrict__ invCov2D_opacities,
                        float2 *__restrict__ means2D,
                        float3 *__restrict__ colors)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_gaussians)
        return;
       
    const float3 mean3D = pos[idx];
    glm::vec3 mean3D_vs = frame.view_matrix * glm::vec4(mean3D.x, mean3D.y, mean3D.z, 1.0f); // transform into view-space (vs)

    if (mean3D_vs.z < frame.near)
    {
        idcs[idx] = -1U;
        depths[idx] = FLT_MAX;
        return;
    }

    const glm::mat3 cov3D = computeCov3D(scale[idx], rot[idx]);
    const glm::mat2 cov2D = computeCov2D(mean3D_vs, cov3D, frame.view_matrix, frame.focal, frame.resolution);

    const glm::mat3x3 proj_matrix = buildProjMat(frame.focal, frame.resolution);
	const glm::vec2 mean2D = glm::vec2(proj_matrix * (mean3D_vs / mean3D_vs.z)) - 0.5f;
	
    // Extent that covers the full Gaussian contribution (up to threshold 1/255)
    const float extent = 3.33f;

    // Compute circular radius, dependent on the largest eigenvalue
	const float min_lambda = 0.01f;
	const float mid = 0.5f * (cov2D[0][0] + cov2D[1][1]);
	const float lambda = mid + sqrt(max(min_lambda, mid * mid - glm::determinant(cov2D)));
	const float circle_extent_radius = extent * sqrt(lambda);
    const float2 circle_extent_square_dims = make_float2(circle_extent_radius);

    // Determine square dimensions in pixel-coordinates (left upper and right lower pixel), bounded by screen resolution
    uint2 rect_min, rect_max;
    getRect(make_float2(mean2D.x, mean2D.y), circle_extent_square_dims, frame.resolution, rect_min, rect_max);
    int pixel_count = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);

    if (pixel_count == 0)
    {
        idcs[idx] = -1U;
        depths[idx] = FLT_MAX;
        return;
    }

    idcs[idx] = idx;
    depths[idx] = mean3D_vs.z;

    const glm::mat2 inv_cov2D = glm::inverse(cov2D);
    invCov2D_opacities[idx] = make_float4(inv_cov2D[0][0], inv_cov2D[0][1], inv_cov2D[1][1], opacity[idx]); // cov2D symmetric
    means2D[idx] = make_float2(mean2D.x, mean2D.y);

    const glm::vec3 viewdir = glm::normalize(glm::vec3(mean3D.x, mean3D.y, mean3D.z) - frame.cam_pos);
    const glm::vec3 color = computeColorFromSH(idx, 3, viewdir, shs);
    colors[idx] = make_float3(color.x, color.y, color.z);
}

__global__ void render(const FrameInfo frame,
                       const uint32_t n_gaussians,
                       const uint32_t *__restrict__ sorted_idcs,

                       const float4 *__restrict__ invCov2D_opacities,
                       const float2 *__restrict__ means2D,
                       const float3 *__restrict__ colors,

                       cudaSurfaceObject_t img_out_surf)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= frame.resolution.x || y >= frame.resolution.y)
        return;

    float T = 1.0f;
    float3 out_color = make_float3(0.0);
    for (int idx = 0; idx < n_gaussians; idx++)
    {
        uint32_t gaussian_idx = sorted_idcs[idx];
        if (gaussian_idx == -1U)
            break;

        const float4 invCov2D_opacity = invCov2D_opacities[gaussian_idx];

        const float3 inv_cov2D = make_float3(invCov2D_opacity); // cov2D is symmetric, so 2x2 matrix only contains 3 unique entries
        const float opacity = invCov2D_opacity.w;

        const float2 mean2D = means2D[gaussian_idx];

        // Evaluate the 2D Gaussian at this pixel position
        const float2 d = {x - mean2D.x, y - mean2D.y};
        const float power = -0.5f * ((inv_cov2D.x * d.x * d.x + inv_cov2D.z * d.y * d.y) + 2.0f * inv_cov2D.y * d.x * d.y);
        if (power > 0.0f)
            continue;

        // alpha = o * exp(-0.5 * ((x-mu)^T x inv_cov x (x-mu)))
        const float alpha = min(0.99f, opacity * exp(power));
        if (alpha < ALPHA_THRESHOLD)
            continue;

        const float3 color = colors[gaussian_idx];

        // Blend the Gaussians front-to-back
        out_color += color * T * alpha;
        T *= (1.0f - alpha);

        if (T < T_THRESHOLD) // early stopping of full opacity reached (= no transmittance left)
            break;
    }

    surf2Dwrite(color_to_uchar4(out_color), img_out_surf, x * sizeof(uchar4), y);
}

Renderer_Reference::Renderer_Reference()
{
}

Renderer_Reference::~Renderer_Reference()
{
}

void Renderer_Reference::setup(const DatasetGPU data, uint2 resolution)
{
    int n_gaussians = data.n_gaussians;
    CUDA_CHECK_THROW(cudaMalloc(&(_d_splats.invCov2D_opacities), sizeof(float4) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&(_d_splats.means2D), sizeof(float2) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&(_d_splats.colors), sizeof(float3) * n_gaussians));

    CUDA_CHECK_THROW(cudaMalloc(&_d_unsorted_depths, sizeof(float) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&_d_unsorted_idcs, sizeof(uint32_t) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&_d_sorted_depths, sizeof(float) * n_gaussians));
    CUDA_CHECK_THROW(cudaMalloc(&_d_sorted_idcs, sizeof(uint32_t) * n_gaussians));

    _d_temp_storage_sort = nullptr;
    _temp_storage_sort_size_bytes = 0;

    // Since we have a fixed size global sort array here, we allocate CUB buffers on setup
    cub::DeviceRadixSort::SortPairs(_d_temp_storage_sort, _temp_storage_sort_size_bytes, _d_unsorted_depths, _d_sorted_depths, _d_unsorted_idcs, _d_sorted_idcs, data.n_gaussians);
    CUDA_CHECK_THROW(cudaMalloc(&_d_temp_storage_sort, _temp_storage_sort_size_bytes));
}

void Renderer_Reference::run(const DatasetGPU data, FrameInfo& frame, cudaSurfaceObject_t img_out_surf, bool warmup)
{
    constexpr uint32_t BLOCK_SIZE = 256;
    prepare<<<(data.n_gaussians + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        data.n_gaussians, frame, _d_unsorted_idcs, _d_unsorted_depths,
        data.d_pos, data.d_scale, data.d_rot, (float*) data.d_shs, data.d_opacity,
        _d_splats.invCov2D_opacities, _d_splats.means2D, _d_splats.colors
    );

    cub::DeviceRadixSort::SortPairs(_d_temp_storage_sort, _temp_storage_sort_size_bytes, _d_unsorted_depths, _d_sorted_depths, _d_unsorted_idcs, _d_sorted_idcs, data.n_gaussians);

    constexpr uint32_t BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 image_grid_size_2D((frame.resolution.x + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, (frame.resolution.y + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D);
    render<<<image_grid_size_2D, block_size_2D>>>(frame, data.n_gaussians, _d_sorted_idcs, _d_splats.invCov2D_opacities, _d_splats.means2D, _d_splats.colors, img_out_surf);
}