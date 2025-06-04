
#pragma once

#include "renderer_base.h"

struct Renderer : public Renderer_Base
{
    Renderer();
    ~Renderer();

    void setup(const DatasetGPU data, uint2 resolution) override;

    void run(const DatasetGPU data, FrameInfo& frame, cudaSurfaceObject_t img_out_surf, bool warmup) override;

private:
    // You are allowed to add additional private methods and members, if you require them.
    // However, do NOT change the public method signatures.

    // Example buffers, used for sorting

    struct Splats
    {
        float4 *invCov2D_opacities = nullptr;
        float2 *means2D = nullptr;
        float3 *colors = nullptr;
    };
    Splats _d_splats;

    uint32_t _buffer_size = 0;
    uint64_t *_d_unsorted_keys;
    uint32_t *_d_unsorted_idcs;
    uint64_t *_d_sorted_keys;
    uint32_t *_d_sorted_idcs;

    uint32_t* _d_tile_range_starts;
    uint32_t* _d_tile_range_ends;

    void* _d_temp_storage_sort = nullptr;
    size_t _temp_storage_sort_size_bytes = 0;

    uint2* _d_rect_min;
    uint2* _d_rect_max;
    float* _d_depths;
    uint32_t* _d_valid;
    uint32_t* _d_tile_offsets;
    uint32_t* _d_tile_counts;
};