
#pragma once

#include "renderer_base.h"

struct Renderer_Reference : public Renderer_Base
{
    Renderer_Reference();
    ~Renderer_Reference();

    void setup(const DatasetGPU data, uint2 resolution) override;

    void run(const DatasetGPU data, FrameInfo& frame, cudaSurfaceObject_t img_out_surf, bool warmup) override;

private:
    struct Splats
    {
        float4 *invCov2D_opacities = nullptr;
        float2 *means2D = nullptr;
        float3 *colors = nullptr;
    };
    Splats _d_splats;
    
    float *_d_unsorted_depths = nullptr;
    uint32_t *_d_unsorted_idcs = nullptr;
    float *_d_sorted_depths = nullptr;
    uint32_t *_d_sorted_idcs = nullptr;

    void* _d_temp_storage_sort = nullptr;
    size_t _temp_storage_sort_size_bytes = 0;
};