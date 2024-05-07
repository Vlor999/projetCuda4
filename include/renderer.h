
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
    uint32_t _buffer_size = 0;
    uint64_t *_d_unsorted_keys;
    uint32_t *_d_unsorted_idcs;
    uint64_t *_d_sorted_keys;
    uint32_t *_d_sorted_idcs;

    void* _d_temp_storage_sort = nullptr;
    size_t _temp_storage_sort_size_bytes = 0;
};