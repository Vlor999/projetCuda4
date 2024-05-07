
#pragma once

#include "dataset.h"

struct Renderer_Base
{
    virtual void setup(const DatasetGPU data, uint2 resolution) = 0;

    virtual void run(const DatasetGPU data, FrameInfo& frame, cudaSurfaceObject_t img_out_surf, bool warmup) = 0;
};