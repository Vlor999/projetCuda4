
#pragma once

#include "cuda_runtime.h"

#include "helper/helper_math.h"

constexpr float T_THRESHOLD = 1e-4f;
constexpr float ALPHA_THRESHOLD = 1.0f / 255.f;

__device__ __host__ inline uint to8bit(const float f)
{
    return min(255, max(0, int(f * 256.f)));
}
__device__ __host__ inline uchar4 color_to_uchar4(const float3 color)
{
    uchar4 data;
    data.x = to8bit(color.x);
    data.y = to8bit(color.y);
    data.z = to8bit(color.z);
    data.w = 255U;
    return data;
}
__device__ __host__ inline uchar4 color_to_uchar4(const float color)
{
    return color_to_uchar4(make_float3(color));
}

__device__ __host__ inline glm::mat3x3 buildProjMat(const float2 focal, const uint2 resolution)
{
    return glm::mat3x3(focal.x, 0, 0,
                       0, focal.y, 0,
                       resolution.x / 2.0f, resolution.y / 2.0f, 1);
}