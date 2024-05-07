
#pragma once

// This file contains code that was adapted from https://github.com/graphdeco-inria/diff-gaussian-rasterization/
// This code may only be used for research purposes, and must notice the original source

#include "cuda_runtime.h"
#include <glm/glm.hpp>

#include "helper/helper_math.h"

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__device__ inline glm::mat3 computeCov3D(const float3 scale, const float4 rot_quat)
{
	// Create scaling matrix (diagonal)
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;

	// Components of the quaternion
	float r = rot_quat.x;
	float x = rot_quat.y;
	float y = rot_quat.z;
	float z = rot_quat.w;

	// Compute rotation matrix from quaternion (glm uses column-major format)
	glm::mat3 R = glm::transpose(glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	));
	glm::mat3 M = R * S;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = M * glm::transpose(M);

    // Covariance is symmetric (could also only store upper right part in 6 values)
    glm::mat3 cov3D(
        Sigma[0][0], Sigma[0][1], Sigma[0][2],
        Sigma[0][1], Sigma[1][1], Sigma[1][2], 
        Sigma[0][2], Sigma[1][2], Sigma[2][2]
    );
    return cov3D;
}

__device__ inline glm::mat2 computeCov2D(const glm::vec3 mean3D_vs, const glm::mat3 cov3D, const glm::mat4x3 view_matrix, float2 focal, uint2 resolution)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	float3 t = make_float3(mean3D_vs.x, mean3D_vs.y, mean3D_vs.z);

	// For a stable calculation of cov2D, limit how far outside the viewport the Gaussian can lie
	// Those Gaussians will probably be discarded later, since their extent does not fall onto the screen
    const float2 tan_fov = 0.5f * make_float2(resolution) / focal;
	const float2 lim = 1.3f * tan_fov;
	t.x = min(lim.x, max(-lim.x, t.x / t.z)) * t.z;
	t.y = min(lim.y, max(-lim.y, t.y / t.z)) * t.z;

	// Equation 29. (transpose since glm is column-major)
	glm::mat3 J = glm::transpose(glm::mat3(
		focal.x / t.z, 			 0.0f, -(focal.x * t.x) / (t.z * t.z),
				 0.0f,  focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
				 0.0f, 			 0.0f, 							 0.0f
	));

	glm::mat3 W = glm::mat3(view_matrix);
	glm::mat3 T = J * W;

	// Equation 31.
	glm::mat3 cov = T * cov3D * glm::transpose(T);

	// Apply low-pass filter: every Gaussian should be at least one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return glm::mat2(cov);
}

// Converting the input spherical harmonics coefficients of each Gaussian to a simple RGB color.
__device__ inline glm::vec3 computeColorFromSH(int idx, int deg, const glm::vec3 dir, const float* shs)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3* sh = ((glm::vec3*) shs) + idx * (deg+1) * (deg+1);
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;
	
	return glm::max(result, 0.0f);
}

__device__ inline void getRect(const float2 center, float2 rect_dims, uint2 screen_dims, uint2& rect_min, uint2& rect_max)
{
	rect_min = make_uint2(
		min(screen_dims.x, max(0, (int) floorf(center.x - rect_dims.x))),
		min(screen_dims.y, max(0, (int) floorf(center.y - rect_dims.y)))
	);
	rect_max = make_uint2(
		min(screen_dims.x, max(0, (int) ceilf(center.x + rect_dims.x))),
		min(screen_dims.y, max(0, (int) ceilf(center.y + rect_dims.y)))
	);
}