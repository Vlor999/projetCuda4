
#pragma once

#include <filesystem>
#include <vector>
namespace fs = std::filesystem;

#include "cuda_runtime.h"
#include "helper/helper_math.h"

#include <glm/glm.hpp>

constexpr uint32_t TILE_SIZE = 16;

typedef float3 Pos;
template<int D>
struct SHs
{
	float shs[(D+1)*(D+1)*3];
};

struct DatasetGPU
{
    int n_gaussians;

	float3* d_pos;
	float3* d_scale;
	float4* d_rot;
	SHs<3>* d_shs;
	float* d_opacity;
};

struct Dataset
{
    int n_gaussians;

	std::vector<float3> pos;
	std::vector<float3> scale;
	std::vector<float4> rot;
	std::vector<SHs<3>> shs;
	std::vector<float> opacity;

    float3 scene_min;
    float3 scene_max;

    bool load(fs::path input_dir);
    DatasetGPU upload();
};


struct FrameInfo
{
    uint2 resolution;
	float2 focal;
	float near;
	glm::vec3 cam_pos;
    glm::mat4x3 view_matrix;

	FrameInfo resize(uint2 new_resolution) const
	{
		FrameInfo new_frame = *this;

		new_frame.focal = focal * (new_resolution.y / (float) resolution.y); // Maintain FOV on y-axis
		new_frame.resolution = new_resolution;
		return new_frame;
	}
};

bool loadCameraFrames(fs::path camera_file_path, std::vector<FrameInfo>& frames);