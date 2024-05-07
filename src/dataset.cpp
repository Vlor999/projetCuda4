
#include "dataset.h"

#include <float.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <json/json.hpp>
using json = nlohmann::json;

#include "helper/cuda_helper_host.h"
#include "helper/helper_math.h"

template<int D>
struct InputPoint
{
	float3 pos;
	float3 n;
	SHs<D> shs;
	float opacity;
	float scale[3];
	float rot[4];
};


float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}
// Load the Gaussians from the given file.
// The following function was adapted from https://github.com/graphdeco-inria/diff-gaussian-rasterization/
template<int D>
int loadPly(const char* filename,
	std::vector<float3>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<float3>& scales,
	std::vector<float4>& rot,
	float3& minn,
	float3& maxx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		std::cerr << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	std::cout << "Loading " << count << " Gaussian splats" << std::endl;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<InputPoint<D>> points(count);
	infile.read((char*)points.data(), count * sizeof(InputPoint<D>));

	minn = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		maxx = fmaxf(maxx, points[i].pos);
		minn = fmaxf(minn, points[i].pos);
	}

	// Resize our SoA data
	pos.resize(count);
	shs.resize(count);
	scales.resize(count);
	rot.resize(count);
	opacities.resize(count);

	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);
	for (int i = 0; i < count; i++)
	{
		int k = i; //mapp[i].second;
		pos[i] = points[k].pos;

		// Normalize quaternion
        rot[i] = normalize(make_float4(points[k].rot[0], points[k].rot[1], points[k].rot[2], points[k].rot[3]));

		// Exponentiate scale
        scales[i] = make_float3(exp(points[k].scale[0]), exp(points[k].scale[1]), exp(points[k].scale[2]));

		// Activate alpha
		opacities[i] = sigmoid(points[k].opacity);

		shs[i].shs[0] = points[k].shs.shs[0];
		shs[i].shs[1] = points[k].shs.shs[1];
		shs[i].shs[2] = points[k].shs.shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[i].shs[j * 3 + 0] = points[k].shs.shs[(j - 1) + 3];
			shs[i].shs[j * 3 + 1] = points[k].shs.shs[(j - 1) + SH_N + 2];
			shs[i].shs[j * 3 + 2] = points[k].shs.shs[(j - 1) + 2 * SH_N + 1];
		}
	}
	return count;
}

bool Dataset::load(fs::path input_file)
{
    n_gaussians = loadPly<3>(input_file.c_str(), pos, shs, opacity, scale, rot, scene_min, scene_max);
    return n_gaussians > 0;
}

DatasetGPU Dataset::upload()
{
    DatasetGPU data_gpu;

    data_gpu.n_gaussians = n_gaussians;
    data_gpu.d_pos = uploadVector(pos);
    data_gpu.d_scale = uploadVector(scale);
    data_gpu.d_rot = uploadVector(rot);
    data_gpu.d_shs = uploadVector(shs);
    data_gpu.d_opacity = uploadVector(opacity);

    return data_gpu;
}


struct FrameInfoInput
{
    uint32_t id;
    std::string img_name;
    uint32_t width;
    uint32_t height;
    float position[3];
    float rotation[3][3];
    float fy;
    float fx;

    FrameInfo toFrameInfo() const
    {
		FrameInfo frame;
		frame.resolution = make_uint2(width, height);
        frame.near = 0.2f;

        glm::mat4 frame_rot = glm::mat4(
			rotation[0][0], rotation[1][0], rotation[2][0], 0,
			rotation[0][1], rotation[1][1], rotation[2][1], 0,
			rotation[0][2], rotation[1][2], rotation[2][2], 0,
            0, 0, 0, 1
        );
        glm::mat4 frame_trans = glm::mat4(1.0f);
        frame_trans[3][0] = position[0];
        frame_trans[3][1] = position[1];
        frame_trans[3][2] = position[2];
        frame.cam_pos = glm::vec3(position[0], position[1], position[2]);

		frame.view_matrix = glm::inverse(frame_trans * frame_rot);
		frame.focal = make_float2(fx, fy);
        return frame;
    }
};
void from_json(const json& j, FrameInfoInput& info)
{
    j.at("id").get_to(info.id);
    j.at("img_name").get_to(info.img_name);
    j.at("width").get_to(info.width);
    j.at("height").get_to(info.height);

    std::vector<float> pos;
    j.at("position").get_to(pos);
    for (int i = 0; i < 3; i++)
        info.position[i] = pos[i];

    std::vector<std::vector<float>> rot;
    j.at("rotation").get_to(rot);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            info.rotation[i][j] = rot[i][j];

    j.at("fy").get_to(info.fy);
    j.at("fx").get_to(info.fx);
}

bool loadCameraFrames(fs::path camera_file_path, std::vector<FrameInfo>& frames)
{
    std::ifstream frameinfo_file(camera_file_path);
    if (!frameinfo_file.is_open())
    {
        return false;
    }
    json frameinfo_json_data = json::parse(frameinfo_file);

	std::vector<FrameInfoInput> frames_input = frameinfo_json_data.get<std::vector<FrameInfoInput>>();
	std::transform(frames_input.cbegin(), frames_input.cend(), std::back_inserter(frames), [](const FrameInfoInput& frame_input){ return frame_input.toFrameInfo(); });
	return true;
}