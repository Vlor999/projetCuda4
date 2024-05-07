
#include "cuda_runtime.h"

#include "dataset.h"
#include "image_buffer.h"
#include "renderer_reference.h"
#include "renderer.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"
#include "helper/GPUTimer.cuh"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "argparse/argparse.hpp"

float mse_to_psnr(float mse)
{
    return -10.0f * std::log(mse) / std::log(10.0f);
}

float computePSNR(ImageBuffer<uchar4, 4> &image_ref, std::vector<uchar4> h_image_ref,
                  ImageBuffer<uchar4, 4> &image, std::vector<uchar4> h_image,
                  std::vector<float> &diff_mse, std::vector<unsigned char> &diff_img_mse)
{
    image_ref.download(h_image_ref.data());
    image.download(h_image.data());

    auto mse_fun = [](uchar4 left, uchar4 right) 
    { 
        float3 diff = (make_float3(left.x, left.y, left.z) - make_float3(right.x, right.y, right.z)) / 255.f; 
        return dot(diff, diff) / 3.0f;
    };
    std::transform(h_image.begin(), h_image.end(), h_image_ref.begin(), diff_mse.begin(), mse_fun);
    std::transform(diff_mse.begin(), diff_mse.end(), diff_img_mse.begin(), [](float val){ return val * 255U; });
    float mse = std::reduce(diff_mse.begin(), diff_mse.end()) / float(diff_mse.size());
    return mse_to_psnr(mse);
}

int main(int argc, char const *argv[])
{
    argparse::ArgumentParser program("3DGS-Full");

    program.add_argument("points-file").help("path to the gaussian point cloud .ply file");
    program.add_argument("camera-file").help("path to the camera transforms .json file");
    program.add_argument("output-dir").help("path to the output directory");

    program.add_argument("-w", "--write-images").help("write images to output directory").default_value(false).implicit_value(true);
    program.add_argument("-r", "--resolution").help("image resolution").default_value(std::vector<uint32_t>{400, 400}).nargs(2).scan<'u', uint32_t>();

    program.add_argument("-it", "--num-benchmark-iterations").help("number of iterations when benchmarking renderer").default_value(1U).scan<'u', uint32_t>();

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        std::cout << "Error - Argument parsing failed!" << std::endl;
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 1;
    }

    fs::path points_file = program.get<std::string>("points-file");
    fs::path cameras_file = program.get<std::string>("camera-file");
    fs::path output_dir = program.get<std::string>("output-dir");

    bool write_images = program.get<bool>("--write-images");
    std::vector<uint32_t> arg_resolution = program.get<std::vector<uint32_t>>("--resolution");
    uint2 image_resolution = {arg_resolution[0], arg_resolution[1]};
    uint32_t n_benchmark_iterations = program.get<unsigned int>("--num-benchmark-iterations");
    uint32_t n_benchmark_iterations_ref = 1U;

	Dataset data;
	if (!data.load(points_file))
    {
        std::cout << "Error - could not load point cloud (.ply)!" << std::endl;
        return 1;
    }

    std::vector<FrameInfo> frames;
    if (!loadCameraFrames(cameras_file, frames))
    {
        std::cout << "Error - Could not load frameinfo file!" << std::endl;
        return 1;
    }
    std::cout << "Loaded Cameras " << frames.size() << std::endl;

    if (write_images)
    {
        fs::create_directories(output_dir);
        if (!fs::exists(output_dir) || !fs::is_directory(output_dir))
        {
            std::cout << "Error - could not create output folder!" << std::endl;
            return 1;
        }
    }
   

	DatasetGPU data_gpu = data.upload();

	Renderer_Reference render_ref;
    render_ref.setup(data_gpu, image_resolution);

	Renderer renderer;
    renderer.setup(data_gpu, image_resolution);


    ImageBuffer<uchar4, 4> img_buffer_ref(image_resolution);
    ImageBuffer<uchar4, 4> img_buffer(image_resolution);

    std::vector<uchar4> h_img_buffer_ref(image_resolution.x * image_resolution.y);
    std::vector<uchar4> h_img_buffer(image_resolution.x * image_resolution.y);
    std::vector<float> h_diff_mse(image_resolution.x * image_resolution.y);
    std::vector<unsigned char> h_diff_img_mse(image_resolution.x * image_resolution.y);

    bool all_success = true;
    float gpu_times_sum = 0.f;
    float psnr_sum = 0.f;

	for (size_t frame_i = 0; frame_i < frames.size(); frame_i++)
	{
        printf("Rendering %3lu/%3lu... ", frame_i + 1, frames.size());
        fflush(stdout);
        FrameInfo frame = frames[frame_i].resize(image_resolution);

        // time reference implementation
        GPUTimer timer_gpu_ref;
        for (uint32_t i = 0; i < n_benchmark_iterations_ref; i++)
        {
            timer_gpu_ref.start();
            render_ref.run(data_gpu, frame, img_buffer_ref.surface(), false);
            (void) timer_gpu_ref.end();
        }
        float gpu_time_ref = timer_gpu_ref.mean();

        renderer.run(data_gpu, frame, img_buffer.surface(), true);
        img_buffer.memsetBuffer(make_uchar4(0, 0, 0, 255));
        CUDA_SYNC_CHECK_THROW();

        // time custom (student) implementation
        GPUTimer timer_gpu_custom;
        for (uint32_t i = 0; i < n_benchmark_iterations; i++)
        {
            timer_gpu_custom.start();
            renderer.run(data_gpu, frame, img_buffer.surface(), false);
            (void) timer_gpu_custom.end();
        }
        float gpu_time_custom = timer_gpu_custom.mean();

        // Measure error
        float psnr = computePSNR(img_buffer_ref, h_img_buffer_ref, img_buffer, h_img_buffer, h_diff_mse, h_diff_img_mse);
        bool success = psnr > 40.f;

        printf("GPU ref = %.6f ms/it | GPU custom = %.6f ms/it | PSNR = %.2f (%s)\n", gpu_time_ref, gpu_time_custom, psnr, success ? "SUCCESS" : "FAILED");
        all_success &= success;
        gpu_times_sum += gpu_time_custom;
        psnr_sum += psnr;

        // Write images
        if (write_images)
        {
            std::string img_filename = "out_gpu_" + std::to_string(frame_i);
            std::cout << "Writing to file " << img_filename << std::endl;
            img_buffer_ref.writeToFile(output_dir / (img_filename + "_ref.png"));
            img_buffer.writeToFile(output_dir / (img_filename + "_custom.png"));
            ImageBuffer<unsigned char, 1>::writeImageToFile(h_diff_img_mse.data(), image_resolution.x, image_resolution.y, 1, output_dir / (img_filename + "_diff.png"));
        }		
	}

    std::cout << "Writing results.csv file ..." << std::endl;
    std::ofstream results_csv;
    results_csv.open("results.csv", std::ios_base::app);
    results_csv << output_dir.filename().c_str() << "," << gpu_times_sum / frames.size() << "," << psnr_sum / frames.size() << "," << (all_success ? "1" : "0") << std::endl;
    results_csv.close();

    return 0;
}