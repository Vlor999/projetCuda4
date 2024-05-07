
#pragma once

#include "cuda_runtime.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <filesystem>
namespace fs = std::filesystem;

template <typename T, int N_CHANNELS>
struct ImageBuffer
{
public:
    ImageBuffer(uint2 resolution)
    {
        resize(resolution);
    }

    void resize(uint2 resolution)
    {
        _resolution = resolution;

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
        CUDA_CHECK_THROW(cudaMallocArray(&(this->_array), &desc, resolution.x, resolution.y, cudaArraySurfaceLoadStore));

        init();
    }

    void init(cudaTextureAddressMode address_mode = cudaAddressModeBorder)
    {
        _surface = createSurface();
        _texture = createTexture(cudaFilterModePoint, address_mode);
    }

    void memsetBuffer(T val)
    {
        std::vector<T> tmp(_resolution.x * _resolution.y, val);
        upload(tmp.data());
    }
    
    void download(T* h_buffer)
    {
        CUDA_CHECK_THROW(cudaMemcpy2DFromArray(h_buffer, _resolution.x * sizeof(T), _array, 0, 0, _resolution.x * sizeof(T), _resolution.y, cudaMemcpyDeviceToHost));
    }

    void upload(T* h_buffer)
    {
        CUDA_CHECK_THROW(cudaMemcpy2DToArray(_array, 0, 0, h_buffer, _resolution.x * sizeof(T), _resolution.x * sizeof(T), _resolution.y, cudaMemcpyHostToDevice));
    }
    
    static void writeImageToFile(void* image, int w, int h, int channels, fs::path filename)
    {
        if (filename.extension() == ".png") {
            stbi_write_png(filename.string().c_str(), w, h, channels, image, w * channels);
        } else if (filename.extension() == ".bmp") {
            stbi_write_bmp(filename.string().c_str(), w, h, channels, image);
        } else if (filename.extension() == ".jpg") {
            stbi_write_jpg(filename.string().c_str(), w, h, channels, image, 100);
        } else {
            throw std::runtime_error("Image file extension not supported!");
        }
    }
    
    void writeToFile(std::filesystem::path filename)
    {
        std::vector<T> image(_resolution.x * _resolution.y);
        download(image.data());

        writeImageToFile(image.data(), _resolution.x, _resolution.y, N_CHANNELS, filename);
    }
    
    cudaTextureObject_t texture() { return _texture; }
    cudaSurfaceObject_t surface() { return _surface; }    

    uint2 resolution() { return _resolution; }

private:
    cudaTextureObject_t createTexture(cudaTextureFilterMode filter_mode, cudaTextureAddressMode address_mode = cudaAddressModeBorder)
    {
        cudaResourceDesc tex_res_desc;
        memset(&tex_res_desc, 0, sizeof(cudaResourceDesc));
        tex_res_desc.resType = cudaResourceTypeArray;
        tex_res_desc.res.array.array = _array;

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(cudaTextureDesc));

        tex_desc.normalizedCoords = false;
        tex_desc.filterMode = filter_mode;
        tex_desc.addressMode[0] = address_mode;
        tex_desc.addressMode[1] = address_mode;
        tex_desc.addressMode[2] = address_mode;
        tex_desc.readMode = cudaReadModeElementType;

        cudaTextureObject_t texture;
        CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &tex_res_desc, &tex_desc, NULL));

        return texture;
    }

    cudaSurfaceObject_t createSurface()
    {
        cudaResourceDesc surf_res_desc;
        memset(&surf_res_desc, 0, sizeof(cudaResourceDesc));
        surf_res_desc.resType = cudaResourceTypeArray;
        surf_res_desc.res.array.array = _array;

        cudaSurfaceObject_t surface;
        CUDA_CHECK_THROW(cudaCreateSurfaceObject(&surface, &surf_res_desc));

        return surface;
    }

    uint2 _resolution{0, 0};
    
    cudaArray_t _array;

    cudaSurfaceObject_t _surface;
    cudaTextureObject_t _texture;
};