#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstdlib>

#include "cuda_helper_host.h"


void inline start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	CUDA_CHECK_THROW(cudaEventCreate(&start));
	CUDA_CHECK_THROW(cudaEventCreate(&end));
	CUDA_CHECK_THROW(cudaEventRecord(start, 0));
}

float inline end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	float time;
	CUDA_CHECK_THROW(cudaEventRecord(end, 0));
	CUDA_CHECK_THROW(cudaEventSynchronize(end));
	CUDA_CHECK_THROW(cudaEventElapsedTime(&time, start, end));
	CUDA_CHECK_THROW(cudaEventDestroy(start));
	CUDA_CHECK_THROW(cudaEventDestroy(end));

	// Returns ms
	return time;
}

struct Result
{
	float mean_{ 0.0f };
	float std_dev_{ 0.0f };
	float median_{ 0.0f };
	float min_{ 0.0f };
	float max_{ 0.0f };
	int num_{ 0 };
};