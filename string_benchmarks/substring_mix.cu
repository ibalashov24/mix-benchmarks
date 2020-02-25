#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdint>

#include "program_options.hpp"
#include "benchmark_utils.cuh"

const int BLOCK_COUNT = 8096;
const int THREAD_COUNT = 500;

__device__
long long threadId()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
} 
	
__global__ 
void find_substring_naive(
	const char *patterns, 
        const int *pattern_borders, 
        int pattern_count,
        __stage(1) const char *data, 
        __stage(1) long long data_length, 
	bool *is_entry) __stage(1)
{
    const long long chunk_size = data_length / (BLOCK_COUNT * THREAD_COUNT); 

    auto thread = threadId();
    for (long long position = thread * chunk_size; position < (thread + 1) * chunk_size; ++position)
    {
    	is_entry[position] = false;
	for (int i = 0; i < pattern_count; ++i)
	{
		int pattern_length = i == 0 ? pattern_borders[0] : pattern_borders[i] - pattern_borders[i - 1];
		if (position + pattern_length >= data_length)
		{
		    continue;
		}

		int pattern_begin = i == 0 ? 0 : pattern_borders[i - 1] + 1;
		for (int j = 0; j < pattern_length; ++j)
		{
		    if (data[j + position] != patterns[pattern_begin + j])
		    {
			return;
		    }
		}

		is_entry[position] = true;
	}
    }
}

/**
 * Executes naive substring search algorithm with usual CUDA memory and return average run time
 */
int  match_naive(
        const char *data, 
        long long data_size, 
        const char *patterns, 
        const int *pattern_borders, 
        int pattern_count,
		int test_runs)
{
    bool *is_entry;
    cudaMalloc((void **) &is_entry, sizeof(bool) * data_size);

	long long time_sum = 0;	
	for (int i = 0; i < test_runs; ++i)
	{
		auto timerBegin = std::chrono::high_resolution_clock::now();
		find_substring_naive<<<BLOCK_COUNT, THREAD_COUNT>>>(patterns, pattern_borders, pattern_count, data, data_size, is_entry);
		cudaDeviceSynchronize();
		auto timerEnd = std::chrono::high_resolution_clock::now();

		time_sum += std::chrono::duration_cast<std::chrono::microseconds>(timerEnd - timerBegin).count();
	}
        //TODO: Check search results in some way
    	cudaFree(is_entry);

	return time_sum / test_runs;
}

int main(int argc, char **argv)
{    
	// Getting benchmark options
	auto arguments = read_arguments(argc, argv);

    std::ifstream string_file(arguments.data_file);
    auto string = read_data_to_gpu(string_file, arguments.data_length);
    string_file.close();

    // Reading pattern
    std::ifstream pattern_file(arguments.pattern_file);
    auto pattern = read_data_to_gpu(pattern_file, arguments.pattern_length * arguments.pattern_count);
    pattern_file.close();

	// Creating service structures
    int *patt_borders = (int *) malloc(arguments.pattern_count * sizeof(int));
    for (int i = 0; i < arguments.pattern_count; ++i)
	    patt_borders[i] = arguments.pattern_length;
    int *cuda_borders;
    cudaMalloc((void **) &cuda_borders, sizeof(int));
    cudaMemcpy(cuda_borders, patt_borders, arguments.pattern_count * sizeof(int), cudaMemcpyHostToDevice);

	// Launching benchmark
    auto time = match_naive(string, arguments.data_length, pattern, cuda_borders, arguments.pattern_count, arguments.test_runs); 

    std::cout << time << std::endl;

    cudaFree(string);
    cudaFree(pattern);
	cudaFree(cuda_borders);
	free(patt_borders);

    return 0;
}
