#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdint>

#include "program_options.hpp"
#include "benchmark_utils.cuh"

const int BLOCK_SIZE = 1024; // threads

__device__
int threadId()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
} 
	
__global__ 
void find_substring_naive(
	const char *patterns, 
        const int *pattern_borders, 
        int pattern_count,
        __stage(1) const char *data, 
        __stage(1) int data_length, 
	bool *is_entry) __stage(1)
{
    auto position = threadId();
    if (position >= data_length)
    {
        return;
    }

    is_entry[position] = false;

    for (int i = 0; i < pattern_count; ++i)
    {
        int pattern_length = i == 0 ? pattern_borders[0] : pattern_borders[i] - pattern_borders[i - 1];
        if (position + pattern_length >= data_length)
        {
            return;
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

/**
 * Executes naive substring search algorithm with usual CUDA memory
 */
void match_naive(
        const char *data, 
        int data_size, 
        const char *patterns, 
        const int *pattern_borders, 
        int pattern_count)
{
    bool *is_entry;
    cudaMalloc((void **) &is_entry, sizeof(bool) * data_size);

    int blocks = data_size / BLOCK_SIZE;
    int threads = BLOCK_SIZE;
    
    find_substring_naive<<<blocks, threads>>>(patterns, pattern_borders, pattern_count, data, data_size, is_entry);
    cudaDeviceSynchronize();

    //TODO: Check search results in some way

    cudaFree(is_entry);
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

    // Executing benchmark
    /*run_benchmarks(
            arguments.benchmark_type, 
            string, 
            arguments.data_length, 
            pattern, 
            arguments.pattern_length);*/
	// Creating service structures
    int *patt_borders = (int *) malloc(arguments.pattern_count * sizeof(int));
    for (int i = 0; i < arguments.pattern_count; ++i)
	    patt_borders[i] = arguments.pattern_length;
    int *cuda_borders;
    cudaMalloc((void **) &cuda_borders, sizeof(int));
    cudaMemcpy(cuda_borders, patt_borders, arguments.pattern_count * sizeof(int), cudaMemcpyHostToDevice);

	// Launching benchmark
    auto timerBegin = std::chrono::high_resolution_clock::now();
    match_naive(string, arguments.data_length, pattern, cuda_borders, arguments.pattern_count); 
    auto timerEnd = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
            timerEnd - timerBegin ).count() << std::endl;

    cudaFree(string);
    cudaFree(pattern);
	cudaFree(cuda_borders);
	free(patt_borders);

    return 0;
}
