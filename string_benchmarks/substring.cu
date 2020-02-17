#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdint>

//#include "program_options.hpp"
#include "benchmark_utils.cuh"

const int BLOCK_SIZE = 1024; // threads

__device__
int threadId()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ 
void find_substring_naive(
        const char *data, 
        int data_length, 
        const char *patterns, 
        const int *pattern_borders, 
        int pattern_count,
	bool *is_entry)
{
    auto position = threadId();
    if (position >= data_length)
    {
        return;
    }

    printf("kek");

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

__global__
void prefix_function(
        const char *data, 
        int data_length, 
        int *prefix_table)
{
    int position = threadId();
    if (position == 0)
    {
        prefix_table[0] = 0;
    }
    if (position >= data_length)
    {
        return;
    }

    int j = prefix_table[position - 1];
    while (j > 0 && data[position] != data[j])
    {
        j = prefix_table[j - 1];
    }

    if (data[position] != data[j])
    {
        ++j;
    }

    prefix_table[position] = j;
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
    
    find_substring_naive<<<10, 10>>>(data, data_size, patterns, pattern_borders, pattern_count, is_entry);
    cudaDeviceSynchronize();

    //TODO: Check search results in some way

    cudaFree(is_entry);
}

/**
 * Executes Knuth-Morris-Pratt substring search algorithm with usual CUDA memory
 */
void match_kmp(
        const char *data, 
        int data_size,
        const char *patterns, 
        int *pattern_borders, 
        int pattern_count)
{
    int *prefix_table;
    cudaMalloc((void **) &prefix_table, sizeof(int) * data_size);

    char *delimiter = (char *) malloc(sizeof(char));
    delimiter[0] = '#';

    // KMP could not handle multiple pattern 
    // in one run effectively, so running for each separately
    for (int i = 0; i < pattern_count; ++i)
    {
        int pattern_length = i == 0 ? pattern_borders[0] : pattern_borders[i] - pattern_borders[i - 1]; 
        int pattern_begin = i == 0 ? 0 : pattern_borders[i - 1] + 1;
        
	// Oh, God...
        char *prefix;
        cudaMalloc((void **) &prefix, sizeof(char) * (pattern_length + 1 + data_size));
        cudaMemcpy(prefix, patterns + pattern_begin, pattern_length, cudaMemcpyDeviceToDevice);
        cudaMemcpy(prefix, delimiter, 1, cudaMemcpyHostToDevice);
        cudaMemcpy(prefix, data, data_size, cudaMemcpyDeviceToDevice);

        int blocks = (data_size + pattern_length + 1) / BLOCK_SIZE;
        int threads = BLOCK_SIZE;

        prefix_function<<<blocks, threads>>>(prefix, pattern_length + data_size + 1, prefix_table);

        cudaFree(prefix);

        //TODO: Check search results in some way
    }

    free(delimiter);
}

void run_benchmarks(
        const std::string &benchmark_type, 
        const char *data, 
        int data_length,
        const char *patterns,
        const int *pattern_lenghts)
{
       

    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{    // Getting benchmark options
//    Arguments arguments = get_args();

    // Reading big data
    //std::ifstream string_file(arguments.data_file);
    //auto string = read_data_to_gpu(string_file, arguments.data_length);
    std::ifstream string_file("data.in");
    auto string = read_data_to_gpu(string_file, 10485760);
    string_file.close();

    // Reading pattern
    //std::ifstream pattern_file(arguments.pattern_file);
    //auto pattern = read_data_to_gpu(pattern_file, arguments.pattern_length);
    std::ifstream pattern_file("pattern.in");
    auto pattern = read_data_to_gpu(pattern_file, 20);
    pattern_file.close();

    // Executing benchmark
    auto timerBegin = std::chrono::high_resolution_clock::now();
    /*run_benchmarks(
            arguments.benchmark_type, 
            string, 
            arguments.data_length, 
            pattern, 
            arguments.pattern_length);*/
    int *patt_borders = (int *) malloc(sizeof(int));
    patt_borders[0] = 10;
    int *cuda_borders;
    cudaMalloc((void **) &cuda_borders, sizeof(int));
    cudaMemcpy(cuda_borders, patt_borders, 10, cudaMemcpyHostToDevice);
    match_naive(string, 10485760, pattern, cuda_borders, 1); 
    auto timerEnd = std::chrono::high_resolution_clock::now();

    std::cout << "Execution time: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
            timerEnd - timerBegin ).count();
    std::cout << std::endl;

//    cudaFree(string);
//    cudaFree(pattern);

    return 0;
}
