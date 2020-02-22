#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdint>

#include "program_options.hpp"
#include "benchmark_utils.hpp"

const int BLOCK_SIZE = 1024; // threads

__global__ 
void find_substring_naive(
        const char *patterns, 
        const uint32_t *pattern_borders, 
        int pattern_count,
        const char *data, 
        unsigned data_length, 
        int *is_entry)
{
    auto position = threadId();
    if (position >= data_length)
    {
        return;
    }

    is_entry[position] = 0;

    for (int i = 0; i < pattern_count; ++i)
    {
        int pattern_length = i == 0 ? pattern_borders[0] : pattern_borders[i] - pat;
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
void prefix(
        const char *data, 
        unsigned data_length, 
        const char *pattern, 
        unsigned pattern_length
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
        uint32_t data_size, 
        const char *patterns, 
        const unsigned *pattern_borders, 
        unsigned pattern_count)
{
    bool *is_entry;
    cudaMalloc((void **) &is_entry, sizeof(bool) * data_size);

    unsigned blocks = data_size / block_size;
    unsigned treads = block_size;

    find_substring_naive<<blocks, threads>>(data, data_size, patterns, pattern_borders, pattern_count, is_entry);

    //TODO: Check search results in some way
}

/**
 * Executes Knuth-Morris-Pratt substring search algorithm with usual CUDA memory
 */
void match_kmp(
        const char *data, 
        uint32_t data_size,
        const char *patterns, 
        int *pattern_borders, 
        unsigned pattern_count)
{
    uint32_t *prefix_table;
    cudaMalloc((void **) &prefix_table, sizeof(uint32_t) * data_size);

    char *delimiter = malloc(sizeof(char));
    delimiter[0] = '#';

    // KMP could not handle multiple pattern 
    // in one run effectively, so running for each separately
    for (int i = 0; i < pattern_count; ++i)
    {
        unsigned blocks = (data_size + pattern_size + 1) / block_size;
        unsigned treads = block_size;

        int pattern_length = i == 0 ? pattern_borders[0] : pattern_borders[i] - pat; 
        int pattern_begin = i == 0 ? 0 : pattern_borders[i - 1] + 1;

        // Oh, God...
        char *prefix;
        cudaMalloc((void **) &prefix, sizeof(char) * (pattern_length + 1 + data_size));
        cudaMemcpy(prefix, patterns + pattern_begin, pattern_length, cudaMemcpyDeviceToDevice);
        cudaMemcpy(prefix, delimiter, 1, cudaMemcpyHostToDevice);
        cudaMemcpy(prefix, data, data_size, cudaMemcpyDeviceToDevice);

        prefix<<blocks, threads>>(prefix, pattern_length + data_size + 1, prefix_table);

        cudaFree(prefix);

        //TODO: Check search results in some way
    }

    free(delimiter);
}

void run_benchmarks(
        const std::string &benchmark_type, 
        const char *data, 
        unsigned data_length,
        const char *patterns,
        const int *pattern_lenghts)
{
    bool *is_entry;
    cudaMalloc((void **) &is_entry, data_length * sizeof(bool));

    find_substring_naive<<<data_length / BLOCK_SIZE, BLOCK_SIZE>>>(
            patterns, pattern_lenghts, 1, data, data_length, is_entry);

    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{    // Getting benchmark options
    Arguments arguments = get_args();

    // Reading big data
	std::ifstream string_file("string.in");
    auto string = read_data_to_gpu(arguments.data_file, arguments.data_length);
    string_file.close();

    // Reading pattern
    std::ifstream pattern_file("pattern.in");
    auto pattern = read_data_to_gpu(arguments.pattern_file, arguments.pattern_length);
    pattern_file.close();

    // Executing benchmark
    auto timerBegin = std::chrono::high_resolution_clock::now();
    run_benchmarks(
            arguments.benchmark_type, 
            string, 
            arguments.data_length, 
            pattern, 
            arguments.pattern_length);
    auto timerEnd = std::chrono::high_resolution_clock::now();

    std::cout << "Execution time: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
            timerEnd - timerBegin ).count();
    std::cout << std::endl;

    cudaFree(string);
    cudaFree(pattern);

    return 0;
}
