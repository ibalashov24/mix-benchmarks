#include <iostream>
#include <fstream>
#include <chrono>


#include "program_options.hpp"
#include "benchmark_utils.hpp"


void match_naive(const std::string &data, const std::string &pattern, bool is_mix, bool is_chunked)
{
    

}

void run_benchmarks(
        const std::string &benchmark_type, 
        const std::string &data, 
        const std::string &pattern,
        bool is_mix = false,
        bool is_chunked = false)
{
    if (benchmark_type == "naive")
    {
        match_naive(data, pattern, is_mix, is_chunked);
    }
    else if (benchmark_type == "naive_const")
    {
        match_naive_const(data, pattern, is_mix, is_chunked);
    }
    else if (benchmark_type == "naive_shared")
    {
        match_naive_shared(data, pattern, is_mix, is_chunked);
    }
    else if (benchmark_type == "kmp")
    {
        match_kmp(data, pattern, is_mix, is_chunked);
    }
    else if (benchmark_type == "kmp_const")
    {
        match_kmp_const(data, pattern, is_mix, is_chunked);
    }
    else 
    {
        std::cerr << "Unrecognized benchmark type" << std::endl;
    }

    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
    // Getting benchmark options
    Arguments arguments = get_args();

    // Reading big data
	std::ifstream string_file("string.in");
    auto string = read_data_to_gpu(arguments.data_file);
    string_file.close();

    // Reading pattern
    std::ifstream pattern_file("pattern.in");
    auto pattern = read_data_to_gpu(arguments.pattern_file);
    pattern_file.close();

    // Executing benchmark
    auto timerBegin = std::chrono::high_resolution_clock::now();
    run_benchmarks(arguments.benchmark_type, string, pattern, arguments.is_mix, arguments.is_chunked);
    auto timerEnd = std::chrono::high_resolution_clock::now();

    std::cout << "Execution time: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
            timerEnd - timerBegin ).count();
    std::cout << std::endl;

    cudaFree(string);
    cudaFree(pattern);

    return 0;
}
