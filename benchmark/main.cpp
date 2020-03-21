#include <iostream>
#include <fstream>

// C++ dependencies
#include "Compiler.h"
#include "program_options.hpp"

// CUDA dependencies
#include "benchmark_utils.cuh"
//#include "substring_mix.cuh"
#include "proxy.cuh"


const int BLOCK_COUNT = 8096;
const int THREAD_COUNT = 500;

int run_benchmark(
        const char *data, 
        long long data_size, 
        const char *patterns, 
        const int *pattern_borders, 
        int pattern_count,
		int test_runs)
{
    bool *is_entry = static_cast<bool *>(alloc_gpu_mem(sizeof(bool) * data_size));

	Compiler C("KEK");
	auto kernel = get_kernel();
	C.setFunction(kernel(&C.getContext(), patterns, pattern_borders, pattern_count, is_entry));
	auto *spec = reinterpret_cast<void (*)(const char *, long long)>(C.compile());

	long long time_sum = 0;	
	for (int i = 0; i < test_runs; ++i)
	{ 
		auto timerBegin = std::chrono::high_resolution_clock::now();
        launch_benchmark(spec, data, data_size, BLOCK_COUNT, THREAD_COUNT);
		auto timerEnd = std::chrono::high_resolution_clock::now();

		time_sum += std::chrono::duration_cast<std::chrono::microseconds>(timerEnd - timerBegin).count();
	}

	//TODO: Check search results in some way
    free_gpu_memory(static_cast<void *>(is_entry));

	return time_sum / test_runs;
}

int main(int argc, char **argv)
{    
	// Getting chmark options
    auto args = read_arguments(argc, argv);

	// Reading data 
    std::ifstream data_file(args.data_file);
    auto string = read_data_to_gpu(data_file, args.data_length);
    data_file.close();


    std::cout << "KEK" << std::endl;
    
    // Reading pattern
    std::ifstream pattern_file(args.pattern_file);
    auto pattern = read_data_to_gpu(pattern_file, args.pattern_length * args.pattern_count);
    pattern_file.close();

	// Creating service structures
    auto cuda_borders = generate_borders(args.pattern_count, args.pattern_length);

	// Launching benchmark
    auto time = run_benchmark(string, args.data_length, pattern, cuda_borders, args.pattern_count, args.test_runs); 
    std::cout << time << std::endl;

    free_gpu_memory(string);
    free_gpu_memory(pattern);
    free_gpu_memory(cuda_borders);
    return 0;
}
