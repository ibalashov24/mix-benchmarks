#include <iostream>
#include <chrono>

#include "Compiler.h"
#include "substring.h"
#include "program_options.hpp"

int main()
{
	std::cout << "Naive substring benchmark" << std::endl;
	
	// Getting chmark options
    	auto args = read_arguments(argc, argv);

	// Reading data 
	std::ifstream data_file(args.data_file);
	auto string = read_data_to_gpu(data_file, args.data_length);
	data_file.close();

	// Reading pattern
	std::ifstream pattern_file(args.pattern_file);
	auto pattern = read_data_to_gpu(pattern_file, args.pattern_length * args.pattern_count);
	pattern_file.close();

	// Init JIT-compiler
	Compiler compiler("Naive");
	compiler.setFunction(mix_find_substring(&compiler.getContext(), pattern, args.pattern_length));
	auto specizalized = compiler.compile()

	// Launch benchmark
	auto timerBegin = std::chrono::high_resolution_clock::now();
	int result = specialized(string, args.data_length);
	auto timerEnd = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::microseconds>(timerEnd - timerBegin).count();

	std::cout << "Execution time: " << time << " ms" << std::endl;

	return 0;
}
