#include <iostream>
#include <fstream>
#include <chrono>

#include "Compiler.h"
#include "substring.h"
#include "program_options.hpp"
#include "utils.hpp"

int main(int argc, char *argv[])
{
	std::cout << "Naive substring benchmark" << std::endl;
	
	// Getting chmark options
    	auto args = read_arguments(argc, argv);

	// Reading data 
	std::ifstream data_file(args.data_file);
	auto string = read_data(data_file, args.data_length);
	data_file.close();

	// Reading pattern
	std::ifstream pattern_file(args.pattern_file);
	auto pattern = read_data(pattern_file, args.pattern_length * args.pattern_count);
	pattern_file.close();

	// Init JIT-compiler
	Compiler compiler("Naive");
	compiler.setFunction(mix_find_substring(&compiler.getContext(), (Char *) pattern, args.pattern_length));
    	auto *spec = reinterpret_cast<int (*)(const Char *, long long)>(compiler.compile());

	// Launch benchmark
	auto timerBegin = std::chrono::high_resolution_clock::now();
	int result = spec((Char *) string, args.data_length);
	auto timerEnd = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::microseconds>(timerEnd - timerBegin).count();

	std::cout << "Execution time: " << time << " ms" << std::endl;

	free(string);
	free(pattern);

	return 0;
}
