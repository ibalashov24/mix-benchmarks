#pragma once
// Generates user-friendly command line arguments for the benchmark

#include <string>

/// Represents arguments for the benchmark
struct Arguments
{
    std::string data_file;
    std::string pattern_file;
    long long data_length;
    int pattern_length;
    int pattern_count;
    int test_runs;
};

/* Parses command line arguments for benchmark
* @returns Parsed arguments
*/
Arguments read_arguments(int argc, char** argv);
