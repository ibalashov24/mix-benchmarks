#pragma once
// Generates user-friendly command line arguments for the benchmark

#include <string>

/// Represents arguments for the benchmark
struct Arguments
{
    std::string benchmark_type;
    std::string data_file;
    std::string pattern_file;
    bool is_mix;
    bool is_chunked;
};

/* Parses command line arguments for benchmark
* @returns Parsed arguments
*/
Arguments get_args();
