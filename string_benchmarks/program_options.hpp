#pragma once
// Generates user-friendly command line arguments for the benchmark

#include <string>

/// Represents arguments for the benchmark
struct Arguments
{
    std::string data_file;
    std::string pattern_file;
    int data_length;
    int pattern_length;
    int pattern_count;
};

/* Parses command line arguments for benchmark
* @returns Parsed arguments
*/
Arguments read_arguments(int argc, char** argv);
