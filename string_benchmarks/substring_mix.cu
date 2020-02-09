#include <stdio.h>
#include <string.h>
#include <chrono>
#include <fstream>
#include <iostream>

#include "benchmark_utils.cu"

#define BLOCKS  16
#define THREADS 64

const int DATA_SIZE = 10485760; // 10MB
const int PATTERN_SIZE = 100; // 100 bytes

enum Benchmark { SPECIALIZED, STANDARD };

__device__ int isSubstring = 0;

__global__ void find_substring_mix(
	     char *string, 
		char *sample, 
	 int stringLength, 
		int patternLength)
{
	
	printf("kek");

	// Number of symbols of the string processed in this thread
    int packSize = stringLength > BLOCKS*THREADS ? stringLength / (BLOCKS*THREADS) : 1;
	// Index of the beginning of the current block of processed symobols
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * packSize;

    if (index + packSize > BLOCKS*THREADS)
    {
        packSize = BLOCKS*THREADS - index;
    }
	// If no more left to process
    if (packSize == 0)
    {
        return;
    }

    for (int i = index; i < index + packSize; ++i)
    {   
        if (i + patternLength >= stringLength)
        {
            break;
        }

        int j;
        for (j = 0; j < patternLength; ++j)
        {
            if (string[i + j] != sample[j])
            {
                break;
            }
        } 

        if (j == patternLength)
        {
            isSubstring = 1;
        }
    }

    return;
}


int main(int argc, char *argv[])
{
    // Reading big data
	std::ifstream string_file("string.in");
    auto string = read_data_to_gpu(string_file, DATA_SIZE);
    string_file.close();

    // Reading pattern
    std::ifstream pattern_file("pattern.in");
    auto pattern = read_data_to_gpu(pattern_file, PATTERN_SIZE);
    pattern_file.close();

    auto timerBegin = std::chrono::high_resolution_clock::now();
    switch(argv[1][0] - '0')
    {
        case SPECIALIZED:
            find_substring_mix<<<BLOCKS, THREADS>>>(string, pattern, DATA_SIZE, PATTERN_SIZE);
            break;
        case STANDARD: 
            break;
        default:
            std::cout << "Wrong benchmark type!" << std::endl;
    } 
    cudaDeviceSynchronize();
    auto timerEnd = std::chrono::high_resolution_clock::now();

    std::cout << isSubstring << std::endl;

    std::cout << "Execution time: ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(
            timerEnd - timerBegin ).count();
    std::cout << std::endl;

    cudaFree(string);
    cudaFree(pattern);

    return 0;
}
