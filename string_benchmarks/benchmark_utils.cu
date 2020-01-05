#include <iostream>
#include <cstdlib>
#include <chrono>

#include "benchmark_utils.cuh"

/**
    Reads data from file to GPU memory in blocks
  */
char *read_data_to_gpu(istream &input, int count)
{
    char *data = nullptr;
    cudaMalloc((void **) &data, count);

    const char *buffer = malloc(BLOCK_SIZE * sizeof(char));
    for (int i = 0; i < count / BLOCK_SIZE + 1; ++i)
    {
        input.read(buffer, BLOCK_SIZE);
        cudaMemcpy(
                (void *) (data + i * BLOCK_SIZE), 
                (const void *) buffer, 
                input.gcount(),
                cudaMemcpyHostToDevice);
    }
    free(buffer);

    return data;
}

