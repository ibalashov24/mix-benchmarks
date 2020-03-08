#include <iostream>
#include <cstdlib>

#include </usr/local/cuda/include/cuda_runtime.h>

#include "benchmark_utils.cuh"

//#define BLOCK_SIZE 2097152; // 2MB
static const int BLOCK_SIZE = 2097152;

/**
    Reads data from file to GPU memory in blocks
  */
char *read_data_to_gpu(std::istream &input, int count)
{
    char *data;
    cudaMalloc((void **) &data, count);

    char *buffer = (char *) malloc(BLOCK_SIZE * sizeof(char));
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

void *alloc_gpu_mem(int count)
{
    void *memory;
    cudaMalloc((void **) &memory, count);
    return memory;
}

void free_gpu_memory(void *pointer)
{
    cudaFree(pointer);
}

void launch_benchmark(
        void (*spec)(const char *, long long),
        const char *data,
        int data_size,
        int block_count,
        int thread_count)
{
	spec<<<block_count, thread_count>>>(data, data_size);
	cudaDeviceSynchronize();
}
