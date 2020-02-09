#include <iostream>
#include <cstdlib>
#include <chrono>

const int BLOCK_SIZE = 2097152; // 2MB

/**
    Reads data from file to GPU memory in blocks
  */
char *read_data_to_gpu(std::istream &input, int count)
{
    char *data = nullptr;
    cudaMalloc((void **) &data, count);

    char *buffer = (char *)malloc(BLOCK_SIZE * sizeof(char));
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

