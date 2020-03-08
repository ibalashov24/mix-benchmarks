#pragma once

#include <iostream>

char *read_data_to_gpu(std::istream &input, int count);
int *generate_borders(int pattern_count);
void *alloc_gpu_mem(int count);
void free_gpu_memory(void *pointer);
void launch_benchmark(
        void (*spec)(const char *, long long),
        const char *data,
        int data_size,
        int block_count,
        int thread_count);
