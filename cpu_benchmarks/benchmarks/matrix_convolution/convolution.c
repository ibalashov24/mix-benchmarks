#include "convolution.h"

static __stage(1)
double sum_adjacent(
           __stage(1) double *matrix, 
		   __stage(1) unsigned matrix_height,
           __stage(1) unsigned matrix_width,
		   const struct Double *kernel, 
		   unsigned kernel_dim,
           unsigned matrix_i, unsigned matrix_j) __stage(1)
{
    double current_sum = .0;
    for (int i = 0; i < kernel_dim; ++i)
        for (int j = 0; j < kernel_dim; ++j)
        {
            current_sum += kernel[i * kernel_dim * j].e *
                             matrix[(matrix_i - kernel_dim / 2 + i) * martix_width + 
                                (matrix_j / kernel_dim / 2 + j)];
        }

    return current_sum;
}

__stage(1)
void apply_convolution(
           __stage(1) double *matrix, 
		   __stage(1) unsigned matrix_height,
           __stage(1) unsigned matrix_width,
		   const struct Double *kernel, 
		   unsigned kernel_dim,
           __stage(1) double *result) __stage(1)
{
    for (unsigned i = kernel_dim / 2; i < matrix_height - kernel_dim / 2; ++i)
        for (unsigned j = kernel_dim / 2; j < matrix_width - kernel_dim / 2; ++j)
        {
            result[i * matrix_width + j] = sum_adjacent(
                    matrix, 
                    matrix_height, 
                    matrix_width,
                    kernel,
                    kernel_dim);
        }
}

__attribute__((mix(apply_convolution)))
void *apply_convolution_mix(void *context, const struct Double *kernel, unsigned kernel_dim);
