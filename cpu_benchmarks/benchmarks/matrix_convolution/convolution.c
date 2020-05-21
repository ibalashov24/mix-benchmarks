#include "convolution.h"

// Calculates the sum of adjacent cells 
// of the matrix multiplied 
// by corresponing kernel elements
 __stage(1)
double sum_adjacent(
           __stage(1) unsigned matrix_height,
           __stage(1) unsigned matrix_width,
           __stage(1) double matrix[matrix_height * matrix_width], 
           unsigned kernel_dim,
           struct Double *kernel,
           __stage(1) unsigned matrix_i, 
           __stage(1) unsigned matrix_j) __stage(1)
{
    double current_sum = .0;
    for (unsigned i = 0; i < kernel_dim; ++i)
        for (unsigned j = 0; j < kernel_dim; ++j)
            current_sum += kernel[i * kernel_dim * j].e *
                             matrix[(matrix_i - kernel_dim / 2 + i) * matrix_width + 
                                (matrix_j - kernel_dim / 2 + j)];

    return current_sum;
}

// Applies convolution with given kernel 
// to the given matrix-like image
void apply_convolution(
           __stage(1) unsigned matrix_height,
           __stage(1) unsigned matrix_width,
           __stage(1) double matrix[matrix_height * matrix_width], 
           unsigned kernel_dim,
           struct Double *kernel, 
           __stage(1) double result[matrix_height * matrix_width]) __stage(1)
{
    for (unsigned i = kernel_dim / 2; i < matrix_height - kernel_dim / 2; ++i)
        for (unsigned j = kernel_dim / 2; j < matrix_width - kernel_dim / 2; ++j)
        {
            result[i * matrix_width + j] = sum_adjacent(
                    matrix_height, 
                    matrix_width,
                    matrix, 
                    kernel_dim,
                    kernel,
                    i, j);
        }
}

// Generates specialized residual function
__attribute__((mix(apply_convolution)))
void *apply_convolution_mix(void *context, unsigned kernel_dim, struct Double *kernel);
