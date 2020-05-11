#pragma once

#ifdef __cplusplus
extern "C" {
#endif
struct Double {
    double e;
} __attribute((packed, staged));

/// Basic non-specialized function
void apply_convolution(
           double *matrix, 
		   unsigned matrix_height,
           unsigned matrix_width,
		   const struct Double *kernel, 
		   unsigned kernel_dim,
           double *result);

/// Generates specialized function on pattern
void *apply_convolution_mix(
        void *context, 
        const struct Double *kernel, 
        unsigned kernel_dim);

#ifdef __cplusplus
}
#endif
