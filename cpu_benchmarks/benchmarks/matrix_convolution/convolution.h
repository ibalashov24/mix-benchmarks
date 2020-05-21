#pragma once

#ifdef __cplusplus
extern "C" {
#endif
struct Double {
    double e;
} __attribute((packed, staged));

/// Basic non-specialized function
void apply_convolution(
		   unsigned matrix_height,
           unsigned matrix_width,
           double matrix[], 
		   unsigned kernel_dim,
		   struct Double *kernel, 
           double result[]);

/// Generates specialized function on pattern
void *apply_convolution_mix(
        void *context, 
        unsigned kernel_dim,
        struct Double *kernel); 

#ifdef __cplusplus
}
#endif
