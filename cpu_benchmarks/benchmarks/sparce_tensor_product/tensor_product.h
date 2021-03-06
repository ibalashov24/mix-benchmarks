#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct Double {
    double c;
} __attribute((packed, staged));

struct CooItem {
    unsigned row;
    unsigned col;
    double value;
};

// Non-specialized function 
void multiply_tensor(
        unsigned matrix_size,
        struct Double *matrix,
        unsigned coo_size,
        struct CooItem coo[],
        struct CooItem out[]); 

// Generates specialized function on dense matrix
void *mix_multiply_tensor(
        void *context,
        unsigned matrix_size,
        struct Double *matrix);

#ifdef __cplusplus
}
#endif
