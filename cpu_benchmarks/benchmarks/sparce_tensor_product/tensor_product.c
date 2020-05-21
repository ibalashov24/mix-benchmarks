#include "tensor_product.h"

// Performs tensor product of the given 
// sparce matrix in COOrdianate representation
// and dense matrix
void multiply_tensor(
        unsigned matrix_size,
        struct Double *matrix,
        __stage(1) unsigned coo_size,
        __stage(1) struct CooItem coo[coo_size],
        __stage(1) struct CooItem out[coo_size + matrix_size * matrix_size]) __stage(1) 
{
    for (size_t i = 0; i < coo_size; ++i)
    {
        for (size_t j = 0; j < matrix_size; ++j)
        {
            for (size_t k = 0; k < matrix_size; ++k)
            {
                size_t current_pos = matrix_size * matrix_size * i + j * matrix_size + k;

                out[current_pos].row = coo[i].row * matrix_size + j;
                out[current_pos].col = coo[i].col * matrix_size + k;
                out[current_pos].value = coo[i].value * matrix[j * matrix_size + k].c;
            }
        }
    }
}

// Generates residual function 
// specialized on dense matrix
__attribute__((mix(multiply_tensor)))
void *mix_multiply_tensor(
        void *context,
        unsigned matrix_size,
        struct Double *matrix); 
