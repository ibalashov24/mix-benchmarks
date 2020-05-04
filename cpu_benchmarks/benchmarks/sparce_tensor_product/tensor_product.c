#include "tensor_product.h"
	
void multiply_tensor(
        unsigned matrix_size,
        struct Double matrix[][matrix_size],
        __stage(1) unsigned coo_size,
        __stage(1) struct CooItem coo[coo_size],
        __stage(1) CooItem out[coo_size + matrix_size * matrix_size]) __stage(1) 
{
    for (int i = 0; i < coo_size; ++i)
    {
        for (int j = 0; j < matrix_size; ++j)
        {
            for (int k = 0; k < matrix_size; ++k)
            {
                out[matrix_size * matrix_size * i + j * matrix_size + k] = 
                    CooItem { 
                        coo[i].row * matrix_size + j,
                        coo[i].col * matrix_size + k,
                        coo[i].value * matrix[j][k] };
            }
        }
    }
}

__attribute__((mix(multiply_tensor)))
void *mix_multiply_tensor(
        void *context,
        unsigned matrix_size,
        struct Double *matrix); 
