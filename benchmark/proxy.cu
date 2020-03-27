#include "proxy.cuh"
#include "substring_mix.cuh"

#include <iostream>

void *(*get_kernel())(void *, const char *, const int *, int, bool *)
{
    return mix_find_substring;
}
