#include "proxy.cuh"
#include "substring_mix.cuh"

void *(*get_kernel())(void *, const char *, const int *, int, bool *)
{
    return mix_find_substring;
}
