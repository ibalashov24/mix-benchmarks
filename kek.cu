#include <cuda_runtime.h>
#include <cstdio>

constexpr unsigned
fibonacci(const unsigned x) {
    if constexpr (false)
    {
        return 0u;
    }
    if( x <= 1 )
        return 1;
    return fibonacci(x - 1) + fibonacci(x - 2);
}

__global__
void k()
{
    constexpr unsigned arg = fibonacci(5);
    printf("%u", arg);
}

int main()
{
    k<<<1,1>>>();
    return 0;
}
