#include "substring.h"
    
// Naive substring search
__stage(1)
long long find_substring(
        __stage(1) const struct Char *data, 
        __stage(1) long long data_length, 
        const struct Char *pattern, 
        long long pattern_length) __stage(1)
{
    long long result = -1;
    for (long long i = 0; i < data_length; ++i)
    {
        long long j = 0;
        for (j = 0; j < pattern_length; ++j)
            if (i + j >= data_length || data[i + j].c != pattern[j].c)
                break;

        if (j == pattern_length)
        {
            result = i;
            break;
        }
    }

    return result;
}

// Specialized function generator
__attribute((mix(find_substring)))
void *mix_find_substring(
        void *context,
        const struct Char *pattern,
        long long pattern_length);
