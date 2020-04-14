#include "substring.h"
	
__stage(1)
int find_substring(
        const struct Char *data, 
        int data_length, 
        __stage(1) const struct Char *pattern, 
        __stage(1) int pattern_length) __stage(1)
{
    int result = -1;
    for (int i = 0; i < data_length; ++i)
    {
        int j = 0;
        for (j = 0; j < pattern_length; ++j)
            if (i + j >= pattern_length || data[i + j].c != pattern[j].c)
                break;

        if (j == pattern_length)
        {
            result = i;
            break;
        }
    }

    return result;
}

__attribute((mix(find_substring)))
void *mix_find_substring(
        void *context,
        const struct Char *pattern,
        int pattern_length);
