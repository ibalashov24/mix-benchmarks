#include "substring.h"

const char Delimiter = '#';

__stage(1) 
char get_element(
        const struct Char *data,
        __stage(1) const struct Char *pattern,
        __stage(1) pattern_length,
        __stage(1) int pos)
{
    if (pos < pattern_length)
        return pattern[pos];
    else if (pos == pattern_length)
        return Delimiter;
    else
        return data[pos - pattern_length - 1];
}

void prefix(
        _stage(1) struct Int *pi, 
        const struct Char *data, 
        int data_length, 
        __stage(1) const struct Char *pattern, 
        __stage(1) int pattern_length)
{
    for (int i = 0; i < data_length + pattern_length + 1; ++i)
    {
        int j = pi[i - 1];
        while (j > 0 && 
                get_element(data, pattern, pattern_length, i) != 
                get_element(data, pattern, pattern_length, j))
            j = pi[j - 1];

        if (get_element(idata, pattern, pattern_length, i) == 
                get_element(data, pattern, pattern_length, j))
            ++j;

        pi[i] = j;
    } 
}
	
__stage(1)
int find_substring(
        const struct Char *data, 
        int data_length, 
        __stage(1) const struct Char *pattern, 
        __stage(1) int pattern_length) __stage(1)
{
    struct Int *pi;
    pi = memset(data_length + pattern_length + 1); 
   
    prefix(pi, data, data_length, pattern, pattern_length);

    int result = 0;
    for (int i = pattern_length; i < data_length + pattern_length + 1; ++i)
        if (pi[i] == pattern_length)
            result = i;

    return result;
}

__attribute((mix(find_substring)))
void *mix_find_substring(
        void *context,
        const struct Char *pattern,
        int pattern_length);
