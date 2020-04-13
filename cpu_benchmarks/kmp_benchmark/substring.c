#include "substring.h"

const char Delimiter = '#';

/// Function for changing string "patters#data"
__stage(1) 
char get_element(
        const struct Char *data,
        __stage(1) const struct Char *pattern,
        __stage(1) int pattern_length,
        __stage(1) int pos) __stage(1)
{
    if (pos < pattern_length)
        return pattern[pos].c;
    else if (pos == pattern_length)
        return Delimiter;
    else
        return data[pos - pattern_length - 1].c;
}

/// Generates prefix function for the Knuth-Morris-Pratt arlgorithm
void prefix(
        __stage(1) int *pi, 
        const struct Char *data, 
        int data_length, 
        __stage(1) const struct Char *pattern, 
        __stage(1) int pattern_length) __stage(1)
{
    for (int i = 0; i < data_length + pattern_length + 1; ++i)
    {
        int j = pi[i - 1];
        while (j > 0 && 
                get_element(data, pattern, pattern_length, i) != 
                get_element(data, pattern, pattern_length, j))
            j = pi[j - 1];

        if (get_element(data, pattern, pattern_length, i) == 
                get_element(data, pattern, pattern_length, j))
            ++j;

        pi[i] = j;
    } 
}
	
/// Knuth-Morris-Pratt algorithm
__stage(1)
int find_substring(
        const struct Char *data, 
        int data_length, 
        __stage(1) const struct Char *pattern, 
        __stage(1) int pattern_length) __stage(1)
{
    int *pi;
    pi = malloc(data_length + pattern_length + 1); 
   
    prefix(pi, data, data_length, pattern, pattern_length);

    int result = 0;
    for (int i = pattern_length; i < data_length + pattern_length + 1; ++i)
        if (pi[i] == pattern_length)
            result = i;

    return result;
}

/// Generator of the specialized function
__attribute((mix(find_substring)))
void *mix_find_substring(
        void *context,
        const struct Char *pattern,
        int pattern_length);
