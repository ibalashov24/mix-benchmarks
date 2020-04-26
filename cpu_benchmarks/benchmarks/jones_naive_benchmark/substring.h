#pragma once

#ifdef __cplusplus
extern "C" {
#endif
struct Char {
    char c;
} __attribute((packed, staged));

/// Basic non-specialized function
long long find_substring(
        const struct Char *data, 
        long long data_length, 
        const struct Char *pattern, 
        long long pattern_length);

/// Generates specialized function on pattern
void *mix_find_substring(
        void *context,
        const struct Char *pattern,
        long long pattern_length);

#ifdef __cplusplus
}
#endif
