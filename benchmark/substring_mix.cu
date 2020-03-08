#include "substring_mix.cuh"

__device__
long long threadId()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
} 
	
__global__ 
void find_substring(
	const char *patterns, 
        const int *pattern_borders, 
        int pattern_count,
        __stage(1) const char *data, 
        __stage(1) long long data_length, 
	bool *is_entry) __stage(1)
{
    const long long chunk_size = data_length / (BLOCK_COUNT * THREAD_COUNT); 

    auto thread = threadId();
    for (long long position = thread * chunk_size; position < (thread + 1) * chunk_size; ++position)
    {
    	is_entry[position] = false;
	for (int i = 0; i < pattern_count; ++i)
	{
		int pattern_length = i == 0 ? pattern_borders[0] : pattern_borders[i] - pattern_borders[i - 1];
		if (position + pattern_length >= data_length)
		{
		    continue;
		}

		int pattern_begin = i == 0 ? 0 : pattern_borders[i - 1] + 1;
		for (int j = 0; j < pattern_length; ++j)
		{
		    if (data[j + position] != patterns[pattern_begin + j])
		    {
			return;
		    }
		}

		is_entry[position] = true;
	}
    }
}

__attribute__((mix(find_substring)))
void *mix_find_substring(
			void *context, 
			const char *patterns,
			const int *pattern_borders,
			int pattern_count,
			bool *is_entry);

