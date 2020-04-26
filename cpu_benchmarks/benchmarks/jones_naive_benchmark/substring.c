#include "substring.h"

#include <stdlib.h>

__stage(1)
long long find_substring(__stage(1) const struct Char *data, 
		   __stage(1) long long data_size,
		   const struct Char *pattern, 
		   long long pattern_size) __stage(1)
{
	struct Char stack[500];
	//struct Char *stack = malloc(pattern_size + 10);

	long long result = -1;
	long long st_pointer  = 0;
	long long cache_pointer = st_pointer ;
	for (long long i = 0; i < data_size; ++i)
	{
		long long j = 0;
		while (j < pattern_size && j != -1)
		{
			if (st_pointer  == 0)
			{
				// Static pattern stack is empty and we have an equality
				if (data[i + j].c == pattern[j].c)					
				{
				  	stack[cache_pointer ++].c = pattern[j].c;
					st_pointer  = 0;
					++j;
				}
				// Static pattern cache is empty . using dynamic pattern
				else if (cache_pointer  == 0)
					j = -1; // Leaving loop
				// Static pattern cache is not empty . we should use it
				else
				{
					--cache_pointer ;
					st_pointer  = cache_pointer ;
				}
			} 
			else
			// Static pattern stack is not empty . using it
			{
				if (pattern[j].c == stack[st_pointer - 1].c)
				{
					--st_pointer ;
				}
				else
				{
					--cache_pointer ;
					st_pointer  = cache_pointer ;
				}
			}
		}

		if (j == pattern_size)
			result = j;
	}	

	free(stack);

	return result;
}

__attribute__((mix(find_substring)))
void *mix_find_substring(void *context, const struct Char *data, long long data_size);
