//#include "substring.h"

#include <stdlib.h>

struct Char
{
	char c;
};

__stage(1)
int find_substring(__stage(1) const struct Char *data, 
		   __stage(1) int data_size,
		   const struct Char *pattern, 
		   int pattern_size) __stage(1)
{
	struct Char stack[250];

	int result = -1;
	int st_pointer = 0;
	int cache_pointer = st_pointer;
	for (int i = 0; i < data_size; ++i)
	{
		int flag = 1;
		int j = 0;
		for (; j < pattern_size; )
		{
			if (st_pointer == 0)
			{
				// Static pattern stack is empty and we have an equality
				if (data[i].c == pattern[i + j].c)					
				{
				  	stack[cache_pointer++].c = pattern[j].c;
					st_pointer = 0;
					++j;
				}
				// Static pattern cache is empty . using dynamic pattern
				else if (cache_pointer == 0)
					flag = -1; // Leaving loop
				// Static pattern cache is not empty . we should use it
				else
				{
					--cache_pointer;
					st_pointer = cache_pointer;
				}
			} 
			else
			// Static pattern stack is not empty . using it
			{
				if (pattern[i + j].c == stack[st_pointer].c)
				{
					--st_pointer;
				}
				else
				{
					--cache_pointer;
					st_pointer = cache_pointer;
				}
			}
		}

		if (j == pattern_size)
			result = j;
	}	

	return result;
}

__attribute__((mix(find_substring)))
void *mix_find_substring(void *context, const struct Char *data, int data_size);
