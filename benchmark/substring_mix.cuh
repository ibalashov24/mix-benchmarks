#pragma once

void find_substring(
		const char *patterns, 
		const int *pattern_borders, 
		int pattern_count,
		const char *data, 
		long long data_length, 
		bool *is_entry);

void *mix_find_substring(
			void *context, 
			const char *patterns,
			const int *pattern_borders,
			int pattern_count,
			bool *is_entry);


