#pragma once

const int BLOCK_SIZE = 2097152; // 2MB

char *read_data_to_gpu(istream &input, int count);
