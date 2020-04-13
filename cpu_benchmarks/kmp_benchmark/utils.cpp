#include <iostream>
#include <cstdlib>

static const int BLOCK_SIZE = 2097152; // 2mb

char *read_data(std::istream &input, int count)
{
    char *data;
    data = (char *) malloc(count);

    for (int i = 0; i < count / BLOCK_SIZE + 1; ++i)
    {
        input.read(data + i * BLOCK_SIZE, BLOCK_SIZE);
    }

    return data;
}
