#include <set>
#include <algorithm>
#include <random>

#include "../Compiler.h"
#include "benchmark/benchmark.h"
#include "tensor_product.h"

const unsigned MaxNonZeroCount = 3276800; // 50mb summary (1 item == 16 bytes)
const unsigned StaticMatrixSize = 5;
const unsigned MaxAbsValue = 100;

const unsigned MinTestSize = 327680; // 5mb summary
const unsigned MaxTestSize = 1966080; // 30mb
const unsigned TestSizeStep = 327680; // 5mb

std::pair<CooItem *, unsigned> generate_sparse_matrix()
{
    std::random_device rd;
    std::mt19937 gen(rd());
   
    std::uniform_real_distribution<> dis_size(0, MaxNonZeroCount);
    std::set<std::pair<unsigned, unsigned>> used_coords;
    while (used_coords.size() != MaxNonZeroCount)
    {
        used_coords.insert(std::make_pair(dis_size(gen), dis_size(gen))); 
    }

    CooItem *result = new CooItem[MaxNonZeroCount];
    std::uniform_real_distribution<> dis_value(-MaxAbsValue, MaxAbsValue);
    int i = 0;
    for (auto item : used_coords)
    {
        result[i++] = CooItem { item.first, item.second, dis_value(gen) };
    }

    return std::make_pair(result, MaxNonZeroCount);
}

double *generate_static_matrix()
{
    double result = new double[StaticMatrixSize * StaticMatrixSize];

    std::random_device rd;
    std::mt19937 gen(rd());
   
    std::uniform_real_distribution<> dis_value(-MaxAbsValue, MaxAbsValue);
    for (int i = 0; i < StaticMatrixSize; ++i)
        for (int j = 0; j < StaticMatrixSize; ++j)
            result[i][j] = dis_value(gen);

    return result;
}

void BM_sparse_multiplication(benchmark::State &state)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_size(0, MaxNonZeroCount - TestSize);

    auto sparse_matrix = generate_sparse_matrix();
    auto static_matrix = generate_static_matrix();

    CooItem *out = new CooItem[sparse_size * StaticMatrixSize * StaticMatrixSize];
    for (auto _ : state)
    {
        unsigned test_begin = dis_size(gen);
        multiply_tensor(StaticMatrixSize, static_matrix, TestSize, sparse_matrix + test_begin, out); 
    }

    delete [] sparse_data;
    delete [] static_matrix;
    delete [] out;
}
BENCHMARK(BM_naive_substring)->DenseRange(MinTestSize, MaxTestSize, TestSizeStep);

void BM_sparse_multiplication_mix(benchmark::State &state)
{
    auto test_size = state.range(0); 

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_size(0, MaxNonZeroCount - test_size);

    auto sparse_matrix = generate_sparse_matrix();
    auto static_matrix = generate_static_matrix();

    // Generate specialized function
    Compiler compiler("Tensor");
    compiler.setFunction(
            mix_multiply_tensor(
                &compiler.getContext(), 
                static_matrix, 
                StaticMatrixSize));
    auto *spec = reinterpret_cast<void (*)(unsigned, CooItem *, CooItem *)>(compiler.compile());

    CooItem *out = new CooItem[sparse_size * StaticMatrixSize * StaticMatrixSize];
    for (auto _ : state)
    {
        auto test_begin = dis_size(gen);
        spec(TestSize, sparse_matrix + test_begin, out); 
    }

    delete [] sparse_data;
    delete [] static_matrix;
    delete [] out;
}
BENCHMARK(BM_naive_substring_mix)->DenseRange(MinTestSize, MaxTestSize, TestSizeStep);

