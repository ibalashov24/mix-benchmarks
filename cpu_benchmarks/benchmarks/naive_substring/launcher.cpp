#include <random>
#include <cstdint>
#include <algorithm>

#include "../Compiler.h"
#include "substring.h"

#include "benchmark/benchmark.h"

const long long DataSourceSize = 10737418240; // 10gb
const long long MaxDataSize = 3355443200;     // 3200mb 
const long long DataSizeStep = 209715200;     // 200mb
const long long PatternLength = 200;          // 200 bytes

char *generate_data(long long data_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // Latin letters + some punctuation signs
    std::uniform_real_distribution<> dis(65, 122);

   auto mem = new char[data_size];
   std::generate_n(mem, data_size, [dis, gen]() { return (char) dis(gen); });

   return mem;
}

void BM_naive_substring(benchmark::State &state)
{
    auto pattern = generate_data(PatternLength);
    auto data_source = generate_data(DataSourceSize);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, DataSourceSize - state.range(0) - 1);

    for (auto _: state) 
    {
        benchmark::DoNotOptimize(
                find_substring(
                    data_source + dis(gen), 
                    state.range(0), 
                    pattern, 
                    PatternLength));
    }
}                       
BENCHMARK(BM_naive_substring)->DenseRange(200, MaxDataSize, DataSizeStep);

void BM_naive_subtring_mix(benchmark::State &state)
{
    auto pattern = generate_data(PatternLength);
    auto data_source = generate_data(DataSourceSize);
 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, DataSourceSize - state.range(0) - 1);

    // Generate specialized function
    Compiler compiler("Naive");
    compiler.setFunction(
            mix_find_substring(
                &compiler.getContext(), 
                pattern, 
                PatternLength));
    auto *spec = reinterpret_cast<int (*)(const Char *, int)>(compiler.compile());

    for (auto _: state)
    {
        int result = spec(data + dis(gen()), state.range(0));
    }
}
BENCHMARK(BM_naive_subtring_mix)->DenseRange(200, MaxDataSize, DataSizeStep);

