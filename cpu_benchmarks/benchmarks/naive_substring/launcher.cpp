#include <random>
#include <cstdint>
#include <algorithm>
#include <fstream>

#include "../Compiler.h"
#include "substring.h"

#include "benchmark/benchmark.h"

const long long StartSize = 52428800;           // 50mb
const long long DataSourceSize = 10737418240;   // 10gb
const long long MaxDataSize = 419430400;        // 400mb 
const long long DataSizeStep = 52428800;        // 50mb
const long long PatternLength = 200;            // 200 bytes

char *generate_data(long long data_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // Latin letters + some punctuation signs
    std::uniform_real_distribution<> dis(65, 122);
    
    auto mem = new char[data_size];
    std::generate_n(mem, data_size, [dis, gen]() mutable { return (char) (dis(gen)); });

    return mem;
}

// Benchmark without specialization
void BM_naive_substring(benchmark::State &state)
{
    auto pattern = (Char *) generate_data(PatternLength);
    auto data_source = generate_data(DataSourceSize);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, DataSourceSize - state.range(0) - 1);

    for (auto _: state) 
    {
        
        int result = find_substring(
                            data_source + (long long) (dis(gen)), 
                            state.range(0), 
                            pattern, 
                            PatternLength);
    }
    
    delete[] pattern;
    delete[] data_source;
}                       
BENCHMARK(BM_naive_substring)->DenseRange(StartSize, MaxDataSize, DataSizeStep);

// Benchmark with specialization on pattern string
void BM_naive_subtring_mix(benchmark::State &state)
{
    auto pattern = reinterpret_cast<Char *>(generate_data(PatternLength));
    auto data_source = reinterpret_cast<Char *>(generate_data(DataSourceSize));
 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, DataSourceSize - state.range(0) - 1);

    // Generating specialized function
    Compiler compiler("Naive");
    compiler.setFunction(
            mix_find_substring(
                &compiler.getContext(), 
                pattern, 
                PatternLength));
    auto *spec = reinterpret_cast<long long (*)(const Char *, long long)>(compiler.compile());

    for (auto _: state)
    {
        auto current_data = data_source + (long long) (dis(gen));
        auto result = spec(reinterpret_cast<const Char *>(current_data), state.range(0));
    }

    delete[] pattern;
    delete[] data_source;
}
BENCHMARK(BM_naive_subtring_mix)->DenseRange(StartSize, MaxDataSize, DataSizeStep);

