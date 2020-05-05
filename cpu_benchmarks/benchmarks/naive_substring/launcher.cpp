#include <random>
#include <cstdint>
#include <algorithm>
#include <fstream>

#include "../Compiler.h"
#include "substring.h"

#include "benchmark/benchmark.h"

//const long long StartSize = 419430400;        // 400mb
//const long long DataSourceSize = 10737418240; // 10gb
//const long long MaxDataSize = 3355443200;     // 3200mb 
//const long long DataSizeStep = 419430400;     // 400mb
//const long long PatternLength = 200;          // 200 bytes

const long long StartSize = 4194304;        // 400mb
const long long DataSourceSize = 10737418240; // 10gb
const long long MaxDataSize = 33554432;     // 3200mb 
const long long DataSizeStep = 4194304;     // 400mb
const long long PatternLength = 200;          // 200 bytes


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

void BM_naive_substring(benchmark::State &state)
{
    auto pattern = (Char *) generate_data(PatternLength);
    auto data_source = (Char *) generate_data(DataSourceSize);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, DataSourceSize - state.range(0) - 1);

//    std::ofstream err("kek.out", std::ios::app);


    for (auto _: state) 
    {
        
        int result =        find_substring(
                    data_source + (long long) (dis(gen)), 
                    state.range(0), 
                    pattern, 
                    PatternLength);

//	err << result << std::endl;
    }

//    err.close();

    delete[] pattern;
    delete[] data_source;
}                       
BENCHMARK(BM_naive_substring)->DenseRange(StartSize, MaxDataSize, DataSizeStep);
//BENCHMARK(BM_naive_substring)->DenseRange(419430400, 419430405, 1);

void BM_naive_subtring_mix(benchmark::State &state)
{
    auto pattern = reinterpret_cast<Char *>(generate_data(PatternLength));
    auto data_source = reinterpret_cast<Char *>(generate_data(DataSourceSize));
 
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
    auto *spec = reinterpret_cast<long long (*)(const Char *, long long)>(compiler.compile());

//    std::ios::sync_with_stdio(false);
//    std::ofstream err("kek.out");

    for (auto _: state)
    {
	auto current_data = data_source + (long long) (dis(gen));
        auto result = spec(reinterpret_cast<const Char *>(current_data), state.range(0));

//	err << result << std::endl;
    }

//    err.close();
    

    delete[] pattern;
    delete[] data_source;
}
//BENCHMARK(BM_naive_subtring_mix)->DenseRange(StartSize, MaxDataSize, DataSizeStep);

