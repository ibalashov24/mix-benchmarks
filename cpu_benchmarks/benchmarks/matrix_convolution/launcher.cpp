#include <random>
#include <cstdint>
#include <algorithm>
#include <fstream>

#include "../Compiler.h"
#include "convolution.h"

#include "benchmark/benchmark.h"

const unsigned MaxMatrixHeight = 2500;      // 3433mb of full matrix
const unsigned MaxMatrixWidth = 2500;  
const unsigned DataSizeStep = 500;          // 95mb
const unsigned KernelDim = 3;

// Put your convolution kernel here
double t_kernel[3][3] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};

// Generates pseudorandom double array with given size
double *generate_data(long long data_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    auto mem = new double[data_size];
    std::generate_n(mem, data_size, [dis, gen]() mutable { return (double) (dis(gen)); });

    return mem;
}

// Benchmark without specialization
void BM_convolution(benchmark::State &state)
{
    auto pattern = (Double *) generate_data(KernelDim * KernelDim);
    auto data_source = (double *) generate_data(MaxMatrixHeight * MaxMatrixWidth);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(
            0, 
            MaxMatrixHeight * MaxMatrixWidth - state.range(0) * state.range(0) - 1);

    auto result = new double[MaxMatrixHeight * MaxMatrixWidth];
    for (auto _: state) 
    {
        apply_convolution(
                    state.range(0), 
                    state.range(0), 
                    data_source + (unsigned) (dis(gen)), 
                    KernelDim,
                    (Double *) t_kernel,
                    result);
    }

    delete[] pattern;
    delete[] data_source;
    delete[] result;
}                       
BENCHMARK(BM_convolution)->DenseRange(DataSizeStep, MaxMatrixHeight, DataSizeStep);

// Benchmark with specialization on the kernel
void BM_convolution_mix(benchmark::State &state)
{
    auto pattern = (Double *) generate_data(KernelDim * KernelDim);
    auto data_source = (double *) generate_data(MaxMatrixHeight * MaxMatrixWidth);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(
            0, 
            MaxMatrixHeight * MaxMatrixWidth - state.range(0) * state.range(0) - 1);

    auto result = new double[MaxMatrixHeight * MaxMatrixWidth];
    
    // Generating specialized function
    Compiler compiler("Naive");
    compiler.setFunction(
            apply_convolution_mix(
                &compiler.getContext(), 
                KernelDim,
                (Double *) t_kernel));
    auto *spec = reinterpret_cast<void (*)(unsigned, unsigned, double *, double *)>(compiler.compile());

    for (auto _: state)
    {
        auto current_data = data_source + (unsigned) (dis(gen));
        spec(
                state.range(0), 
                state.range(0),
                (double *) (current_data), 
                result);
    }

    delete[] pattern;
    delete[] data_source;
    delete[] result;
}
BENCHMARK(BM_convolution_mix)->DenseRange(DataSizeStep, MaxMatrixHeight, DataSizeStep);

