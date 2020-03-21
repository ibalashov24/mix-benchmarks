CC=clang
CXX=clang++
CXX_STANDARD=14
CFLAGS= --ferror-limit=100 -Wall 
CUDA_PATH=/usr/local/cuda/lib64
LINKER_FLAGS=L/usr/local/cuda/lib64 -L/usr/local/lib -ldl -lrt -pthread -lcudart_static -lcudart `llvm-config --cppflags --cflags --cxxflags --ldflags --system-libs --libs --link-shared --ignore-libllvm` -lboost_program_options

all: compiler start benchmark
	$(CXX) $(CFLAGS) Compiler.o Dumper.o Environment.o main.o program_options.o benchmark_utils.o substring_mix.o $(LINKER_FLAGS)

compiler:
	$(CXX) $(CFLAGS) Compiler.cpp Dumper.cpp Environment.cpp -std=c++$(CXX_STANDARD)

start:
	$(CXX) $(CFLAGS) main.cpp program_options.cpp -std=c++$(CXX_STANDARD)

benchmark:
	$(CXX) $(CFLAGS) -O1 benchmark_utils.cu substring_mix.cu --cuda-gpu-arch=sm_60


