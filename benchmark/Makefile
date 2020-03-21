CC=clang
CXX=clang++
CXX_STANDARD=14
CFLAGS=-O1 --ferror-limit=100 -Wall 
CUDA_PATH=/usr/local/cuda/lib64

all: compiler options benchmark

compiler:
	$(CXX) $(CFLAGS) Compiler.cpp Dumper.cpp Environment.cpp


