CC=clang
CXX=clang++
EXECUTABLE=benchmark
LINKER_FLAGS=-L/usr/local/cuda/lib64 -L/usr/local/lib -ldl -lrt -pthread -lcudart_static -lcudart `llvm-config --cppflags --cflags --cxxflags --ldflags --system-libs --libs --link-shared --ignore-libllvm` -fno-rtti

all: spec env
	$(CXX) -o $(EXECUTABLE) substring.o launcher.o Compiler.o Dumper.o Environment.o $(LINKER_FLAGS)

spec: 
	$(CC) -O1 -c substring.c

env: 
	$(CXX) -std=c++14 -c launcher.cpp Compiler.cpp Dumper.cpp Environment.cpp -fno-rtti

clean: 
	rm *.o
	rm $(EXECUTABLE)

