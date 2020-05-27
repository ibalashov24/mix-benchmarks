# Бенчмарк специализации алгоритмов на базе LLVM.mix

### Типы бенчмарков
* cuda_benchmarks - бенчмарки на CUDA с демонстрацией проблем
* cpu_benchmarks - бенчмарки на CPU

### Сборка и запуск (Ubuntu 18.04)

1. Выбрать директорию бенчмарка нужного типа (CUDA/CPU) и войти в неё
2. `git clone https://github.com/google/benchmark deps`
2. `mkdir build && cd build`
3. `cmake -DCMAKE_BUILD_TYPE=Release ..` 
4. `cmake --build .`
5. `./benchmarks/{cpu или cuda}_benchmarks`

### Ссылки
* [Репозиторий Google Benchmark](https://github.com/google/benchmark)
* [Репозиторий LLVM.mix](https://github.com/eush77/llvm.mix)
* [Репозиторий Clang.mix](https://github.com/eush77/clang.mix)

