# Бенчмарк специализации алгоритмов на базе LLVM.mix

### Типы бенчмарков
* cuda_benchmarks - бенчмарки на CUDA с демонстрацией проблем
* cpu_benchmarks - бенчмарки на CPU

### Сборка и запуск (Ubuntu 18.04)

1. Выбрать и войти в директорию с бенчмарком нужного типа (CUDA/CPU)
2. `mkdir build`
3. `cmake -DCMAKE_BUILD_TYPE=Release ..` 
4. `cmake --build .`
5. `./benchmarks/{cpu или cuda}_benchmarks`

### Ссылки
* [Репозиторий LLVM.mix](https://github.com/eush77/llvm.mix)
* [Репозиторий Clang.mix](https://github.com/eush77/clang.mix)

