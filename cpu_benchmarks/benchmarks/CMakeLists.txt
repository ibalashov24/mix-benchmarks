set(CMAKE_C_COMPILER ${CLANG})
set(CMAKE_C_FLAGS "-std=c11 -fPIC -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O1")
set(CMAKE_CXX_FLAGS
  "-std=c++14 -fno-exceptions -fno-rtti -g -Wall -Wextra -Wno-unused-parameter")

add_executable(cpu-benchmarks Compiler.cpp Dumper.cpp Environment.cpp)
target_include_directories(cpu-benchmarks PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(naive_substring)
add_subdirectory(jones_naive_benchmark)
#add_subdirectory(kmp_benchmark)
