cmake_minimum_required(VERSION 3.10)
project(SUBSTRING LANGUAGES C CXX)

find_package(LLVM REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
llvm_map_components_to_libnames(LLVM_LIBS core executionengine ipo native object orcjit support)
link_libraries(${LLVM_LIBS})

add_subdirectory(deps)
link_libraries(benchmark_main)

find_program(CLANG clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
if(EXISTS ${CLANG})
  message(STATUS "Found ${CLANG}, building C examples")
  add_subdirectory(benchmarks)
endif()
