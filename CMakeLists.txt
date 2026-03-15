cmake_minimum_required(VERSION 3.8)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake_config.cmake)

project(cu_operator LANGUAGES CXX CUDA)

enable_language(CUDA)

add_library(cu_operator STATIC
	src/sgemm.cu
)

target_compile_options(cu_operator PRIVATE -O3 --keep -src-in-ptx)

target_include_directories(cu_operator PUBLIC src/)

set_target_properties(cu_operator PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)


add_executable(test_sgemm test/sgemm.cpp)
target_compile_options(test_sgemm PRIVATE -O2)
target_link_libraries(test_sgemm cu_operator)

set_target_properties(test_sgemm PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

