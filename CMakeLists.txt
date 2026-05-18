cmake_minimum_required(VERSION 3.8)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake_config.cmake)

project(cu_operator LANGUAGES CXX CUDA)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS
        /usr/include
        /usr/local/cuda/include
        /usr/local/cudnn/include
        $ENV{CUDNN_ROOT}/include
)
find_library(CUDNN_LIBRARY cudnn
    HINTS
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/cudnn/lib64
        $ENV{CUDNN_ROOT}/lib64
        $ENV{CUDNN_ROOT}/lib
)

if (NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found. Please set CUDNN_ROOT or install cuDNN.")
endif()

file(GLOB FA_SRC CONFIGURE_DEPENDS "src/fa/*")

add_library(cu_operator STATIC
	src/sgemm.cu
    src/hgemm.cu
    ${FA_SRC}
)
target_link_libraries(cu_operator PUBLIC
    CUDA::cuda_driver
    CUDA::nvrtc
    ${CUDNN_LIBRARY}
)

target_compile_options(cu_operator PRIVATE -O2 --keep -src-in-ptx -lineinfo -Xcompiler -Wno-unused-variable)

target_include_directories(cu_operator PUBLIC src/ ${CUDNN_INCLUDE_DIR} cudnn-frontend/include)

set_target_properties(cu_operator PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)


add_library(cu_expriment STATIC
    src/expr/lsu_test.cu
    src/expr/wavefront_test.cu
    src/expr/cp_async_test.cu
    src/expr/ldg128_test.cu
)

target_compile_options(cu_expriment PRIVATE -O3 --keep -src-in-ptx -lineinfo)
target_include_directories(cu_expriment PUBLIC src/expr)
target_link_libraries(cu_expriment PRIVATE cu_operator)

add_executable(test_fa test/fa.cu)
target_compile_options(test_fa PRIVATE -O2)
target_link_libraries(test_fa cu_operator cudnn)

add_executable(test_hgemm test/hgemm.cu)
target_compile_options(test_hgemm PRIVATE -O2)
target_link_libraries(test_hgemm cu_operator cublas)

add_executable(test_sgemm test/sgemm.cu)
target_compile_options(test_sgemm PRIVATE -O2)
target_link_libraries(test_sgemm cu_operator cublas)

set_target_properties(test_sgemm PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

##### experiment -- lsu #####
add_executable(test_lsu test/lsu.cpp)
target_compile_options(test_lsu PRIVATE -O2)
target_link_libraries(test_lsu cu_expriment)

set_target_properties(test_lsu PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

##### experiment -- wavefront #####
add_executable(test_wavefront test/wavefront.cpp)
target_compile_options(test_wavefront PRIVATE -O2)
target_link_libraries(test_wavefront cu_expriment)

set_target_properties(test_wavefront PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

##### experiment -- cp_async #####
add_executable(test_cp_async test/cp_async.cpp)
target_compile_options(test_cp_async PRIVATE -O2)
target_link_libraries(test_cp_async cu_expriment)

##### experiment -- ldg128 #####
add_executable(test_ldg128 test/ldg128.cpp)
target_compile_options(test_ldg128 PRIVATE -O2)
target_link_libraries(test_ldg128 cu_expriment)

##### experiment -- pcie bandwidth #####
add_executable(test_pcie_bandwidth test/pcie_bandwidth.cu)
target_compile_options(test_pcie_bandwidth PRIVATE -O2)

##### experiment -- ex2_fp16x2 throughoutput #####
add_executable(test_ex2_fp16x2_throughoutput test/ex2_fp16x2.cu)
target_compile_options(test_ex2_fp16x2_throughoutput PRIVATE -O2)

set_target_properties(test_cp_async PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
