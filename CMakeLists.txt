# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_ARCHITECTURES 80)

# Set the project name
project(gemmlib LANGUAGES CUDA CXX)

set(CUTLASS_NVCC_ARCHS "80;90a")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# fetch the cutlass library from github
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG        a75b4ac483166189a45290783cb0a18af5ff0ea5 # v3.3.0
)

FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  FetchContent_Populate(cutlass)
endif()

FetchContent_Declare(
  gsl
  URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip
  URL_HASH SHA1=cf368104cd22a87b4dd0c80228919bb2df3e2a14
)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(gsl googletest)

list(APPEND INC_PATH ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${cutlass_SOURCE_DIR}/include ${cutlass_SOURCE_DIR}/tools/util/include ${gsl_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/mickey)

message(STATUS "include paths: ${INC_PATH}")

add_library(gemmlib ${PROJECT_SOURCE_DIR}/src/blkq4_f16_gemm_sm80.cu)
target_include_directories(gemmlib PUBLIC ${INC_PATH})

# Tests

enable_testing()

add_executable(gemmtest ${PROJECT_SOURCE_DIR}/test/blkq4_fp16_gemm_sm80_test.cu)
target_include_directories(gemmtest PUBLIC ${INC_PATH})

target_link_libraries(gemmtest PRIVATE gemmlib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(gemmtest)
