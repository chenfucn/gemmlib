/**
 * Standalone executable for experimenting with CUDA code.
*/

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>

#include <cstdint>
#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"

#include "cute/layout.hpp"

union Float16 {
    uint16_t bits;
    half value;
};

__global__ void test_kernel(int const* input,
                    int* output){
    __shared__ int smem[128];
    cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
            smem, nullptr, false);
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    output[0] = smem[0];

    Float16 f16;
    f16.value = 1032.0f;
    printf("1032: %x\n", f16.bits);
    f16.value = -72.0f;
    printf("-72: %x\n", f16.bits);
}


int main() {

    // Allocate device memory
    thrust::host_vector<int> h_input(128);
    h_input[0] = 1;
    thrust::host_vector<int> h_output(128);
    h_output[0] = 111;

    // Copy data to device
    thrust::device_vector<int> d_input(h_input);
    thrust::device_vector<int> d_output(h_output);

    test_kernel<<<1, 1>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()));

    h_output = d_output;

    printf("Output: %d\n", h_output[0]);

    using Swizzled16 = decltype(
        cute::composition(cute::Swizzle<1,0,3>{}, 
                          cute::Layout<cute::Shape<cute::_2, cute::_16>,
                                       cute::Stride<cute::_1, cute::_2>>{}));
    Swizzled16 swizzled;
    cute::print_layout(swizzled);



    return 0;
}
