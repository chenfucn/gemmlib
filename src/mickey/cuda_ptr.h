/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    matrix_layout.h
 *
 * Abstract:
 *   Utils for simplifying positioning and striding in tensors. Inspired
 *   by CUTLASS, striving for 0 runtime cost while promote safety.
 *
 *   Only supports 2D tensors (matrix) for now.
 */

#pragma once

#include <cstdint>
#include <gsl/gsl>
#include <cuda_runtime_api.h>

namespace mickey {

namespace detail {

auto  myCudaMalloc = [](size_t mySize) { void* ptr; cudaMalloc((void**)&ptr, mySize); return ptr; };
auto deleter = [](void* ptr) { cudaFree(ptr); /*std::cout<<"\nDeleted3\n";*/ };

}  // namespace detail

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, decltype(detail::deleter)>;

template <typename T>
cuda_unique_ptr<T> make_cuda_unique() {
  return cuda_unique_ptr<T>(nullptr, detail::deleter);
}

template <typename T>
cuda_unique_ptr<T> make_cuda_unique(size_t num_elements) {
  return cuda_unique_ptr<T>(static_cast<T*>(detail::myCudaMalloc(num_elements * sizeof(T))), detail::deleter);
}


}  // namespace mickey