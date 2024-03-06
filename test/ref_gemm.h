/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    ref_gemm.h
 *
 * Abstract:
 *   Reference implementation of GEMM
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/util/reference/device/gemm.h"


namespace onnxruntime {
namespace cuda {
namespace test {
/**
 * @brief Reference implementation of GEMM
 *        Copied directly from cutlass util/reference/device/gemm.h
 *        for the strange reason that compiler insists on asking
 *        for explicit stream argument in kernel launch.
*/
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename AccumulatorType
>
void compute_gemm_ref(
  cutlass::gemm::GemmCoord problem_size,
  ScalarType alpha,
  cutlass::TensorRef<ElementA, LayoutA> tensor_a,
  cutlass::TensorRef<ElementB, LayoutB> tensor_b,
  ScalarType beta,
  cutlass::TensorRef<ElementC, LayoutC> tensor_c,
  cutlass::TensorRef<ElementC, LayoutC> tensor_d,
  AccumulatorType initial_accum = AccumulatorType(0)) {

  // Blocking structure potentially improves performance of reference implementation
  // with a minor increase in complexity.
  //
  // Note, this reference implementation is NOT expected to approach peak performance.
  using OutputTile = cutlass::MatrixShape<4, 4>;

  dim3 block(16, 8);

  dim3 grid(
    (problem_size.m() + block.x * OutputTile::kRow - 1) / (block.x * OutputTile::kRow),
    (problem_size.n() + block.y * OutputTile::kColumn - 1) / (block.y * OutputTile::kColumn)
  );

  // Launch a GEMM kernel
  cutlass::reference::device::kernel::Gemm<
    cutlass::TensorRef<ElementA, LayoutA>,
    cutlass::TensorRef<ElementB, LayoutB>,
    cutlass::TensorRef<ElementC, LayoutC>,
    ScalarType,
    AccumulatorType,
    OutputTile,
    cutlass::multiply_add<AccumulatorType>,
    cutlass::NumericConverter<ElementC, ScalarType>
  ><<<grid, block, 0, 0>>>(
    problem_size,
    alpha,
    tensor_a,
    tensor_b,
    beta,
    tensor_c,
    tensor_d,
    initial_accum
  );
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
