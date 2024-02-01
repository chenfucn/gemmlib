/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    blkq4_fp16_gemm_sm80_testcu.cu
 *
 * Abstract:
 *   Test code for invoking block-wise quantized 4b GEMM kernels.
 *   This part requires CUTLASS header files, which do not play
 *   well with gtest headers.
 */

#include "ms_blkq4gemm.h"
#include "matrix_layout.h"
#include "blk_q4/f16_gemm_sm80.h"
#include "blkq4_fp16_gemm_sm80.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace cuda{
namespace test{

bool sm80_supported(){
  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "Unable to obtain GPU device properties: " << cudaGetErrorString(error);
    return false;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Device compute capability mismatch, desired 8.0, actual " << props.major << "." << props.minor;
    return false;
  }
  return true;
}

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
////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Converting cutlass tensor to MatrixRef
//

template <
  typename Element,
  typename Layout>
__forceinline__
mickey::MatrixRef<Element, Layout, true> make_MatrixRef(cutlass::HostTensor<Element, Layout> const& tensor) {
  static_assert(std::is_same<Layout, cutlass::layout::ColumnMajor>::value
                || std::is_same<Layout, cutlass::layout::RowMajor>::value);
  auto shape = cutlass::make_Coord(tensor.extent().row(), tensor.extent().column());
  auto* ptr = const_cast<typename std::remove_const<Element>::type *>(tensor.host_data());
  return mickey::MatrixRef<Element, Layout, true>(ptr, tensor.capacity(), shape);
}

template <
  typename Element,
  typename Layout>
__forceinline__
mickey::MatrixRef<Element const, Layout, true> make_ConstMatrixRef(cutlass::HostTensor<Element, Layout> const& tensor) {
  static_assert(std::is_same<Layout, cutlass::layout::ColumnMajor>::value
                || std::is_same<Layout, cutlass::layout::RowMajor>::value);
  auto shape = cutlass::make_Coord(tensor.extent().row(), tensor.extent().column());
  return mickey::MatrixRef<Element const, Layout, true>(tensor.host_data(), tensor.capacity(), shape);
}

//
// Invoking the kernel
//

template<
    int block_size,
    bool column_wise_blocking,
    bool small_m,
    bool has_offsets>
void run_blkq4_gemm(int m, int n, int k) {

  using ElementDequant = cutlass::half_t;
  using QuantBlocking =
      std::conditional_t<column_wise_blocking,
                         cutlass::MatrixShape<block_size, 1>,
                         cutlass::MatrixShape<1, block_size>>;

  using GemmRunner = mickey::BlkQ4F16GemmImpl<ElementDequant, QuantBlocking, small_m, has_offsets>;

  using ElementAccumulator = typename GemmRunner::ElementAccumulator;
  using ElementComputeEpilogue = typename GemmRunner::ElementComputeEpilogue;
  using ElementInputA = typename GemmRunner::ElementInputA;
  using ElementOutput = typename GemmRunner::ElementOutput;
  using ElementW = typename GemmRunner::ElementW;
  using ElementWPack = typename GemmRunner::ElementWPack;
  using ElementQScale = typename GemmRunner::ElementQScale;
  using ElementQOffset = typename GemmRunner::ElementQOffset;

  using LayoutInputA = typename GemmRunner::LayoutInputA;
  using LayoutOutput = typename GemmRunner::LayoutOutput;
  using LayoutInputWPack = typename GemmRunner::LayoutInputWPack;
  using LayoutInputQScale = typename GemmRunner::LayoutInputQScale;

  const cutlass::gemm::GemmCoord problem_size = {m, n, k};

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K

  // Create weight matrix with dimensions K x N.
  // Actual weight type is int4, we use ElementW = uint8 to avoid possible compilation
  // troubles. Since the layout is column major, we are packing 2 weights in a column
  // into one int8
  auto q_weight_shape = cutlass::make_Coord(problem_size.k()/2, problem_size.n());
  auto q_meta_shape = cutlass::make_Coord(problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn);
  cutlass::HostTensor<ElementW, LayoutInputWPack> tensor_weight(q_weight_shape);
  // Create weight quantization scale and offset with dimensions K x N
  cutlass::HostTensor<ElementQScale, LayoutInputQScale> tensor_scale(q_meta_shape);
  cutlass::HostTensor<ElementQOffset, LayoutInputQScale> tensor_offset(q_meta_shape);

  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      2);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros

  //
  // For testing quantization and dequantization, it is not straight
  // forward to avoid flaky tests due to rounding errors. The way we
  // try to achieve this is to:
  // 1. Generate a set of quantized weights, scales and offsets
  // 2. Dequantize the weights
  // 3. Quantize the dequantized weights
  // 4. Compare the dequantied-and-then-quantized weights with
  //    the original quantized weights
  //
  // Random filling of the initial values are key to get this right.
  // For weights, we must ensure each block gets a full range of
  // values, i.e. must contain 0 and 15. And for scales, they must
  // all be positive.
  //

  int v = 7;
  for (int c = 0; c < tensor_weight.extent()[1]; c++) {
    for (int r = 0; r < tensor_weight.extent()[0]; ++r) {
      uint8_t v0 = static_cast<uint8_t>(v);
      v = (v + 5) % 16;
      if (v == 11 || v == 7 || v == 3) {
        // making the cycle 13 instead of 16, avoiding same values in a row
        v = (v + 5) % 16;
      }
      uint8_t v1 = 0;
      v1 = static_cast<uint8_t>(v);
      v = (v + 5) % 16;
      if (v == 11 || v == 7 || v == 3) {
        // making the cycle 13 instead of 16, avoiding same values in a row
        v = (v + 5) % 16;
      }

      tensor_weight.at({r, c}) = ElementW((v1 << 4) | v0);
    }
  }

  for (int c = 0; c < tensor_scale.extent()[1]; c++) {
    for (int r = 0; r < tensor_scale.extent()[0]; ++r) {
      int f = (((c * v + r + v / 3 ) % 63) + 1);
      v += 41;
      int m = c * v + r + v * 3;
      tensor_scale.at({r, c}) = ElementQScale(static_cast<float>(f) / static_cast<float>(1 << (4 + (m % 2))));
      if (has_offsets) {
        tensor_offset.at({r, c}) = ElementQOffset(((f + m + v) % 8) + 4);
      }
    }
  }

//   // Fill tensor_weight with the patterned data, so that we can use
//   // print to make sure the layout matches after loaded to registers
//   int loop_val = 0;
//   int offset = 3;
//   for (int col_tile = 0; col_tile < tensor_weight.extent().column()/8; ++col_tile) {
//     for (int row_tile = 0; row_tile < tensor_weight.extent().row()/4; ++row_tile) {
//       for (int col = 0; col < 8; ++col) {
//         for (int row = 0; row < 4; ++row) {
//           auto weight_cord = cutlass::make_Coord(row_tile * 4 + row, col_tile * 8 + col);
//           auto val = (loop_val + offset) % 256;
//           tensor_weight.at(weight_cord) = ElementW(val);
//           loop_val++;
//           if (loop_val == 256) {
//             loop_val = 0;
//             offset += 11;
//           }
//         }
//       }
//     }
//   }
//   for (int col = 0; col < tensor_scale.extent().column(); ++col){
//     int c =  col * QuantBlocking::kColumn;
//     for (int row = 0; row < tensor_scale.extent().row(); ++row){
//       int r = row * QuantBlocking::kRow;
//       auto weight_cord = cutlass::make_Coord(r/2, c);
//       int w = 0;
//       if (r % 2 == 0) {
//         w = int(tensor_weight.at(weight_cord) & 0x0f);
//       } else {
//         w = int(tensor_weight.at(weight_cord) >> 4);
//       }
//       tensor_scale.at({row, col}) = w;
// #ifdef USE_QUANT_OFFSET
//       tensor_offset.at({row, col}) = ElementQOffset(w);
// #endif
//     }
//   }

  // int fill_val = -512;
  // int factor = 1;
  // for (int col = 0; col < tensor_scale.extent().column(); ++col){
  //   for (int row = 0; row < tensor_scale.extent().row(); ++row){
  //     tensor_scale.at({row, col}) = ElementQScale((float)fill_val * float(factor));
  //     fill_val++;
  //     if (fill_val == 512) {
  //       fill_val = -512;
  //       factor += 1;
  //     }
  //   }
  // }

  // Dequantize weights and save into matrix B for reference
  using ElementInputB = ElementInputA;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create dequantized matrix B with dimensions K x N
  for (int col = 0; col < tensor_b.extent().column(); ++col){
    for (int row = 0; row < tensor_b.extent().row(); ++row) {
      auto weight_cord = cutlass::make_Coord(row/2, col);
      auto scale_cord = cutlass::make_Coord(row / QuantBlocking::kRow, col / QuantBlocking::kColumn);
      const uint8_t offset = has_offsets ? tensor_offset.at(scale_cord) : 8;
      int w = 0;
      if (row % 2 == 0) {
        w = int(tensor_weight.at(weight_cord) & 0x0f) - offset;
      } else {
        w = int(tensor_weight.at(weight_cord) >> 4) - offset;
      }
      auto scale = tensor_scale.at(scale_cord);
      tensor_b.at({row, col}) = scale * float(w);
    }
  }
  tensor_b.sync_device();

  // std::cout << "Matrix B:\n" << tensor_b.host_view() << "\n";
  // std::cout << "Matrix Weight:\n" << tensor_weight.host_view() << "\n";
  // std::cout << "Matrix Scale:\n" << tensor_scale.host_view() << "\n";
  // if constexpr (has_offsets) {
  //   std::cout << "Matrix Offset:\n" << tensor_offset.host_view() << "\n";
  // }

  // Prepacking weight matrix and quantization meta data ...
  ElementW *o_elements_dev_ptr = nullptr;
  cudaMalloc(&o_elements_dev_ptr, q_weight_shape.product() * sizeof(ElementW));
  ElementInputB *o_scales_dev_ptr = nullptr;
  cudaMalloc(&o_scales_dev_ptr, q_meta_shape.product() * sizeof(ElementInputB));
  uint8_t *o_zp_dev_ptr = nullptr;
  cudaMalloc(&o_zp_dev_ptr, q_meta_shape.product() * sizeof(uint8_t));

  auto err = mickey::blkq4_fp16_quant_sm80_dispatch(
    block_size,
    column_wise_blocking,
    problem_size.k(), problem_size.n(), problem_size.k(),
    0,
    gsl::make_span(reinterpret_cast<half const*>(tensor_b.device_data()), tensor_b.size()),
    gsl::make_span(o_elements_dev_ptr, q_weight_shape.product()),
    gsl::make_span(reinterpret_cast<half*>(o_scales_dev_ptr), q_meta_shape.product()),
    has_offsets ? gsl::make_span(o_zp_dev_ptr, q_meta_shape.product()) : gsl::span<uint8_t>()); 

#if 0

  // Code for verifying the prepacked weights and quantization meta data
  // Leave it here for debugging purpose in case quantization and prepacking code changes failing the tests
  std::vector<ElementW> o_elements(q_weight_shape.product());
  mickey::MatrixRef<ElementW, cutlass::layout::ColumnMajor, true> tensor_o_elements(o_elements, cutlass::make_Coord(problem_size.k(), problem_size.n()/2));
  cudaMemcpy(o_elements.data(), o_elements_dev_ptr, q_weight_shape.product() * sizeof(ElementW), cudaMemcpyDeviceToHost);

  cutlass::HostTensor<ElementW, LayoutInputWPack> tensor_weight_prepacked(
    cutlass::make_Coord(problem_size.k(), problem_size.n()/2));
  prepack_weights_ref(problem_size.k(), problem_size.n(),
                      make_ConstMatrixRef(tensor_weight),
                      make_MatrixRef(tensor_weight_prepacked));
  for (int col = 0; col < tensor_weight_prepacked.extent().column(); ++col){
    for (int row = 0; row < tensor_weight_prepacked.extent().row(); ++row) {
      auto weight_cord = cutlass::make_Coord(row, col);
      EXPECT_EQ(tensor_weight_prepacked.at(weight_cord), tensor_o_elements.at(weight_cord)) << "[" << row << ", " << col << "]";
    }
  }

  cutlass::HostTensor<ElementQScale, LayoutInputQScale> tensor_scale_prepacked(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});
  cutlass::HostTensor<ElementQOffset, LayoutInputQScale> tensor_offset_prepacked(
      {problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn});

  auto scale_ref = make_ConstMatrixRef(tensor_scale);
  prepack_quant_scales_ref<ElementQScale, typename decltype(scale_ref)::Layout, QuantBlocking>(
      problem_size.k(), problem_size.n(), scale_ref,
      make_MatrixRef(tensor_scale_prepacked));
  if constexpr (has_offsets) {
    auto offset_ref = make_ConstMatrixRef(tensor_offset);
    prepack_quant_offsets_ref<typename decltype(offset_ref)::Layout, QuantBlocking>(
        problem_size.k(), problem_size.n(), offset_ref,
        make_MatrixRef(tensor_offset_prepacked));
  }

  std::vector<ElementQScale> packed_scales(q_meta_shape.product());
  mickey::MatrixRef<ElementQScale, LayoutInputQScale, true> tensor_packed_scales(
      packed_scales, q_meta_shape);
  std::vector<ElementQOffset> packed_zp(q_meta_shape.product());
  mickey::MatrixRef<ElementQOffset, LayoutInputQScale, true> tensor_packed_zp(
      packed_zp, q_meta_shape);
  cudaMemcpy(packed_scales.data(), o_scales_dev_ptr, q_meta_shape.product() * sizeof(ElementQScale), cudaMemcpyDeviceToHost);
  cudaMemcpy(packed_zp.data(), o_zp_dev_ptr, q_meta_shape.product() * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  for (int col = 0; col < tensor_scale_prepacked.extent().column(); ++col){
    for (int row = 0; row < tensor_scale_prepacked.extent().row(); ++row) {
      auto scale_cord = cutlass::make_Coord(row, col);
      EXPECT_EQ(tensor_scale_prepacked.at(scale_cord), tensor_packed_scales.at(scale_cord)) << "[" << row << ", " << col << "]";
      if constexpr (has_offsets) {
        EXPECT_EQ(tensor_offset_prepacked.at(scale_cord), tensor_packed_zp.at(scale_cord)) << "[" << row << ", " << col << "]";
      }
    }
  }

#endif

  // Copy data from host to GPU...
  tensor_a.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  cutlass::TensorRef<ElementWPack const, LayoutInputWPack> ref_W(
    reinterpret_cast<ElementWPack const *>(o_elements_dev_ptr),
    LayoutInputWPack::packed({problem_size.k()/2, problem_size.n()/2}));
  cutlass::TensorRef<ElementQScale const, LayoutInputQScale> ref_scales(
    reinterpret_cast<ElementQScale const *>(o_scales_dev_ptr),
    LayoutInputQScale::packed({problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn}));
  cutlass::TensorRef<ElementQOffset const, LayoutInputQScale> ref_offsets(
    o_zp_dev_ptr,
    LayoutInputQScale::packed({problem_size.k()/QuantBlocking::kRow, problem_size.n()/QuantBlocking::kColumn}));

  // Construct events
  cudaEvent_t finish_gemm_event;
  auto cuda_err = cudaEventCreate(&finish_gemm_event);
  ASSERT_EQ(cuda_err, cudaSuccess) << "Failed to create CUDA event: " << cudaGetErrorString(cuda_err);

  // run GEMM
  cutlass::Status status;
  if constexpr (has_offsets){
    status = GemmRunner::run(
      nullptr, problem_size, tensor_a.device_ref(), ref_W,
      ref_scales, ref_offsets,
      tensor_c.device_ref(), tensor_d.device_ref());
  } else {
    status = GemmRunner::run(
      nullptr, problem_size, tensor_a.device_ref(), ref_W,
      ref_scales,
      tensor_c.device_ref(), tensor_d.device_ref());
  }
  ASSERT_EQ(status, cutlass::Status::kSuccess) << "Kernel execution failed: " << cutlassGetStatusString(status);

  // Record an event when the GEMMs are complete
  cuda_err = cudaEventRecord(finish_gemm_event);
  ASSERT_EQ(cuda_err, cudaSuccess) << "Failed to record CUDA event: " << cudaGetErrorString(cuda_err);

  // Wait for work on the device to complete.
  cuda_err = cudaEventSynchronize(finish_gemm_event);
  ASSERT_EQ(cuda_err, cudaSuccess) << "Failure during sync CUDA event: " << cudaGetErrorString(cuda_err);

  cudaEventDestroy(finish_gemm_event);

  // Preparing reference kernel arguments

  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  compute_gemm_ref<ElementInputA, LayoutInputA,
               ElementInputB, LayoutInputB,
               ElementOutput, LayoutOutput,
               ElementComputeEpilogue, ElementAccumulator>(
      problem_size,
      alpha,
      tensor_a.device_ref(),
      tensor_b.device_ref(),
      beta,
      tensor_c.device_ref(),
      tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());
  ASSERT_TRUE(passed) << "Gemm kernel result wrong!";
}

TEST(BlkQ4Fp16Gemm, Sm80Test) {
  if (!sm80_supported()) {
    return;
  }

  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(32, 32, 64);
  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, true>(32, 32, 64);

  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(32, 96, 64);
  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, true>(32, 96, 64);

  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(32, 96, 192);
  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, true>(32, 96, 192);

  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(256, 672, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, true>(256, 672, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(512, 2048 + 32, 960);
  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(512, 2048 + 32, 960);

  onnxruntime::cuda::test::run_blkq4_gemm<16, false, false, false>(256, 672, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<16, false, false, true>(256, 672, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<64, false, false, false>(256, 1024, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<64, false, false, true>(256, 1024, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<16, true, false, false>(256, 672, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<16, true, false, true>(256, 672, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<64, true, false, false>(256, 1024, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<64, true, false, true>(256, 1024, 576);

  // small m
  onnxruntime::cuda::test::run_blkq4_gemm<16, false, true, false>(16, 704, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<16, false, true, true>(16, 704, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<64, false, true, false>(16, 1024, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<64, false, true, true>(16, 1024, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<16, true, true, false>(16, 672, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<16, true, true, true>(16, 672, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<64, true, true, false>(16, 1024, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<64, true, true, true>(16, 1024, 576);
}


}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
