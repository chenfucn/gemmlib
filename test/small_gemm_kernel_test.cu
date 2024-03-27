/**
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file small_gemm_kernel_test.cu
 * 
 * Test the mixed precision gemm kernel specifically for the small matrices
 * with sliced k.
 */

#include <cuda.h>
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/arch/mma.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/debug.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "matrix_layout.h"
#include "blkq4_fp16_util.h"
#include "blkq4_fp16_gemm_sm80.h"
#include "ref_gemm.h"

#include "gemm/kernel/quant_b4_gemm.h"

#include "gtest/gtest.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace onnxruntime {
namespace cuda {
namespace test {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename QuantBlocking_,              ///! Shape of the quantization block, either 1xb or bx1
  typename WarpShape_,                  ///! Warp-scoped matrix multiply-accumulate
  int SplitKSerial_ = 1,                ///! How many warps to split the K dimension in the same MxN block
  int Stages_ = 4                       ///! Stages of the pipelined mainloop
>
class QuantB4GemmTestDevKernel {
 public:
  using QuantBlocking = QuantBlocking_;
  using WarpShape = WarpShape_;
  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kStages = Stages_;

  using TestKernel = mickey::gemm::kernel::QuantB4Gemm<QuantBlocking, false, WarpShape, kSplitK, kStages>;
  using Args = typename TestKernel::Params;

  cutlass::Status run(
    cudaStream_t stream,
    cutlass::gemm::GemmCoord const & problem_size,
    void* ptr_output,
    int output_byte_stride,
    void const *ptr_a,
    int a_byte_stride,
    void const *ptr_packed_b,
    int b_byte_stride,
    void const *ptr_scales,
    int scales_byte_stride) {

    Args args(problem_size, ptr_output, output_byte_stride,
              ptr_a, a_byte_stride, ptr_packed_b, b_byte_stride,
              ptr_scales, scales_byte_stride);
    cutlass::Status status = TestKernel::can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    dim3 grid(args.grid_tiled_shape_.m(), args.grid_tiled_shape_.n(), args.grid_tiled_shape_.k());
    dim3 block(TestKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename TestKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<TestKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        std::cerr << "Failed to obtain maximum shared memory size " << smem_size << " for kernel: "
                  << cudaGetErrorString(result) << "\n";
        return cutlass::Status::kErrorInternal;
      }
    }
   
    cutlass::Kernel<TestKernel><<<grid, block, smem_size, stream>>>(args);

    return cutlass::Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename LayoutT, typename ElementT>
void print_tiled_tensor(cutlass::HostTensor<ElementT, LayoutT>& t) {
  for (int row = 0; row < t.extent()[0]; ++row) {
    for (int col = 0; col < t.extent()[1]; ++col) {
      printf("%f, ", static_cast<float>(t.at({row, col})));
      if (col % 8 == 7) {
        printf(", ");
      }
    }
    printf("\n");
    if (row % 8 == 7) {
      printf("\n");
    }
  }
}


template <typename QuantBlocking, typename WarpShape, int kSplitK, int kStages>
void test_quantb4_gemm(int m, int n, int k) {
  std::cout << "Testing Blocking: " << QuantBlocking::kRow << "x" << QuantBlocking::kColumn 
            << " WarpShape: " << WarpShape::kM << "x" << WarpShape::kN << "x" << WarpShape::kK
            << ", kSplitK: " << kSplitK << ", kStages: " << kStages;
  std::cout << ", m: " << m << ", n: " << n << ", k: " << k << std::endl;

  using Test = QuantB4GemmTestDevKernel<QuantBlocking, WarpShape, kSplitK, kStages>;
  Test test;
  cutlass::gemm::GemmCoord problem_size(m, n, k);

  constexpr bool has_offsets = false;
  using QuantBaseT = onnxruntime::test::BlkQuantizationRef<QuantBlocking, has_offsets>;
  using LayoutQMeta = typename QuantBaseT::LayoutQMeta;

  // fill the tensor with reduced bits fp16 seems to be necessary to avoid rounding errors
  // during test. Need to investigate further why.
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_a({m, k});
  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 174321, 1.5f, -1.125f, 6);

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_b({k, n});
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 193456, 1.75f, -1.25f, 8);
  cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor> q4_weights;
  cutlass::HostTensor<cutlass::half_t, LayoutQMeta> scales;
  cutlass::HostTensor<uint8_t, LayoutQMeta> offsets;

  QuantBaseT::QuantizeFp16To4Bit(tensor_b, q4_weights, scales, offsets);
  QuantBaseT::Dequantize4BitToFp16(tensor_b, q4_weights, scales, offsets);
  QuantBaseT::QuantizeFp16To4Bit(tensor_b, q4_weights, scales, offsets);
  cutlass::reference::host::TensorFill(tensor_b.host_view(), cutlass::half_t(0));

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> dst;
  QuantBaseT::Dequantize4BitToFp16(dst, q4_weights, scales, offsets);

  // Allocate result tensor
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_d(
      problem_size.mn());
  cutlass::reference::host::TensorFill(tensor_d.host_view());

#if 0
  // Debug print the weights tensor detail
  for (int col = 0; col < n; ++col) {
  for (int row = 0; row < k; ++row) {
  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; ++col) {
    for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; ++col) {
      auto weight_pos = cutlass::make_Coord(row/2, col);
      auto meta_pos = cutlass::make_Coord(row / QuantBlocking::kRow, col / QuantBlocking::kColumn);
      const float scale = static_cast<float>(scales.at(meta_pos));
      const uint8_t offset = has_offsets ? offsets.at(meta_pos) : 8;
      const int w = (row % 2 == 0) ? (q4_weights.at(weight_pos) & 0xf) : (q4_weights.at(weight_pos) >> 4);

      const float f = scale * (w - offset);
      printf("%f=%2dx%f,  ", float(dst.at({row, col})), w, scale);
      ASSERT_EQ(dst.at({row, col}), cutlass::half_t(f));
    }
    printf("\n");
  }

  // Debug print the tensor A
  for (int row = 0; row < tensor_a.extent()[0]; ++row) {
    for (int col = 0; col < tensor_a.extent()[1]; ++col) {
      printf("%f, ", float(tensor_a.at({row, col})));
    }
    printf("\n");
  }
#endif

  std::vector<uint8_t> packed_w_ref(k * n / 2);
  mickey::MatrixRef<uint8_t, cutlass::layout::ColumnMajor, true> tensor_packed_w_ref(
      packed_w_ref, cutlass::make_Coord(k, n / 2));
  onnxruntime::cuda::test::prepack_weights_ref(k, n, onnxruntime::test::make_ConstMatrixRef(q4_weights), tensor_packed_w_ref);

  int meta_tensor_stride = scales.stride(0);
  thrust::device_vector<cutlass::half_t> packed_scale_dev;

  if constexpr (std::is_same<LayoutQMeta, cutlass::layout::ColumnMajor>::value) {
    std::vector<cutlass::half_t> packed_scales_ref(scales.size());
    mickey::MatrixRef<cutlass::half_t, LayoutQMeta, true> tensor_packed_s_ref =
        mickey::make_MatrixRef<cutlass::half_t, LayoutQMeta, true>(packed_scales_ref, scales.extent());
    onnxruntime::cuda::test::prepack_quant_scales_ref<cutlass::half_t, LayoutQMeta, QuantBlocking>(
        k, n, onnxruntime::test::make_ConstMatrixRef(scales), tensor_packed_s_ref);
    packed_scale_dev = packed_scales_ref;
  
    // std::vector<uint8_t> packed_zp_ref(meta_shape.product());
    // mickey::MatrixRef<uint8_t, LayoutQMeta, true> tensor_packed_zp_ref =
    //     mickey::make_MatrixRef<ElementQOffset, LayoutQMeta, true>(packed_zp_ref, meta_shape);
    // onnxruntime::cuda::test::prepack_quant_offsets_ref<LayoutQMeta, QuantBlocking>(
    //       rows, columns, tensor_offset.const_ref(), tensor_packed_zp_ref);
  } else {
    packed_scale_dev.resize(scales.size());
    thrust::copy(scales.host_data(), scales.host_data() + scales.size(), packed_scale_dev.begin());
  }

  thrust::device_vector<uint8_t> packed_w_dev(packed_w_ref);
  tensor_d.sync_device();
  tensor_a.sync_device();

  cutlass::Status status = test.run(nullptr, problem_size,
                                    tensor_d.device_data(), tensor_d.stride(0) * sizeof(cutlass::half_t),
                                    tensor_a.device_data(), tensor_a.stride(0) * sizeof(cutlass::half_t),
                                    thrust::raw_pointer_cast(packed_w_dev.data()), problem_size.k(),
                                    thrust::raw_pointer_cast(packed_scale_dev.data()), meta_tensor_stride * sizeof(cutlass::half_t));
  ASSERT_EQ(status, cutlass::Status::kSuccess);
  tensor_d.sync_host();

  // Run reference kernel
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel
  cutlass::reference::host::TensorFill(tensor_ref_d.host_view());
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_c(
      problem_size.mn());
  cutlass::reference::host::TensorFill(tensor_c.host_view());

  tensor_ref_d.sync_device();
  tensor_c.sync_device();
  dst.sync_device();

  // Initialize alpha and beta for dot product computation
  float alpha = 1.0f;
  float beta = 0.0f;

  compute_gemm_ref<cutlass::half_t, cutlass::layout::RowMajor,
                   cutlass::half_t, cutlass::layout::RowMajor,
                   cutlass::half_t, cutlass::layout::RowMajor,
                   float, float>(
      problem_size,
      alpha,
      tensor_a.device_ref(),
      dst.device_ref(),
      beta,
      tensor_c.device_ref(),
      tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  tensor_ref_d.sync_host();

  for (int row = 0; row < tensor_d.extent()[0]; ++row) {
    for (int col = 0; col < tensor_d.extent()[1]; ++col) {
      float expected = tensor_ref_d.at({row, col});
      float actual = tensor_d.at({row, col});
      if (expected == actual) {
        continue;
      }
      float diff = fabs(expected - actual);
      if (diff < 2e-7) {
        continue;
      }
      float diff_ratio = fabs(expected - actual) / max(fabs(expected), fabs(actual)); 
      if (diff_ratio > 3e-3) {
        std::cerr << "Mismatch found at (" << row << ", " << col << "): " << expected << " != " << actual << " ratio: " << diff_ratio << std::endl;
        ASSERT_TRUE(false);
      }
    }
  }
}

TEST(QuantB4Gemm, PackedBTest) {
  test_quantb4_gemm<cutlass::MatrixShape<1, 16>, cutlass::gemm::GemmShape<16, 64, 16>, 4, 2>(31, 128 + 32, 1024 + 16);
  test_quantb4_gemm<cutlass::MatrixShape<128, 1>, cutlass::gemm::GemmShape<16, 64, 16>, 8, 3>(67, 128 + 16, 4096 + 128);

  test_quantb4_gemm<cutlass::MatrixShape<128,1>, cutlass::gemm::GemmShape<16, 16, 64>, 2, 3>(65, 48, 1024 + 128);
  test_quantb4_gemm<cutlass::MatrixShape<1, 64>, cutlass::gemm::GemmShape<16, 16, 64>, 4, 4>(1, 128, 4096 + 16);

  test_quantb4_gemm<cutlass::MatrixShape<1, 16>, cutlass::gemm::GemmShape<32, 32, 32>, 1, 3>(35, 48, 32 * 4 + 16);
  test_quantb4_gemm<cutlass::MatrixShape<16, 1>, cutlass::gemm::GemmShape<32, 32, 32>, 1, 2>(35, 48, 32 * 3 + 16);
  test_quantb4_gemm<cutlass::MatrixShape<1, 128>, cutlass::gemm::GemmShape<16, 32, 32>, 8, 3>(70, 128, 4096 + 16);
  test_quantb4_gemm<cutlass::MatrixShape<64, 1>, cutlass::gemm::GemmShape<64, 32, 32>, 2, 2>(70, 48, 64 * 7);

  test_quantb4_gemm<cutlass::MatrixShape<1, 32>, cutlass::gemm::GemmShape<64, 64, 128>, 1, 4>(68, 160, 4096 + 16);
  test_quantb4_gemm<cutlass::MatrixShape<32, 1>, cutlass::gemm::GemmShape<128, 128, 128>, 1, 2>(170, 176, 2048 + 32);
}

} // namespace test
} // namespace cuda
} // namespace onnxruntime
