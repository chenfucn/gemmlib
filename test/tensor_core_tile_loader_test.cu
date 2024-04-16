/**
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file tensor_core_tile_loader_test.cu
 */

#include <cuda.h>
#include "cutlass/aligned_buffer.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "matrix_layout.h"
#include "blkq4_fp16_util.h"
#include "blkq4_fp16_gemm_sm80.h"

#include "gemm/warp/tensor_core_tile_loader.h"
#include "gemm/warp/quantb_meta_loader.h"

#include "gtest/gtest.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace onnxruntime {
namespace cuda {
namespace test {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename QuantBlocking_,       ///! Shape of the quantization block, either 1xb or bx1
  bool     has_quant_offset_,    ///! Whether the quantization has offset
  typename TBShape_,             ///! Threadblock-scoped matrix multiply-accumulate block shape
  int SplitKSerial_ = 1,         ///! Split the K dimension in the same MxN block to multiple TBs
  int Stages_ = 4                ///! Stages of the pipelined mainloop
>
struct LoadPackedBTestKernel {
 public:
  //
  // Type definitions
  //

  using QuantBlocking = QuantBlocking_;
  using TBShape = TBShape_;
  static constexpr bool has_quant_offset = has_quant_offset_;
  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kStages = Stages_;

  static constexpr bool kDebugPrint = false;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementT = cutlass::half_t;
  static constexpr int kElementSize = 2;
  static_assert(kElementSize == sizeof(ElementT), "Only support 16b float now");

  static_assert(TBShape::kN % 64 == 0); // 4 tiles to fully utilize ldmatrix inst
  static_assert(TBShape::kK % 16 == 0); // int4 unit tile is 16x16

  using WarpBShape = cutlass::gemm::GemmShape<1, 64, 32>; // TODO!! test 1, 64, 16 too
  using PackedBLoader = mickey::gemm::warp::TensorCoreTileLoader<WarpBShape, TBShape::kK>;
  using MetaLoader = mickey::gemm::warp::QuantBScaleLoader<QuantBlocking, WarpBShape, ElementT, false>;

  // When computing mma, each time we use a (16, WarpBShape::kN) of B, that's
  // (WarpBShape::kN / 8) * 2 tiles, each tile has 2 elements per thread.
  // So we need to load (WarpBShape::kN / 8) * 2 * 2 element of B.
  using FragmentB = cutlass::Array<ElementT, 2 * (WarpBShape::kN / 8) * 2>;

  static constexpr int kNWarps = TBShape::kN / WarpBShape::kN;
  static constexpr int kKWarps = TBShape::kK / WarpBShape::kK;
  static constexpr int kWarps = kNWarps * kKWarps;
  static constexpr int kThreadCount = 32 * kWarps;
  static constexpr int kMmaIterations = TBShape::kK / InstructionShape::kK;
  static constexpr int kWarpMmaIterations = kMmaIterations / kKWarps;
  static_assert(kWarpMmaIterations == 1 || kWarpMmaIterations == 2 || kWarpMmaIterations == 4);

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size_;

    cutlass::gemm::GemmCoord grid_tiled_shape_;
    void* const ptr_output_;
    const int output_byte_stride_;
    void const * const ptr_packed_b_;
    const int b_byte_stride_;
    void const * const ptr_scales_;
    const int scales_byte_stride_;
    void const * const ptr_offsets_;
    const int offsets_byte_stride_;
    int gemm_k_size_{0};

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      void* ptr_output,
      int output_byte_stride,
      void const *ptr_packed_b,
      int b_byte_stride,
      void const *ptr_scales,
      int scales_byte_stride,
      void const *ptr_offsets = nullptr,
      int offsets_byte_stride = 0
    ):
      problem_size_(problem_size),
      ptr_output_(ptr_output),
      output_byte_stride_(output_byte_stride),
      ptr_packed_b_(ptr_packed_b),
      b_byte_stride_(b_byte_stride),
      ptr_scales_(ptr_scales),
      scales_byte_stride_(scales_byte_stride),
      ptr_offsets_(ptr_offsets),
      offsets_byte_stride_(offsets_byte_stride),
      gemm_k_size_(mickey::round_up(mickey::div_up(problem_size.k(), kSplitK), TBShape::kK)),
      grid_tiled_shape_(cutlass::gemm::GemmCoord(
        1, mickey::div_up(problem_size.n(), TBShape::kN), kSplitK
      )) { }
  };

  /// Shared memory storage structure
  struct SharedStorage {
    /// Buffer for prepacked weights
    static constexpr int kPackedBSizePerIter = PackedBLoader::kByteSize;
    static constexpr int kPackedBSizePerWarp = kPackedBSizePerIter * kStages;
    static constexpr int kPackedBSize = kPackedBSizePerWarp * kWarps;
    cutlass::AlignedBuffer<uint8_t, kPackedBSize> operand_B;

    static constexpr int kMetaSizePerIter = MetaLoader::kSmemSize;
    static constexpr int kMetaSizePerWarp = kMetaSizePerIter * kStages;
    static constexpr int kMetaSize = kMetaSizePerWarp * kWarps;
    cutlass::AlignedBuffer<ElementT, kMetaSize> shared_Scale;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  LoadPackedBTestKernel() { }

  /// Determines whether kernel satisfies alignment
  static cutlass::Status can_implement(const Params &params) {
    if ((params.problem_size_.k() % QuantBlocking::kRow != 0) ||
        (params.problem_size_.n() % QuantBlocking::kColumn) != 0){
      std::cerr << "LoadPackedBTestKernel validation fail: partial quantization block not supported!" << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (reinterpret_cast<uintptr_t>(params.ptr_packed_b_) % 128) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.ptr_packed_b_ is not aligned to 128 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.b_byte_stride_ % 128) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.b_byte_stride_ is not aligned to 128 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (reinterpret_cast<uintptr_t>(params.ptr_scales_) % 16) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.ptr_scales_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.scales_byte_stride_ % 16) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.scales_byte_stride_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if constexpr (has_quant_offset) {
      if (params.ptr_offsets_ == nullptr || params.offsets_byte_stride_ == 0) {
        std::cerr << "LoadPackedBTestKernel validation fail: Required quantization offsets are not provided!" << std::endl;
        return cutlass::Status::kErrorInvalidProblem;
      }
      if (reinterpret_cast<uintptr_t>(params.ptr_offsets_) % 16) {
        std::cerr << "LoadPackedBTestKernel validation fail: params.ptr_offsets_ is not aligned to 16 bytes!" << std::endl;
        return cutlass::Status::kErrorMisalignedOperand;
      }
      if (params.offsets_byte_stride_ % 16) {
        std::cerr << "LoadPackedBTestKernel validation fail: params.offsets_byte_stride_ is not aligned to 16 bytes!" << std::endl;
        return cutlass::Status::kErrorMisalignedOperand;
      }
    } else {
      if (params.ptr_offsets_ != nullptr || params.offsets_byte_stride_ != 0) {
        std::cerr << "LoadPackedBTestKernel validation fail: quantization offsets are provided to scale only kernel!" << std::endl;
        return cutlass::Status::kErrorInvalidProblem;
      }
    }

    if (reinterpret_cast<uintptr_t>(params.ptr_output_) % 16) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.ptr_output_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.output_byte_stride_ % 16) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.output_byte_stride_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.problem_size_.k() % 16 != 0) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.problem_size_.k() is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (params.problem_size_.k() > params.b_byte_stride_) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.problem_size_.k() is greater than params.b_byte_stride_!" << std::endl;
      // for gemm of 16b floats, weights is packed to shape (k/2,n/2), column major
      // so stride should be greater or equal to k/2, with element size 2, it should be k
      return cutlass::Status::kErrorInvalidProblem;
    }

    if constexpr (kSplitK > 1){
      if (params.gemm_k_size_ < TBShape::kK * kStages * 2) {
        // spliting too small, may not get enough iterations to rampup pipeline
        std::cerr << "LoadPackedBTestKernel validation fail: kSplitK is too small, k: " << params.gemm_k_size_ << " is smaller than " << (TBShape::kK * kStages * 4) << std::endl;
        return cutlass::Status::kErrorNotSupported;
      }
    }

    return cutlass::Status::kSuccess;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    // Early exit if CTA is out of range
    if (params.grid_tiled_shape_.m() <= blockIdx.x ||
      params.grid_tiled_shape_.n() <= blockIdx.y ||
      params.grid_tiled_shape_.k() <= blockIdx.z){
      // should not happen
      if (threadIdx.x == 0) {
        printf("CTA out of range %d, %d, %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
      }
      return;
    }

    //
    // Initialization phase: locating our position
    //
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
#ifndef NDEBUG
    bool assert_pass = true;
    if (warp_idx >= kWarps) {
      assert_pass = false;
      if (lane_idx == 0) {
        printf("warp_idx %d exceeds kWarps %d! Should use %d threads per threadblock for kernel launch!\n",
          warp_idx, kWarps, kThreadCount);
      }
    }
    assert(assert_pass);
#endif

    typename PackedBLoader::Fragment fragment_packed_b;
    typename MetaLoader::FragmentScales fragment_scales;
    FragmentB fragment_b;

    const int warp_n_idx = warp_idx / kKWarps;
    const int warp_k_idx = warp_idx % kKWarps;
    const int n_start = blockIdx.y * TBShape::kN + warp_n_idx * WarpBShape::kN;
    const int n_end = min(params.problem_size_.n(), n_start + WarpBShape::kN);  
    const int k_start = blockIdx.z * params.gemm_k_size_ + warp_k_idx * WarpBShape::kK;
    const int k_end = min(params.problem_size_.k(), (blockIdx.z + 1) * params.gemm_k_size_);

    PackedBLoader packed_b_loader{
      params.ptr_packed_b_,
      params.b_byte_stride_,
      n_start,
      n_end,
      k_start,
      k_end,
      lane_idx};

    MetaLoader meta_loader{
      lane_idx,
      params.ptr_scales_,
      params.scales_byte_stride_,
      n_start, n_end};

    if constexpr (kDebugPrint) {
      if (lane_idx == 0) {
        printf("Warp: %d, k_start %d, k_end %d, n_start %d, n_end %d\n",
          warp_idx, k_start, k_end, n_start, n_end);
      }
    }

    int load_k = k_start; // current k index for loading from global memory to shared memory
    int proc_k = k_start; // current k index for reading from shared memory and processing
    int smem_write_stage = 0;
    int smem_read_stage = 0;
    uint8_t* packed_b_shared_ptr = packed_b_loader.get_smem_lane_ptr(shared_storage.operand_B.data() + 
        SharedStorage::kPackedBSizePerWarp * warp_idx, lane_idx);

    ElementT* shared_scale_ptr = shared_storage.shared_Scale.data() + SharedStorage::kMetaSizePerWarp * warp_idx;

    //
    // Prologue
    //
    CUTLASS_PRAGMA_UNROLL
    for (; smem_write_stage < kStages - 1; ++smem_write_stage, load_k += TBShape::kK) {
      uint8_t* packed_b_smem_ptr = packed_b_shared_ptr + smem_write_stage * SharedStorage::kPackedBSizePerIter;
      ElementT* scale_smem_ptr = shared_scale_ptr + smem_write_stage * SharedStorage::kMetaSizePerIter;
    
      meta_loader.load_to_smem(lane_idx, load_k, min(k_end - load_k, WarpBShape::kK), scale_smem_ptr);
      packed_b_loader.load_to_smem(packed_b_smem_ptr);
      ++packed_b_loader;

      cutlass::arch::cp_async_fence();
    }    

    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<kStages - 2>();
    //__syncthreads(); is this necessary since the loader is warp based?
    if constexpr(kDebugPrint) {
      if (lane_idx == 0) {
        printf("Prologue, warp: %d, ShapredPtr: %p, WarpPtr: %p\n",
          warp_idx, shared_storage.operand_B.data(), packed_b_shared_ptr);
      }
      cutlass::debug::dump_shmem(shared_storage.operand_B.data(), SharedStorage::kPackedBSize);
    }

    //
    // Mainloop
    //
    for (; proc_k < k_end; smem_write_stage = (smem_write_stage + 1) % kStages, smem_read_stage = (smem_read_stage + 1) % kStages, proc_k += TBShape::kK, load_k += TBShape::kK){
      typename MetaLoader::FragmentScales fragment_addon;
  
      const uint8_t* packed_b_smem_read_ptr = packed_b_shared_ptr + smem_read_stage * SharedStorage::kPackedBSizePerIter;
      uint8_t* packed_b_smem_write_ptr = packed_b_shared_ptr + smem_write_stage * SharedStorage::kPackedBSizePerIter;

      const ElementT* scale_smem_read_ptr = shared_scale_ptr + smem_read_stage * SharedStorage::kMetaSizePerIter;
      ElementT* scale_smem_write_ptr = shared_scale_ptr + smem_write_stage * SharedStorage::kMetaSizePerIter;

      meta_loader.load_to_smem(lane_idx, load_k, min(k_end - load_k, WarpBShape::kK), scale_smem_write_ptr);
      meta_loader.load_fragment(lane_idx, fragment_scales, scale_smem_read_ptr);

      meta_loader.process(fragment_scales, fragment_addon);

      packed_b_loader.load_to_smem(packed_b_smem_write_ptr);
      ++packed_b_loader;

      // Load from shared memory to fragments/registers, and compute mma, 16 k at a time, dictated by Ampere mma shape
      CUTLASS_PRAGMA_UNROLL
      for (int mma_iter = 0; mma_iter < kWarpMmaIterations; ++mma_iter)  {
        PackedBLoader::load_to_register(lane_idx, mma_iter, packed_b_smem_read_ptr, fragment_packed_b);

        meta_loader.dequant_k16(mma_iter, fragment_packed_b, fragment_scales, fragment_addon, fragment_b);
        CUTLASS_PRAGMA_UNROLL
        for (int b_tile_n = 0; b_tile_n < (WarpBShape::kN/8); ++b_tile_n) {
          int n = n_start + b_tile_n * 8 + (lane_idx / 4);
          int k = proc_k + mma_iter * InstructionShape::kK + (lane_idx % 4) * 2;
          if (n < n_end && k < k_end) {
            int stride = params.output_byte_stride_ / kElementSize;
            ElementT* dst = reinterpret_cast<ElementT*>(params.ptr_output_) + k * stride + n;
            const int frag_b_idx = b_tile_n * 4;
            *dst = fragment_b[frag_b_idx];
            dst += stride;
            *dst = fragment_b[frag_b_idx + 1];
            dst += stride * 7;
            *dst = fragment_b[frag_b_idx + 2];
            dst += stride;
            *dst = fragment_b[frag_b_idx + 3];
          }
        }
      }

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();

      // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
      cutlass::arch::cp_async_wait<kStages - 2>();
      //__syncthreads(); is this necessary since the loader is warp based?
      if constexpr(kDebugPrint) {
        if (lane_idx == 0) {
          printf("Mainloop, warp: %d, proc_k %d, load_k %d\nWritePtr: %p, ReadPtr: %p\n",
            warp_idx, proc_k, load_k, packed_b_smem_write_ptr, packed_b_smem_read_ptr);
        }
        cutlass::debug::dump_shmem(shared_storage.operand_B.data(), SharedStorage::kPackedBSize);
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename QuantBlocking_,              ///! Shape of the quantization block, either 1xb or bx1
  typename TBShape_,                  ///! Warp-scoped matrix multiply-accumulate
  int SplitKSerial_ = 1,                ///! How many warps to split the K dimension in the same MxN block
  int Stages_ = 4                       ///! Stages of the pipelined mainloop
>
class LoadPackedBTest {
 public:
  using QuantBlocking = QuantBlocking_;
  using TBShape = TBShape_;
  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kStages = Stages_;

  using TestKernel = LoadPackedBTestKernel<QuantBlocking, false, TBShape, kSplitK, kStages>;
  using Args = typename TestKernel::Params;

  cutlass::Status run(
    cudaStream_t stream,
    cutlass::gemm::GemmCoord const & problem_size,
    void* ptr_output,
    int output_byte_stride,
    void const *ptr_packed_b,
    int b_byte_stride,
    void const *ptr_scales,
    int scales_byte_stride) {

    Args args(problem_size, ptr_output, output_byte_stride, ptr_packed_b, b_byte_stride, ptr_scales, scales_byte_stride);
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

template <typename QuantBlocking, typename TBShape, int kSplitK, int kStages>
void test_load_packed_b(int m, int n, int k) {
  std::cout << "Testing Blocking: " << QuantBlocking::kRow << "x" << QuantBlocking::kColumn 
            << " TBShape: " << TBShape::kM << "x" << TBShape::kN << "x" << TBShape::kK
            << ", kSplitK: " << kSplitK << ", kStages: " << kStages;
  std::cout << ", m: " << m << ", n: " << n << ", k: " << k << std::endl;

  using Test = LoadPackedBTest<QuantBlocking, TBShape, kSplitK, kStages>;
  Test test;
  cutlass::gemm::GemmCoord problem_size(m, n, k);

  constexpr bool has_offsets = false;
  using QuantBaseT = onnxruntime::test::BlkQuantizationRef<QuantBlocking, has_offsets>;
  using LayoutQMeta = typename QuantBaseT::LayoutQMeta;

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_b({k, n});
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 51, -1.75f, 1.9f);
  cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor> q4_weights;
  cutlass::HostTensor<cutlass::half_t, LayoutQMeta> scales;
  cutlass::HostTensor<uint8_t, LayoutQMeta> offsets;

  QuantBaseT::QuantizeFp16To4Bit(tensor_b, q4_weights, scales, offsets);
  QuantBaseT::Dequantize4BitToFp16(tensor_b, q4_weights, scales, offsets);
  QuantBaseT::QuantizeFp16To4Bit(tensor_b, q4_weights, scales, offsets);
  cutlass::reference::host::TensorFill(tensor_b.host_view(), cutlass::half_t(0));

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> dst;
  QuantBaseT::Dequantize4BitToFp16(dst, q4_weights, scales, offsets);

#if 0
  // Debug print the weights tensor detail
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
#endif

  // Pack the weights, each int4 16x16 block is packed to a 128b vector on the k dimension
  std::vector<uint8_t> packed_w_ref(k * n / 2);
  mickey::MatrixRef<uint8_t, cutlass::layout::ColumnMajor, true> tensor_packed_w_ref(
      packed_w_ref, cutlass::make_Coord(k * (128 / 16), n / 16));
  onnxruntime::cuda::test::pack_q4_128b(k, n, onnxruntime::test::make_ConstMatrixRef(q4_weights), tensor_packed_w_ref);

  int packed_b_stride = tensor_packed_w_ref.stride(0);
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
  tensor_b.sync_device();

  int dequant_stride = tensor_b.stride(0);
  ASSERT_EQ(dequant_stride, problem_size.n());
  cutlass::Status status = test.run(nullptr, problem_size,
                                    tensor_b.device_data(), dequant_stride * sizeof(cutlass::half_t),
                                    thrust::raw_pointer_cast(packed_w_dev.data()), packed_b_stride,
                                    thrust::raw_pointer_cast(packed_scale_dev.data()), meta_tensor_stride * sizeof(cutlass::half_t));
  ASSERT_EQ(status, cutlass::Status::kSuccess);
  tensor_b.sync_host();
  cudaDeviceSynchronize();
  bool passed = cutlass::reference::host::TensorEquals(dst.host_view(), tensor_b.host_view());
  if (!passed) {
    std::cerr << "Mismatch found in test_load_packed_b!" << std::endl;
    std::cerr << "Expected:" << std::endl;
    std::cerr << dst.host_view() << std::endl;
    std::cerr << "Actual:" << std::endl;
    std::cerr << tensor_b.host_view() << std::endl;
  }
  ASSERT_TRUE(passed);
}

TEST(TensorCoreLoader, PackedBTest) {
  test_load_packed_b<cutlass::MatrixShape<32, 1>, cutlass::gemm::GemmShape<1, 256, 64>, 4, 3>(1, 512 - 32, 64 * 20 + 32);
  test_load_packed_b<cutlass::MatrixShape<1, 32>, cutlass::gemm::GemmShape<1, 128, 64>, 4, 3>(1, 128 + 32, 64 * 22 - 32);
  test_load_packed_b<cutlass::MatrixShape<1, 64>, cutlass::gemm::GemmShape<1, 256, 64>, 4, 3>(1, 512 - 64, 64 * 20 + 16);
  test_load_packed_b<cutlass::MatrixShape<16, 1>, cutlass::gemm::GemmShape<1, 128, 64>, 4, 3>(1, 512 + 32, 64 * 22 - 32);

  // test_load_packed_b<cutlass::MatrixShape<1, 16>, cutlass::gemm::GemmShape<1, 16, 64>, 1, 4>(1, 48, 1024 + 16);
  // test_load_packed_b<cutlass::MatrixShape<16, 1>, cutlass::gemm::GemmShape<1, 16, 64>, 2, 3>(1, 48, 1024 + 16);
  // test_load_packed_b<cutlass::MatrixShape<128,1>, cutlass::gemm::GemmShape<1, 16, 64>, 2, 3>(1, 48, 1024 + 128);
  // test_load_packed_b<cutlass::MatrixShape<1, 64>, cutlass::gemm::GemmShape<1, 16, 64>, 4, 4>(1, 128, 4096 + 16);

  // test_load_packed_b<cutlass::MatrixShape<1, 32>, cutlass::gemm::GemmShape<1, 32, 32>, 1, 4>(1, 32 * 3, 1024 + 16);
  // test_load_packed_b<cutlass::MatrixShape<32, 1>, cutlass::gemm::GemmShape<1, 32, 32>, 1, 4>(1, 48, 1024 + 32);
  // test_load_packed_b<cutlass::MatrixShape<128,1>, cutlass::gemm::GemmShape<1, 32, 32>, 2, 3>(1, 48, 1024 + 128);
  // test_load_packed_b<cutlass::MatrixShape<1, 64>, cutlass::gemm::GemmShape<1, 32, 32>, 4, 4>(1, 128, 4096 + 16);

  // test_load_packed_b<cutlass::MatrixShape<1, 32>, cutlass::gemm::GemmShape<1, 64, 128>, 1, 4>(1, 160, 4096 + 16);
  // test_load_packed_b<cutlass::MatrixShape<32, 1>, cutlass::gemm::GemmShape<1, 128, 128>, 1, 4>(1, 176, 4096 + 32);
}

} // namespace test
} // namespace cuda
} // namespace onnxruntime
