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

#include "gemm/warp/tensor_core_tile_loader.h"

#include "gtest/gtest.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace onnxruntime {
namespace cuda {
namespace test {

/////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
constexpr int div_up(int a, int b) {
  return (a + b - 1) / b;
}

CUTLASS_HOST_DEVICE
constexpr int round_up(int a, int b) {
  return div_up(a, b) * b;
}

template <
  typename QuantBlocking_,              ///! Shape of the quantization block, either 1xb or bx1
  bool     has_quant_offset_,           ///! Whether the quantization has offset
  typename WarpShape_,                  ///! Warp-scoped matrix multiply-accumulate
  int SplitKSerial_ = 1,                ///! How many warps to split the K dimension in the same MxN block
  int Stages_ = 4                       ///! Stages of the pipelined mainloop
>
struct LoadPackedBTestKernel {
 public:
  using QuantBlocking = QuantBlocking_;
  using WarpShape = WarpShape_;
  static constexpr bool has_quant_offset = has_quant_offset_;
  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kStages = Stages_;

  static_assert(kSplitK > 0 && ((kSplitK - 1) & kSplitK) == 0,
     "kSplitK must be positive and a power of 2");

  static constexpr bool kDebugPrint = false;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementT = cutlass::half_t;
  static constexpr int kElementSize = sizeof(ElementT);
  static_assert(kElementSize == 2, "Only support 16b float now");

  // Quantized weights are packed int4, each 16x16 tile of int4
  // is packed into 8x8 tile of 16b (i.e. 8x16 tile of bytes)
  static_assert(WarpShape::kN % 16 == 0 && WarpShape::kK % 16 == 0,
    "Weight B is packed as 16x16 tiles, warp shape must contain whole tiles!");
  using WarpPackedBShape = cutlass::gemm::GemmShape<1, WarpShape::kN/2, WarpShape::kK>;

  // decide per warp tile loader shape, it loads 1, 2 or 4 tiles at a time
  static constexpr int kNTilesPerLoad = std::min(4, div_up(WarpPackedBShape::kN, 8));
  static constexpr int kKTilesPerLoad = std::min(4/kNTilesPerLoad, div_up(WarpPackedBShape::kK, 16));
  using PackedBLoader = mickey::gemm::warp::TensorCoreTileLoader<kNTilesPerLoad, kKTilesPerLoad>;

  static_assert((WarpPackedBShape::kN % PackedBLoader::kMNStride) == 0);
  static_assert((WarpPackedBShape::kK % PackedBLoader::kKStride) == 0);

  static constexpr int kNloads = WarpPackedBShape::kN / PackedBLoader::kMNStride;
  static constexpr int kKloads = WarpPackedBShape::kK / PackedBLoader::kKStride;

  static constexpr int kWarps = kSplitK; // TODO! more warps when we have a larger thread block shape
  static int const kThreadCount = 32 * kWarps;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size_;

    // Decide thread block level partitioning. Here the K value is always 1,
    // as we don't split K dimension at thread block level. Instead, we split
    // K dimension at warp level based on template parameter SplitKSerial_.
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

    //
    // Methods
    //

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
      gemm_k_size_(round_up(div_up(problem_size.k(), kSplitK), WarpShape::kK)),
      // TODO! grid_tiled_shape_ should be based on thread block shape
      grid_tiled_shape_(cutlass::gemm::GemmCoord(
        1, div_up(problem_size.n(), WarpShape::kN), 1
      )) { }
  };

  /// Shared memory storage structure
  struct SharedStorage {
    /// Buffer for prepacked weights
    static constexpr int kPackedBSizePerIter = kNloads * kKloads * PackedBLoader::kByteSize;
    static constexpr int kPackedBSizePerWarp = kPackedBSizePerIter * kStages;
    static constexpr int kPackedBSize = kPackedBSizePerWarp * kWarps;

    cutlass::AlignedBuffer<uint8_t, kPackedBSize> operand_B;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  LoadPackedBTestKernel() { }

  /// Determines whether kernel satisfies alignment
  static cutlass::Status can_implement(const Params &params) {
    if (reinterpret_cast<uintptr_t>(params.ptr_packed_b_) % 16) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.ptr_packed_b_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.b_byte_stride_ % 16) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.b_byte_stride_ is not aligned to 16 bytes!" << std::endl;
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
      // TODO! Use thread block shape
      int remain = params.problem_size_.k() % params.gemm_k_size_;
      if (remain > 0 && remain < WarpShape::kK * kStages * 2) {
        // spliting too small, may not get enough iterations to rampup pipeline
        std::cerr << "LoadPackedBTestKernel validation fail: kSplitK is too small, k: " << remain << " is smaller than " << (WarpShape::kK * kStages * 4) << std::endl;
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
      params.grid_tiled_shape_.n() <= blockIdx.y) {
      // should not happen
      if (threadIdx.x == 0) {
        printf("CTA out of range %d, %d\n", blockIdx.x, blockIdx.y);
      }
      return;
    }

    //
    // Initialization phase: locating our position
    //
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    const int warp_idx_k = warp_idx % kSplitK;
    const int lane_b_k_offset = lane_idx % 4;
    const int lane_b_n_offset = lane_idx / 4;
#ifndef NDEBUG
    bool assert_pass = true;
    if (warp_idx >= kWarps) {
      assert_pass = false;
      if (lane_idx == 0) {
        printf("warp_idx %d exceeds kWarps %d! Should use %d threads per threadblock for kernel launch!\n",
          warp_idx, kWarps, kThreadCount);
      }
    }
    if (warp_idx_k != warp_idx) {
      assert_pass = false;
      if (lane_idx == 0) {
        printf("warp_idx_k %d should be equal to warp_idx %d while we don't yet specify thread block shape larger than warp shape!\n",
          warp_idx_k, warp_idx);
      }
    }
    assert(assert_pass);
#endif

    // Fragments (hopefully in registers) for A and B
    using FragmentPackedB = cutlass::Array<unsigned, PackedBLoader::kTiles>;
    FragmentPackedB fragment_packed_b[kNloads];

    //
    // for gemm input B size (k,n), packed b is (k/2,n/2), element size 2, column major.
    // so lead dimension byte size is coincidentally k/2 * 2 = k
    // and next dimension size is n/2
    //
    const int n_start = blockIdx.y * WarpShape::kN;   // TODO! change to thread block shape
    const int n_end = min(params.problem_size_.n(), (blockIdx.y + 1) * WarpShape::kN);
    const int packed_n_start = (n_start) / 2;
    const int packed_n_end = n_end / 2;
  
    const int k_start = warp_idx_k * params.gemm_k_size_;
    const int k_end = min(params.problem_size_.k(), (warp_idx_k + 1) * params.gemm_k_size_);

    PackedBLoader packed_b_loader{
      params.ptr_packed_b_,
      params.b_byte_stride_,
      packed_n_start,
      packed_n_end,
      k_start,
      k_end,
      lane_idx};

    if constexpr (kDebugPrint) {
      if (lane_idx == 0) {
        printf("Warp: %d, k_start %d, k_end %d, packed_n_start %d, packed_n_end %d\n",
          warp_idx, k_start, k_end, packed_n_start, packed_n_end);
      }
    }

    int load_k = k_start; // current k index for loading from global memory to shared memory
    int proc_k = k_start; // current k index for reading from shared memory and processing
    int smem_write_stage = 0;
    int smem_read_stage = 0;
    uint8_t* packed_b_shared_ptr = shared_storage.operand_B.data() + 
      SharedStorage::kPackedBSizePerWarp * warp_idx;

    //
    // Prologue
    //
    CUTLASS_PRAGMA_UNROLL
    for (; smem_write_stage < kStages - 1; ++smem_write_stage, load_k += WarpShape::kK) {
      uint8_t* packed_b_smem_ptr = packed_b_shared_ptr + smem_write_stage * SharedStorage::kPackedBSizePerIter;

      CUTLASS_PRAGMA_UNROLL
      for (int k_load = 0; k_load < kKloads; ++k_load) {
        packed_b_loader.load_to(packed_b_smem_ptr);
        packed_b_smem_ptr += PackedBLoader::kByteSize;
        CUTLASS_PRAGMA_UNROLL
        for (int n_load = 1; n_load < kNloads; ++n_load) {
          packed_b_loader.load_with_mn_offset(packed_b_smem_ptr, n_load * PackedBLoader::kMNStride);
          packed_b_smem_ptr += PackedBLoader::kByteSize;
        }
        ++packed_b_loader;
      }

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }    

    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<kStages - 2>();
    //__syncthreads(); is this necessary since the loader is warp based?
    if constexpr(kDebugPrint) {
      if (lane_idx == 0) {
        printf("Prologue, warp: %d, k_start %d, k_end %d, packed_n_start %d, packed_n_end %d\nShapredPtr: %p, WarpPtr: %p\n",
          warp_idx, k_start, k_end, packed_n_start, packed_n_end, shared_storage.operand_B.data(), packed_b_shared_ptr);
      }
      cutlass::debug::dump_shmem(shared_storage.operand_B.data(), SharedStorage::kPackedBSize);
    }

    //
    // Mainloop
    //
    for (; proc_k < k_end; smem_write_stage = (smem_write_stage + 1) % kStages, smem_read_stage = (smem_read_stage + 1) % kStages){
      uint8_t* packed_b_smem_read_ptr = packed_b_shared_ptr + smem_read_stage * SharedStorage::kPackedBSizePerIter;
      uint8_t* packed_b_smem_write_ptr = packed_b_shared_ptr + smem_write_stage * SharedStorage::kPackedBSizePerIter;

      CUTLASS_PRAGMA_UNROLL
      for (int k_load = 0; k_load < kKloads; ++k_load) {
        CUTLASS_PRAGMA_UNROLL
        for (int n_load = 0; n_load < kNloads; ++n_load) {
          PackedBLoader::ldmatrix_sync(fragment_packed_b[n_load], lane_idx, packed_b_smem_read_ptr);
          packed_b_smem_read_ptr += PackedBLoader::kByteSize;

          if constexpr (kDebugPrint) {
            uint8_t const* ptr = reinterpret_cast<uint8_t const*>(fragment_packed_b[n_load].data());
            printf("Warp: %d, lane %2d, smem_read_ptr %p, %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d\n",
              warp_idx, lane_idx, packed_b_smem_read_ptr, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8], ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15]);
          }

          int n_idx = packed_n_start + n_load * PackedBLoader::kMNStride;
          int tt = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int t_k = 0; t_k < PackedBLoader::kKTiles; ++t_k) {
            CUTLASS_PRAGMA_UNROLL
            for (int t_n = 0; t_n < PackedBLoader::kMNTiles; ++t_n, ++tt) {
              int n = n_idx + t_n * 8 + lane_b_n_offset;
              int k = proc_k + t_k * 16 + lane_b_k_offset * sizeof(unsigned);
              if (n < packed_n_end && k < k_end) {
                unsigned* dst = reinterpret_cast<unsigned*>(reinterpret_cast<uint8_t*>(params.ptr_output_) + n * params.output_byte_stride_ + k);
                *dst = fragment_packed_b[n_load][tt];
              }
            }
          }

        }

        proc_k += PackedBLoader::kKStride;

        if (load_k < k_end) {
          packed_b_loader.load_to(packed_b_smem_write_ptr);
          packed_b_smem_write_ptr += PackedBLoader::kByteSize;
          CUTLASS_PRAGMA_UNROLL
          for (int n_load = 1; n_load < kNloads; ++n_load) {
            packed_b_loader.load_with_mn_offset(packed_b_smem_write_ptr, n_load * PackedBLoader::kMNStride);
            packed_b_smem_write_ptr += PackedBLoader::kByteSize;
          }
          ++packed_b_loader;
        }
        load_k += PackedBLoader::kKStride;
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
  typename WarpShape_,                  ///! Warp-scoped matrix multiply-accumulate
  int SplitKSerial_ = 1,                ///! How many warps to split the K dimension in the same MxN block
  int Stages_ = 4                       ///! Stages of the pipelined mainloop
>
class LoadPackedBTest {
 public:
  using QuantBlocking = QuantBlocking_;
  using WarpShape = WarpShape_;
  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kStages = Stages_;

  using TestKernel = LoadPackedBTestKernel<QuantBlocking, false, WarpShape, kSplitK, kStages>;
  using Args = typename TestKernel::Params;

  cutlass::Status run(
    cudaStream_t stream,
    cutlass::gemm::GemmCoord const & problem_size,
    void const *ptr_packed_b,
    int64_t b_byte_stride,
    void* ptr_output,
    int64_t output_byte_stride) {

    Args args(problem_size, ptr_output, output_byte_stride, ptr_packed_b, b_byte_stride, nullptr, 0);
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

template <typename WarpShape, int kSplitK, int kStages>
void test_load_packed_b(int m, int n, int k) {
  using Test = LoadPackedBTest<cutlass::MatrixShape<1,16>, WarpShape, kSplitK, kStages>;
  Test test;
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor> tensor_weight_prepacked(
      cutlass::make_Coord(problem_size.k(), problem_size.n()/2));
  cutlass::reference::host::TensorFillRandomUniform(tensor_weight_prepacked.host_view(), 51, 1, 250);

  cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor> q4_weights_copy(
      cutlass::make_Coord(problem_size.k(), problem_size.n()/2));
  cutlass::reference::host::TensorFill(q4_weights_copy.host_view(), uint8_t(0));

  tensor_weight_prepacked.sync_device();
  q4_weights_copy.sync_device();

  cutlass::Status status = test.run(nullptr, problem_size, tensor_weight_prepacked.device_data(), problem_size.k(), q4_weights_copy.device_data(), problem_size.k());
  ASSERT_EQ(status, cutlass::Status::kSuccess);
  cudaDeviceSynchronize();

  q4_weights_copy.sync_host();

  bool pass = cutlass::reference::host::TensorEquals(tensor_weight_prepacked.host_view(), q4_weights_copy.host_view());
  // if (!pass) {
  //   std::cerr << "Mismatched!\n";
  //   std::cerr << "------------ tensor_weight_prepacked -----------------------\n" << std::endl;
  //   for (int c = 0; c < tensor_weight_prepacked.extent().column(); ++c) {
  //     for (int r = 0; r < tensor_weight_prepacked.extent().row(); ++r) {
  //       printf("%3d ", int(tensor_weight_prepacked.at({r, c})));
  //     }
  //     printf("\n");
  //   }
    
  //   std::cerr << "------------ q4_weights_copy -----------------------\n" << q4_weights_copy.host_view() << std::endl;
  // }
  ASSERT_TRUE(pass);

}

TEST(TensorCoreLoader, PackedBTest) {
  test_load_packed_b<cutlass::gemm::GemmShape<1, 16, 64>, 2, 3>(1, 40, 1024 + 16);
  test_load_packed_b<cutlass::gemm::GemmShape<1, 32, 32>, 2, 3>(1, 80, 1024 + 16);
  test_load_packed_b<cutlass::gemm::GemmShape<1, 16, 64>, 2, 4>(1, 80, 2048 + 16);
}

} // namespace test
} // namespace cuda
} // namespace onnxruntime
