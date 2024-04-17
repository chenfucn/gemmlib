/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file kernel/quant_b4_gemv.h
 * @brief Fused GEMM kernel for fp16 x int4, where B matrix is blockwise quantized to 4bits.
 *
 **************************************************************************************************/

#pragma once

#include "cuda_fp16.h"

#include "cutlass/cutlass.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_policy.h"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

#include "gemm/warp/tensor_core_tile_loader.h"
#include "gemm/warp/vec_loader.h"
#include "gemm/warp/quantb_meta_loader.h"
#include "int_util.h"

namespace mickey {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

union __b64 {
  unsigned long long u64;
  float2 f64;
};
static_assert(sizeof(__b64) == 8, "sizeof(__b64) must be 8 bytes.");

CUTLASS_DEVICE
void float2_fmadd(float2 &d, float2 a, float2 b) {
  d.x += a.x * b.x;
  d.y += a.y * b.y;
}

/**
 * @brief Gemm shape 8x16x16, result is 16x16, same as B,
 *       reduction on the k dimension later
 */
CUTLASS_DEVICE
void mma_op(cutlass::Array<__half2, 2> const &a,
            cutlass::Array<__half2, 4> const &b,
            cutlass::Array<float2, 4> &accumulators) {
  float2_fmadd(accumulators[0], __half22float2(a[0]), __half22float2(b[0]));
  float2_fmadd(accumulators[1], __half22float2(a[1]), __half22float2(b[1]));
  float2_fmadd(accumulators[2], __half22float2(a[0]), __half22float2(b[2]));
  float2_fmadd(accumulators[3], __half22float2(a[1]), __half22float2(b[3]));
}

/**
 * @brief Fused GEMM kernel for fp16 x int4, where B matrix is blockwise quantized to 4bits.
 */
template <
  typename QuantBlocking_,     ///! Shape of the quantization block, either 1xb or bx1
  bool     has_quant_offset_,  ///! Whether the quantization has offset
  int SplitKSerial_ = 1,       ///! How many warps to split the K dimension in the same MxN block
  int Stages_ = 4              ///! Stages of the pipelined mainloop
>
struct QuantB4Gemv {
 public:
  //
  // Type definitions
  //

  using QuantBlocking = QuantBlocking_;
  using ElementT = cutlass::half_t;
  using ElementT2 = __half2;
  static constexpr bool has_quant_offset = has_quant_offset_;
  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kStages = Stages_;
  static constexpr int kElementSize = sizeof(ElementT);

  // HBM load best to be 128 bytes, so K must be 64,
  // N must be multiple of 16, as we pack 16x16 of quantized
  // weights into a 8x8 tile (128 bytes)
  using WarpShape = cutlass::gemm::GemmShape<1, 16, 64>;

  //
  // Type constraints verifications:
  //
  static_assert(kSplitK > 0 && ((kSplitK - 1) & kSplitK) == 0,
     "SplitK must be positive and a power of 2");
  static_assert(kStages > 1, "Number of pipeline stages must be greater than 1.");
  static_assert(kElementSize == 2, "Only support 16b float types.");


  /// switches for debug print
  static constexpr bool kDebugPrintB = false;
  static constexpr bool kDebugPrintA = false;
  static constexpr bool kDebugPrintC = false;
  static constexpr bool kDebugPrintSteps = false;

  using ATileLoader = mickey::gemm::warp::VectorLoader;
  using MetaLoader = mickey::gemm::warp::QuantBScaleLoader<QuantBlocking, WarpShape, ElementT, false>;
  using PackedBLoader = mickey::gemm::warp::TensorCoreTileLoader<WarpShape, WarpShape::kK>;

  struct MainLoopSharedBuffer{
    /// Buffer for prepacked weights
    static constexpr int kPackedBSizePerIter = PackedBLoader::kByteSize;
    static constexpr int kPackedBSize = kPackedBSizePerIter * kStages;
    cutlass::AlignedBuffer<uint8_t, kPackedBSize> shared_B;

    /// Buffer for A tensor
    static constexpr int kASizePerIter = ATileLoader::kBlockSize / kElementSize;
    static constexpr int kASize = kASizePerIter * kStages;
    cutlass::AlignedBuffer<ElementT, kASize> shared_A;

    /// Buffer for quantization meta data
    static constexpr int kMetaSizePerIter = MetaLoader::kSmemSize;
    static constexpr int kMetaSize = kMetaSizePerIter * kStages;
    cutlass::AlignedBuffer<ElementT, kMetaSize> shared_Scale;
  };

  static constexpr int kWarps = kSplitK; // TODO! more warps when we have a larger thread block shape
  static constexpr int kThreads = 32 * kWarps;

  // One stage loads k = 64. We compute k = 16 mma at a time in the main loop 
  static constexpr int kMmaIterations = WarpShape::kK / 16;

  //
  // Each warp has its own shared memory buffer, and writes partial results
  // to shared_Acc only after the main loop. Thus we can use `union' to save
  // shared memory space.
  // 
  // On the other hand, we also wasted a little bit of shared memory.
  // Technically, we only need (kWarps - 1) shared_Acc buffers. But we
  // declare kWarps of those buffers, so that we can isolate the shared
  // memory buffer for each warp. Since different warps finish the main
  // loop at different times, we don't want the warp that finishes early
  // to overwrite the shared memory buffer of the warp that still working
  // on the main loop.  An extra __syncthreads can be used to avoid this,
  // but we don't like the performance impact of it.
  //
  union WarpSmemT {
    MainLoopSharedBuffer main_loop;

    /// Buffer for accumulators of the partial results after the main loop
    static constexpr int kAccSizePerWarp = 4 * WarpShape::kN;
    cutlass::AlignedBuffer<float, kAccSizePerWarp> shared_Acc;
  };

  /// Shared memory storage structure
  struct SharedStorage {
    WarpSmemT smem[kWarps];
  };

  // Fragments of quantized weights
  using FragmentPackedB = cutlass::Array<
      unsigned,  // 8 of int4 weights each tile (becomes 4 tiles when de-quantized)
      4>;

  using FragmentB = cutlass::Array<ElementT2, 4>;  // 16x16 tile
  using FragmentB_flat = cutlass::Array<ElementT, 8>;  // same with FragmentB
  using FragmentA = cutlass::Array<ElementT2, 4>;  // 8x32 tile, two iter of 8x16
  using FragmentC = cutlass::Array<float2, 4>;  // same as B, reduction on k later

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size_;

    // Decide thread block level partitioning. Here the K value is always 1,
    // as we don't split K dimension at thread block level. Instead, we split
    // K dimension at warp level based on template parameter SplitKSerial_.
    cutlass::gemm::GemmCoord grid_tiled_shape_;
    void* const ptr_output_;
    const int output_byte_stride_;
    void const * const ptr_a_;
    const int a_byte_stride_;
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
      void const *ptr_a,
      int a_byte_stride,
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
      ptr_a_(ptr_a),
      a_byte_stride_(a_byte_stride),
      ptr_packed_b_(ptr_packed_b),
      b_byte_stride_(b_byte_stride),
      ptr_scales_(ptr_scales),
      scales_byte_stride_(scales_byte_stride),
      ptr_offsets_(ptr_offsets),
      offsets_byte_stride_(offsets_byte_stride),
      gemm_k_size_(mickey::round_up(mickey::div_up(problem_size.k(), kSplitK), WarpShape::kK)),
      grid_tiled_shape_(cutlass::gemm::GemmCoord(
        mickey::div_up(problem_size.m(), WarpShape::kM),
        mickey::div_up(problem_size.n(), WarpShape::kN),
        1)) { }
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  QuantB4Gemv() { }

  /// Determines whether kernel satisfies alignment
  static cutlass::Status can_implement(const Params &params) {
    if ((reinterpret_cast<uintptr_t>(params.ptr_a_) % 16)) {
      std::cerr << "QuantB4Gemm validation fail: params.ptr_a_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.a_byte_stride_ % 16) {
      std::cerr << "QuantB4Gemm validation fail: params.a_byte_stride_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if ((params.problem_size_.k() % QuantBlocking::kRow != 0) ||
        (params.problem_size_.n() % QuantBlocking::kColumn) != 0){
      std::cerr << "QuantB4Gemm validation fail: partial quantization block not supported!" << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (reinterpret_cast<uintptr_t>(params.ptr_packed_b_) % 16) {
      std::cerr << "QuantB4Gemm validation fail: params.ptr_packed_b_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.b_byte_stride_ % 16) {
      std::cerr << "QuantB4Gemm validation fail: params.b_byte_stride_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (reinterpret_cast<uintptr_t>(params.ptr_scales_) % 16) {
      std::cerr << "QuantB4Gemm validation fail: params.ptr_scales_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.scales_byte_stride_ % 16) {
      std::cerr << "QuantB4Gemm validation fail: params.scales_byte_stride_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if constexpr (has_quant_offset) {
      if (params.ptr_offsets_ == nullptr || params.offsets_byte_stride_ == 0) {
        std::cerr << "QuantB4Gemm validation fail: Required quantization offsets are not provided!" << std::endl;
        return cutlass::Status::kErrorInvalidProblem;
      }
      if (reinterpret_cast<uintptr_t>(params.ptr_offsets_) % 16) {
        std::cerr << "QuantB4Gemm validation fail: params.ptr_offsets_ is not aligned to 16 bytes!" << std::endl;
        return cutlass::Status::kErrorMisalignedOperand;
      }
      if (params.offsets_byte_stride_ % 16) {
        std::cerr << "QuantB4Gemm validation fail: params.offsets_byte_stride_ is not aligned to 16 bytes!" << std::endl;
        return cutlass::Status::kErrorMisalignedOperand;
      }
    } else {
      if (params.ptr_offsets_ != nullptr || params.offsets_byte_stride_ != 0) {
        std::cerr << "QuantB4Gemm validation fail: quantization offsets are provided to scale only kernel!" << std::endl;
        return cutlass::Status::kErrorInvalidProblem;
      }
    }

    if (reinterpret_cast<uintptr_t>(params.ptr_output_) % 16) {
      std::cerr << "QuantB4Gemm validation fail: params.ptr_output_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.output_byte_stride_ % 16) {
      std::cerr << "QuantB4Gemm validation fail: params.output_byte_stride_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.problem_size_.n() > (params.output_byte_stride_ / kElementSize)) {
      std::cerr << "QuantB4Gemm validation fail: params.problem_size_.n() is greater than params.output_byte_stride_!" << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (params.problem_size_.k() % 16 != 0) {
      std::cerr << "QuantB4Gemm validation fail: params.problem_size_.k() is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }
    if (params.problem_size_.k() > params.b_byte_stride_) {
      std::cerr << "QuantB4Gemm validation fail: params.problem_size_.k() is greater than params.b_byte_stride_!" << std::endl;
      // for gemm of 16b floats, weights is packed to shape (k/2,n/2), column major
      // so stride should be greater or equal to k/2, with element size 2, it should be k
      return cutlass::Status::kErrorInvalidProblem;
    }

    if constexpr (kSplitK > 1){
      if (params.gemm_k_size_ < WarpShape::kK * kStages * 2) {
        // spliting too small, may not get enough iterations to rampup pipeline
        std::cerr << "QuantB4Gemm validation fail: kSplitK is too big, segment k: " << params.gemm_k_size_ << " is smaller than " << (WarpShape::kK * kStages * 2) << std::endl;
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
    const int warp_idx = div_power2<32>(threadIdx.x);
    const int lane_idx = mod_power2<32>(threadIdx.x);

#ifndef NDEBUG
    bool assert_pass = true;
    if (warp_idx >= kWarps) {
      assert_pass = false;
      if (lane_idx == 0) {
        printf("warp_idx %d exceeds kWarps %d! Should use %d threads per threadblock for kernel launch!\n",
          warp_idx, kWarps, kThreads);
      }
    }
    const int warp_idx_k = mod_power2<kSplitK>(warp_idx);
    if (warp_idx_k != warp_idx) {
      assert_pass = false;
      if (lane_idx == 0) {
        printf("warp_idx_k %d should be equal to warp_idx %d while we don't yet specify thread block shape larger than warp shape!\n",
          warp_idx_k, warp_idx);
      }
    }
    assert(assert_pass);
#endif

    //
    // for gemm input B size (k,n), packed b is (k/2,n/2), element size 2, column major.
    // so lead dimension byte size is coincidentally k/2 * 2 = k
    // and next dimension size is n/2
    //
    const int n_start = mul_power2<WarpShape::kN>(blockIdx.y);   // TODO! change to thread block shape
    const int n_end = min(params.problem_size_.n(), mul_power2<WarpShape::kN>(blockIdx.y + 1));
  
    const int k_start = warp_idx * params.gemm_k_size_;
    const int k_end = min(params.problem_size_.k(), (warp_idx + 1) * params.gemm_k_size_);

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

    ATileLoader a_tile_loader{
      params.ptr_a_,
      mul_power2<kElementSize>(k_start), mul_power2<kElementSize>(k_end), // convert to byte based index
      lane_idx};

    //
    // Prologue: start loading from global memory to shared memory
    //

    int load_k = k_start; // current k index for loading from global memory to shared memory
    int smem_write_stage = 0;
    uint8_t* packed_b_shared_ptr = packed_b_loader.get_smem_lane_ptr(shared_storage.smem[warp_idx].main_loop.shared_B.data(), lane_idx);
    ElementT* a_shared_ptr = shared_storage.smem[warp_idx].main_loop.shared_A.data();
    ElementT* scales_shared_ptr = shared_storage.smem[warp_idx].main_loop.shared_Scale.data();

    if constexpr (kDebugPrintSteps) {
      if (lane_idx == 0) {
        printf("Warp: %d, n_start %d, n_end %d, k_start %d, k_end %d, PackedB: %p, A: %p, Scales: %p\n",
          warp_idx, n_start, n_end, k_start, k_end, packed_b_shared_ptr, a_shared_ptr, scales_shared_ptr);
      }
    }

    uint8_t* packed_b_smem_write_ptr = packed_b_shared_ptr;
    ElementT* a_smem_write_ptr = a_shared_ptr;
    ElementT* scales_smem_write_ptr = scales_shared_ptr;

    CUTLASS_PRAGMA_UNROLL
    for (; smem_write_stage < kStages - 1; ++smem_write_stage, load_k += WarpShape::kK) {
      meta_loader.load_to_smem(lane_idx, load_k, min(k_end - load_k, WarpShape::kK), scales_smem_write_ptr);
      scales_smem_write_ptr += MainLoopSharedBuffer::kMetaSizePerIter;

      // Load packed b
      packed_b_loader.load_to_smem(packed_b_smem_write_ptr);
      packed_b_smem_write_ptr += MainLoopSharedBuffer::kPackedBSizePerIter;
      ++packed_b_loader;

      // Load A
      a_tile_loader.load_to_smem(lane_idx, a_smem_write_ptr);
      a_smem_write_ptr += MainLoopSharedBuffer::kASizePerIter;
      ++a_tile_loader;

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }    

    // Prepare for the main loop, declare fragments and accumulators,
    // hopefully allocated in registers
    FragmentPackedB fragment_packed_b[2];
    typename MetaLoader::FragmentScales fragment_scales[2];
    FragmentB fragment_b;
    FragmentA fragment_a[2];
    FragmentC accumulators;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentC::kElements; ++i) {
      accumulators[i] = float2{0.0f, 0.0f};
    }
  
    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<kStages - 2>();
    //__syncthreads(); is this necessary since the loader is warp based?
    if constexpr(kDebugPrintA) {
      if (lane_idx == 0) {
        printf("Prologue, warp: %d, WarpPtr: %p\n",
          warp_idx, a_shared_ptr);
        printf("\n********Dumping the shared memory of Warp %d*******\n\n", warp_idx);

        for (int i = 0; i < MainLoopSharedBuffer::kASize; i += 8) {
          for (int j = 0; j < 8; ++j) {
            printf("%f, ", float(a_shared_ptr[i + j]));
          }
          printf("\n");
        }
      }
    }

    //
    // Prefix of the Mainloop, pre-loading the double buffer in registers
    //
    uint8_t const* packed_b_smem_read_ptr = packed_b_shared_ptr;
    ElementT const* a_smem_read_ptr = a_shared_ptr;
    ElementT const* scales_smem_read_ptr = scales_shared_ptr;

    if constexpr (kDebugPrintSteps) {
      if (lane_idx == 0) {
        printf("Prefix: PackedB[%d] <- %p <- %p,  A[%d] <- %p <- %p,  fragment_scales[%d] <- load_k %d <- %p <- %p\n",
          0, packed_b_smem_read_ptr, packed_b_smem_write_ptr, 0, a_smem_read_ptr, a_smem_write_ptr, 0, load_k, scales_smem_read_ptr, scales_smem_write_ptr);
      }
    }

    meta_loader.load_fragment(lane_idx, fragment_scales[0], scales_smem_read_ptr);
    meta_loader.load_to_smem(lane_idx, load_k, min(k_end - load_k, WarpShape::kK), scales_smem_write_ptr);

    // Each stage processes k = 64. For packed B, each ldmatrix is 16x64
    packed_b_loader.load_to_register(lane_idx, 0, packed_b_smem_read_ptr, fragment_packed_b[0]);
    packed_b_loader.load_to_smem(packed_b_smem_write_ptr);

    // For vector A, each ldmatrix is 8x32, the vector is replicated 8 times
    a_tile_loader.load_fragment_k64(lane_idx, a_smem_read_ptr, 0, fragment_a[0]);
    a_tile_loader.load_to_smem(lane_idx, a_smem_write_ptr);

    //
    // Main loop
    // proc_k = load_k - (kStages - 1) * WarpShape::kK
    //
    while (load_k < k_end + (kStages - 1) * WarpShape::kK){

      // One stage has kMmaIterations, we unroll the main loop by 2,
      // as the meta data is loaded only once every stage, need 2 stages
      // to complete a double buffer cycle. This is necessary to make
      // all indices compile time constants.
      CUTLASS_PRAGMA_UNROLL
      for (int iter2 = 0; iter2 < kMmaIterations * 2; ++iter2) {
        // To speedup de-quantization, instead of using f = s * (q - z),
        // we use f = s * q + (s * -z) to take advantage of the fma
        // instruction. This is the storage of (s * -z)
        typename MetaLoader::FragmentScales fragment_addon;

        // Advance to the next stage of the main loop just one step eariler
        const int next_iter2 = (iter2 + 1) % (kMmaIterations * 2);
        const int next_iter = next_iter2 % kMmaIterations;
        if (next_iter == 0) {
          cutlass::arch::cp_async_fence();

          const int read_stage_diff = (smem_write_stage == (kStages - 2)) ? (1 - kStages) : 1;
          smem_write_stage = (smem_write_stage + 1) % kStages;
          scales_smem_write_ptr = const_cast<ElementT*>(scales_smem_read_ptr);
          packed_b_smem_write_ptr = const_cast<uint8_t*>(packed_b_smem_read_ptr);
          a_smem_write_ptr = const_cast<ElementT*>(a_smem_read_ptr);
          scales_smem_read_ptr += read_stage_diff * MainLoopSharedBuffer::kMetaSizePerIter;
          packed_b_smem_read_ptr += read_stage_diff * MainLoopSharedBuffer::kPackedBSizePerIter;
          a_smem_read_ptr += read_stage_diff * MainLoopSharedBuffer::kASizePerIter;
          ++packed_b_loader;
          ++a_tile_loader;

          cutlass::arch::cp_async_wait<kStages - 2>();
          //__syncthreads(); is this necessary since the loader is warp based?

          load_k += WarpShape::kK;
          if constexpr (kDebugPrintSteps) {
            if (lane_idx == 0) {
              printf("fragment_scales[%d] <- load_k %d <- %p <- %p\n", (next_iter2 / kMmaIterations) % 2, load_k, scales_smem_read_ptr, scales_smem_write_ptr);
              printf("PackedB[%d] <- %p <- %p\n", (next_iter2 / kMmaIterations) % 2, packed_b_smem_read_ptr, packed_b_smem_write_ptr);
            }
          }
          meta_loader.load_to_smem(lane_idx, load_k, min(k_end - load_k, WarpShape::kK), scales_smem_write_ptr);
          meta_loader.load_fragment(lane_idx, fragment_scales[(next_iter2 / kMmaIterations) % 2], scales_smem_read_ptr);

          if constexpr(kDebugPrintB) {
            if (lane_idx == 0) {
              printf("Mainloop, warp: %d, proc_k %d, load_k %d\nWritePtr: %p, ReadPtr: %p\n",
                warp_idx, load_k - (kStages - 1) * WarpShape::kK, load_k, packed_b_smem_write_ptr, packed_b_smem_read_ptr);
            }
            cutlass::debug::dump_shmem(packed_b_shared_ptr, MainLoopSharedBuffer::kPackedBSize);
          }
          packed_b_loader.load_to_register(lane_idx, 0, packed_b_smem_read_ptr, fragment_packed_b[(next_iter2 / kMmaIterations) % 2]);
          packed_b_loader.load_to_smem(packed_b_smem_write_ptr);
          a_tile_loader.load_to_smem(lane_idx, a_smem_write_ptr);
        }

        if ((iter2 % kMmaIterations)== 0) {
          meta_loader.process(fragment_scales[(iter2 / kMmaIterations) % 2], fragment_addon);
        }

        if ((next_iter2 % 2) == 0) {
          if constexpr (kDebugPrintSteps) {
            if (lane_idx == 0) {
              printf("A[%d] <- %p <- %p\n",  (next_iter2 / 2) % 2, a_smem_read_ptr, a_smem_write_ptr);
            }
          }
          a_tile_loader.load_fragment_k64(lane_idx, a_smem_read_ptr, next_iter * 16 * kElementSize, fragment_a[(next_iter2 / 2) % 2]);
        }

        // Dequantize weights block (16, WarpShape::kN)
        if constexpr (kDebugPrintSteps) {
          if (lane_idx == 0) {
            printf("Mma(PackedB[%d], fragment_scales[%d], A[%d])\n", (iter2 / kMmaIterations) % 2, (iter2 / kMmaIterations) % 2, iter2 % 2);
          }
        }
        FragmentB_flat* fragb_ptr = reinterpret_cast<FragmentB_flat*>(fragment_b.data());
        meta_loader.dequant_k16(iter2 % kMmaIterations,
                                fragment_packed_b[(iter2 / kMmaIterations) % 2],
                                fragment_scales[(iter2 / kMmaIterations) % 2],
                                fragment_addon, fragb_ptr[0]);

        // GEMM operation, covering a shape of (8, 16, 16)
        static_assert(sizeof(cutlass::Array<ElementT2, 2>) * 2 == sizeof(FragmentA));
        const auto* half_a = reinterpret_cast<cutlass::Array<ElementT2, 2> const*>(&fragment_a[(iter2 / 2) % 2]);
        if constexpr (kDebugPrintA) {
          const int lane_id = threadIdx.x % 32;
          if (lane_id == 0) {
            printf("====  A tiles =======\n");
          }
          const char* const format = (lane_id == 31) ? "%f, %f\n\n" : ((lane_id % 4) == 3) ? "%f, %f\n" : "%f, %f, ";
            printf(format, float(half_a[iter2 % 2][0].x), float(half_a[iter2 % 2][0].y));
            printf(format, float(half_a[iter2 % 2][1].x), float(half_a[iter2 % 2][1].y));
          if (lane_id == 0) {
            printf("====  B tiles =======\n");
          }
          printf(format, float(fragment_b[0].x), float(fragment_b[0].y));
          printf(format, float(fragment_b[1].x), float(fragment_b[1].y));
          printf(format, float(fragment_b[2].x), float(fragment_b[2].y));
          printf(format, float(fragment_b[3].x), float(fragment_b[3].y));
        }

        mma_op(half_a[iter2 % 2], fragment_b, accumulators);
      }  // next k block (stride = 16)
    }  // Main loop: next stage

    // Partial k reduction, within a thread
    __b64 v;
    v.f64.x = accumulators[0].x + accumulators[0].y + accumulators[1].x + accumulators[1].y;
    v.f64.y = accumulators[2].x + accumulators[2].y + accumulators[3].x + accumulators[3].y;
    __b64 other;
    other.u64 = __shfl_down_sync(0xffffffff, v.u64, 2);
    v.f64.x += other.f64.x;
    v.f64.y += other.f64.y;
    other.u64 = __shfl_down_sync(0xffffffff, v.u64, 1);
    v.f64.x += other.f64.x;
    v.f64.y += other.f64.y;

    cutlass::arch::cp_async_wait<0>();
    // ========================== Finish the main loop ==========================
    // !!!!! SHOULD NOT ACCESS main_loop SHARED MEMORY AFTER THIS POINT !!!!!
  
    // Finished the main loop, now each warp stores the partial results
    // to shared memory. Later warp 0 should gather them to form the final result.
    float* d_smem_ptr = shared_storage.smem[warp_idx].shared_Acc.data();
    if (lane_idx % 4 == 0) {
      d_smem_ptr[lane_idx / 4] = v.f64.x;
      d_smem_ptr[lane_idx / 4 + 8] = v.f64.y;
    }

    if constexpr (kWarps > 1) {
      __syncthreads();
    }

    if (warp_idx != 0) {
      return;
    }

    //
    // Only warp 0 gathers the result from all other warps and stores it to global memory
    // Be extra careful with synchronization code below, as only a subset of threads
    // are active!

    int n = n_start + lane_idx;
    ElementT* output_ptr = reinterpret_cast<ElementT*>(params.ptr_output_);
    float val{0.0f};
    if (n < n_end) {
      CUTLASS_PRAGMA_UNROLL
      for (int warp = 0; warp < kWarps; ++warp){
        const auto* smem_ptr = shared_storage.smem[warp].shared_Acc.data();
        val += smem_ptr[lane_idx];
      }
      *(output_ptr + n) = ElementT(val);
    }
  }

};


}  // namespace kernel
}  // namespace gemm
}  // namespace mickey
