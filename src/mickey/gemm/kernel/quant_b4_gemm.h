/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file kernel/quant_b4_gemm.h
 * @brief Fused GEMM kernel for fp16 x int4, where B matrix is blockwise quantized to 4bits.
 *
 **************************************************************************************************/

#pragma once

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
#include "gemm/warp/swizzle_tile_loader.h"
#include "gemm/warp/quantb_meta_loader.h"
#include "int_util.h"

namespace mickey {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Fused GEMM kernel for fp16 x int4, where B matrix is blockwise quantized to 4bits.
 */
template <
  typename QuantBlocking_,     ///! Shape of the quantization block, either 1xb or bx1
  bool     has_quant_offset_,  ///! Whether the quantization has offset
  typename ThreadblockShape_,  ///! Warp-scoped matrix multiply-accumulate
  int SplitKSerial_ = 1,       ///! How many warps to split the K dimension in the same MxN block
  int Stages_ = 4              ///! Stages of the pipelined mainloop
>
struct QuantB4Gemm {
 public:
  //
  // Type definitions
  //

  using QuantBlocking = QuantBlocking_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = cutlass::gemm::GemmShape<ThreadblockShape::kM, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementT = cutlass::half_t;
  static constexpr bool has_quant_offset = has_quant_offset_;
  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kStages = Stages_;
  static constexpr int kElementSize = 2;

  //
  // Type constraints verifications:
  //
  static_assert(kStages > 1, "Number of pipeline stages must be greater than 1.");
  static_assert(kElementSize == sizeof(ElementT), "Only support 16b float types.");
  static_assert(ThreadblockShape::kN % WarpShape::kN == 0);
  static_assert(ThreadblockShape::kK % WarpShape::kK == 0);
  static_assert(ThreadblockShape::kK == 64);  // keeping global memory stride at 128B
  static_assert((WarpShape::kM % InstructionShape::kM == 0)
                 && (WarpShape::kN % InstructionShape::kN == 0)
                 && (WarpShape::kK % InstructionShape::kK == 0));

  /// switches for debug print
  static constexpr bool kDebugPrintB = false;
  static constexpr bool kDebugPrintA = false;
  static constexpr bool kDebugPrintC = false;
  static constexpr bool kDebugPrintSteps = false;

  // We partition warps based on B tiles. Each warp processes a
  // a sub block of B, gmem -> smem -> register -> dequant
  // All kNWarps need to process all of A tiles on the corresponding k offset
  // Accumulators in the same N warps are added together via shared
  // memory after the main loop.
  static constexpr int kNWarps = ThreadblockShape::kN / WarpShape::kN;
  static constexpr int kKWarps = ThreadblockShape::kK / WarpShape::kK;
  static constexpr int kWarps = kNWarps * kKWarps;
  static constexpr int kThreads = 32 * kWarps;
  static constexpr int kMmaIterations = ThreadblockShape::kK / InstructionShape::kK;
  static constexpr int kWarpMmaIterations = kMmaIterations / kKWarps;
  static_assert(kWarpMmaIterations == 1 || kWarpMmaIterations == 2 || kWarpMmaIterations == 4);
  
  using ATileLoader = mickey::gemm::warp::SwizzleTileLoader<
          ThreadblockShape::kM, ThreadblockShape::kK * kElementSize, kThreads>;
  using MetaLoader = mickey::gemm::warp::QuantBScaleLoader<QuantBlocking, WarpShape, ElementT, false>;
  using PackedBLoader = mickey::gemm::warp::TensorCoreTileLoader<WarpShape, ThreadblockShape::kK>;

  struct MmaSharedStorage {
    /// Buffer for A tensor
    static constexpr int kASizePerIter = ATileLoader::kBlockSize / kElementSize;
    static constexpr int kASize = kASizePerIter * kStages;
    cutlass::AlignedBuffer<ElementT, kASize> shared_A;

    /// Buffer for prepacked weights
    static constexpr int kPackedBSizePerIter = PackedBLoader::kByteSize;
    static constexpr int kPackedBSize = kPackedBSizePerIter * kStages;
    cutlass::AlignedBuffer<uint8_t, kPackedBSize> shared_B[kWarps];

    /// Buffer for quantization meta data
    static constexpr int kMetaSizePerIter = MetaLoader::kSmemSize;
    static constexpr int kMetaSize = kMetaSizePerIter * kStages;
    cutlass::AlignedBuffer<ElementT, kMetaSize> shared_Scale[kWarps];
  };

  union SharedStorage {
    MmaSharedStorage main_loop;

    /// End of mma loop, warps from different k group need
    /// to add their accumulators together
    cutlass::AlignedBuffer<float, ThreadblockShape::kM * ThreadblockShape::kN> shared_Acc[kKWarps];
  };

  // Fragments for operand A and dequantized B
  // In each main loop iteration, we use a (WarpShape::kM, 16) block
  // of A and (16, WarpShape::kN) block of B for mma. 
  // For B, that is (WarpShape::kN / 8) * 2 tiles, each tile has 2
  // elements per thread.
  using FragmentB = cutlass::Array<ElementT, (WarpShape::kN / 8) * 4>;
  using FragmentA = cutlass::Array<ElementT, (WarpShape::kM / 8) * 4>;

  //
  // The way we use the cutlass MmaTensorOp class below is confusing, because:
  //
  // MmaTensorOp from cutlass is really convoluted. It iterates over the m,n
  // dimension to run mma instructions the following number of times:
  // (WarpShape::kM / InstructionShape::kM) * (WarpShape::kN / InstructionShape::kN).
  // So, the operation always cover a shape of
  // (WarpShape::kM, WarpShape::kN, InstructionShape::kK).
  // Unfortunately, it does not reach that conclusion in a straight forward
  // way. Instead, it asks you to provide a shared memory layout for both A
  // and B, and construct shared memory tile iterators based on these layout.
  // The solo purpose of these iterators is to compute the k dimension size.
  // And they don't access shared memory at all. What's worse, the layout
  // must be a certain swizzled shape, for it to compute the current k, or
  // else the operation can not be used. This is a serious abstraction leak
  // that makes this class difficult to use.
  //
  using MmaPolicy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<InstructionShape, 32, ElementT,
                         cutlass::layout::RowMajor, ElementT,
                         cutlass::layout::ColumnMajor, float,
                         cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1> >;

  using MmaOp = cutlass::gemm::warp::MmaTensorOp<
      cutlass::gemm::GemmShape<WarpShape::kM, WarpShape::kN, InstructionShape::kK>, ElementT, cutlass::layout::RowMajor, ElementT,
      cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor,
      MmaPolicy>;

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
      gemm_k_size_(mickey::round_up(mickey::div_up(problem_size.k(), kSplitK), ThreadblockShape::kK)),
      // TODO! grid_tiled_shape_ should be based on thread block shape
      grid_tiled_shape_(cutlass::gemm::GemmCoord(
        mickey::div_up(problem_size.m(), ThreadblockShape::kM),
        mickey::div_up(problem_size.n(), ThreadblockShape::kN),
        kSplitK)) { }
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  QuantB4Gemm() { }

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
    if (reinterpret_cast<uintptr_t>(params.ptr_packed_b_) % 128 != 0 || params.b_byte_stride_ % 128 != 0) {
      std::cerr << "QuantB4Gemm validation fail: params.ptr_packed_b_ " << params.ptr_packed_b_
                << " is not aligned to 128 bytes, with stride " << params.b_byte_stride_ << "!" << std::endl;
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
      // TODO! Use thread block shape
      if (params.gemm_k_size_ < ThreadblockShape::kK * kStages + 2) {
        // spliting too small, may not get enough iterations to rampup pipeline
        std::cerr << "QuantB4Gemm validation fail: kSplitK is too small, k: " << params.gemm_k_size_ << " is smaller than " << (ThreadblockShape::kK * kStages + 2) << std::endl;
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
      params.grid_tiled_shape_.k() <= blockIdx.z) {
      // should not happen
      if (threadIdx.x == 0) {
        printf("CTA out of range %d, %d, %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
      }
      return;
    }

    //
    // Initialization phase: locating our position
    //
    const int warp_idx = div_power2<32>(threadIdx.x);
    const int lane_idx = mod_power2<32>(threadIdx.x);
    const int warp_n_idx = div_power2<kKWarps>(warp_idx);
    const int warp_k_idx = mod_power2<kKWarps>(warp_idx);

#ifndef NDEBUG
    bool assert_pass = true;
    if (warp_idx >= kWarps) {
      assert_pass = false;
      if (lane_idx == 0) {
        printf("warp_idx %d exceeds kWarps %d! Should use %d threads per threadblock for kernel launch!\n",
          warp_idx, kWarps, kThreads);
      }
    }
    assert(assert_pass);
#endif

    const int m_start = blockIdx.x * ThreadblockShape::kM;
    const int m_end = min(params.problem_size_.m(), (blockIdx.x + 1) * ThreadblockShape::kM);
    const int n_start = mul_power2<ThreadblockShape::kN>(blockIdx.y) + warp_n_idx * WarpShape::kN;
    const int n_end = min(params.problem_size_.n(), n_start + WarpShape::kN);  
    const int k_start = blockIdx.z * params.gemm_k_size_;
    const int warp_k_offset = warp_k_idx * WarpShape::kK;
    const int k_end = min(params.problem_size_.k(), (blockIdx.z + 1) * params.gemm_k_size_);

    PackedBLoader packed_b_loader{
      params.ptr_packed_b_,
      params.b_byte_stride_,
      n_start,
      n_end,
      k_start + warp_k_offset,
      k_end,
      lane_idx};

    MetaLoader meta_loader{
      lane_idx,
      params.ptr_scales_,
      params.scales_byte_stride_,
      n_start, n_end};

    ATileLoader a_tile_loader{
      params.ptr_a_,
      params.a_byte_stride_,
      m_start, m_end,
      mul_power2<kElementSize>(k_start), mul_power2<kElementSize>(k_end), // convert to byte based index
      threadIdx.x};

    //
    // Prologue: start loading from global memory to shared memory
    //

    int load_k = k_start; // current k index for loading from global memory to shared memory
    int smem_write_stage = 0;
    uint8_t* packed_b_shared_ptr = packed_b_loader.get_smem_lane_ptr(shared_storage.main_loop.shared_B[warp_idx].data(), lane_idx);
    ElementT* a_shared_ptr = shared_storage.main_loop.shared_A.data();
    ElementT* scales_shared_ptr = shared_storage.main_loop.shared_Scale[warp_idx].data();

    if constexpr (kDebugPrintSteps) {
      if (lane_idx == 0) {
        printf("Warp: %d, m_start %d, m_end %d, n_start %d, n_end %d, k_start %d, k_end %d\n    PackedB: %p, A: %p, Scales: %p\n",
          warp_idx, m_start, m_end, n_start, n_end, k_start, k_end, packed_b_shared_ptr, a_shared_ptr, scales_shared_ptr);
      }
    }

    uint8_t* packed_b_smem_write_ptr = packed_b_shared_ptr;
    ElementT* a_smem_write_ptr = a_shared_ptr;
    ElementT* scales_smem_write_ptr = scales_shared_ptr;

    CUTLASS_PRAGMA_UNROLL
    for (; smem_write_stage < kStages - 1; ++smem_write_stage, load_k += ThreadblockShape::kK) {
      const int warp_load_k = load_k + warp_k_offset;
      meta_loader.load_to_smem(lane_idx, warp_load_k, min(k_end - warp_load_k, WarpShape::kK), scales_smem_write_ptr);
      scales_smem_write_ptr += MmaSharedStorage::kMetaSizePerIter;

      // Load packed b
      packed_b_loader.load_to_smem(packed_b_smem_write_ptr);
      packed_b_smem_write_ptr += MmaSharedStorage::kPackedBSizePerIter;
      ++packed_b_loader;

      // Load A
      a_tile_loader.load_to_smem(threadIdx.x, a_smem_write_ptr);
      a_smem_write_ptr += MmaSharedStorage::kASizePerIter;
      ++a_tile_loader;

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }    

    // Prepare for the main loop, declare fragments and accumulators,
    // hopefully allocated in registers
    typename PackedBLoader::Fragment fragment_packed_b[2];
    typename MetaLoader::FragmentScales fragment_scales[2];
    FragmentB fragment_b;
    FragmentA fragment_a[2];
    typename MmaOp::FragmentC accumulators;
    accumulators.clear();
  
    MmaOp mma_op;

    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<kStages - 2>();
    __syncthreads();
    if constexpr(kDebugPrintA) {
      if (threadIdx.x == 0) {
        printf("\n****** Shared memory of A %p ******\n", a_shared_ptr);
        for (int i = 0; i < MmaSharedStorage::kASize; i += 64) {
          for (int j = 0; j < 64; ++j) {
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
    meta_loader.load_to_smem(lane_idx, load_k + warp_k_offset, min(k_end - load_k - warp_k_offset, WarpShape::kK), scales_smem_write_ptr);
    packed_b_loader.load_to_register(lane_idx, 0, packed_b_smem_read_ptr, fragment_packed_b[0]);

    a_tile_loader.load_fragment_k32(lane_idx, a_smem_read_ptr, warp_k_offset * kElementSize, fragment_a[0].data());
    a_tile_loader.load_to_smem(threadIdx.x, a_smem_write_ptr);

    //
    // Main loop
    // proc_k = load_k - (kStages - 1) * ThreadblockShape::kK
    //
    while (load_k < k_end + (kStages - 1) * ThreadblockShape::kK){

      // One stage has kMmaIterations, we unroll the main loop by 2,
      // as the meta data is loaded only once every stage, need 2 stages
      // to complete a double buffer cycle. This is necessary to make
      // all indices compile time constants.
      CUTLASS_PRAGMA_UNROLL
      for (int iter2 = 0; iter2 < kWarpMmaIterations * 2; ++iter2) {
        const int iter = iter2 % kWarpMmaIterations;
        const int next_iter2 = (iter2 + 1) % (kWarpMmaIterations * 2);
        const int next_iter = next_iter2 % kWarpMmaIterations;

        // To speedup de-quantization, instead of using f = s * (q - z),
        // we use f = s * q + (s * -z) to take advantage of the fma
        // instruction. This is the storage of (s * -z)
        typename MetaLoader::FragmentScales fragment_addon;

        if (iter == 0) {
          packed_b_loader.load_to_smem(packed_b_smem_write_ptr);
        }

        if (next_iter == 0) {
          cutlass::arch::cp_async_fence();  // Advance to the next stage

          const int read_stage_diff = (smem_write_stage == (kStages - 2)) ? (1 - kStages) : 1;
          smem_write_stage = (smem_write_stage + 1) % kStages;
          scales_smem_write_ptr = const_cast<ElementT*>(scales_smem_read_ptr);
          packed_b_smem_write_ptr = const_cast<uint8_t*>(packed_b_smem_read_ptr);
          a_smem_write_ptr = const_cast<ElementT*>(a_smem_read_ptr);
          scales_smem_read_ptr += read_stage_diff * MmaSharedStorage::kMetaSizePerIter;
          packed_b_smem_read_ptr += read_stage_diff * MmaSharedStorage::kPackedBSizePerIter;
          a_smem_read_ptr += read_stage_diff * MmaSharedStorage::kASizePerIter;
          ++packed_b_loader;
          ++a_tile_loader;

          cutlass::arch::cp_async_wait<kStages - 2>();
          __syncthreads();

          load_k += ThreadblockShape::kK;
          if constexpr (kDebugPrintSteps) {
            if (lane_idx == 0) {
              printf("fragment_scales[%d] <- load_k %d <- %p <- %p\n", (next_iter2 / kWarpMmaIterations) % 2, load_k, scales_smem_read_ptr, scales_smem_write_ptr);
            }
          }
          meta_loader.load_to_smem(lane_idx, load_k + warp_k_offset, min(k_end - load_k - warp_k_offset, WarpShape::kK), scales_smem_write_ptr);
          meta_loader.load_fragment(lane_idx, fragment_scales[(next_iter2 / kWarpMmaIterations) % 2], scales_smem_read_ptr);
          a_tile_loader.load_to_smem(threadIdx.x, a_smem_write_ptr);

          if constexpr(kDebugPrintB) {
            if (lane_idx == 0) {
              printf("Mainloop, warp: %d, proc_k %d, load_k %d\nWritePtr: %p, ReadPtr: %p\n",
                warp_idx, load_k - (kStages - 1) * ThreadblockShape::kK, load_k, packed_b_smem_write_ptr, packed_b_smem_read_ptr);
            }
            cutlass::debug::dump_shmem(packed_b_shared_ptr, MmaSharedStorage::kPackedBSize);
          }
        }

        if (iter == 0) {
          meta_loader.process(fragment_scales[(iter2 / kWarpMmaIterations) % 2], fragment_addon);
        }

        if constexpr (kDebugPrintSteps) {
          if (lane_idx == 0) {
            printf("PackedB[%d] <- %p <- %p\n", (iter2 + 1) % 2, packed_b_smem_read_ptr, packed_b_smem_write_ptr);
          }
        }
        packed_b_loader.load_to_register(lane_idx, next_iter, packed_b_smem_read_ptr, fragment_packed_b[(iter2 + 1) % 2]);

        if constexpr (kDebugPrintSteps) {
          if (lane_idx == 0) {
            printf("A[%d] <- %p <- %p\n",  (iter2 + 1) % 2, a_smem_read_ptr, a_smem_write_ptr);
          }
        }
        a_tile_loader.load_fragment_k32(lane_idx, a_smem_read_ptr,
                                        (warp_k_offset + next_iter * InstructionShape::kK) * kElementSize,
                                        fragment_a[(iter2 + 1) % 2].data());

        if constexpr (kDebugPrintA) {
          const int lane_id = threadIdx.x % 32;
          if (lane_id == 0) {
            printf("====  A tiles =======\n");
          }
          const char* const format = (lane_id == 31) ? "%f, %f\n\n" : ((lane_id % 4) == 3) ? "%f, %f\n" : "%f, %f, ";
          const ElementT* a_ptr = fragment_a[iter2 % 2].data();
          for (int m2_tile = 0; m2_tile < (ThreadblockShape::kM / InstructionShape::kM); ++m2_tile, a_ptr += 8) {
            printf(format, float(a_ptr[0]), float(a_ptr[1]));
            printf(format, float(a_ptr[2]), float(a_ptr[3]));
            printf(format, float(a_ptr[4]), float(a_ptr[5]));
            printf(format, float(a_ptr[6]), float(a_ptr[7]));
          }
        }

        // Dequantize weights block (16, ThreadblockShape::kN)
        if constexpr (kDebugPrintSteps) {
          if (lane_idx == 0) {
            printf("Mma(PackedB[%d], fragment_scales[%d], A[%d])\n", (iter2 / kWarpMmaIterations) % 2, (iter2 / kWarpMmaIterations) % 2, iter2 % 2);
          }
        }
        meta_loader.dequant_k16(iter, fragment_packed_b[iter2 % 2], fragment_scales[(iter2 / kWarpMmaIterations) % 2], fragment_addon, fragment_b);

        // GEMM operation, covering a shape of (ThreadblockShape::kM, ThreadblockShape::kN, InstructionShape::kK)
        mma_op(accumulators, fragment_a[iter2 % 2], fragment_b, accumulators);
      }  // next k block (stride = 16)
    }  // Main loop: next stage

    if constexpr (kDebugPrintC) {
      static_assert(MmaOp::FragmentC::kElements == (WarpShape::kN / InstructionShape::kN) * (WarpShape::kM / InstructionShape::kM) * 4);
      for (int warp = 0; warp < kWarps; ++warp) {
        if (warp_idx == warp) {
          const float* c_ptr = accumulators.data();
          const int lane_id = threadIdx.x % 32;
          if (lane_id == 0) {
            printf("======= C tiles in warp %d =======\n", warp_idx);
          }
          const char* const format = (lane_id == 31) ? "%f, %f\n\n" : ((lane_id % 4) == 3) ? "%f, %f\n" : "%f, %f, ";
          for (int n_tile = 0; n_tile < (ThreadblockShape::kN / InstructionShape::kN); ++n_tile) {
            for (int m_tile = 0; m_tile < (ThreadblockShape::kM / InstructionShape::kM); ++m_tile, c_ptr += 4) {
              // since InstructionShape::kM is 16, we can print 2 tiles
              printf(format, float(c_ptr[0]), float(c_ptr[1]));
              printf(format, float(c_ptr[2]), float(c_ptr[3]));
            }
          }
        }
        __syncthreads();
      }
    }

    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    // ========================== Finish the main loop ==========================
    // !!!!! SHOULD NOT ACCESS main_loop SHARED MEMORY AFTER THIS POINT !!!!!
  
    // Store partial result to shared memory
    float2* const pacc_smem_ptr = reinterpret_cast<float2*>(shared_storage.shared_Acc[warp_k_idx].data());
    CUTLASS_PRAGMA_UNROLL
    for (int m_tile = 0; m_tile < (WarpShape::kM / 8); ++m_tile) {
      const int m = div_power2<4>(lane_idx) + m_tile * 8;  // assuming no m split among warps
      CUTLASS_PRAGMA_UNROLL
      for (int n_tile = 0; n_tile < (WarpShape::kN / 8); ++n_tile) {
        const int n = warp_n_idx * WarpShape::kN + (mod_power2<4>(lane_idx) << 1) + n_tile * 8;
        const float2* c_ptr = reinterpret_cast<float2 const*>(accumulators.data()) + m_tile + n_tile * (WarpShape::kM / 8);
        *(pacc_smem_ptr + m * (ThreadblockShape::kN / 2) + n/2) = c_ptr[0];
      }
    }

    if constexpr (kKWarps > 1) {
      __syncthreads();
    }

    // ========================== Warp reduction ==========================
    // Loading TB::kM x TB::kN row major from shared memory.
    // Each smem load 16 bytes, i.e. 4 floats. Each warp loads 4 x 32 = 128
    // floats.
    static_assert(ThreadblockShape::kN >= (4 * 32)); // make math simpler
    constexpr int kAccLoadsN = ThreadblockShape::kN / (4 * 32);
    static_assert(ThreadblockShape::kM % kWarps == 0);
    constexpr int kAccLoadsM = ThreadblockShape::kM / kWarps;

    cutlass::Array<float2, 2> other_acc[kAccLoadsM][kAccLoadsN];
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < kAccLoadsM; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kAccLoadsN; ++n) {
        const int row_idx = m * kWarps + warp_idx;
        const int col_idx = n * (4 * 32) + lane_idx * 4;
        const int offset = row_idx * ThreadblockShape::kN + col_idx;
        for (int k = 0; k < kKWarps; ++k) {
          if (k == 0) {
            other_acc[m][n][0].x = shared_storage.shared_Acc[k].data()[offset + 0];
            other_acc[m][n][0].y = shared_storage.shared_Acc[k].data()[offset + 1];
            other_acc[m][n][1].x = shared_storage.shared_Acc[k].data()[offset + 2];
            other_acc[m][n][1].y = shared_storage.shared_Acc[k].data()[offset + 3];
          } else {
            other_acc[m][n][0].x += shared_storage.shared_Acc[k].data()[offset + 0];
            other_acc[m][n][0].y += shared_storage.shared_Acc[k].data()[offset + 1];
            other_acc[m][n][1].x += shared_storage.shared_Acc[k].data()[offset + 2];
            other_acc[m][n][1].y += shared_storage.shared_Acc[k].data()[offset + 3];
          }
        }
      }
    }

    // Store the thread block result to global memory
    using half4 = cutlass::Array<__half2, 2>;
    auto* output_ptr = reinterpret_cast<ElementT*>(params.ptr_output_);
    int output_stride = params.output_byte_stride_ / sizeof(ElementT);
    const int tb_m_start = blockIdx.x * ThreadblockShape::kM;
    const int tb_m_end = min(params.problem_size_.m(), mul_power2<ThreadblockShape::kM>(blockIdx.x + 1));
    const int tb_n_start = mul_power2<ThreadblockShape::kN>(blockIdx.y);
    const int tb_n_end = min(params.problem_size_.n(), mul_power2<ThreadblockShape::kN>(blockIdx.y + 1));  

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < kAccLoadsM; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kAccLoadsN; ++n) {
        const int row_idx = tb_m_start + m * kWarps + warp_idx;
        const int col_idx = tb_n_start + n * (4 * 32) + lane_idx * 4;
        if (row_idx < tb_m_end && col_idx < tb_n_end) {
          // printf("%2d, %2d, (%2d, %2d)\n", warp_idx, lane_idx, row_idx, col_idx);
          half4 tmp;
          tmp[0] = __float22half2_rn(other_acc[m][n][0]);
          tmp[1] = __float22half2_rn(other_acc[m][n][1]);
          half4* dst_ptr = reinterpret_cast<half4*>(output_ptr + row_idx * output_stride + col_idx);
          *dst_ptr = tmp;
        }
      }
    }
  }
};


}  // namespace kernel
}  // namespace gemm
}  // namespace mickey
