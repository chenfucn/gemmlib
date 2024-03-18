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

#include "gemm/warp/tensor_core_tile_loader.h"
#include "gemm/warp/quantb_meta_loader.h"

#include "int_util.h"

namespace mickey {
namespace gemm {
namespace kernel {

template<typename ElementT, typename Shape, int Stages>
struct MmaLoopSharedBuffer{
  // Quantized weights are packed int4, each 16x16 tile of int4
  // is packed into 8x8 tile of 16b (i.e. 8x16 tile of bytes)
  using PackedBShape = cutlass::MatrixShape<Shape::kK, Shape::kN/2>;
  static_assert(sizeof(ElementT) == 2, "Only support 16b float types.");

  /// Buffer for prepacked weights
  static constexpr int kPackedBSizePerIter = PackedBShape::kCount;
  static constexpr int kPackedBSize = kPackedBSizePerIter * Stages;
  cutlass::AlignedBuffer<uint8_t, kPackedBSize> shared_B;

  /// Buffer for A tensor
  static constexpr int kASizePerIter = Shape::kM * Shape::kK;
  static constexpr int kASize = kASizePerIter * Stages;
  cutlass::AlignedBuffer<ElementT, kASize> shared_A;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Fused GEMM kernel for fp16 x int4, where B matrix is blockwise quantized to 4bits.
 */
template <
  typename QuantBlocking_,     ///! Shape of the quantization block, either 1xb or bx1
  bool     has_quant_offset_,  ///! Whether the quantization has offset
  typename WarpShape_,         ///! Warp-scoped matrix multiply-accumulate
  int SplitKSerial_ = 1,       ///! How many warps to split the K dimension in the same MxN block
  int Stages_ = 4              ///! Stages of the pipelined mainloop
>
struct QuantB4Gemm {
 public:
  //
  // Type definitions
  //

  using QuantBlocking = QuantBlocking_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ElementT = cutlass::half_t;
  static constexpr bool has_quant_offset = has_quant_offset_;
  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kStages = Stages_;
  static constexpr int kElementSize = sizeof(ElementT);

  //
  // Type constraints verifications:
  //
  static_assert(kSplitK > 0 && ((kSplitK - 1) & kSplitK) == 0,
     "SplitK must be positive and a power of 2");
  static_assert(kStages > 1, "Number of pipeline stages must be greater than 1.");
  static_assert(kElementSize == 2, "Only support 16b float types.");

  // Quantized weights are packed int4, each 16x16 tile of int4
  // is packed into 8x8 tile of 16b (i.e. 8x16 tile of bytes)
  static_assert(WarpShape::kN % 16 == 0 && WarpShape::kK % 16 == 0,
    "Weight B is packed as 16x16 tiles, warp shape must contain whole tiles!");

  // Need to explore the way to relax this for very small m value.
  static_assert(WarpShape::kM % 16 == 0,
      "Stride M smaller than mma instruction shape is not yet supported!");

  /// switches for debug print
  static constexpr bool kDebugPrintB = false;
  static constexpr bool kDebugPrintFragA = false;
  static constexpr bool kDebugPrintC = false;

  using WarpPackedBShape = cutlass::gemm::GemmShape<1, WarpShape::kN/2, WarpShape::kK>;
  static constexpr int kNTilesPerLoad = std::min(4, WarpPackedBShape::kN / 8);
  static constexpr int kKTilesPerLoad = std::min(4/kNTilesPerLoad, WarpPackedBShape::kK / 16);
  using PackedBLoader = mickey::gemm::warp::TensorCoreTileLoader<kNTilesPerLoad, kKTilesPerLoad>;

  static_assert((WarpPackedBShape::kN % PackedBLoader::kMNStride) == 0);
  static_assert((WarpPackedBShape::kK % PackedBLoader::kKStride) == 0);

  static constexpr int kB_Nloads = WarpPackedBShape::kN / PackedBLoader::kMNStride;
  static constexpr int kB_Kloads = WarpPackedBShape::kK / PackedBLoader::kKStride;

  using MetaLoader = mickey::gemm::warp::QuantBScaleLoader<QuantBlocking, WarpShape, ElementT, false>;

  // load A to shared memory, 2x2 tile to match the tensorcore shape 16x8x16
  using ATileLoader = mickey::gemm::warp::TensorCoreTileLoader<2, 2>;
  static_assert(ATileLoader::kMNStride == InstructionShape::kM);
  static_assert(ATileLoader::kKStride == InstructionShape::kK * kElementSize);
  static constexpr int kA_Mloads = WarpShape::kM / InstructionShape::kM;
  static constexpr int kA_Kloads = WarpShape::kK / InstructionShape::kK;

  static constexpr int kWarps = kSplitK; // TODO! more warps when we have a larger thread block shape
  static int const kThreadCount = 32 * kWarps;

  using MainLoopSharedBuffer = MmaLoopSharedBuffer<ElementT, WarpShape, kStages>;
  static_assert(MainLoopSharedBuffer::kPackedBSizePerIter == kB_Nloads * kB_Kloads * PackedBLoader::kByteSize);
  static_assert(MainLoopSharedBuffer::kASizePerIter == kA_Mloads * kA_Kloads * ATileLoader::kByteSize / kElementSize);

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
  // TODO!! need to reconsider this when we have a thread block shape
  // larger than warp shape.
  //
  union WarpSmemT {
    MainLoopSharedBuffer main_loop;

    /// Buffer for accumulators of the partial results after the main loop
    static constexpr int kAccSizePerWarp = WarpShape::kM * WarpShape::kN;
    cutlass::AlignedBuffer<float, kAccSizePerWarp> shared_Acc;
  };

  /// Shared memory storage structure
  struct SharedStorage {
    WarpSmemT smem[kWarps];
  };

  // Fragments of quantized weights, keep entire warp tile in registers to maximize
  // the reuse of shared scale/offsets
  using FragmentPackedB = cutlass::Array<
      unsigned,  // 8 of int4 weights each tile (becomes 4 tiles when de-quantized)
      PackedBLoader::kTiles * kB_Nloads * kB_Kloads>;

  // Fragments for operand A and dequantized B, each tile has 2 elements per thread.
  // In each main loop iteration, we use a (WarpShape::kM, 16) block of A and
  // (16, WarpShape::kN) block of B for mma
  using FragmentA = cutlass::Array<ElementT, 2 * (WarpShape::kM / 8) * 2>;
  using FragmentB = cutlass::Array<ElementT, 2 * (WarpShape::kN / 8) * 2>;

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
      gemm_k_size_(mickey::round_up(mickey::div_up(problem_size.k(), kSplitK), WarpShape::kK)),
      // TODO! grid_tiled_shape_ should be based on thread block shape
      grid_tiled_shape_(cutlass::gemm::GemmCoord(
        mickey::div_up(problem_size.m(), WarpShape::kM),
        mickey::div_up(problem_size.n(), WarpShape::kN),
        1)) { }
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
      // TODO! Use thread block shape
      int remain = params.problem_size_.k() % params.gemm_k_size_;
      if (remain > 0 && remain < WarpShape::kK * kStages * 2) {
        // spliting too small, may not get enough iterations to rampup pipeline
        std::cerr << "QuantB4Gemm validation fail: kSplitK is too small, k: " << remain << " is smaller than " << (WarpShape::kK * kStages * 4) << std::endl;
        return cutlass::Status::kErrorNotSupported;
      }
    }

    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE
  static void advance_stage(int &stage, uint8_t* &packed_b_smem_ptr, ElementT* &a_smem_ptr) {
    if (stage == kStages - 1) {
      packed_b_smem_ptr -= kStages * MainLoopSharedBuffer::kPackedBSizePerIter;
      a_smem_ptr -= kStages * MainLoopSharedBuffer::kASizePerIter;
      stage = 0;
    } else {
      ++stage;
    }
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

    // Local fragments hopefully allocated in registers
    FragmentPackedB fragment_packed_b;
    typename MetaLoader::FragmentScales fragment_scales[kStages];
    FragmentB fragment_b;
    FragmentA fragment_a;
  
    typename MmaOp::FragmentC accumulators;
    accumulators.clear();
  
    MmaOp mma_op;

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

    const int m_start = blockIdx.x * WarpShape::kM;  // TODO! change to thread block shape
    const int m_end = min(params.problem_size_.m(), (blockIdx.x + 1) * WarpShape::kM);

    PackedBLoader packed_b_loader{
      params.ptr_packed_b_,
      params.b_byte_stride_,
      packed_n_start,
      packed_n_end,
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
      params.a_byte_stride_,
      m_start, m_end,
      k_start * kElementSize, k_end * kElementSize, // need to convert to byte based index
      lane_idx};

    if constexpr (kDebugPrintB || kDebugPrintFragA || kDebugPrintC) {
      if (lane_idx == 0) {
        printf("Warp: %d, m_start %d, m_end %d, n_start %d, n_end %d, k_start %d, k_end %d, packed_n_start %d, packed_n_end %d\n",
          warp_idx, m_start, m_end, n_start, n_end, k_start, k_end, packed_n_start, packed_n_end);
      }
    }

    int load_k = k_start; // current k index for loading from global memory to shared memory
    int smem_write_stage = 0;
    uint8_t* packed_b_shared_ptr = packed_b_loader.get_smem_lane_ptr(shared_storage.smem[warp_idx].main_loop.shared_B.data());
    ElementT* a_shared_ptr = a_tile_loader.get_smem_lane_ptr(shared_storage.smem[warp_idx].main_loop.shared_A.data());

    uint8_t* packed_b_smem_write_ptr = packed_b_shared_ptr;
    ElementT* a_smem_write_ptr = a_shared_ptr;

    //
    // Prologue
    //
    CUTLASS_PRAGMA_UNROLL
    for (; smem_write_stage < kStages - 1; ++smem_write_stage, load_k += WarpShape::kK) {
      meta_loader.load(fragment_scales[smem_write_stage], load_k, min(k_end, load_k + WarpShape::kK));

      // Load packed b
      CUTLASS_PRAGMA_UNROLL
      for (int k_load = 0; k_load < kB_Kloads; ++k_load) {
        packed_b_loader.load_lateral_n<kB_Nloads>(packed_b_smem_write_ptr);
        packed_b_smem_write_ptr += PackedBLoader::kByteSize * kB_Nloads;
        ++packed_b_loader;
      }

      // Load A
      CUTLASS_PRAGMA_UNROLL
      for (int ka_load = 0; ka_load < kA_Kloads; ++ka_load) {
        a_tile_loader.load_lateral_n<kA_Mloads>(a_smem_write_ptr);
        a_smem_write_ptr += ATileLoader::kByteSize * kA_Mloads / kElementSize;
        ++a_tile_loader;
      }

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }    

    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<kStages - 2>();
    //__syncthreads(); is this necessary since the loader is warp based?
    if constexpr(kDebugPrintB) {
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

    int smem_read_stage = 0;
    uint8_t* packed_b_smem_read_ptr = packed_b_shared_ptr;
    ElementT* a_smem_read_ptr = a_shared_ptr;

    //
    // Mainloop
    //
    CUTLASS_PRAGMA_UNROLL
    for (int proc_k = k_start; proc_k < k_end; proc_k += WarpShape::kK){
      typename MetaLoader::FragmentScales fragment_addon;
  
      meta_loader.load(fragment_scales[smem_write_stage], load_k, min(k_end, load_k + WarpShape::kK));
      cutlass::Array<unsigned, PackedBLoader::kTiles>* packed_b_tile_frag_ptr =
          reinterpret_cast<cutlass::Array<unsigned, PackedBLoader::kTiles>*>(fragment_packed_b.data());

      meta_loader.process(fragment_scales[smem_read_stage], fragment_addon);

      // If PackedBLoader::kKStride > 16, then kNLoads must be 1. Because we don't want a
      // over-complicated tile visiting pattern. We always want to visit the all the
      // packed B tiles on the N dimension in a contiguous manner, and then move to the next
      // K dimension.
      static_assert(PackedBLoader::kKStride <= 16 || kB_Nloads == 1);

      // Load from shared memory to fragments/registers, and compute mma, 16 k at a time, dictated by Ampere mma shape
      CUTLASS_PRAGMA_UNROLL
      for (int warp_k_offset = 0; warp_k_offset < WarpShape::kK; warp_k_offset += InstructionShape::kK) {
        // Load packed weights. They are smaller in size, so they are loaded in bigger blocks
        if ((warp_k_offset % PackedBLoader::kKStride) == 0) {
          CUTLASS_PRAGMA_UNROLL
          for (int n_load = 0; n_load < kB_Nloads; ++n_load, ++packed_b_tile_frag_ptr) {
            PackedBLoader::ldmatrix_sync(*packed_b_tile_frag_ptr, packed_b_smem_read_ptr);
            packed_b_smem_read_ptr += PackedBLoader::kByteSize;

            if constexpr (kDebugPrintB) {
              uint8_t const* ptr = reinterpret_cast<uint8_t const*>(packed_b_tile_frag_ptr->data());
              printf("Warp: %d, lane %2d, smem_read_ptr %p, %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d\n",
                warp_idx, lane_idx, packed_b_smem_read_ptr, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8], ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15]);
            }
          }

          packed_b_loader.load_lateral_n<kB_Nloads>(packed_b_smem_write_ptr);
          packed_b_smem_write_ptr += PackedBLoader::kByteSize * kB_Nloads;
          ++packed_b_loader;
          load_k += PackedBLoader::kKStride;
        }

        static_assert(ATileLoader::kTiles * sizeof(unsigned) * kA_Mloads == FragmentA::kElements * sizeof(ElementT));
        cutlass::Array<unsigned, ATileLoader::kTiles>* a_tile_frag_ptr =
            reinterpret_cast<cutlass::Array<unsigned, ATileLoader::kTiles>*>(fragment_a.data());

        CUTLASS_PRAGMA_UNROLL
        for (int m_load = 0; m_load < kA_Mloads; ++m_load, ++a_tile_frag_ptr) {
          ATileLoader::ldmatrix_sync(*a_tile_frag_ptr, a_smem_read_ptr);
          a_smem_read_ptr += ATileLoader::kByteSize / kElementSize;
        }

        a_tile_loader.load_lateral_n<kA_Mloads>(a_smem_write_ptr);
        a_smem_write_ptr += ATileLoader::kByteSize * kA_Mloads / kElementSize;
        ++a_tile_loader;

        if constexpr (kDebugPrintFragA) {
          const int lane_id = threadIdx.x % 32;
          if (lane_id == 0) {
            printf("====  A tiles =======\n");
          }
          const char* const format = (lane_id == 31) ? "%f, %f\n\n" : ((lane_id % 4) == 3) ? "%f, %f\n" : "%f, %f, ";
          const ElementT* a_ptr = fragment_a.data();
          for (int m2_tile = 0; m2_tile < (WarpShape::kM / InstructionShape::kM); ++m2_tile, a_ptr += 8) {
            printf(format, float(a_ptr[0]), float(a_ptr[1]));
            printf(format, float(a_ptr[2]), float(a_ptr[3]));
            printf(format, float(a_ptr[4]), float(a_ptr[5]));
            printf(format, float(a_ptr[6]), float(a_ptr[7]));
          }
        }

        // Dequantize weights block (16, WarpShape::kN)
        meta_loader.dequant_k16(warp_k_offset, fragment_packed_b, fragment_scales[smem_read_stage], fragment_addon, fragment_b);

        // GEMM operation, covering a shape of (WarpShape::kM, WarpShape::kN, InstructionShape::kK)
        mma_op(accumulators, fragment_a, fragment_b, accumulators);
      }

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();

      advance_stage(smem_write_stage, packed_b_smem_write_ptr, a_smem_write_ptr);
      advance_stage(smem_read_stage, packed_b_smem_read_ptr, a_smem_read_ptr);

      // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
      cutlass::arch::cp_async_wait<kStages - 2>();
      //__syncthreads(); is this necessary since the loader is warp based?

      if constexpr(kDebugPrintB) {
        if (lane_idx == 0) {
          printf("Mainloop, warp: %d, proc_k %d, load_k %d\nWritePtr: %p, ReadPtr: %p\n",
            warp_idx, proc_k, load_k, packed_b_smem_write_ptr, packed_b_smem_read_ptr);
        }
        cutlass::debug::dump_shmem(packed_b_shared_ptr, MainLoopSharedBuffer::kPackedBSize);
      }
    }

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
          for (int n_tile = 0; n_tile < (WarpShape::kN / InstructionShape::kN); ++n_tile) {
            for (int m_tile = 0; m_tile < (WarpShape::kM / InstructionShape::kM); ++m_tile, c_ptr += 4) {
              // since InstructionShape::kM is 16, we can print 2 tiles
              printf(format, float(c_ptr[0]), float(c_ptr[1]));
              printf(format, float(c_ptr[2]), float(c_ptr[3]));
            }
          }
        }
        __syncthreads();
      }
    }

    // ========================== Finish the main loop ==========================
    // !!!!! SHOULD NOT ACCESS main_loop SHARED MEMORY AFTER THIS POINT !!!!!
  
    // Finished the main loop, now each warp (except warp 0) stores the partial results
    // to shared memory. Later warp 0 should gather them to form the final result
    using Float4 = cutlass::Array<float, 4>;  // hopefully utilize 128b st.shared.b128
    constexpr int kAccLoads = MmaOp::FragmentC::kElements / 4;
    static_assert(kAccLoads * 4 == MmaOp::FragmentC::kElements);
    if (warp_idx != 0){
      Float4* d_smem_ptr = reinterpret_cast<Float4*>(shared_storage.smem[warp_idx].shared_Acc.data());
      d_smem_ptr += lane_idx;
      Float4* f4s = reinterpret_cast<Float4*>(accumulators.data());
      CUTLASS_PRAGMA_UNROLL
      for (int acc_l = 0; acc_l < kAccLoads; ++acc_l) {
        d_smem_ptr[0] = f4s[acc_l];
        d_smem_ptr += 32;
      }
    }

    cutlass::arch::cp_async_wait<0>();
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
    //
    Float4 other_acc[2];
    int double_buffer_idx = 0;
    int frag_idx = 0;

    CUTLASS_PRAGMA_UNROLL
    for (int warp = 1; warp < kWarps; ++warp) {
      Float4* d_smem_ptr = reinterpret_cast<Float4*>(shared_storage.smem[warp].shared_Acc.data()) + lane_idx;

      if constexpr (kDebugPrintC) {
        if (lane_idx == 0) {
          printf("======= C gatered from warp %d =======\n", warp);
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int acc_l = 0; acc_l < kAccLoads; ++acc_l, double_buffer_idx ^= 1) {
        other_acc[double_buffer_idx] = d_smem_ptr[0];
        d_smem_ptr += 32;

        if constexpr (kDebugPrintC) {
          const char* const format = (lane_idx == 31) ? "%f, %f\n\n" : ((lane_idx % 4) == 3) ? "%f, %f\n" : "%f, %f, ";
          printf(format, float(other_acc[double_buffer_idx][0]), float(other_acc[double_buffer_idx][1]));
          printf(format, float(other_acc[double_buffer_idx][2]), float(other_acc[double_buffer_idx][3]));
        }

        if (warp == 1 && acc_l == 0) {
          continue;
        }

        const int read_idx = double_buffer_idx ^ 1;
        accumulators[frag_idx + 0] += other_acc[read_idx][0];
        accumulators[frag_idx + 1] += other_acc[read_idx][1];
        accumulators[frag_idx + 2] += other_acc[read_idx][2];
        accumulators[frag_idx + 3] += other_acc[read_idx][3];
        frag_idx += 4;
        frag_idx = frag_idx % MmaOp::FragmentC::kElements;
      }
    }
    {
      const int read_idx = double_buffer_idx ^ 1;
      accumulators[frag_idx + 0] += other_acc[read_idx][0];
      accumulators[frag_idx + 1] += other_acc[read_idx][1];
      accumulators[frag_idx + 2] += other_acc[read_idx][2];
      accumulators[frag_idx + 3] += other_acc[read_idx][3];
    }

    // Store the result
    __half2* output_ptr = reinterpret_cast<__half2*>(params.ptr_output_);
    int output_stride = params.output_byte_stride_ / sizeof(__half2);
    const float2* c_ptr = reinterpret_cast<float2 const*>(accumulators.data());

    int n = n_start + lane_b_k_offset * 2;
    CUTLASS_PRAGMA_UNROLL
    for (int n_tile = 0; n_tile < (WarpShape::kN / 8); ++n_tile, n += 8) {
      int m = m_start + lane_b_n_offset;
      CUTLASS_PRAGMA_UNROLL
      for (int m_tile = 0; m_tile < (WarpShape::kM / 8); ++m_tile, m += 8, ++c_ptr) {
        if (n < n_end && m < m_end) {
          *(output_ptr + m * output_stride + n/2) = __float22half2_rn(c_ptr[0]);
        }
      }
    }

  }
};


}  // namespace kernel
}  // namespace gemm
}  // namespace mickey
