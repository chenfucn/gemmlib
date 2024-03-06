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

#include "cutlass/arch/mma.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_policy.h"

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

#include "gemm/warp/tensor_core_tile_loader.h"
#include "gemm/warp/quantb_meta_loader.h"

#include "gtest/gtest.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace onnxruntime {
namespace cuda {
namespace test {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename QuantBlocking_,              ///! Shape of the quantization block, either 1xb or bx1
  bool     has_quant_offset_,           ///! Whether the quantization has offset
  typename WarpShape_,                  ///! Warp-scoped matrix multiply-accumulate
  int SplitKSerial_ = 1,                ///! How many warps to split the K dimension in the same MxN block
  int Stages_ = 4                       ///! Stages of the pipelined mainloop
>
struct LoadPackedBTestKernel {
 public:
  //
  // Type definitions
  //

  using QuantBlocking = QuantBlocking_;
  using WarpShape = WarpShape_;
  static constexpr bool has_quant_offset = has_quant_offset_;
  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kStages = Stages_;

  static_assert(kSplitK > 0 && ((kSplitK - 1) & kSplitK) == 0,
     "kSplitK must be positive and a power of 2");

  static constexpr bool kDebugPrint = false;
  static constexpr bool kDebugPrintFragA = false;
  static constexpr bool kDebugPrintC = true;

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

  // Since int4 weights are packed (16x16) -> (8x8), each tile is expanded to 4 tiles when
  // de-quantized to 16b float.

  // Fragments of quantized weights, keep all warp tile in registers for it's easier to
  // locate corresponding scale
  using FragmentPackedB = cutlass::Array<
      unsigned,  // 8 of int4 weights each tile (becomes 4 tiles when de-quantized)
      PackedBLoader::kTiles * kB_Nloads * kB_Kloads>;

  // Fragments for operand A and B, each tile has 2 elements per thread. In each iteration,
  // we use a (WarpShape::kM, 16) block of A and (16, WarpShape::kN) block of B for mma
  using FragmentA = cutlass::Array<ElementT, 2 * (WarpShape::kM / 8) * 2>;
  using FragmentB = cutlass::Array<ElementT, 2 * (WarpShape::kN / 8) * 2>;

  static constexpr int kWarps = kSplitK; // TODO! more warps when we have a larger thread block shape
  static int const kThreadCount = 32 * kWarps;

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

  /// Shared memory storage structure
  struct SharedStorage {
    /// Buffer for prepacked weights
    static constexpr int kPackedBSizePerIter = kB_Nloads * kB_Kloads * PackedBLoader::kByteSize;
    static constexpr int kPackedBSizePerWarp = kPackedBSizePerIter * kStages;
    static constexpr int kPackedBSize = kPackedBSizePerWarp * kWarps;
    cutlass::AlignedBuffer<uint8_t, kPackedBSize> operand_B;

    /// Buffer for A tensor
    static constexpr int kASizePerIter = kA_Mloads * kA_Kloads * ATileLoader::kByteSize / kElementSize;
    static constexpr int kASizePerWarp = kASizePerIter * kStages;
    static constexpr int kASize = kASizePerWarp * kWarps;
    cutlass::AlignedBuffer<ElementT, kASize> operand_A;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  LoadPackedBTestKernel() { }

  /// Determines whether kernel satisfies alignment
  static cutlass::Status can_implement(const Params &params) {
    if ((reinterpret_cast<uintptr_t>(params.ptr_a_) % 16)) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.ptr_a_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (params.a_byte_stride_ % 16) {
      std::cerr << "LoadPackedBTestKernel validation fail: params.a_byte_stride_ is not aligned to 16 bytes!" << std::endl;
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if ((params.problem_size_.k() % QuantBlocking::kRow != 0) ||
        (params.problem_size_.n() % QuantBlocking::kColumn) != 0){
      std::cerr << "LoadPackedBTestKernel validation fail: partial quantization block not supported!" << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }
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
    ElementT* a_shared_ptr = shared_storage.operand_A.data() + 
      SharedStorage::kASizePerWarp * warp_idx;

    //
    // Prologue
    //
    CUTLASS_PRAGMA_UNROLL
    for (; smem_write_stage < kStages - 1; ++smem_write_stage, load_k += WarpShape::kK) {
      uint8_t* packed_b_smem_ptr = packed_b_shared_ptr + smem_write_stage * SharedStorage::kPackedBSizePerIter;
      ElementT* a_smem_ptr = a_shared_ptr + smem_write_stage * SharedStorage::kASizePerIter;

      meta_loader.load(fragment_scales[smem_write_stage], load_k, min(k_end, load_k + WarpShape::kK));

      // Load packed b
      CUTLASS_PRAGMA_UNROLL
      for (int k_load = 0; k_load < kB_Kloads; ++k_load) {
        packed_b_loader.load_lateral_n<kB_Nloads>(packed_b_smem_ptr);
        packed_b_smem_ptr += PackedBLoader::kByteSize * kB_Nloads;
        ++packed_b_loader;
      }

      // Load A
      CUTLASS_PRAGMA_UNROLL
      for (int ka_load = 0; ka_load < kA_Kloads; ++ka_load) {
        a_tile_loader.load_lateral_n<kA_Mloads>(a_smem_ptr);
        a_smem_ptr += ATileLoader::kByteSize * kA_Mloads / kElementSize;
        ++a_tile_loader;
      }

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }    

    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<kStages - 2>();
    //__syncthreads(); is this necessary since the loader is warp based?
    if constexpr(kDebugPrint) {
      if (lane_idx == 0) {
        printf("Prologue, warp: %d, ShapredPtr: %p, WarpPtr: %p\n",
          warp_idx, shared_storage.operand_A.data(), a_shared_ptr);
        printf("\n********Dumping the shared memory of Warp %d*******\n\n", warp_idx);

        for (int i = 0; i < SharedStorage::kASizePerWarp; i += 8) {
          for (int j = 0; j < 8; ++j) {
            printf("%f, ", float(a_shared_ptr[i + j]));
          }
          printf("\n");
        }
      }
    }

    //
    // Mainloop
    //
    for (; proc_k < k_end; smem_write_stage = (smem_write_stage + 1) % kStages, smem_read_stage = (smem_read_stage + 1) % kStages, proc_k += WarpShape::kK){
      typename MetaLoader::FragmentScales fragment_addon;
  
      uint8_t* packed_b_smem_read_ptr = packed_b_shared_ptr + smem_read_stage * SharedStorage::kPackedBSizePerIter;
      uint8_t* packed_b_smem_write_ptr = packed_b_shared_ptr + smem_write_stage * SharedStorage::kPackedBSizePerIter;
      ElementT* a_smem_read_ptr = a_shared_ptr + smem_read_stage * SharedStorage::kASizePerIter;
      ElementT* a_smem_write_ptr = a_shared_ptr + smem_write_stage * SharedStorage::kASizePerIter;

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
            PackedBLoader::ldmatrix_sync(*packed_b_tile_frag_ptr, lane_idx, packed_b_smem_read_ptr);
            packed_b_smem_read_ptr += PackedBLoader::kByteSize;

            if constexpr (kDebugPrint) {
              uint8_t const* ptr = reinterpret_cast<uint8_t const*>(packed_b_tile_frag_ptr->data());
              printf("Warp: %d, lane %2d, smem_read_ptr %p, %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d\n",
                warp_idx, lane_idx, packed_b_smem_read_ptr, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8], ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15]);
            }
          }

          if (load_k < k_end) {
            packed_b_loader.load_lateral_n<kB_Nloads>(packed_b_smem_write_ptr);
            packed_b_smem_write_ptr += PackedBLoader::kByteSize * kB_Nloads;
            ++packed_b_loader;
          }
          load_k += PackedBLoader::kKStride;
        }

        static_assert(ATileLoader::kTiles * sizeof(unsigned) * kA_Mloads == FragmentA::kElements * sizeof(ElementT));
        cutlass::Array<unsigned, ATileLoader::kTiles>* a_tile_frag_ptr =
            reinterpret_cast<cutlass::Array<unsigned, ATileLoader::kTiles>*>(fragment_a.data());

        CUTLASS_PRAGMA_UNROLL
        for (int m_load = 0; m_load < kA_Mloads; ++m_load, ++a_tile_frag_ptr) {
          ATileLoader::ldmatrix_sync(*a_tile_frag_ptr, lane_idx, a_smem_read_ptr);
          a_smem_read_ptr += ATileLoader::kByteSize / kElementSize;
        }

        if constexpr (kDebugPrintFragA) {
          const int lane_id = threadIdx.x % 32;
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
        CUTLASS_PRAGMA_UNROLL
        for (int b_tile_n = 0; b_tile_n < (WarpShape::kN/8); ++b_tile_n) {
          int n = n_start + b_tile_n * 8 + lane_b_n_offset;
          int k = proc_k + warp_k_offset + lane_b_k_offset * 2;
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

        // GEMM operation, covering a shape of (WarpShape::kM, WarpShape::kN, InstructionShape::kK)
        mma_op(accumulators, fragment_a, fragment_b, accumulators);
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

    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    if constexpr (kDebugPrintC) {
      static_assert(MmaOp::FragmentC::kElements == (WarpShape::kN / InstructionShape::kN) * (WarpShape::kM / InstructionShape::kM) * 4);
      const float* c_ptr = accumulators.data();
      const int lane_id = threadIdx.x % 32;
      const char* const format = (lane_id == 31) ? "%f, %f\n\n" : ((lane_id % 4) == 3) ? "%f, %f\n" : "%f, %f, ";
      for (int n_tile = 0; n_tile < (WarpShape::kN / InstructionShape::kN); ++n_tile) {
        for (int m_tile = 0; m_tile < (WarpShape::kM / InstructionShape::kM); ++m_tile, c_ptr += 4) {
          // since InstructionShape::kM is 16, we can print 2 tiles
          printf(format, float(c_ptr[0]), float(c_ptr[1]));
          printf(format, float(c_ptr[2]), float(c_ptr[3]));
        }
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

template <typename QuantBlocking, typename WarpShape, int kSplitK, int kStages>
void test_load_packed_b(int m, int n, int k) {
  std::cout << "Testing Blocking: " << QuantBlocking::kRow << "x" << QuantBlocking::kColumn 
            << " WarpShape: " << WarpShape::kM << "x" << WarpShape::kN << "x" << WarpShape::kK
            << ", kSplitK: " << kSplitK << ", kStages: " << kStages;
  std::cout << ", m: " << m << ", n: " << n << ", k: " << k << std::endl;

  using Test = LoadPackedBTest<QuantBlocking, WarpShape, kSplitK, kStages>;
  Test test;
  cutlass::gemm::GemmCoord problem_size(m, n, k);

  constexpr bool has_offsets = false;
  using QuantBaseT = onnxruntime::test::BlkQuantizationRef<QuantBlocking, has_offsets>;
  using LayoutQMeta = typename QuantBaseT::LayoutQMeta;

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_a({m, k});
  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 51, -1.75f, 1.9f);

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
  tensor_b.sync_device();
  tensor_a.sync_device();

  int dequant_stride = tensor_b.stride(0);
  ASSERT_EQ(dequant_stride, problem_size.n());
  cutlass::Status status = test.run(nullptr, problem_size,
                                    tensor_b.device_data(), dequant_stride * sizeof(cutlass::half_t),
                                    tensor_a.device_data(), tensor_a.stride(0) * sizeof(cutlass::half_t),
                                    thrust::raw_pointer_cast(packed_w_dev.data()), problem_size.k(),
                                    thrust::raw_pointer_cast(packed_scale_dev.data()), meta_tensor_stride * sizeof(cutlass::half_t));
  ASSERT_EQ(status, cutlass::Status::kSuccess);
  tensor_b.sync_host();

  // Run reference kernel
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel
  cutlass::reference::host::TensorFill(tensor_ref_d.host_view());
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_c(
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
                   float, cutlass::layout::RowMajor,
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
  std::cout << "========  Reference kernel result:  ============" << std::endl;
  for (int row = 0; row < tensor_ref_d.extent()[0]; ++row) {
    for (int col = 0; col < tensor_ref_d.extent()[1]; ++col) {
      printf("%f, ", tensor_ref_d.at({row, col}));
      if (col % 8 == 7) {
        printf(", ");
      }
    }
    printf("\n");
    if (row % 8 == 7) {
      printf("\n");
    }
  }


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

  test_load_packed_b<cutlass::MatrixShape<16, 1>, cutlass::gemm::GemmShape<32, 32, 32>, 1, 4>(32, 32, 32);

  // test_load_packed_b<cutlass::gemm::GemmShape<1, 16, 64>, 1, 3>(1, 16, 64);
}

} // namespace test
} // namespace cuda
} // namespace onnxruntime
