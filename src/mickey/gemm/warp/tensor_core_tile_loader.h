/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file warp/tensor_core_tile_loader.h
 * @brief Load matrix tiles from global memory to shared memory, in a way the shared memory
 * tiles can be readily loaded using ldmatrix instruction while avoiding bank conflicts.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"

#include "int_util.h"

namespace mickey {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Load packed quantized int4 matrix tiles from global memory to shared memory.
 *        Dimensions are of the dequantized (fp16) tiles.
 *        WarpShape::kM is ignored, using kN and kK to specify B dimension only.
 * 
 *        Tested shapes: (M, 16, 64), (M, 32, 32), (M, 64, 32)
*/
template <typename WarpShape_, int TBStrideK_>
class TensorCoreTileLoader {
 public:
  using WarpShape = WarpShape_;
  static constexpr int kThreadBlockStrideK = TBStrideK_;

  static_assert(WarpShape::kN == 16 || WarpShape::kN == 32 || WarpShape::kN == 64);
  static_assert(WarpShape::kK % 16 == 0); // packing restriction
  static_assert(kThreadBlockStrideK % WarpShape::kK == 0);

  // Q4 matrix is column major. Each 16x16 fp16 block is quantized
  // and packed into a 128 byte vector that fits into a cache line.
  // Each thread loads 16 bytes. 8 threads load a line/block. So the
  // thread layout looks like:
  //  0   8  16  24
  //  1   9  17  25 
  //  2  10  18  26
  //  3  11  19  27
  //  4  12  20  28
  //  5  13  21  29
  //  6  14  22  30
  //  7  15  23  31

  static constexpr int kBlkDim = 16;
  static constexpr int kCacheLineSize = 128;
  static constexpr int kAccessSize = 16;  // cp.async 16 bytes at a time
  static constexpr int kThreadsPerCacheLine = kCacheLineSize / kAccessSize; // 8 threads per cache line/block
  static constexpr int kThreads = 32;     // 32 threads in a warp

  static constexpr int kNBlks = WarpShape::kN / kBlkDim;  // number of blocks in N dimension
  static constexpr int kKBlks = WarpShape::kK / kBlkDim;  // number of blocks in K dimension
  static constexpr int kNumBlks = kNBlks * kKBlks;
  static constexpr int kByteSize = kNumBlks * kCacheLineSize;

  // A warp loads 4 cache lines at a time, this requires the warp shape to be
  // (16, 64), (32, 32), (64, 16), or multiple of these
  static_assert(kNumBlks % 4 == 0); 

  // each cp_async loads of the participating threads must load entire N width
  static constexpr int kKBlksPerLoad = kThreads / (kThreadsPerCacheLine * kNBlks);
  static constexpr int kStrideKPerLoad = kKBlksPerLoad * kBlkDim;
  static constexpr int kLoads = kKBlks / kKBlksPerLoad;
  static_assert(kThreadsPerCacheLine * kNBlks * kKBlksPerLoad == kThreads);

  // Register to store loaded tiles
  using Fragment = cutlass::Array<uint32_t, 4>;

 private:
    /// Pointer to global memory to load data from
    uint8_t const* g_ptr_{nullptr};
    uint8_t const* g_ptr_end_{nullptr};

    /// Stride in bytes to advance to next row in n dimension
    const int stride_;

 public:

  CUTLASS_HOST_DEVICE
  static const cutlass::MatrixCoord get_lane_coord(int lane_id) {
    // 16x16 block packed into 128 byte cache line
    const int blk_id = lane_id / kThreadsPerCacheLine;
    const int lane_in_blk = lane_id % kThreadsPerCacheLine;
    const int n_id = blk_id % kNBlks;
    if constexpr (kKBlksPerLoad > 1) {
      const int k_id = lane_id / (kThreadsPerCacheLine * kNBlks);
      return cutlass::MatrixCoord(k_id * kThreadsPerCacheLine + lane_in_blk, n_id);
    } else {
      return cutlass::MatrixCoord(lane_in_blk, n_id);
    }
  }

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  TensorCoreTileLoader(
    void const* data_ptr,  ///< Pointer to the global memory tiles
    int byte_stride,       ///< Stride in bytes to advance to next row
    int n_start,           ///< Starting position in the M or N dimension
    int n_end,             ///< End position in the M or N dimension
    int k_start,           ///< Starting position in the K dimension
    int k_end,             ///< End position in the K dimension
    int lane_id)           ///< ID of each participating thread
    : stride_(byte_stride) {
#ifndef NDEBUG
    bool assertion_pass = true;
    if (reinterpret_cast<uintptr_t>(data_ptr) % 128 != 0 && byte_stride % 128 != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("data_ptr: %p and stride %d are not aligned to 128B boundary!\n", data_ptr, byte_stride);
      }
    }
    if (k_start % WarpShape::kK != 0 || k_end % kBlkDim != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("Not well formed k_start: %d and k_end %d!\n", k_start, k_end);
      }
    }
    if ((n_start % WarpShape::kN) != 0 || (n_end % kBlkDim) != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("Not well formed n_start: %d and n_end %d!\n", n_start, n_end);
      }
    }
    if (lane_id < 0 || lane_id >= kThreads) {
      assertion_pass = false;
      printf("lane_id should be [0-%d) but it is: %d!\n", kThreads, lane_id);
    }
    assert(assertion_pass);
#endif

    int n_blks = min(kNBlks, (n_end - n_start) / kBlkDim);
    auto coord = get_lane_coord(lane_id);
    if (k_start >= k_end || coord[1] >= n_blks) {
      return;
    }

    uint8_t const* n_base = reinterpret_cast<uint8_t const*>(data_ptr) + (n_start / kBlkDim + coord[1]) * byte_stride;
    g_ptr_ = n_base + (k_start * (kCacheLineSize / kBlkDim)) + (coord[0] * kAccessSize);
    g_ptr_end_ = n_base + k_end * (kCacheLineSize / kBlkDim);
  }

  /// Advances to the next position in the K dimension
  CUTLASS_DEVICE
  TensorCoreTileLoader& operator++() {
    g_ptr_ += (kThreadBlockStrideK / kBlkDim) * kCacheLineSize;
    return *this;
  }

  /**
   * @brief Get the pointer to the shared memory location for the current lane
   * @param smem_ptr pointer to the shared memory location for the warp.
  */
  template<typename T>
  CUTLASS_DEVICE
  static T* get_smem_lane_ptr(T* smem_ptr, int lane_id) {
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(smem_ptr) + mul_power2<kAccessSize>(lane_id));
  }

  template<typename T>
  CUTLASS_DEVICE
  static T* get_smem_warp_base_ptr(T* smem_lane_ptr, int lane_id) {
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(smem_lane_ptr) - mul_power2<kAccessSize>(lane_id));
  }

  /// Loads a tile from global memory to shared memory
  CUTLASS_DEVICE
  void load_to_smem(void* smem_lane_ptr) const {
    const uint8_t *ptr = g_ptr_;
    uint8_t *smem_lane_ptr_ = reinterpret_cast<uint8_t*>(smem_lane_ptr);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoads; ++i) {
      cutlass::arch::cp_async<kAccessSize, cutlass::arch::CacheOperation::Global>(
          smem_lane_ptr_, ptr, ptr < g_ptr_end_);
      smem_lane_ptr_ += kThreads * kAccessSize;
      ptr += kKBlksPerLoad * kCacheLineSize;  // stride on the k dimension
    }
  }

  /**
   * @brief Load a (16, WarpShape::kN) belt from shared memory to register
  */
  CUTLASS_DEVICE
  static void load_to_register(int lane_id, int k_iter, uint8_t const* smem_lane_ptr, Fragment& reg) {
    assert(k_iter < kKBlks);
    assert((k_iter % kKBlksPerLoad) == 0);
    assert(lane_id >= 0 && lane_id < 32);

    uint8_t const* smem_ptr = smem_lane_ptr + k_iter * kNBlks * kCacheLineSize;
    cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(reg, smem_ptr);
  }

};

}  // namespace warp
}  // namespace gemm
}  // namespace mickey
