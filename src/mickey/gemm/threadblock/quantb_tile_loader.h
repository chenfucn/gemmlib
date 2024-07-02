/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file threadblock/quantb_tile_loader.h
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
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Load packed quantized int4 matrix tiles from global memory to shared memory.
 *        Dimensions are of the dequantized (fp16) tiles.
 *        Shape::kM is ignored, using kN and kK to specify B dimension only.
 */
template <typename ThreadblockShape_, unsigned NumThreads_>
class QuantBTileLoader {
 public:
  using ThreadblockShape = ThreadblockShape_;
  static constexpr unsigned kThreads = NumThreads_;

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

  static constexpr int kBlkDim = 16;  // 16x16 block dictated by Q4 packing format
  static constexpr int kCacheLineSize = 128;  // also block size
  static constexpr int kAccessSize = 16;  // cp.async 16 bytes at a time
  static constexpr int kThreadsPerCacheLine = 8; // kCacheLineSize / kAccessSize

  static constexpr int kNBlks = ThreadblockShape::kN / kBlkDim;  // number of blocks in N dimension
  static constexpr int kKBlks = ThreadblockShape::kK / kBlkDim;  // number of blocks in K dimension
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

  /// @brief Compute the (k,n) coordinate in the threadblock
  /// @param thread_id 
  /// @return (k,n) coordinate in the threadblock
  CUTLASS_HOST_DEVICE
  static const cutlass::MatrixCoord get_lane_coord(unsigned thread_id) {
    const unsigned blk_id = thread_id / kThreadsPerCacheLine;
    const unsigned lane_in_blk = thread_id % kThreadsPerCacheLine;
    const unsigned n_id = blk_id % kNBlks;
    if constexpr (kKBlksPerLoad > 1) {
      const unsigned k_id = thread_id / (kThreadsPerCacheLine * kNBlks);
      return cutlass::MatrixCoord(
          static_cast<int>(k_id * kThreadsPerCacheLine + lane_in_blk),
          static_cast<int>(n_id));
    } else {
      return cutlass::MatrixCoord(static_cast<int>(lane_in_blk), static_cast<int>(n_id));
    }
  }

  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  QuantBTileLoader(
    void const* data_ptr,  ///< Pointer to the global memory tiles
    int byte_stride,       ///< Stride in bytes to advance to next row
    int n_start,           ///< Starting position in the M or N dimension
    int n_end,             ///< End position in the M or N dimension
    int k_start,           ///< Starting position in the K dimension
    int k_end,             ///< End position in the K dimension
    unsigned thread_id)    ///< ID of each participating thread
    : stride_(byte_stride) {
#ifndef NDEBUG
    bool assertion_pass = true;
    if (reinterpret_cast<uintptr_t>(data_ptr) % 128 != 0 && byte_stride % 128 != 0) {
      assertion_pass = false;
      if (thread_id == 0) {
        printf("data_ptr: %p and stride %d are not aligned to 128B boundary!\n", data_ptr, byte_stride);
      }
    }
    if (k_start % ThreadblockShape::kK != 0 || k_end % kBlkDim != 0) {
      assertion_pass = false;
      if (thread_id == 0) {
        printf("Not well formed k_start: %d and k_end %d!\n", k_start, k_end);
      }
    }
    if ((n_start % ThreadblockShape::kN) != 0 || (n_end % kBlkDim) != 0) {
      assertion_pass = false;
      if (thread_id == 0) {
        printf("Not well formed n_start: %d and n_end %d!\n", n_start, n_end);
      }
    }
    if (thread_id >= kThreads) {
      assertion_pass = false;
      printf("thread_id should be [0-%d) but it is: %d!\n", kThreads, thread_id);
    }
    assert(assertion_pass);
#endif

    int n_blks = min(kNBlks, (n_end - n_start) / kBlkDim);
    auto coord = get_lane_coord(thread_id);
    if (k_start >= k_end || coord[1] >= n_blks) {
      return;
    }

    uint8_t const* n_base = reinterpret_cast<uint8_t const*>(data_ptr) + (n_start / kBlkDim + coord[1]) * byte_stride;
    g_ptr_ = n_base + (k_start * (kCacheLineSize / kBlkDim)) + (coord[0] * kAccessSize);
    g_ptr_end_ = n_base + k_end * (kCacheLineSize / kBlkDim);
  }

  /// Advances to the next position in the K dimension
  CUTLASS_DEVICE
  QuantBTileLoader& operator++() {
    g_ptr_ += kKBlks * kCacheLineSize;
    return *this;
  }

  /// Loads a tile from global memory to shared memory
  CUTLASS_DEVICE
  void load_to_smem(void* smem_ptr, unsigned thread_id) const {
    const uint8_t *ptr = g_ptr_;
    uint8_t *smem_lane_ptr_ = reinterpret_cast<uint8_t*>(smem_ptr) + mul_power2<kAccessSize>(thread_id);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kLoads; ++i) {
      cutlass::arch::cp_async<kAccessSize, cutlass::arch::CacheOperation::Global>(
          smem_lane_ptr_, ptr, ptr < g_ptr_end_);
      smem_lane_ptr_ += kThreads * kAccessSize;
      ptr += kKBlksPerLoad * kCacheLineSize;  // stride on the k dimension
    }
  }

  /**
   * @brief Load a (16, 64) block, i.e. 4 tiles from shared memory to register
   * Each n_iter covers 4 columns of the block (64), each k_iter covers 1 rows of the block (16)
   */
  CUTLASS_DEVICE
  static void load_fragment_tile4(unsigned thread_id, uint8_t const* smem_lane_ptr, int n_iter, int k_iter, Fragment& reg) {
    assert(k_iter < kKBlks);
    assert(n_iter < (kNBlks / 4));
    assert(thread_id < kThreads);

    unsigned lane_id = thread_id % 32;
    uint8_t const* smem_ptr = smem_lane_ptr + mul_power2<kAccessSize>(lane_id) + n_iter * 4 * kCacheLineSize + k_iter * kNBlks * kCacheLineSize;
    cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(reg, smem_ptr);
  }

};

}  // namespace threadblock
}  // namespace gemm
}  // namespace mickey
