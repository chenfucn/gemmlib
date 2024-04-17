/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file warp/vector_loader.h
 * @brief Load an activation vector from global memory to shared memory, used by GEMV kernel.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

#include "int_util.h"

namespace mickey {
namespace gemm {
namespace warp {

/**
 * @brief Load a vector from global memory to shared memory, used by GEMV kernel.
*/
class VectorLoader {
 public:

  // consider generalize to 2, 4 batch size (need swizzle pattern??)
  static constexpr int SmemDimM = 1;
  static constexpr int SmemDimK = 128;    // HBM load 128 bytes per access
  static constexpr int kAccessSize = 16;  // one cp.async loads 16 bytes
  static constexpr int kBlockSize = SmemDimM * SmemDimK;

  static constexpr int kThreads = kBlockSize / kAccessSize;

 private:
  /// Pointer to global memory to load data from
  uint8_t const* g_ptr_{nullptr};
  uint8_t const* g_ptr_end_{nullptr};

 public:
  CUTLASS_DEVICE
  VectorLoader(
      void const* data_ptr,  ///< Pointer to the global memory tiles
      int k_start,           ///< Starting position in the K dimension
      int k_end,             ///< End position in the K dimension
      int lane_id) {         ///< ID of each participating thread
#ifndef NDEBUG
  bool assertion_pass = true;
  if (reinterpret_cast<uintptr_t>(data_ptr) % kAccessSize != 0) {
    assertion_pass = false;
    if (lane_id == 0) {
      printf("data_ptr: %p is not aligned to 16B boundary!\n", data_ptr);
    }
  }
  if (k_start % kAccessSize != 0) {
    assertion_pass = false;
    if (lane_id == 0) {
      printf("k_start: %d is not aligned to 16B boundary!\n", k_start);
    }
  }
  if (k_end % kAccessSize != 0) {
    assertion_pass = false;
    if (lane_id == 0) {
      printf("k_end: %d is not aligned to 16B boundary!\n", k_end);
    }
  }
  if (lane_id < 0 || lane_id >= 32) {
    assertion_pass = false;
    if (lane_id == 0) {
      printf("Warp based loader, thread_id should be [0-%d) but it is: %d!\n", kThreads, lane_id);
    }
  }
  assert(assertion_pass);
#endif

    k_start += lane_id * kAccessSize;
    if (k_start >= k_end || lane_id >= kThreads) {
      return;
    }

    g_ptr_ = reinterpret_cast<uint8_t const*>(data_ptr) + k_start;
    g_ptr_end_ = reinterpret_cast<uint8_t const*>(data_ptr) + k_end;
  }

  CUTLASS_DEVICE
  void load_to_smem(const int lane_id, void* smem) {
    uint8_t* smem_ptr0 = reinterpret_cast<uint8_t*>(smem) + (lane_id % kThreads) * kAccessSize;
    cutlass::arch::cp_async<kAccessSize, cutlass::arch::CacheOperation::Global>(
        smem_ptr0, g_ptr_, g_ptr_ < g_ptr_end_);
  }

  CUTLASS_DEVICE
  VectorLoader& operator++() {
    g_ptr_ += kAccessSize * kThreads;
    return *this;
  }

  template <typename Fragment>
  CUTLASS_DEVICE
  static void load_fragment_k64(const int lane_id, void const* smem, int offset_k, Fragment& frag) {
#ifndef NDEBUG
    bool assert_fail = false;
    if (offset_k != 0 && offset_k != 64) {
      assert_fail = true;
      if (lane_id == 0) {
        printf("Invalid offset_k: %d!\n", offset_k);
      }
    }
    if (lane_id < 0 || lane_id >= 32) {
      assert_fail = true;
      printf("Warp based loader, lane_id should be [0-32) but it is: %d!\n", lane_id);
    }
    assert(assert_fail == false);
#endif
    constexpr int kLdsmTiles = 4; // fully utilize ldmatrix instruction
    static_assert(sizeof(cutlass::Array<uint32_t, kLdsmTiles>) == sizeof(Fragment), "Fragment size mismatch!");
    auto* frag_ptr = reinterpret_cast<cutlass::Array<uint32_t, kLdsmTiles>*>(&frag);
    int k_lane_id = lane_id / 8 + offset_k / 16;
    const uint8_t* smem_ptr = reinterpret_cast<const uint8_t*>(smem) + k_lane_id * kAccessSize;
    cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(frag_ptr[0], smem_ptr);
  }
};

} // namespace warp
} // namespace gemm
} // namespace mickey
