/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file threadblock/swizzle_tile_loader.h
 * @brief Load matrix tiles from global memory to shared memory, in a way the shared memory
 * tiles can be readily loaded using ldmatrix instruction while avoiding bank conflicts.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"
#include "cute/layout.hpp"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

#include "int_util.h"

namespace mickey {
namespace gemm {
namespace threadblock {

/**
 * @brief Load a row major tile (SmemDimM, SmemDimK) from global memory to shared
 *        memory and then to fragment with ldmatrix instruction, with swizzling
 *        to avoid bank conflicts.
 */
template <int SmemDimM, int SmemDimK, int NumThreads>
class SwizzleTileLoader;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int SmemDimM_, int NumThreads_>
class SwizzleTileLoader<SmemDimM_, 128, NumThreads_> {
  public:
    static constexpr int SmemDimM = SmemDimM_;
    static constexpr int SmemDimK = 128;
    static constexpr int kThreads = NumThreads_;
    static constexpr int kAccessSize = 16;  // one cp.async loads 16 bytes
    static constexpr int kBlockSize = SmemDimM * SmemDimK;
    static constexpr int kTiles = (SmemDimM / 8) * (SmemDimK / 16);

    static_assert(kThreads % 32 == 0); // whole warp only
    static_assert(kTiles % 4 == 0);   // 4 tiles per warp per cp.async or ldmatrix

    // Swizzle pattern is 8x8
    static constexpr int kSwizzleK = SmemDimK / kAccessSize;
    static_assert(kSwizzleK == cute::_8::value);
    static constexpr int kSwizzleM = cute::_8::value;
    static constexpr int kSwizzleTileSize = kSwizzleK * kSwizzleM;
    using Swizzled128 = decltype(
        cute::composition(cute::Swizzle<3,0,3>{}, 
                          cute::Layout<cute::Shape<cute::_8, cute::_8>,
                                       cute::Stride<cute::_1, cute::_8>>{}));

    // Number of rows loaded by a single cp.async
    static constexpr int kGloadStrideM = kThreads / kSwizzleK;
    static_assert(SmemDimM % kGloadStrideM == 0);

    // How many cp.async instructions to load a tile
    static constexpr int kGloadSplit = SmemDimM / kGloadStrideM;

 private:
    /// Pointer to global memory to load data from
    uint8_t const* g_ptr_{nullptr};
    /// Iteration boundaries in the M or N dimension
    int mn_cnt_{0};
    /// Iteration boundaries in the K dimension, in strides of 16
    int k_cnt_{0};
    /// Stride in bytes to advance to next row in m or n dimension
    const int stride_;

 public:
    CUTLASS_DEVICE
    SwizzleTileLoader(
        void const* data_ptr,  ///< Pointer to the global memory tiles
        int byte_stride,       ///< Stride in bytes to advance to next row
        int mn_start,          ///< Starting position in the M or N dimension
        int mn_end,            ///< End position in the M or N dimension
        int k_start,           ///< Starting position in the K dimension
        int k_end,             ///< End position in the K dimension
        int thread_id)         ///< ID of each participating thread
    : stride_(byte_stride) {
    #ifndef NDEBUG
        bool assertion_pass = true;
        if (reinterpret_cast<uintptr_t>(data_ptr) % kAccessSize != 0) {
            assertion_pass = false;
            if (thread_id == 0) {
                printf("data_ptr: %p is not aligned to 16B boundary!\n", data_ptr);
            }
        }
        if (byte_stride % kAccessSize != 0) {
            assertion_pass = false;
            if (thread_id == 0) {
                printf("byte_stride: %d is not aligned to 16B boundary!\n", byte_stride);
            }
        }
        if (k_start % kAccessSize != 0) {
            assertion_pass = false;
            if (thread_id == 0) {
                printf("k_start: %d is not aligned to 16B boundary!\n", k_start);
            }
        }
        if (k_end % kAccessSize != 0) {
            assertion_pass = false;
            if (thread_id == 0) {
                printf("k_end: %d is not aligned to 16B boundary!\n", k_end);
            }
        }
        if (thread_id >= kThreads) {
            assertion_pass = false;
            if (thread_id == 0) {
                printf("Warp based loader, thread_id should be [0-%d) but it is: %d!\n", kThreads, thread_id);
            }
        }
        assert(assertion_pass);
    #endif

        int lane_m = thread_id / kSwizzleK;
        int lane_k = thread_id % kSwizzleK;
        mn_start += lane_m;
        k_start += lane_k * kAccessSize;

        mn_cnt_ = div_up(mn_end - mn_start, kGloadStrideM);
        k_cnt_ = div_up(k_end - k_start, kSwizzleK * kAccessSize);
        if (mn_cnt_ <= 0 || k_cnt_ <= 0) {
            mn_cnt_ = 0;
            k_cnt_ = 0;
            g_ptr_ = nullptr;
            return;
        }
        g_ptr_ = reinterpret_cast<uint8_t const*>(data_ptr) + mn_start * byte_stride + k_start;
        // if (thread_id == 0)
        //   printf("thread_id: %d, mn_start: %d, mn_end: %d, k_start: %d, k_end: %d, g_ptr: %p\n", thread_id, mn_start, mn_end, k_start, k_end, g_ptr_);
    }

    /**
     * @brief Load a row major tile (SmemDimM, 128) from global memory to shared memory 
    */
    CUTLASS_DEVICE
    void load_to_smem(const int thread_id, void* smem) {
        const uint8_t* data_ptr = g_ptr_;

        if constexpr (kThreads == 32) {
            // The swizzle pattern is 8x8, but we only have 32 threads,
            // covering half of the swizzle pattern
            uint8_t* smem_ptr0 = reinterpret_cast<uint8_t*>(smem) + Swizzled128{}(thread_id) * kAccessSize;
            uint8_t* smem_ptr1 = reinterpret_cast<uint8_t*>(smem) + Swizzled128{}(thread_id + kThreads) * kAccessSize;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kGloadSplit;) {
                cutlass::arch::cp_async<kAccessSize, cutlass::arch::CacheOperation::Global>(
                    smem_ptr0, data_ptr, g_ptr_ != nullptr && i < mn_cnt_);
                data_ptr += stride_ * kGloadStrideM;
                smem_ptr0 += kSwizzleTileSize * kAccessSize;
                ++i;

                cutlass::arch::cp_async<kAccessSize, cutlass::arch::CacheOperation::Global>(
                    smem_ptr1, data_ptr, g_ptr_ != nullptr && i < mn_cnt_);
                data_ptr += stride_ * kGloadStrideM;
                smem_ptr1 += kSwizzleTileSize * kAccessSize;
                ++i;
            }
        } else {
            // kThreads is 64, 128, 256, etc.
            // The swizzle pattern is 8x8, and we have enough threads to cover it
            const int pattern_offset = (thread_id / 64) * kSwizzleTileSize * kAccessSize;
            uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(smem) + Swizzled128{}(thread_id % 64) * kAccessSize + pattern_offset;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kGloadSplit; ++i) {
                cutlass::arch::cp_async<kAccessSize, cutlass::arch::CacheOperation::Global>(
                    smem_ptr, data_ptr, g_ptr_ != nullptr && i < mn_cnt_);
                data_ptr += stride_ * kGloadStrideM;
                smem_ptr += kGloadStrideM * kSwizzleK * kAccessSize;
            }
        }
    }

    CUTLASS_DEVICE
    void load_to_smem_split(const unsigned int thread_id, void* smem, const int split_idx){
        const uint8_t* split_ptr = g_ptr_ + split_idx * stride_ * kGloadStrideM;
        if (split_idx >= kGloadSplit) {
            return;
        }

        if constexpr (kThreads == 32) {
            const int offset = (split_idx >> 1) * kSwizzleTileSize * kAccessSize;
            const int swizzled = Swizzled128{}(thread_id + (split_idx & 1) * kThreads) * kAccessSize;
            uint8_t* split_smem_ptr = reinterpret_cast<uint8_t*>(smem) + swizzled + offset;

            cutlass::arch::cp_async<kAccessSize, cutlass::arch::CacheOperation::Global>(
                split_smem_ptr, split_ptr, g_ptr_ != nullptr && split_idx < mn_cnt_);
        } else {
            const int pattern_offset = (thread_id / 64) * kSwizzleTileSize * kAccessSize;
            const int swizzled = Swizzled128{}(thread_id % 64) * kAccessSize;
            uint8_t* split_smem_ptr = reinterpret_cast<uint8_t*>(smem) + swizzled + pattern_offset + split_idx * kGloadStrideM * kSwizzleK * kAccessSize;

            cutlass::arch::cp_async<kAccessSize, cutlass::arch::CacheOperation::Global>(
                split_smem_ptr, split_ptr, g_ptr_ != nullptr && split_idx < mn_cnt_);
        }
    }

    /**
     * @brief Advance global memory pointer to the next tile in the K dimension
    */
    CUTLASS_DEVICE
    SwizzleTileLoader& operator++() {    
        --k_cnt_;
        if (k_cnt_ > 0) {
            g_ptr_ += kAccessSize * kSwizzleK;
        } else {
            g_ptr_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Load a ribbin of (16, 32) from shared memory to fragment,
     *        This is what a single ldmatrix instruction can load:
     *        4 tiles shaped as 2x2 fp16 tiles, fitting the A tensor
     *        input layout of the tensor core.
     */
    CUTLASS_DEVICE
    static void load_fragment_Atiles(const int lane_id, void const* smem, int offset_m, int offset_k, cutlass::Array<unsigned, 4>& frag) {
        static_assert((SmemDimM % 16) == 0);
#ifndef NDEBUG
        bool assert_fail = false;
        if ((offset_k % 32) != 0) {
            assert_fail = true;
            if (lane_id == 0) {
                printf("Invalid offset_k: %d!\n", offset_k);
            }
        }
        if ((offset_m % 16) != 0) {
            assert_fail = true;
            if (lane_id == 0) {
                printf("Invalid offset_m: %d!\n", offset_m);
            }
        }
        if (lane_id < 0 || lane_id >= 32) {
            assert_fail = true;
            if (lane_id == 0) {
                printf("Warp based loader, lane_id should be 0-32 but it is: %d!\n", lane_id);
            }
        }
        assert(assert_fail == false);
#endif

        constexpr int kStrideM = 16 / kSwizzleM;  // Span 2 swizzle patterns on M dim
        int m_lane_id = lane_id % 16;
        int k_tile_id = (lane_id + offset_k) / 16;

        int m_tile_id = m_lane_id / kSwizzleM;
        int m_tile_offset = m_lane_id % kSwizzleM;
        int swizzled_id = Swizzled128{}(k_tile_id, m_tile_offset) + m_tile_id * kSwizzleTileSize;
        // printf("lane_id: %d, m_lane_id: %d, k_tile_id: %d, swizzled_id: %d\n", lane_id, m_lane_id, k_tile_id, swizzled_id);
        const uint8_t* smem_ptr = reinterpret_cast<const uint8_t*>(smem) + swizzled_id * kAccessSize;
        smem_ptr += offset_m * (kSwizzleTileSize * kStrideM * kAccessSize) / 16;

        cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(frag, smem_ptr);
    }

};  // class SwizzleTileLoader<SmemDimM_, 128, NumThreads_>

} // namespace threadblock
} // namespace gemm
} // namespace mickey
