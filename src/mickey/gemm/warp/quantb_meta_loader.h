/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file warp/quantb_meta_loader.h
 * @brief Load quantization scales and offsets from global memory to fragments.
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

namespace detail {

/**
 * @brief Convert 4b weights to fp16 using bits operations.
*/
CUTLASS_DEVICE
void weights2Half([[maybe_unused]] uint32_t const weights,
                  cutlass::Array<cutlass::half_t, 8>& dest)
{
  // 4b weights are arranged as [0, 2, 4, 6, 1, 3, 5, 7], so that adjacent
  // weights are in adjacent 16b half words.
  //   w & 0x000f000f --> take out element 0, 1
  //   w & 0x00f000f0 --> take out element 2, 3
  //   (w >> 8) & 0x000f000f --> take out element 4, 5
  //   (w >> 8) & 0x00f000f0 --> take out element 6, 7
  //
  // For element 0, 1, 4, 5, we have 0x000?000?, set the high bits
  // to 0x6400, essentially we set the exponent bits to 25, effective
  // exp = 25 - 15 = 10, with explicity hight bit, the value is
  //   2^10 + q_w.
  //
  // Similarly for element 2, 3, 6, 7, we have 0x00?000?, set the
  // high bits to 0x5400, essentially we set the exponent bits to 21,
  // effective exp = 21 - 15 = 6, with explicity hight bit, the value
  // is 2^6 + q_w.
  //
  // 1.125 instruction per weight, 9 instructions in total.

  uint32_t*      b32s   = reinterpret_cast<uint32_t*>(dest.data());
  half2*         pairs  = reinterpret_cast<half2*>(dest.data());
  const uint32_t high_8s = weights >> 8;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 500))
  asm volatile(
    "  lop3.b32      %0, %4, 0x000f000f, 0x64006400, 0xea;\n"
    "  lop3.b32      %1, %4, 0x00f000f0, 0x54005400, 0xea;\n"
    "  lop3.b32      %2, %5, 0x000f000f, 0x64006400, 0xea;\n"
    "  lop3.b32      %3, %5, 0x00f000f0, 0x54005400, 0xea;\n"
    : "=r"(b32s[0]), "=r"(b32s[1]), "=r"(b32s[2]), "=r"(b32s[3])
    : "r"(weights), "r"(high_8s));
#else
  assert(false);
#endif

  constexpr __half_raw kKilo{0x6400};
  constexpr half2 onek(kKilo, kKilo);
  constexpr __half_raw k64{0x5400};
  constexpr half2 sixtyfour{k64, k64};

  pairs[0] = __hsub2(pairs[0], onek);
  pairs[1] = __hsub2(pairs[1], sixtyfour);
  pairs[2] = __hsub2(pairs[2], onek);
  pairs[3] = __hsub2(pairs[3], sixtyfour);
}

template <int N>
CUTLASS_DEVICE
void compute_addon(cutlass::Array<cutlass::half_t, N> const &frag_scales,
                   cutlass::Array<cutlass::half_t, N> &frag_addon) {
  static_assert(N % 2 == 0, "N must be even");
  //half -8.0: c800
  constexpr __half_raw kMinus8{0xc800};
  constexpr half2 mm8(kMinus8, kMinus8);
  const half2* scales_pair = reinterpret_cast<half2 const*>(frag_scales.data());
  half2* addon_pair = reinterpret_cast<half2*>(frag_addon.data());

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < frag_scales.size() / 2; ++i) {
    addon_pair[i] = __hmul2_rn(scales_pair[i], mm8);
  }
}

}  // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Loader for blockwise quantization scales
template <
    typename QuantBlocking_,  ///! Shape of the quant block (concept: MatrixShape)
    typename WarpShape_,      ///! Shape of the warp tile (concept: GemmShape kM ignored)
    typename ElementT_ = cutlass::half_t,  ///! Data type of the scales and dequantized B
    bool DebugPrint = false>
struct QuantBScaleLoader;


/// Specialization for column-wise quantization, i.e. QuantBlocking::kColumn == 1
template <
    int block_size_,
    typename WarpShape_,
    typename ElementT_,
    bool DebugPrint>
struct QuantBScaleLoader<cutlass::MatrixShape<block_size_, 1>, WarpShape_, ElementT_, DebugPrint> {
  //
  // Type definitions
  //
  using QuantBlocking = cutlass::MatrixShape<block_size_, 1>;
  using WarpShape = WarpShape_;
  using ElementT = ElementT_;

  static_assert((QuantBlocking::kColumn == 1));
  static_assert((WarpShape::kN % 16) == 0 && (WarpShape::kK % 16) == 0,
                "Warp tile size must be multiple of 16x16, the unit of packed weights.");
  static_assert(sizeof(ElementT) == 2, "Quantization only supports 16-bit float types");

  //
  // Column-wise blocking --> kColumn == 1, every column has its own
  // scale/offset, there are far less rows than columns in a warp tile.
  // So we use row-major layout to maximize continuous memory access in
  // a warp.
  //
  // Warp thread layout: As dictated by 16b tensor core layout, 32
  // threads in a warp is divided int 8 groups of 4 threads, each group
  // is responsible for a column, and each thread is responsible for 2
  // rows, forming a 8x8 tile.
  //

  // Number of continuous elements of scale/offset that a warp need to load.
  static constexpr int kMetaFragSize = WarpShape::kN / 8;
  static constexpr int kMetaChunkCount = div_up(WarpShape::kK, QuantBlocking::kRow);

  using FragmentScales = cutlass::Array<ElementT, kMetaFragSize * kMetaChunkCount>;
  // using FragmentOffsets = cutlass::Array<uint8_t, kMetaFragSize * kMetaChunkCount>;

  //
  // Data members
  //
  const int lane_b_k_offset;
  const int lane_b_n_offset;
  const int n_cnt;

  const ElementT * const scales_p;
  const int scales_stride;

  //
  // Methods
  //
  CUTLASS_DEVICE
  static const ElementT* get_scales_p(const void* ptr_scales, int scales_byte_stride, int k, int n) {
    return reinterpret_cast<ElementT const*>(
        reinterpret_cast<uint8_t const*>(ptr_scales) + k * scales_byte_stride + n * sizeof(ElementT));
  }

  /// Initializes the scale loader, pointing to the start of the scales tensor
  CUTLASS_DEVICE
  QuantBScaleLoader(
      int lane_idx,
      void const *ptr_scales,
      int scales_byte_stride,
      int start_n,
      int end_n)
      : lane_b_k_offset(mod_power2<4>(lane_idx) * 2),
        lane_b_n_offset(lane_idx / 4),
        n_cnt(end_n - start_n),
        scales_p(get_scales_p(ptr_scales, scales_byte_stride, 0, (start_n + lane_b_n_offset))),
        scales_stride(scales_byte_stride / sizeof(ElementT))
  {
    assert(ptr_scales != nullptr);
    assert(scales_byte_stride > 0 && mod_power2<16>(scales_byte_stride) == 0);
    assert(scales_stride >= end_n);
  }

  /// Loads [start_k, end_k) x [start_n, end_n) scales from global memory to fragment
  /// [start_n, end_n) was specified in the constructor
  CUTLASS_DEVICE
  void load(FragmentScales &frag_scales, int start_k, int end_k) const {
    if (end_k <= start_k) {
      return;
    }

    // Column-wise quantization, every column has its own scale/offset
    const int meta_k = (start_k + lane_b_k_offset) / QuantBlocking::kRow;
    const ElementT* scales_ptr = scales_p + meta_k * scales_stride;;
    const int k_loads = div_up(end_k - start_k, QuantBlocking::kRow);

    CUTLASS_PRAGMA_UNROLL
    for (int k_blk_idx = 0; k_blk_idx < kMetaChunkCount; ++k_blk_idx) {
      // k-dimension loop
      CUTLASS_PRAGMA_UNROLL
      for (int n_tile_idx = 0; n_tile_idx < (WarpShape::kN / 8); ++n_tile_idx) {
        // n-dimension loop
        if ((k_blk_idx < k_loads) && ((n_tile_idx * 8) < n_cnt)) {
          frag_scales[k_blk_idx * kMetaFragSize + n_tile_idx] = scales_ptr[n_tile_idx * 8];
        } else {
          frag_scales[k_blk_idx * kMetaFragSize + n_tile_idx] = ElementT(0);
        }
      }
      scales_ptr += scales_stride;
    }
  }

  /**
   * @brief Compute an addon value for dequantization using fma instructions.
   * 
   * Dequantization forumular is:
   *     f_weight = scale * (q_weight - 8)
   * To use fma instruction, we need to compute:
   *    addon = scale * (-8)
   *    f_weight = scale * q_weight + addon
   */
  CUTLASS_DEVICE
  void process(FragmentScales const &frag_scales, FragmentScales &frag_addon) const {
    detail::compute_addon(frag_scales, frag_addon);
  }

  using FragmentB = cutlass::Array<ElementT, 2 * (WarpShape::kN / 8) * 2>;

  /// Dequantize a block of (16, WarpShape::kN) packed int4 weights to 16b float.
  /// This block has (WarpShape::kN / 8) * 2 tiles, each tile has 2 elements per thread,
  /// thus the FragmentB has (WarpShape::kN / 8) * 2 * 2 elements.

  template <int PackedBSize>
  CUTLASS_DEVICE
  void dequant_k16(
      const int k_offset,
      cutlass::Array<unsigned, PackedBSize> const &frag_pack_b,
      FragmentScales const &frag_scales,
      FragmentScales const &frag_addon,
      FragmentB &frag_b) const {
#ifndef NDEBUG
    if ((k_offset % 16) != 0) {
      if (lane_b_k_offset == 0 && lane_b_n_offset == 0) {
        printf("k_offset must be multiple of 16\n");
      }
      assert(false);
    }
#endif

    // Each 32b number in packed B represent a 16x16 tile
    constexpr int kPackedBNTiles = WarpShape::kN / 16;
    constexpr int kPackedBKStride = PackedBSize / kPackedBNTiles;
    static_assert(kPackedBKStride * kPackedBNTiles == PackedBSize);

    // We are processing 16xWarpShape::kN weights at a time, assuming each column has
    // only one scale/offset, so the block size cannot be smaller than 16.
    static_assert(QuantBlocking::kRow >= 16);

    const int meta_k = k_offset / QuantBlocking::kRow;
    int b_idx = mod_power2<kPackedBKStride>(k_offset >> 4) * kPackedBNTiles;
    half2* fb_pair = reinterpret_cast<half2*>(frag_b.data());
    half const* scales = reinterpret_cast<half const*>(frag_scales.data() + meta_k * kMetaFragSize);
    half const* addon = reinterpret_cast<half const*>(frag_addon.data() + meta_k * kMetaFragSize);

    // Column-wise quantization, every column has its own scale/offset
    CUTLASS_PRAGMA_UNROLL
    for (int nn = 0; nn < (WarpShape::kN / 8); nn += 2, ++b_idx, fb_pair += 4) {
      half2 scale_pair = __half2half2(scales[nn]);
      half2 scale_pair1 = __half2half2(scales[nn + 1]);
      half2 addon_pair = __half2half2(addon[nn]);
      half2 addon_pair1 = __half2half2(addon[nn + 1]);

      cutlass::Array<ElementT, 8> ws;
      detail::weights2Half(frag_pack_b[b_idx], ws);
      const half2* const weight_pair = reinterpret_cast<half2 const*>(ws.data());

      fb_pair[0] = __hfma2(scale_pair, weight_pair[0], addon_pair);
      fb_pair[1] = __hfma2(scale_pair, weight_pair[1], addon_pair);
      fb_pair[2] = __hfma2(scale_pair1, weight_pair[2], addon_pair1);
      fb_pair[3] = __hfma2(scale_pair1, weight_pair[3], addon_pair1);

      if constexpr (DebugPrint) {
        const int lane_id = threadIdx.x % 32;
        const char* const format = ((lane_id % 4) == 3) ? "%f=%fx%f, %f=%fx%f\n" : "%f=%fx%f, %f=%fx%f, ";
        printf(format, float(fb_pair[0].x), float(weight_pair[0].x), float(scale_pair.x),
               float(fb_pair[0].y), float(weight_pair[0].y), float(scale_pair.y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[1].x), float(weight_pair[1].x), float(scale_pair.x),
               float(fb_pair[1].y), float(weight_pair[1].y), float(scale_pair.y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[2].x), float(weight_pair[2].x), float(scale_pair1.x),
               float(fb_pair[2].y), float(weight_pair[2].y), float(scale_pair1.y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[3].x), float(weight_pair[3].x), float(scale_pair1.x),
               float(fb_pair[3].y), float(weight_pair[3].y), float(scale_pair1.y));
        if (lane_id == 31) {
          printf("\n");
        }
      }
    }
  }

};


/// Specialization for row-wise quantization, i.e. QuantBlocking::kRow == 1
template <
    int block_size_,
    typename WarpShape_,
    typename ElementT_,
    bool DebugPrint>
struct QuantBScaleLoader<cutlass::MatrixShape<1, block_size_>, WarpShape_, ElementT_, DebugPrint> {
  //
  // Type definitions
  //
  using QuantBlocking = cutlass::MatrixShape<1, block_size_>;
  using WarpShape = WarpShape_;
  using ElementT = ElementT_;

  static_assert(QuantBlocking::kRow == 1);
  static_assert((WarpShape::kN % 16) == 0 && (WarpShape::kK % 16) == 0,
                "Warp tile size must be multiple of 16x16, the unit of packed weights.");
  static_assert(sizeof(ElementT) == 2, "Quantization only supports 16-bit float types");

  //
  // Row-wise blocking --> kRow == 1, every row has its own
  // scale/offset, there are far less columns than rows in a warp tile.
  // So we use column-major layout to maximize continuous memory access in
  // a warp.
  //

  // Number of continuous elements of scale/offset that a warp need to load
  static constexpr int kMetaFragSize = (WarpShape::kK / 8) * 2;  // row wise quant, every row has its own scale/offset
  static constexpr int kMetaChunkCount = div_up(WarpShape::kN, QuantBlocking::kColumn);

  using FragmentScales = cutlass::Array<ElementT, kMetaFragSize * kMetaChunkCount>;
  // using FragmentOffsets = cutlass::Array<uint8_t, kMetaFragSize * kMetaChunkCount>;

  //
  // Data members
  //
  const int lane_b_k_offset;
  const int lane_b_n_offset;
  const int n_cnt;

  const ElementT * const scales_p;
  const int scales_stride;

  //
  // Methods
  //
  CUTLASS_DEVICE
  static const ElementT* get_scales_p(const void* ptr_scales, int scales_byte_stride, int k, int n) {
    return reinterpret_cast<ElementT const*>(
        reinterpret_cast<uint8_t const*>(ptr_scales) + n * scales_byte_stride + k * sizeof(ElementT));
  }

  /// Initializes the scale loader, pointing to the start of the scales tensor
  CUTLASS_DEVICE
  QuantBScaleLoader(
      int lane_idx,
      void const *ptr_scales,
      int scales_byte_stride,
      int start_n,
      int end_n)
      : lane_b_k_offset(mod_power2<4>(lane_idx) * 2),
        lane_b_n_offset(lane_idx / 4),
        n_cnt(div_up(end_n - start_n, QuantBlocking::kColumn)),
        scales_p(get_scales_p(ptr_scales, scales_byte_stride, 0, (start_n + lane_b_n_offset) / QuantBlocking::kColumn)),
        scales_stride(scales_byte_stride / sizeof(ElementT))
  {
    assert(ptr_scales != nullptr);
    assert(scales_byte_stride > 0 && mod_power2<16>(scales_byte_stride) == 0);
  }

  /// Loads [start_k, end_k) x [start_n, end_n) scales from global memory to fragment
  /// [start_n, end_n) was specified in the constructor
  CUTLASS_DEVICE
  void load(FragmentScales &frag_scales, int start_k, int end_k) const {
    assert(scales_stride >= end_k);
    if (end_k <= start_k) {
      return;
    }

    // Row-wise quantization, every row has its own scale/offset, elements have been rearraged
    // such that we can load two tile at a time.
    // T0        T0
    // T1        T0
    // T2        T1
    // T3   =>   T1
    // T0        T2
    // T1        T2
    // T2        T3
    // T3        T3
    const int meta_k = start_k + lane_b_k_offset * 2;
    const ElementT* scales_ptr = scales_p + meta_k;
    const int k_loads = (end_k - start_k) / 16;
    if (mod_power2<16>(end_k - start_k) != 0) {
      if (lane_b_k_offset == 0 && lane_b_n_offset == 0) {
        printf("k-dimension must be multiple of 16\n");
      }
      assert(false);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int n_blk_idx = 0; n_blk_idx < kMetaChunkCount; ++n_blk_idx) {
      // n-dimension loop
      CUTLASS_PRAGMA_UNROLL
      for (int k_tile_idx = 0; k_tile_idx < (WarpShape::kK / 16); ++k_tile_idx) {
        // k-dimension loop
        static_assert((kMetaFragSize / 4) == (WarpShape::kK / 16));
        const int dst_idx = n_blk_idx * kMetaFragSize + k_tile_idx * 4;
        const int src_idx = k_tile_idx * 16;
        if ((n_blk_idx < n_cnt) && (k_tile_idx < k_loads)) {
          frag_scales[dst_idx] = scales_ptr[src_idx];
          frag_scales[dst_idx + 1] = scales_ptr[src_idx + 1];
          frag_scales[dst_idx + 2] = scales_ptr[src_idx + 2];
          frag_scales[dst_idx + 3] = scales_ptr[src_idx + 3];
        } else {
          frag_scales[dst_idx] = ElementT(0);
          frag_scales[dst_idx + 1] = ElementT(0);
          frag_scales[dst_idx + 2] = ElementT(0);
          frag_scales[dst_idx + 3] = ElementT(0);
        }
      }
      scales_ptr += scales_stride;
    }
  }

  /**
   * @brief Compute an addon value for dequantization using fma instructions.
   * 
   * Dequantization forumular is:
   *     f_weight = scale * (q_weight - 8)
   * To use fma instruction, we need to compute:
   *    addon = scale * (-8)
   *    f_weight = scale * q_weight + addon
   */
  CUTLASS_DEVICE
  void process(FragmentScales const &frag_scales, FragmentScales &frag_addon) const {
    detail::compute_addon(frag_scales, frag_addon);
  }

  using FragmentB = cutlass::Array<ElementT, 2 * (WarpShape::kN / 8) * 2>;

  /// Dequantize a block of (16, WarpShape::kN) packed int4 weights to 16b float.
  /// This block has (WarpShape::kN / 8) * 2 tiles, each tile has 2 elements per thread,
  /// thus the FragmentB has (WarpShape::kN / 8) * 2 * 2 elements.
  template<int PackedBSize>
  CUTLASS_DEVICE
  void dequant_k16(
      const int k_offset,
      cutlass::Array<unsigned, PackedBSize> const &frag_pack_b,
      FragmentScales const &frag_scales,
      FragmentScales const &frag_addon,
      FragmentB &frag_b) const {
  #ifndef NDEBUG
    if ((k_offset % 16) != 0) {
      if (lane_b_k_offset == 0 && lane_b_n_offset == 0) {
        printf("k_offset must be multiple of 16\n");
      }
      assert(false);
    }
  #endif

    // Each 32b number in packed B represent a 16x16 tile
    constexpr int kPackedBNTiles = WarpShape::kN / 16;
    constexpr int kPackedBKStride = PackedBSize / kPackedBNTiles;
    static_assert(kPackedBKStride * kPackedBNTiles == PackedBSize);

    // Row-wise quantization, every row has its own scale/offset
    int b_idx = mod_power2<kPackedBKStride>(k_offset >> 4) * kPackedBNTiles;
    half2* fb_pair = reinterpret_cast<half2*>(frag_b.data());
    half2 const* scale_pair = nullptr;
    half2 const* addon_pair = nullptr;

    CUTLASS_PRAGMA_UNROLL
    for (int nn = 0; nn < WarpShape::kN; nn += 16, ++b_idx, fb_pair += 4) {
      if (mod_power2<QuantBlocking::kColumn>(nn) == 0) {
        int meta_n = nn / QuantBlocking::kColumn;
        scale_pair = reinterpret_cast<half2 const*>(frag_scales.data() + (meta_n * kMetaFragSize + k_offset / 4)); // k_offset / 16 * 4
        addon_pair = reinterpret_cast<half2 const*>(frag_addon.data() + (meta_n * kMetaFragSize + k_offset / 4)); // k_offset / 16 * 4
      }

      cutlass::Array<ElementT, 8> ws;
      detail::weights2Half(frag_pack_b[b_idx], ws);
      const half2* const weight_pair = reinterpret_cast<half2 const*>(ws.data());

      fb_pair[0] = __hfma2(scale_pair[0], weight_pair[0], addon_pair[0]);
      fb_pair[1] = __hfma2(scale_pair[1], weight_pair[1], addon_pair[1]);
      fb_pair[2] = __hfma2(scale_pair[0], weight_pair[2], addon_pair[0]);
      fb_pair[3] = __hfma2(scale_pair[1], weight_pair[3], addon_pair[1]);

      if constexpr (DebugPrint) {
        const int lane_id = threadIdx.x % 32;
        const char* const format = ((lane_id % 4) == 3) ? "%f=%fx%f, %f=%fx%f\n" : "%f=%fx%f, %f=%fx%f, ";
        printf(format, float(fb_pair[0].x), float(weight_pair[0].x), float(scale_pair[0].x),
               float(fb_pair[0].y), float(weight_pair[0].y), float(scale_pair[0].y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[1].x), float(weight_pair[1].x), float(scale_pair[1].x),
               float(fb_pair[1].y), float(weight_pair[1].y), float(scale_pair[1].y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[2].x), float(weight_pair[2].x), float(scale_pair[0].x),
               float(fb_pair[2].y), float(weight_pair[2].y), float(scale_pair[0].y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[3].x), float(weight_pair[3].x), float(scale_pair[1].x),
               float(fb_pair[3].y), float(weight_pair[3].y), float(scale_pair[1].y));
        if (lane_id == 31) {
          printf("\n");
        }
      }
    }
  }

};

}  // namespace warp
}  // namespace gemm
}  // namespace mickey
