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

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Loader for blockwise quantization scales
template <
    typename QuantBlocking_,  ///! Shape of the quant block (concept: MatrixShape)
    typename WarpShape_,      ///! Shape of the warp tile (concept: GemmShape kM ignored)
    typename ElementT_ = cutlass::half_t,  ///! Data type of the scales and dequantized B
    bool DebugPrint = false>
struct QuantBScaleLoader {
  //
  // Type definitions
  //
  using QuantBlocking = QuantBlocking_;
  using WarpShape = WarpShape_;
  using ElementT = ElementT_;

  static_assert((QuantBlocking::kRow == 1) || (QuantBlocking::kColumn == 1),
                "Only row-wise or column-wise quantization is supported");
  static_assert((WarpShape::kN % 16) == 0 && (WarpShape::kK % 16) == 0,
                "Warp tile size must be multiple of 16x16, the unit of packed weights.");
  static_assert(sizeof(ElementT) == 2, "Quantization only supports 16-bit float types");

  //
  // Layout of quantization scale/offset:
  // Quantization block either comes from a single row or a single column
  //   using QuantBlocking = typename std::conditional_t<column_wise_blocking,
  //              cutlass::MatrixShape<block_size, 1>,
  //              cutlass::MatrixShape<1, block_size>>;
  //
  // Column-wise blocking --> kColumn == 1, every column has its own
  // scale/offset, there are far less rows than columns in a warp tile.
  // So we use row-major layout to maximize continuous memory access in
  // a warp. Conversely, row-wise blocking uses column-major layout.
  //
  // Warp thread layout: As dictated by 16b tensor core layout, 32
  // threads in a warp is divided int 8 groups of 4 threads, each group
  // is responsible for a column, and each thread is responsible for 2
  // rows, forming a 8x8 tile.
  //

  // Number of continuous elements of scale/offset that a warp need to load
  static constexpr int kMetaFragSize = (QuantBlocking::kColumn == 1)
        ? WarpShape::kN / 8  // column wise quant, every column has its own scale/offset
        : (WarpShape::kK / 8) * 2;  // row wise quant, every row has its own scale/offset

  static constexpr int kMetaChunkCount = (QuantBlocking::kColumn == 1)
        ? div_up(WarpShape::kK, QuantBlocking::kRow)
        : div_up(WarpShape::kN, QuantBlocking::kColumn);

  using FragmentScales = cutlass::Array<ElementT, kMetaFragSize * kMetaChunkCount>;
  // using FragmentOffsets = cutlass::Array<uint8_t, kMetaFragSize * kMetaChunkCount>;

  static constexpr int kBTiles = (WarpShape::kK / 8) * (WarpShape::kN / 8);
  using FragmentPackedB = cutlass::Array<unsigned, kBTiles / 4>;

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
    if constexpr (QuantBlocking::kColumn == 1) {
      return reinterpret_cast<ElementT const*>(
          reinterpret_cast<uint8_t const*>(ptr_scales) + k * scales_byte_stride + n * sizeof(ElementT));
    } else {
      return reinterpret_cast<ElementT const*>(
          reinterpret_cast<uint8_t const*>(ptr_scales) + n * scales_byte_stride + k * sizeof(ElementT));
    }
  }

  CUTLASS_DEVICE
  static void expand_int4_to_int16(unsigned src, cutlass::Array<int16_t, 8>& dst) {
    unsigned* dst_ptr = reinterpret_cast<unsigned*>(dst.data());
    dst_ptr[0] = src & 0x000f000f;
    dst_ptr[1] = (src >> 4) & 0x000f000f;
    dst_ptr[2] = (src >> 8) & 0x000f000f;
    dst_ptr[3] = (src >> 12) & 0x000f000f;
  }

  /// Initializes the scale loader, pointing to the start of the scales tensor
  CUTLASS_DEVICE
  QuantBScaleLoader(
      int lane_idx,
      void const *ptr_scales,
      int scales_byte_stride,
      int start_n,
      int end_n)
      : lane_b_k_offset((lane_idx % 4) * 2),
        lane_b_n_offset(lane_idx / 4),
        n_cnt(div_up(end_n - start_n, QuantBlocking::kColumn)),
        scales_p(get_scales_p(ptr_scales, scales_byte_stride, 0, (start_n + lane_b_n_offset) / QuantBlocking::kColumn)),
        scales_stride(scales_byte_stride / sizeof(ElementT))
  {
    assert(ptr_scales != nullptr);
    assert(scales_byte_stride > 0 && (scales_byte_stride % 16) == 0);
    if constexpr (QuantBlocking::kColumn == 1)
    {
      assert(scales_stride >= end_n);
    }
  }

  /// Loads [start_k, end_k) x [start_n, end_n) scales from global memory to fragment
  /// [start_n, end_n) was specified in the constructor
  CUTLASS_DEVICE
  void load(FragmentScales &frag_scales, int start_k, int end_k) const {
    if (end_k <= start_k) {
      return;
    }

    if constexpr (QuantBlocking::kColumn == 1) {
      // Column-wise quantization, every column has its own scale/offset
      const int meta_k = (start_k + lane_b_k_offset) / QuantBlocking::kRow;
      const ElementT* scales_ptr = scales_p + meta_k * scales_stride;;
      const int k_loads = div_up(end_k - start_k, QuantBlocking::kRow);

      for (int k_blk_idx = 0; k_blk_idx < kMetaChunkCount; ++k_blk_idx) {
        // k-dimension loop
        for (int n_tile_idx = 0; n_tile_idx < kMetaFragSize; ++n_tile_idx) {
          // n-dimension loop
          if ((k_blk_idx < k_loads) && ((n_tile_idx * 8) < n_cnt)) {
            frag_scales[k_blk_idx * kMetaFragSize + n_tile_idx] = scales_ptr[n_tile_idx * 8];
          } else {
            frag_scales[k_blk_idx * kMetaFragSize + n_tile_idx] = ElementT(0);
          }
        }
        scales_ptr += scales_stride;
      }
    } else {
      assert(scales_stride >= end_k);

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
      if ((end_k - start_k) % 16 != 0) {
        if (lane_b_k_offset == 0 && lane_b_n_offset == 0) {
          printf("k-dimension must be multiple of 16\n");
        }
        assert(false);
      }

      for (int n_blk_idx = 0; n_blk_idx < kMetaChunkCount; ++n_blk_idx) {
        // n-dimension loop
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
  }

  using FragmentB = cutlass::Array<ElementT, 2 * (WarpShape::kN / 8) * 2>;

  /// Dequantize a block of (16, WarpShape::kN) packed int4 weights to 16b float.
  /// This block has (WarpShape::kN / 8) * 2 tiles, each tile has 2 elements per thread,
  /// thus the FragmentB has (WarpShape::kN / 8) * 2 * 2 elements.
  CUTLASS_DEVICE
  void dequant_k16(const int k_offset, FragmentPackedB const &frag_pack_b, FragmentScales const &frag_scales, FragmentB &frag_b) const {
  #ifndef NDEBUG
    if ((k_offset % 16) != 0) {
      if (lane_b_k_offset == 0 && lane_b_n_offset == 0) {
        printf("k_offset must be multiple of 16\n");
      }
      assert(false);
    }
  #endif

    int b_idx = (k_offset >> 4) * (WarpShape::kN / 16);
    ElementT* fb_ptr = frag_b.data();
    if (QuantBlocking::kColumn == 1) {
      // Column-wise quantization, every column has its own scale/offset
      static_assert(QuantBlocking::kColumn > 1 || QuantBlocking::kRow >= 16);
      int meta_k = k_offset / QuantBlocking::kRow;
      for (int nn = 0; nn < WarpShape::kN / 8; nn += 2, ++b_idx, fb_ptr += 8) {
        ElementT scale = frag_scales[meta_k * kMetaFragSize + nn];
        ElementT scale2 = frag_scales[meta_k * kMetaFragSize + nn + 1];

        cutlass::Array<int16_t, 8> ws;
        expand_int4_to_int16(frag_pack_b[b_idx], ws);
        fb_ptr[0] = ElementT(ws[0] - 8) * scale;
        fb_ptr[1] = ElementT(ws[1] - 8) * scale;
        fb_ptr[2] = ElementT(ws[2] - 8) * scale;
        fb_ptr[3] = ElementT(ws[3] - 8) * scale;
        fb_ptr[4] = ElementT(ws[4] - 8) * scale2;
        fb_ptr[5] = ElementT(ws[5] - 8) * scale2;
        fb_ptr[6] = ElementT(ws[6] - 8) * scale2;
        fb_ptr[7] = ElementT(ws[7] - 8) * scale2;

        if constexpr (DebugPrint) {
          const int lane_id = threadIdx.x % 32;
          const char* const format = ((lane_id % 4) == 3) ? "%f=%2dx%f, %f=%2dx%f\n" : "%f=%2dx%f, %f=%2dx%f, ";
          printf(format, float(fb_ptr[0]), int(ws[0]), float(scale),
                float(fb_ptr[1]), int(ws[1]), float(scale));
          if (lane_id == 31) {
            printf("\n");
          }
          printf(format, float(fb_ptr[2]), int(ws[2]), float(scale),
                float(fb_ptr[3]), int(ws[3]), float(scale));
          if (lane_id == 31) {
            printf("\n");
          }
          printf(format, float(fb_ptr[4]), int(ws[4]), float(scale2),
                float(fb_ptr[5]), int(ws[5]), float(scale2));
          if (lane_id == 31) {
            printf("\n");
          }
          printf(format, float(fb_ptr[6]), int(ws[6]), float(scale2),
                float(fb_ptr[7]), int(ws[7]), float(scale2));
          if (lane_id == 31) {
            printf("\n");
          }
        }
      }
    } else {
      // Row-wise quantization, every row has its own scale/offset
      ElementT const* scales = nullptr;
      for (int nn = 0; nn < WarpShape::kN; nn += 16, ++b_idx, fb_ptr += 8) {
        if (nn % QuantBlocking::kColumn == 0) {
          int meta_n = nn / QuantBlocking::kColumn;
          scales = frag_scales.data() + (meta_n * kMetaFragSize + k_offset / 4); // k_offset / 16 * 4
        }

        cutlass::Array<int16_t, 8> ws;
        expand_int4_to_int16(frag_pack_b[b_idx], ws);
        fb_ptr[0] = ElementT(ws[0] - 8) * scales[0];
        fb_ptr[1] = ElementT(ws[1] - 8) * scales[1];
        fb_ptr[2] = ElementT(ws[2] - 8) * scales[2];
        fb_ptr[3] = ElementT(ws[3] - 8) * scales[3];
        fb_ptr[4] = ElementT(ws[4] - 8) * scales[0];
        fb_ptr[5] = ElementT(ws[5] - 8) * scales[1];
        fb_ptr[6] = ElementT(ws[6] - 8) * scales[2];
        fb_ptr[7] = ElementT(ws[7] - 8) * scales[3];

        if constexpr (DebugPrint) {
          const int lane_id = threadIdx.x % 32;
          const char* const format = ((lane_id % 4) == 3) ? "%f=%2dx%f, %f=%2dx%f\n" : "%f=%2dx%f, %f=%2dx%f, ";
          printf(format, float(fb_ptr[0]), int16_t(ws[0]), float(scales[0]),
                float(fb_ptr[1]), int16_t(ws[1]), float(scales[1]));
          if (lane_id == 31) {
            printf("\n");
          }
          printf(format, float(fb_ptr[2]), int16_t(ws[2]), float(scales[2]),
                float(fb_ptr[3]), int16_t(ws[3]), float(scales[3]));
          if (lane_id == 31) {
            printf("\n");
          }
          printf(format, float(fb_ptr[4]), int16_t(ws[4]), float(scales[0]),
                float(fb_ptr[5]), int16_t(ws[5]), float(scales[1]));
          if (lane_id == 31) {
            printf("\n");
          }
          printf(format, float(fb_ptr[6]), int16_t(ws[6]), float(scales[2]),
                float(fb_ptr[7]), int16_t(ws[7]), float(scales[3]));
          if (lane_id == 31) {
            printf("\n");
          }
        }
      }
    }
  }

};

}  // namespace warp
}  // namespace gemm
}  // namespace mickey
