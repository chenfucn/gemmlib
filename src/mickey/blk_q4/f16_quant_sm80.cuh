/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    blk_q4/f16_quant_sm80.cuh
 *
 * Abstract:
 *    Prepack weights and quantization parameters (scales and offsets) for
 *    GEMM, where activations are fp16 or bf16, and weights are block-wise
 *    4b quantized values, specifically for Ampere GPUs.
 *
 *    Prepacking enables faster loading of weights and quantization parameters
 *    into tensor cores, and faster dequantization of weights.
 *
 *    Only supports fp16 for now, bfloat16 support will be added later.
 */

#pragma once

#include <cuda_fp16.h>
#include "matrix_layout.h"

namespace mickey {

template <
    typename ElementT,
    int block_size,
    int qbits,
    bool Columnwise,
    bool ExtraBoundsCheck = false>
__global__
void block_quantize_kernel(
    uint8_t* dst,
    ElementT* scales,
    uint8_t* zero_points,
    const ElementT* src,
    int rows,
    int columns,
    int leadingDimension);

template <
    typename ElementT,
    int block_size,
    int qbits,
    bool Columnwise,
    bool ExtraBoundsCheck = false>
__global__
void prepack_weights_kernel(
    int rows,
    int columns,
    const uint8_t* weights,      // <- int4 weights, column major
    uint8_t* weights_prepacked,  // <- int4 prepacked weights tensor, same size buffer
    size_t buf_size);

template <
    typename ElementT,
    int block_size,
    int qbits,
    bool Columnwise,
    bool ExtraBoundsCheck = false>
__global__
void prepack_scales_kernel(
    int meta_rows,
    int meta_columns,
    const ElementT* scales,     // <- quant scales, column major layout
    ElementT* scales_prepacked, // <- quant scales prepacked, same size buffer
    const uint8_t* offsets,     // <- quant offsets, int4, column major layout
    uint8_t* offsets_prepacked, // <- quant offsets prepacked, double size buffer
    size_t buf_size);

template <int qbits>
struct BitsTraits {
    static_assert(qbits <= 8, "Only BitsTraits are for small number of bits!");

    static constexpr int kBits = qbits;
    static constexpr int kMax = (1 << qbits) - 1;
    static constexpr int kMid = 1 << (qbits - 1);
    static constexpr float kMaxFp = static_cast<float>(kMax);

    // number of qbit elements to pack into whole bytes
    static constexpr int kPackSize = (qbits == 8) ? 1 : (qbits == 4) ? 2 : (qbits == 2) ? 4 : 0;
    static_assert(kPackSize != 0, "Packing to whole bytes not supported for this qbits!");
};

/**
 * @brief Rectify min/max from a set of weights, and convert to scale and zero point
 *        for quantization
 * @tparam ScaleT   type of scale, usually floating point of various bits
 * @tparam qbits  number of int bits used for zero point value
 * @param[in]   min
 * @param[in]   max
 * @param[out]  scale
 * @param[out]  zp
 */
template <typename ScaleT, int qbits>
CUTLASS_DEVICE
void
range2scalezp(float min, float max, ScaleT& scale, uint8_t& zp)
{
    constexpr int zp_max = BitsTraits<qbits>::kMax;
    constexpr float zp_max_fp = BitsTraits<qbits>::kMaxFp;

    min = ::min(min, 0.0f);
    max = ::max(max, 0.0f);

    float scale_f = (max - min) / zp_max;

    float zero_point_fp = min;
    if (scale_f != 0.0f) {
        zero_point_fp = 0.f - min / scale_f;
    }

    if (zero_point_fp < 0.0f) {
        zp = 0;
    } else if (zero_point_fp > zp_max_fp) {
        zp = zp_max;
    } else {
        zp = (uint8_t)roundf(zero_point_fp);
    }
    scale = ScaleT(scale_f);
}

template <typename ScaleT, int qbits>
CUTLASS_DEVICE
void
range2scale(float min, float max, ScaleT& scale)
{
    constexpr int mid_v = BitsTraits<qbits>::kMid;
    constexpr float mid_fp = static_cast<float>(-mid_v);

    max = fabsf(max) > fabsf(min) ? max : min;

    scale = ScaleT(max / mid_fp);
};

CUTLASS_DEVICE
float clamp(float x, float a, float b)
{
  return max(a, min(b, x));
}

/**
 * @brief Blockwise quantization methods
 * @tparam ElementT       source data type, fp16
 * @tparam block_size     number of elemenets quantized together
 * @tparam qbits          number of bits in each quantized element
 * @tparam Columnwise     true:  elements in a block come from one single column
 *                        false: elements in a block come from one single row
 */
template <
    typename ElementT,
    int block_size,
    int qbits,
    bool Columnwise,
    bool ExtraBoundsCheck = false>
struct BlockwiseQuantization {
  static_assert(qbits == 4, "Only 4b block quantization is supported!");
  static_assert(sizeof(ElementT) == 2, "Only 16b floating point types are supported!");

  using QuantBlocking =
      std::conditional_t<Columnwise,
                         cutlass::MatrixShape<block_size, 1>,
                         cutlass::MatrixShape<1, block_size>>;

  using ElementW = uint8_t;  // <- Weight is int4, uint8 for two of them
  // We pack 4 weights into one 16b element, so we can leverage cutlass tile iterators
  // for async share memory loading, and minimizing bank conflict during matrix loading
  using ElementWPack = ElementT;
  using LayoutWPack = cutlass::layout::ColumnMajor;  // <- layout of packed weight, must be column major

  // Current Ampere kernel use 8b zero point, need to shrink it to 4b in the future
  using ElementQOffset = uint8_t;

  // Layout of the quantization parameters (scales and zero points)
  // Major on the dimension that has the most parameters per squarish weight block.
  // E.g. for column-wise quantization, a [64, 64] block has [2, 64] parameters,
  // where each row has more data, so we use row major layout so that warp threads
  // can use less load instructions to load more parameters.
  using LayoutQmeta =
      typename std::conditional<Columnwise,
                                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type;

  /**
   * @brief  Get quantized weight tensor dimensions.
   * Actual weight type is int4, we use ElementW = uint8 to avoid possible compilation
   * troubles. Since the layout is column major, we are packing 2 weights in a column
   * into one int8
   */
  CUTLASS_HOST_DEVICE
  static auto get_quant_weights_shape(int rows, int columns) {
    return cutlass::make_Coord(rows / 2, columns);
  }

  CUTLASS_HOST_DEVICE
  static auto get_quant_meta_shape(int rows, int columns) {
    return cutlass::make_Coord(rows / QuantBlocking::kRow, columns / QuantBlocking::kColumn);
  }

  using ThreadBlk = cutlass::MatrixShape<QuantBlocking::kRow * BitsTraits<qbits>::kPackSize, QuantBlocking::kColumn>;

  /**
   * @brief Cuda Kernel Thread Partition for weight tensor quantization
   */
  CUTLASS_HOST_DEVICE
  static int get_num_blks_quant_weights(int rows, int columns) {
    // Thread partitioning
    const auto thrd_row_blks = (rows + ThreadBlk::kRow - 1) / ThreadBlk::kRow;
    const auto thrd_col_blks = (columns + ThreadBlk::kColumn - 1) / ThreadBlk::kColumn;
    const auto total_thrd_blks = thrd_row_blks * thrd_col_blks;
    return total_thrd_blks;
  }

  /**
   * @brief Device code for quantized a single block of weight elements, resulting quantized
   *        and packed data are stored in column major (transposed)
   * @param[out] dst           pointer to the quantized weights, column major: [columns, rows]
   * @param[out] scale         pointer to the scales, column major: [columns/QuantBlk::kColumn, rows/QuantBlk::kRow]
   * @param[out] zero_points   pointer to the zero points, same shape as scale
   * @param[in]  src           pointer to the source matrix, column major: [columns, rows]
   * @param rows
   * @param columns
   * @param leadingDimension   stride of the source matrix, i.e. distance from one column to the next
   */
  CUTLASS_DEVICE
  static void dev_quantize_blk(
      uint8_t* dst,
      ElementT* scales,
      uint8_t* zero_points,
      const ElementT* src,
      int rows,
      int columns,
      int leadingDimension,
      int block_idx)
  {
    // Thread partitioning
    const auto thrd_row_blks = (rows + ThreadBlk::kRow - 1) / ThreadBlk::kRow;
    const auto thrd_col_blks = (columns + ThreadBlk::kColumn - 1) / ThreadBlk::kColumn;
    const auto total_thrd_blks = thrd_row_blks * thrd_col_blks;
    if (block_idx >= total_thrd_blks) {
      return;
    }

    const auto row_blks = (rows + QuantBlocking::kRow - 1) / QuantBlocking::kRow;

    auto q_shape = get_quant_weights_shape(rows, columns);

    const int r_blk_idx = static_cast<int32_t>(block_idx / thrd_col_blks);
    const int c_blk_idx = static_cast<int32_t>(block_idx % thrd_col_blks);

    const int r = r_blk_idx * ThreadBlk::kRow;
    const int c = c_blk_idx * ThreadBlk::kColumn;

    const int r_end = min(r + ThreadBlk::kRow, rows);
    const int c_end = min(c + ThreadBlk::kColumn, columns);

    const int meta_row = r / QuantBlocking::kRow;
    const int meta_col = c / QuantBlocking::kColumn;

    // compute scale and zero point
    for (int kpack = 0; kpack < BitsTraits<qbits>::kPackSize; kpack++) {
      // scan a single block to extract range [min, max]
      float min = __FLT_MAX__; // std::numeric_limits<float>::max();
      float max = -min;
      const int row_start = r + kpack * QuantBlocking::kRow;
      const int row_end = ::min(row_start + QuantBlocking::kRow, r_end);
      for (int i = row_start; i < row_end; ++i) {
          for (int j = c; j < c_end; ++j) {
              const float v = static_cast<float>(src[i + j * leadingDimension]);
              if (v < min) min = v;
              if (v > max) max = v;
          }
      }

      // store scale and zero point at quant parameter matrix position
      if (row_start < row_end) {
          const int32_t meta_idx = meta_col * row_blks + meta_row + kpack;
          if (zero_points == nullptr) {
              range2scale<ElementT, qbits>(min, max, scales[meta_idx]);
          } else {
              range2scalezp<ElementT, qbits>(min, max, scales[meta_idx], zero_points[meta_idx]);
          }
      }
    }

    for (int32_t j = c; j < c_end; ++j) {
      const int32_t meta_c = j / QuantBlocking::kColumn;
      for (int32_t i = r; i < r_end; i += 2) {
        const int32_t meta_r = i / QuantBlocking::kRow;
        const int meta_idx = meta_c * row_blks + meta_r;
        float scale = static_cast<float>(scales[meta_idx]);
        uint8_t zp = zero_points ? zero_points[meta_idx] : 8;
        float reciprocal_scale = scale ? 1.0f / scale : 0.0f;

        const float v0 = static_cast<float>(src[i + j * leadingDimension]);
        const uint8_t vi0 = (uint8_t)clamp(roundf(v0 * reciprocal_scale + zp),
                                                0.0f, BitsTraits<qbits>::kMaxFp);

        uint8_t vi1 = (uint8_t)zp;
        if (i + 1 < r_end) {
          if constexpr (QuantBlocking::kRow == 1) {
              scale = static_cast<float>(scales[meta_idx + 1]);
              reciprocal_scale = scale ? 1.0f / scale : 0.0f;
              zp = zero_points ? zero_points[meta_idx + 1] : 8;
          }
          const float v1 = static_cast<float>(src[(i + 1) + j * leadingDimension]);
          vi1 = (uint8_t)clamp(roundf(v1 * reciprocal_scale + zp), 0.0f,
                                    BitsTraits<qbits>::kMaxFp);
        }

        // !! 4b specific code
        dst[j * q_shape[0] + i / 2] = (vi0 & 0xf) | (vi1 << 4);
      }
    }
  }

  static void block_quantize(
      uint8_t* dst,
      ElementT* scales,
      uint8_t* zero_points,
      const ElementT* src,
      int rows,
      int columns,
      int leadingDimension,
      cudaStream_t stream)
  {
    constexpr int ThreadsPerBlock = 256;
    auto total_thrd_blks = get_num_blks_quant_weights(rows, columns);
    auto grid = (total_thrd_blks + ThreadsPerBlock - 1) / ThreadsPerBlock;
    block_quantize_kernel<ElementT, block_size, qbits, Columnwise, ExtraBoundsCheck>
        <<<grid, ThreadsPerBlock, 0, stream>>>(dst, scales, zero_points, src, rows, columns, leadingDimension);
  }

  /**
   * @brief Prepack weight matrix to facilitate matrix loading, depending on MMA
   * instruction layout.
   *
   * The weight matrix is int4, yet we want to leverage existing fp16/bf16
   * tile loading and MMA layout code in CUTLASS. So we group 4 int4 into 2
   * bytes, pretending it's fp16. This grouping must be done in a way to be
   * easily unpacked into tiles that match the MMA instruction layout.
   * For MMA instruction <16, 8, 16>, each instruction processes 2 8x8 tiles,
   * vertically stacked on the K dimension. And MmaTensorOpMultiplicandTileIterator
   * loads a <InstructionShape::kK, WarpShape::kN> tile.
   *
   * So we stack 2x2 tiles on a 3rd dimeansion, and reshape them in a HWC fashion:
   * T0, T2
   * T1, T3
   * ==>
   * T0[0, 0], T1[0, 0], T2[0, 0], T3[0, 0]
   * T0[1, 0], T1[1, 0], T2[1, 0], T3[1, 0]
   * T0[2, 0], T1[2, 0], T2[2, 0], T3[2, 0]
   * T0[3, 0], T1[3, 0], T2[3, 0], T3[3, 0]
   * ...
   * T0[0, 7], T1[0, 7], T2[0, 7], T3[0, 7]
   * T0[1, 7], T1[1, 7], T2[1, 7], T3[1, 7]
   * T0[2, 7], T1[2, 7], T2[2, 7], T3[2, 7]
   * T0[3, 7], T1[3, 7], T2[3, 7], T3[3, 7]
   *
   * This pack a 8x16 int8 tile into a 16x8 int8 tile, i.e. a 8x8 16b tile
   */
  CUTLASS_DEVICE
  static void dev_prepack_weights(
    int rows,
    int columns,
    const uint8_t* weights,     // <- int4 weights, column major
    uint8_t* weights_prepacked, // <- int4 prepacked weights tensor, same size buffer
    size_t buf_size
  ) {
    // locate the coordinate of the tile in the weight matrix
    int row_dtile = blockIdx.x * blockDim.x + threadIdx.x;
    int col_dtile = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_dtile >= rows / 16 || col_dtile >= columns / 16) {
      return;
    }

    const MatrixRef<uint8_t const, cutlass::layout::ColumnMajor, ExtraBoundsCheck>
        tensor_weight(weights, buf_size, cutlass::make_Coord(rows / 2, columns));
    const MatrixRef<uint8_t, LayoutWPack, ExtraBoundsCheck>
        tensor_weight_prepacked(weights_prepacked, buf_size, cutlass::make_Coord(rows, columns / 2));

    auto t0_base = cutlass::make_Coord(0, 0);
    auto t1_base = cutlass::make_Coord(4, 0);
    auto t2_base = cutlass::make_Coord(0, 8);
    auto t3_base = cutlass::make_Coord(4, 8);

    // below is the loop body of 
    //   for (int col_dtile = 0; col_dtile < columns / 16; ++col_dtile) {
    //     for (int row_dtile = 0; row_dtile < rows / 16; ++row_dtile) {
    //       HERE...
    //     }
    //   }

    // Packing from a 8x16 tile to a 16x8 tile
    auto dtile_base = cutlass::make_Coord(row_dtile * 8, col_dtile * 16);
    auto packed_tile_base = cutlass::make_Coord(row_dtile * 16, col_dtile * 8);
    for (int col = 0; col < 8; ++col) {
      for (int row = 0; row < 4; ++row) {
        auto cord = cutlass::make_Coord(row, col);
        auto packed_cord = packed_tile_base + cutlass::make_Coord(row * 4, col);  // packed tile is 16x8
        uint8_t buf[4];
        buf[0] = tensor_weight.at(dtile_base + t0_base + cord);
        buf[1] = tensor_weight.at(dtile_base + t1_base + cord);
        buf[2] = tensor_weight.at(dtile_base + t2_base + cord);
        buf[3] = tensor_weight.at(dtile_base + t3_base + cord);

        // [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7] so that each pair of adjacent weights
        // are in different b16 register at the same positions. This makes it easier to convert to
        // fp16x2 format in a b32 register

        tensor_weight_prepacked.at(packed_cord) = (buf[0] & 0x0f) | ((buf[1] & 0x0f) << 4);
        tensor_weight_prepacked.at(packed_cord + cutlass::make_Coord(1, 0)) = (buf[2] & 0x0f) | ((buf[3] & 0x0f) << 4);
        tensor_weight_prepacked.at(packed_cord + cutlass::make_Coord(2, 0)) = ((buf[0] & 0xf0) >> 4) | (buf[1] & 0xf0);
        tensor_weight_prepacked.at(packed_cord + cutlass::make_Coord(3, 0)) = ((buf[2] & 0xf0) >> 4) | (buf[3] & 0xf0);
      }
    }
  }

  static std::string prepack_weights(
      int rows,
      int columns,
      const gsl::span<uint8_t const>& weights,     // <- int4 weights, column major
      const gsl::span<uint8_t>& weights_prepacked, // <- int4 prepacked weights tensor, same size buffer
      cudaStream_t stream = 0
  ) {
    if (!((rows % 16) == 0 && (columns % 16) == 0 &&
          (rows % QuantBlocking::kRow) == 0 &&
          (columns % QuantBlocking::kColumn) == 0)) {
      return "Does not support odd number of rows or columns!";
    }

    if (weights.size() != size_t(rows * columns / 2)) {
      return "Weight tensor shape mismatch!";
    }
    if (weights_prepacked.size() != weights.size()) {
      return "Prepacked Weight tensor buffer should be the same size!";
    }

    constexpr int thread_blk_size = 16;
    constexpr int grid_stride = thread_blk_size * 16; // one thread process 16 items, one thread block has 16 threads
    dim3 block_dim(thread_blk_size, thread_blk_size);
    dim3 grid_dim((rows + grid_stride - 1) / grid_stride, (columns + grid_stride - 1) / grid_stride);

    prepack_weights_kernel<ElementT, block_size, qbits, Columnwise, ExtraBoundsCheck>
        <<<grid_dim, block_dim, 0, stream>>>(rows, columns, weights.data(), weights_prepacked.data(), weights.size());

    return std::string();
  }

  /**
   * @brief We rearrange the values of the quantization scale and offset tensors
   * to facilitate faster loading to tensor core, only 16b gemm, and (1,n)
   * block quantization.
   */
  static constexpr bool ShouldRearrangeMeta = sizeof(ElementT) == 2 && QuantBlocking::kRow == 1;

  CUTLASS_DEVICE
  static void dev_prepack_scales(
    int meta_rows,
    int meta_columns,
    const ElementT* scales,     // <- quant scales, column major layout
    ElementT* scales_prepacked, // <- quant scales prepacked, same size buffer
    const uint8_t* offsets,     // <- quant offsets, int4, column major layout
    uint8_t* offsets_prepacked, // <- quant offsets prepacked, double
    size_t buf_size
  ) {
    int row_blk = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_blk >= meta_rows || col >= meta_columns) {
      return;
    }

    auto meta_shape = cutlass::make_Coord(meta_rows, meta_columns);
    const MatrixRef<ElementT const, cutlass::layout::ColumnMajor, ExtraBoundsCheck>
        tensor_scale(scales, buf_size, meta_shape);
    const MatrixRef<ElementT, LayoutQmeta, ExtraBoundsCheck>
        tensor_scale_prepacked(scales_prepacked, buf_size, meta_shape);

    if constexpr (ShouldRearrangeMeta) {
      // In Ampere tensor op, each operand B tile is 8 x 8, in a warp of 32 threads, each thread
      // holds a fragment of the tile containing 2 elements in the k dimension. Most often we use
      // mma instruction shape of 16x8x16, which means 2 B tiles are stacked in the k dimension,
      // as shown below (T stands for thread):
      // T0, T4, T8, T12
      // T1, T5, T9, T13
      // T2, T6, T10, T14
      // T3, T7, T11, T15
      // T0, T4, T8, T12
      // T1, T5, T9, T13
      // T2, T6, T10, T14
      // T3, T7, T11, T15
      //
      // We need to deliver quantization scale and offset elements to the corresponding threads,
      // so we can perform dequantization efficiently. With a column major layout, each thread
      // needs two separate loads for a mma instruction, due to the tile fragment layout shown
      // above. To reduce the number of loads, we rearrange each column as below, so we can use
      // a single load to load fragments for two tiles:
      // T0        T0
      // T1        T0
      // T2        T1
      // T3   =>   T1
      // T0        T2
      // T1        T2
      // T2        T3
      // T3        T3
      for (int thread_id = 0; thread_id < 4; thread_id++) {
        const int dst_idx = row_blk + thread_id * 4;
        const int src_idx = row_blk + thread_id * 2;
        tensor_scale_prepacked.at(dst_idx + 0, col) = tensor_scale.at(src_idx + 0, col);
        tensor_scale_prepacked.at(dst_idx + 1, col) = tensor_scale.at(src_idx + 1, col);
        tensor_scale_prepacked.at(dst_idx + 2, col) = tensor_scale.at(src_idx + 8, col);
        tensor_scale_prepacked.at(dst_idx + 3, col) = tensor_scale.at(src_idx + 9, col);
        if (offsets) {
        }
      }
    } else {
      // Potential transpose if the prepacked layout is different from the original layout
      for (int row = 0; row < min(16, meta_rows - row_blk); ++row) {
        tensor_scale_prepacked.at(row_blk + row, col) = tensor_scale.at(row_blk + row, col);
      }
    }

    if (offsets != nullptr) {
      MatrixRef<uint8_t const, cutlass::layout::ColumnMajor, ExtraBoundsCheck>
          tensor_offset(offsets, buf_size, meta_shape);
      MatrixRef<uint8_t, LayoutQmeta, ExtraBoundsCheck> tensor_offset_prepacked(offsets_prepacked, buf_size, meta_shape);
      if constexpr (ShouldRearrangeMeta) {
        for (int thread_id = 0; thread_id < 4; thread_id++) {
          const int dst_idx = row_blk + thread_id * 4;
          const int src_idx = row_blk + thread_id * 2;

          // [a, b, c, d] => [a, c, b, d] so that adjacent weights are in their own
          // 16b element: [a, x, b, x] and [x, c, x, d], which makes it easier to
          // convert to fp16x2 format in a b32 register
          tensor_offset_prepacked.at(dst_idx + 0, col) = tensor_offset.at(src_idx + 0, col);
          tensor_offset_prepacked.at(dst_idx + 1, col) = tensor_offset.at(src_idx + 8, col);
          tensor_offset_prepacked.at(dst_idx + 2, col) = tensor_offset.at(src_idx + 1, col);
          tensor_offset_prepacked.at(dst_idx + 3, col) = tensor_offset.at(src_idx + 9, col);
        }
      } else {
        // Potential transpose if the prepacked layout is different from the original layout
        for (int row = 0; row < min(16, meta_rows - row_blk); ++row) {
          tensor_offset_prepacked.at(row_blk + row, col) = tensor_offset.at(row_blk + row, col);
        }
      }
    }
  }

  static std::string prepack_quant_meta(
      size_t rows,
      size_t columns,
      const gsl::span<ElementT const>& scales,     // <- quant scales, column major layout
      const gsl::span<ElementT>& scales_prepacked, // <- quant scales prepacked, same size buffer
      const gsl::span<uint8_t const>& offsets,     // <- quant offsets, int4, column major layout
      const gsl::span<uint8_t>& offsets_prepacked, // <- quant offsets prepacked, double size buffer
      cudaStream_t stream = 0
  ) {
    if ((rows % 16) != 0 || (columns % 16) != 0) {
      return "Does not support odd number of rows or columns!";
    }

    auto meta_shape = get_quant_meta_shape(rows, columns);
    if (!offsets.empty()) {
      if (offsets.size() != size_t(meta_shape.product())) {
        return "Quantization offset tensor shape mismatch! Expected " +
              std::to_string(meta_shape.product()) + " elements, got " +
              std::to_string(offsets.size()) + " elements!";
      }
      if (offsets_prepacked.size() != size_t(meta_shape.product())) {
        return "Wrong buffer size for prepacked quantization offsets!";
      }
    }
    if (scales.size() != size_t(meta_shape.product())) {
      return "Quantization scale tensor shape mismatch!";
    }
    if (scales_prepacked.size() != size_t(meta_shape.product())) {
      return "Prepacked quantization scale tensor buffer should be the same size!";
    }

    // 2D grid of 2D thread blocks
    constexpr int thread_blk_size = 16; // 256 threads per thread block
    constexpr int grid_stride = thread_blk_size * 16; // one thread process 16 items, one thread block has 16 threads
    dim3 block_dim(thread_blk_size, thread_blk_size);
    dim3 grid_dim((meta_shape[0] + grid_stride - 1) / grid_stride, (columns + thread_blk_size - 1) / thread_blk_size);
    prepack_scales_kernel<ElementT, block_size, qbits, Columnwise, ExtraBoundsCheck>
        <<<grid_dim, block_dim, 0, stream>>>(
            meta_shape[0], meta_shape[1], scales.data(), scales_prepacked.data(), offsets.data(), offsets_prepacked.data(), scales.size());
    return std::string();
  }

};

template <
    typename ElementT,
    int block_size,
    int qbits,
    bool Columnwise,
    bool ExtraBoundsCheck>
__global__
void block_quantize_kernel(
    uint8_t* dst,
    ElementT* scales,
    uint8_t* zero_points,
    const ElementT* src,
    int rows,
    int columns,
    int leadingDimension)
{
  using QuantType = BlockwiseQuantization<ElementT, block_size, qbits, Columnwise, ExtraBoundsCheck>;
  int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_idx >= QuantType::get_num_blks_quant_weights(rows, columns)) {
    return;
  }
  QuantType::dev_quantize_blk(dst, scales, zero_points, src, rows, columns, leadingDimension, block_idx);
}

template <
    typename ElementT,
    int block_size,
    int qbits,
    bool Columnwise,
    bool ExtraBoundsCheck>
__global__
void prepack_weights_kernel(
    int rows,
    int columns,
    const uint8_t* weights,      // <- int4 weights, column major
    uint8_t* weights_prepacked,  // <- int4 prepacked weights tensor, same size buffer
    size_t buf_size)
{
  using QuantType = BlockwiseQuantization<ElementT, block_size, qbits, Columnwise, ExtraBoundsCheck>;
  QuantType::dev_prepack_weights(rows, columns, weights, weights_prepacked, buf_size);
}

template <
    typename ElementT,
    int block_size,
    int qbits,
    bool Columnwise,
    bool ExtraBoundsCheck>
__global__
void prepack_scales_kernel(
    int meta_rows,
    int meta_columns,
    const ElementT* scales,     // <- quant scales, column major layout
    ElementT* scales_prepacked, // <- quant scales prepacked, same size buffer
    const uint8_t* offsets,     // <- quant offsets, int4, column major layout
    uint8_t* offsets_prepacked, // <- quant offsets prepacked, double size buffer
    size_t buf_size)
{
  using QuantType = BlockwiseQuantization<ElementT, block_size, qbits, Columnwise, ExtraBoundsCheck>;
  QuantType::dev_prepack_scales(meta_rows, meta_columns, scales, scales_prepacked, offsets, offsets_prepacked, buf_size);
}

}  // namespace mickey
