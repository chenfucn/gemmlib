/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4_dq.cpp

Abstract:

    This module contains the data structures and implementations
    for blocked int4 quantization and dequantization.

    Int4 block quantization is used to compress weight tensors of large
    language models.

--*/

#include "matrix_layout.h"


/***************************************************************
 * The quantization format that pack data and quantization
 * parameters into separate buffers.
 */


template <
    int Row_,    ///< rows of a matrix
    int Column_  ///< columns of a matrix
    >
struct Shape2D {
    static int const kRow = Row_;              ///< rows of a matrix
    static int const kColumn = Column_;        ///< columns of a matrix
    static int const kCount = Row_ * Column_;  ///< total number of elements in a matrix
};


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
M_FORCEINLINE
void
range2scalezp(float min, float max, ScaleT& scale, uint8_t& zp)
{
    constexpr int zp_max = BitsTraits<qbits>::kMax;
    constexpr float zp_max_fp = BitsTraits<qbits>::kMaxFp;

    min = std::min(min, 0.0f);
    max = std::max(max, 0.0f);

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
M_FORCEINLINE
void
range2scale(float min, float max, ScaleT& scale)
{
    constexpr int mid_v = BitsTraits<qbits>::kMid;
    constexpr float mid_fp = static_cast<float>(-mid_v);

    max = fabsf(max) > fabsf(min) ? max : min;

    scale = ScaleT(max / mid_fp);
};


/**
 * @brief Blockwise quantization methods
 * @tparam ElementT       source data type, e.g. fp32/fp16
 * @tparam block_size     number of elemenets quantized together
 * @tparam qbits          number of bits in each quantized element
 * @tparam Columnwise     true:  elements in a block come from one single column
 *                        false: elements in a block come from one single row
 */
template <
    typename ElementT,
    int32_t block_size,
    int32_t qbits,
    bool Columnwise>
struct BlockwiseQuantizer {
    // To support other qbits, need to add bit packing code for
    // storing to dst and zero points
    static_assert(qbits == 4, "Only 4b block quantization is supported!");

    using QuantBlk = std::conditional_t<Columnwise, Shape2D<block_size, 1>, Shape2D<1, block_size>>;
    using ThreadBlk = Shape2D<QuantBlk::kRow * BitsTraits<qbits>::kPackSize, QuantBlk::kColumn>;

    static
    M_FORCEINLINE
    void quantizeMetaShape(int rows, int columns, int& meta_rows, int& meta_cols)
    {
        meta_rows = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;
        meta_cols = (columns + QuantBlk::kColumn - 1) / QuantBlk::kColumn;
    }

    static
    M_FORCEINLINE
    void quantizedShape(int rows, int columns, int& q_rows, int& q_cols) {
        int meta_rows;
        int meta_cols;
        quantizeMetaShape(rows, columns, meta_rows, meta_cols);

        // quantized matrix is stored in column major, packed by column
        q_rows = (meta_rows * QuantBlk::kRow * qbits + 7) / 8;
        q_cols = meta_cols * QuantBlk::kColumn;
    }

    static M_FORCEINLINE void quantizedBufferSizes(
        int rows, int columns, size_t& data_bytes, size_t& scale_num_elements, size_t* zero_point_bytes
    )
    {
        int meta_rows, meta_cols;
        quantizeMetaShape(rows, columns, meta_rows, meta_cols);
        int q_rows, q_cols;
        quantizedShape(rows, columns, q_rows, q_cols);

        data_bytes = q_rows * q_cols;
        scale_num_elements = meta_rows * meta_cols;

        if (zero_point_bytes) {
            // this works for qbits == 4 but may need to be updated for other qbits values
            *zero_point_bytes = ((meta_rows * qbits + 7) / 8) * meta_cols;
        }
    }

    /**
     * @brief Quantized a Matrix shape [rows, columns], resulting quantized
     *        and packed data are stored in column major (transposed)
     * @param[out] dst           pointer to the quantized weights, column major: [columns, rows]
     * @param[out] scale         pointer to the scales, column major: [columns/QuantBlk::kColumn, rows/QuantBlk::kRow]
     * @param[out] zero_points   pointer to the zero points, same shape as scale
     * @param[in]  src           pointer to the source matrix, row major: [rows, columns]
     * @param rows
     * @param columns
     * @param leadingDimension   stride of the source matrix, i.e. distance from one row to the next
     */
    static void quantizeAndTranspose(
        uint8_t* dst,
        ElementT* scales,
        uint8_t* zero_points,
        const ElementT* src,
        int32_t rows,
        int32_t columns,
        int32_t leadingDimension)
    {
        // Thread partitioning
        const auto thrd_row_blks = (rows + ThreadBlk::kRow - 1) / ThreadBlk::kRow;
        const auto thrd_col_blks = (columns + ThreadBlk::kColumn - 1) / ThreadBlk::kColumn;
        const auto total_thrd_blks = thrd_row_blks * thrd_col_blks;

        const auto row_blks = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;

        int q_rows, q_cols;
        quantizedShape(rows, columns, q_rows, q_cols);

        for (int block_idx = 0; block_idx < total_thrd_blks; block_idx++){
                uint8_t zp_bytes[BitsTraits<qbits>::kPackSize];
                std::fill_n(zp_bytes, BitsTraits<qbits>::kPackSize, (uint8_t)8);

                const int32_t r_blk_idx = static_cast<int32_t>(block_idx / thrd_col_blks);
                const int32_t c_blk_idx = static_cast<int32_t>(block_idx % thrd_col_blks);

                const int32_t r = r_blk_idx * ThreadBlk::kRow;
                const int32_t c = c_blk_idx * ThreadBlk::kColumn;

                const int32_t r_end = std::min(r + ThreadBlk::kRow, rows);
                const int32_t c_end = std::min(c + ThreadBlk::kColumn, columns);

                const int meta_row = r / QuantBlk::kRow;
                const int meta_col = c / QuantBlk::kColumn;

                // compute scale and zero point
                for (int kpack = 0; kpack < BitsTraits<qbits>::kPackSize; kpack++) {

                    // scan a single block to extract range [min, max]
                    float min = std::numeric_limits<float>::max();
                    float max = -min;
                    const int row_start = r + kpack * QuantBlk::kRow;
                    const int row_end = std::min(row_start + QuantBlk::kRow, r_end);
                    for (int i = row_start; i < row_end; ++i) {
                        for (int j = c; j < c_end; ++j) {
                            const float v = static_cast<float>(src[i * leadingDimension + j]);
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
                            range2scalezp<ElementT, qbits>(min, max, scales[meta_idx], zp_bytes[kpack]);
                        }
                    }
                }

                // !! 4b specific code as we need to pack 2 4b numbers into one byte
                if (zero_points != nullptr) {
                    const int32_t meta_idx = meta_col * ((row_blks + 1) / 2) + meta_row / 2;
                    zero_points[meta_idx] = (zp_bytes[0] & 0xf) | (zp_bytes[1] << 4);
                }

                for (int32_t j = c; j < c_end; ++j) {
                    const int32_t meta_c = j / QuantBlk::kColumn;
                    for (int32_t i = r; i < r_end; i += 2) {
                        const int32_t meta_r = i / QuantBlk::kRow;
                        const float scale = static_cast<float>(scales[meta_c * row_blks + meta_r]);
                        const float reciprocal_scale = scale ? 1.0f / scale : 0.0f;
                        const int8_t zp = zp_bytes[meta_r & 1];
                        const int8_t zp1 = zp_bytes[((i + 1) / QuantBlk::kRow) & 1];

                        const float v0 = static_cast<float>(src[i * leadingDimension + j]);
                        const uint8_t vi0 = (uint8_t)std::clamp(roundf(v0 * reciprocal_scale + zp),
                                                                0.0f, BitsTraits<qbits>::kMaxFp);

                        uint8_t vi1 = (uint8_t)zp;
                        if (i + 1 < r_end) {
                            float reciprocal_scale1 = reciprocal_scale;
                            if constexpr (QuantBlk::kRow == 1) {
                                const float scale1 =
                                    static_cast<float>(scales[meta_c * row_blks + meta_r + 1]);
                                reciprocal_scale1 = scale1 ? 1.0f / scale1 : 0.0f;
                            }
                            const float v1 = static_cast<float>(src[(i + 1) * leadingDimension + j]);
                            vi1 = (uint8_t)std::clamp(roundf(v1 * reciprocal_scale1 + zp1), 0.0f,
                                                      BitsTraits<qbits>::kMaxFp);
                        }

                        // !! 4b specific code
                        dst[j * q_rows + i / 2] = (vi0 & 0xf) | (vi1 << 4);
                    }
                }

        }
    }

    /**
     * @brief Dequantize a column major quantized matrix, and store the result in a column major
     * matrix for use in GEMM
     * @param[out] dst           pointer to the dequantized matrix, column major: [columns, rows]
     * @param[in]  weights       pointer to the quantized weights, column major: [columns, rows]
     * @param[in]  scales        pointer to the scales of quantized blocks, column major layout
     * @param[in]  zero_points   pointer to the zero points of quantized blocks, packed column major
     *                           scales
     * @param[in]  rows
     * @param[in]  columns
     */
/*    static void dequantize(
        ElementT* dst,
        const uint8_t* weights,
        const ElementT* scales,
        const uint8_t* zero_points,
        int32_t rows,
        int32_t columns,
        MLAS_THREADPOOL* thread_pool)
    {
        // Thread partitioning
        const auto thrd_row_blks = (rows + ThreadBlk::kRow - 1) / ThreadBlk::kRow;
        const auto thrd_col_blks = (columns + ThreadBlk::kColumn - 1) / ThreadBlk::kColumn;
        const auto total_thrd_blks = thrd_row_blks * thrd_col_blks;

        const auto row_blks = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;

        int q_rows, q_cols;
        quantizedShape(rows, columns, q_rows, q_cols);

        MlasTryBatchParallel(
            thread_pool, total_thrd_blks,
            [&](ptrdiff_t block_idx) {
                int32_t r_blk_idx = static_cast<int32_t>(block_idx / thrd_col_blks);
                int32_t c_blk_idx = static_cast<int32_t>(block_idx % thrd_col_blks);

                int32_t r = r_blk_idx * ThreadBlk::kRow;
                int32_t c = c_blk_idx * ThreadBlk::kColumn;

                int32_t r_end = std::min(r + ThreadBlk::kRow, rows);
                int32_t c_end = std::min(c + ThreadBlk::kColumn, columns);

                for (int32_t j = c; j < c_end; ++j) {
                    const int32_t meta_col = j / QuantBlk::kColumn;

                    // !! 4b specific code
                    // the whole loop is 4b specific due to sub 8 bit packing
                    // and unpacking. We can potentially make this qbits generic
                    // by wraping the packing/unpacking code like cutlass::Array
                    for (int32_t i = r; i < r_end; i += 2) {
                        const int32_t meta_row = i / QuantBlk::kRow;

                        const float scale0 =
                            static_cast<float>(scales[meta_col * row_blks + meta_row]);

                        const int zp_pair =
                            (zero_points == nullptr)
                                ? 0x88
                                : zero_points[meta_col * ((row_blks + 1) / 2) + meta_row / 2];
                        const int zp0 = (meta_row & 1) ? (zp_pair >> 4) : (zp_pair & 0xf);

                        const uint8_t vi0 = weights[j * q_rows + i / 2] & 0xf;
                        const float v0 = (static_cast<float>(vi0) - zp0) * scale0;

                        dst[j * rows + i] = static_cast<ElementT>(v0);
                        if ((i + 1) < r_end) {
                            float scale1 = scale0;
                            int zp1 = zp0;
                            if constexpr (QuantBlk::kRow == 1) {
                                scale1 =
                                    static_cast<float>(scales[meta_col * row_blks + meta_row + 1]);
                                zp1 = (zp_pair >> 4) & 0xf;
                            }
                            const uint8_t vi1 = weights[j * q_rows + i / 2] >> 4;
                            const float v1 = (static_cast<float>(vi1) - zp1) * scale1;
                            dst[j * rows + (i + 1)] = static_cast<ElementT>(v1);
                        }
                    }
                }
            });
    } */
};


template <typename T, int qbits>
void
MlasBlockwiseQuantMetaShape(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& meta_rows,
    int& meta_cols
    )
{
    switch (block_size) {
        case 16: {
            if (columnwise) {
                BlockwiseQuantizer<T, 16, qbits, true>::quantizeMetaShape(rows, columns, meta_rows, meta_cols);
            } else {
                BlockwiseQuantizer<T, 16, qbits, false>::quantizeMetaShape(rows, columns, meta_rows, meta_cols);
            }
            break;
        }
        case 32: {
            if (columnwise) {
                BlockwiseQuantizer<T, 32, qbits, true>::quantizeMetaShape(rows, columns, meta_rows, meta_cols);
            } else {
                BlockwiseQuantizer<T, 32, qbits, false>::quantizeMetaShape(
                                    rows, columns, meta_rows, meta_cols);
            }
            break;
        }
        case 64: {
            if (columnwise) {
                BlockwiseQuantizer<T, 64, qbits, true>::quantizeMetaShape(rows, columns, meta_rows,
                                                                      meta_cols);
            } else {
                BlockwiseQuantizer<T, 64, qbits, false>::quantizeMetaShape(rows, columns, meta_rows,
                                                                       meta_cols);
            }
            break;
        }
        case 128: {
            if (columnwise) {
                BlockwiseQuantizer<T, 128, qbits, true>::quantizeMetaShape(rows, columns, meta_rows,
                                                                      meta_cols);
            } else {
                BlockwiseQuantizer<T, 128, qbits, false>::quantizeMetaShape(rows, columns, meta_rows,
                                                                       meta_cols);
            }
            break;
        }
        case 256: {
            if (columnwise) {
                BlockwiseQuantizer<T, 256, qbits, true>::quantizeMetaShape(rows, columns, meta_rows,
                                                                      meta_cols);
            } else {
                BlockwiseQuantizer<T, 256, qbits, false>::quantizeMetaShape(rows, columns, meta_rows,
                                                                       meta_cols);
            }
            break;
        }
        default:
            meta_rows = 0;
            meta_cols = 0;
            break;
    }
}



template <typename T, int qbits>
void
MlasBlockwiseQuantizedShape(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& q_rows,
    int& q_cols
    )
{
    switch (block_size) {
        case 16: {
            if (columnwise) {
                BlockwiseQuantizer<T, 16, qbits, true>::quantizedShape(rows, columns, q_rows, q_cols);
            } else {
                BlockwiseQuantizer<T, 16, qbits, false>::quantizedShape(rows, columns, q_rows, q_cols);
            }
            break;
        }
        case 32: {
            if (columnwise) {
                BlockwiseQuantizer<T, 32, qbits, true>::quantizedShape(rows, columns, q_rows, q_cols);
            } else {
                BlockwiseQuantizer<T, 32, qbits, false>::quantizedShape(
                                    rows, columns, q_rows, q_cols);
            }
            break;
        }
        case 64: {
            if (columnwise) {
                BlockwiseQuantizer<T, 64, qbits, true>::quantizedShape(rows, columns, q_rows, q_cols);
            } else {
                BlockwiseQuantizer<T, 64, qbits, false>::quantizedShape(rows, columns, q_rows, q_cols);
            }
            break;
        }
        case 128: {
            if (columnwise) {
                BlockwiseQuantizer<T, 128, qbits, true>::quantizedShape(rows, columns, q_rows, q_cols);
            } else {
                BlockwiseQuantizer<T, 128, qbits, false>::quantizedShape(rows, columns, q_rows, q_cols);
            }
            break;
        }
        case 256: {
            if (columnwise) {
                BlockwiseQuantizer<T, 256, qbits, true>::quantizedShape(rows, columns, q_rows, q_cols);
            } else {
                BlockwiseQuantizer<T, 256, qbits, false>::quantizedShape(rows, columns, q_rows, q_cols);
            }
            break;
        }
        default:
            q_rows = 0;
            q_cols = 0;
            break;
    }
}



template <typename T, int qbits>
void
MlasQuantizeBlockwise(
    uint8_t* dst,
    T* scales,
    uint8_t* zero_points,
    const T* src,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int leading_dimension
    )
{
    switch (block_size) {
        case 16:
            if (columnwise) {
                BlockwiseQuantizer<T, 16, qbits, true>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension);
            } else {
                BlockwiseQuantizer<T, 16, qbits, false>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension);
            }
            break;

        case 32:
            if (columnwise) {
                BlockwiseQuantizer<T, 32, qbits, true>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension);
            } else {
                BlockwiseQuantizer<T, 32, qbits, false>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension);
            }
            break;

        case 64:
            if (columnwise) {
                BlockwiseQuantizer<T, 64, qbits, true>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension);
            } else {
                BlockwiseQuantizer<T, 64, qbits, false>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension);
            }
            break;

        case 128:
            if (columnwise) {
                BlockwiseQuantizer<T, 128, qbits, true>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension);
            } else {
                BlockwiseQuantizer<T, 128, qbits, false>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension);
            }
            break;

        case 256:
            if (columnwise) {
                BlockwiseQuantizer<T, 256, qbits, true>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension);
            } else {
                BlockwiseQuantizer<T, 256, qbits, false>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension);
            }
            break;

        default:
            // Only block size 16, 32, 64, 128, 256 are supported.
            break;
    }
}
