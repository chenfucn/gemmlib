/**
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file blkq4_fp16_util.h
 * @brief Utility functions for quantization and dequantization of fp16 data
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/layout/layout.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "matrix_layout.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test{

template <typename QuantBlocking, bool has_offsets>
struct BlkQuantizationRef {
  // layout of quantization scale and offset tensors
  using LayoutQMeta = 
    typename std::conditional<QuantBlocking::kRow == 1,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>::type;

  template<typename SrcLayout>
  static void QuantizeFp16To4Bit(
      cutlass::HostTensor<cutlass::half_t, SrcLayout>& src,
      cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor>& q4_weights,
      cutlass::HostTensor<cutlass::half_t, LayoutQMeta>& scales,
      cutlass::HostTensor<uint8_t, LayoutQMeta>& offsets) {
    int rows = src.extent().row();
    int cols = src.extent().column();
    ASSERT_TRUE((rows % 2) == 0) << "Tests error: can not process odd number of rows for 4-bit quantization";
    q4_weights.reset({rows / 2, cols});

    ASSERT_TRUE((rows % QuantBlocking::kRow) == 0 && (cols % QuantBlocking::kColumn) == 0) << "Tests error: can not have partial quantization block";

    int meta_rows = (rows / QuantBlocking::kRow);
    int meta_cols = (cols / QuantBlocking::kColumn);
    scales.reset({meta_rows, meta_cols});
    if (has_offsets) offsets.reset({meta_rows, meta_cols});

    constexpr int n_bits = 4;
    for (int col = 0; col < cols; col += QuantBlocking::kColumn) {
      for (int row = 0; row < rows; row += QuantBlocking::kRow) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = -min_val;
        for (int r = 0; r < QuantBlocking::kRow; ++r) {
          for (int c = 0; c < QuantBlocking::kColumn; ++c) {
            auto pos = cutlass::make_Coord(row + r, col + c);
            float f = static_cast<float>(src.at(pos));
            min_val = std::min(min_val, f);
            max_val = std::max(max_val, f);
          }
        }
        min_val = std::min(0.0f, min_val);
        max_val = std::max(0.0f, max_val);

        float pscale;
        uint8_t zp;
        if constexpr (has_offsets){
          int max_int = (1 << n_bits) - 1; // 4-bit range
          int min_int = 0;
          float scale = static_cast<float>(cutlass::half_t((max_val - min_val) / (max_int - min_int)));
          ASSERT_TRUE(scale >= 0.0f) << "Tests error: scale must be positive";

          pscale = (scale < 1e-6) ? 0.0f : float(max_int - min_int) / (max_val - min_val);
          pscale = static_cast<float>(cutlass::half_t(pscale));
          float offset_fp = min_val;
          if (scale >= 1e-6) {
            offset_fp = 0.0f - min_val * pscale;
          }
          // Handle clamping
          if (offset_fp < 0.0f) {
              zp = 0;
          } else if (offset_fp > 15.0f) {
              zp = 15;
          } else {
              zp = (uint8_t)roundf(offset_fp);
          }
          scales.at({row / QuantBlocking::kRow, col / QuantBlocking::kColumn}) = cutlass::half_t(scale);
          offsets.at({row / QuantBlocking::kRow, col / QuantBlocking::kColumn}) = zp;
        } else {
          // No zero-point
          max_val = (fabs(max_val) > fabs(min_val)) ? max_val : min_val;
          const cutlass::half_t scale = cutlass::half_t(max_val / (-8));
          pscale = (fabs(max_val) > 1e-7) ? ((-8) / max_val) : 0.0f;
          pscale = static_cast<float>(cutlass::half_t(pscale));
          scales.at({row / QuantBlocking::kRow, col / QuantBlocking::kColumn}) = scale;
        }

        for (int r = 0; r < QuantBlocking::kRow; ++r) {
          for (int c = 0; c < QuantBlocking::kColumn; ++c) {
            auto pos = cutlass::make_Coord(row + r, col + c);
            float f = static_cast<float>(src.at(pos));
            float q_f = has_offsets ? roundf(f * pscale + zp) : (f * pscale + 8.5f);
            const uint8_t q = (uint8_t)std::min(15.0f, std::max(0.0f, q_f));

            uint8_t& w = q4_weights.at({pos[0] / 2, pos[1]});
            if (pos[0] % 2 == 0) {
              w = (w & 0xf0) | q;
            } else {
              w = (w & 0x0f) | (q << 4);
            }

          }
        }
      }
    }
  }

  template<typename DstLayout>
  static void Dequantize4BitToFp16(
      cutlass::HostTensor<cutlass::half_t, DstLayout>& dst,
      cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor>& q4_weights,
      cutlass::HostTensor<cutlass::half_t, LayoutQMeta>& scales,
      cutlass::HostTensor<uint8_t, LayoutQMeta>& offsets) {
    int rows = q4_weights.extent().row() * 2;  // K
    int cols = q4_weights.extent().column();  // N
    dst.reset({rows, cols});

    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < rows; ++row) {
        auto weight_pos = cutlass::make_Coord(row/2, col);
        auto meta_pos = cutlass::make_Coord(row / QuantBlocking::kRow, col / QuantBlocking::kColumn);
        const float scale = static_cast<float>(scales.at(meta_pos));
        const uint8_t offset = has_offsets ? offsets.at(meta_pos) : 8;
        const int w = (row % 2 == 0) ? (q4_weights.at(weight_pos) & 0xf) : (q4_weights.at(weight_pos) >> 4);

        const float f = scale * (w - offset);
        dst.at({row, col}) = cutlass::half_t(f);
      }
    }
  }
};


//
// Converting cutlass tensor to MatrixRef
//

template <
  typename Element,
  typename Layout>
__forceinline__
mickey::MatrixRef<Element, Layout, true> make_MatrixRef(cutlass::HostTensor<Element, Layout> const& tensor) {
  static_assert(std::is_same<Layout, cutlass::layout::ColumnMajor>::value
                || std::is_same<Layout, cutlass::layout::RowMajor>::value);
  auto shape = cutlass::make_Coord(tensor.extent().row(), tensor.extent().column());
  auto* ptr = const_cast<typename std::remove_const<Element>::type *>(tensor.host_data());
  return mickey::MatrixRef<Element, Layout, true>(ptr, tensor.capacity(), shape);
}

template <
  typename Element,
  typename Layout>
__forceinline__
mickey::MatrixRef<Element const, Layout, true> make_ConstMatrixRef(cutlass::HostTensor<Element, Layout> const& tensor) {
  static_assert(std::is_same<Layout, cutlass::layout::ColumnMajor>::value
                || std::is_same<Layout, cutlass::layout::RowMajor>::value);
  auto shape = cutlass::make_Coord(tensor.extent().row(), tensor.extent().column());
  return mickey::MatrixRef<Element const, Layout, true>(tensor.host_data(), tensor.capacity(), shape);
}

}  // namespace test
}  // namespace onnxruntime