/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    blkq4_fp16_gemm_sm80.h
 *
 * Abstract:
 *   Bridge between gtest code and gemm kernel implementation.
 *   Gemm kernel requires CUTLASS header files, which causes strange
 *   compilation errors with RE2 header files, which are required
 *   by gtest.
 */

#pragma once

#include "matrix_layout.h"
#include "cutlass/cutlass.h"

namespace onnxruntime {
namespace cuda {
namespace test {

// Pytorch linear layer stores the weight matrix in shape (out_features, in_features)
// The input matrix should be in shape (batch_size, in_features). The output matrix
// is in shape (batch_size, out_features).
//
// In blas M, N, K notation, where we have matrix multiplication:
//    C = A * B
// where A is the input matrix in shape (M, K), B is the weight matrix in shape (K, N),
// and C is the output matrix in shape (M, N)
// M = batch_size, N = out_features, K = in_features
//
// So B the weight matrix is in shape (K, N) = (in_features, out_features)
// The weight matrix in Pytorch linear layer can be seen as the transpose of B,
// i.e. column major layout of B.


static inline void prepack_weights_ref(
    int in_features,
    int out_features,
    const mickey::MatrixRef<uint8_t const, cutlass::layout::ColumnMajor, true>& tensor_weight,
    const mickey::MatrixRef<uint8_t, cutlass::layout::ColumnMajor, true>& tensor_weight_prepacked) {
  if (tensor_weight.shape()[0] != in_features / 2 || tensor_weight.shape()[1] != out_features) {
    throw std::runtime_error("Unexpected tensor_weight shape! Expected: (" +
                             std::to_string(in_features / 2) + ", " + std::to_string(out_features) + "), Got: (" +
                             std::to_string(tensor_weight.shape()[0]) + ", " + std::to_string(tensor_weight.shape()[1]) + ").");
  }
  if (tensor_weight_prepacked.shape()[0] != in_features || tensor_weight_prepacked.shape()[1] != out_features / 2) {
    throw std::runtime_error("Unexpected tensor_weight_prepacked shape! Expected: (" +
                             std::to_string(in_features) + ", " + std::to_string(out_features / 2) + "), Got: (" +
                             std::to_string(tensor_weight_prepacked.shape()[0]) + ", " + std::to_string(tensor_weight_prepacked.shape()[1]) + ").");
  }

  auto t0_base = cutlass::make_Coord(0, 0);
  auto t1_base = cutlass::make_Coord(4, 0);
  auto t2_base = cutlass::make_Coord(0, 8);
  auto t3_base = cutlass::make_Coord(4, 8);
  for (int col_dtile = 0; col_dtile < out_features / 16; ++col_dtile) {
    for (int row_dtile = 0; row_dtile < in_features / 16; ++row_dtile) {
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
  }
}

template <
    typename ScaleElementT,
    typename Layout,
    typename QuantBlocking>
void prepack_quant_scales_ref(
    int in_features,
    int out_features,
    const mickey::MatrixRef<ScaleElementT const, Layout, true>& tensor_scale,
    const mickey::MatrixRef<ScaleElementT, Layout, true>& tensor_scale_prepacked) {
  if (tensor_scale.shape()[0] != (in_features / QuantBlocking::kRow) || tensor_scale.shape()[1] != (out_features / QuantBlocking::kColumn)) {
    throw std::runtime_error("Unexpected tensor_scale shape! Expected: (" +
                             std::to_string(in_features / QuantBlocking::kRow) + ", " + std::to_string(out_features / QuantBlocking::kColumn) + "), Got: (" +
                             std::to_string(tensor_scale.shape()[0]) + ", " + std::to_string(tensor_scale.shape()[1]) + ").");
  }
  if (tensor_scale_prepacked.shape() != tensor_scale.shape()) {
    throw std::runtime_error("Unexpected tensor_scale_prepacked shape! Expected: (" +
                             std::to_string(in_features / QuantBlocking::kRow) + ", " + std::to_string(out_features / QuantBlocking::kColumn) + "), Got: (" +
                             std::to_string(tensor_scale_prepacked.shape()[0]) + ", " + std::to_string(tensor_scale_prepacked.shape()[1]) + ").");
  }

  // Only prepacking scale and offset tensors for a often used special case:
  //    16b gemm (2 elements per 32b register, operand tile shape 8x8)
  //    2 B operand tiles per mma instruction stacked on k dimension
  //    (1,n) quantization blocking
  if constexpr (sizeof(ScaleElementT) == 2 && QuantBlocking::kRow == 1) {
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
    // needs two seperate loads for a mma instruction, due to the tile fragment layout shown
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

    for (int col = 0; col < tensor_scale.shape()[1]; ++col) {
      for (int row_blk = 0; row_blk < tensor_scale.shape()[0]; row_blk += 16) {
        for (int thread_id = 0; thread_id < 4; thread_id++) {
          const int dst_idx = row_blk + thread_id * 4;
          const int src_idx = row_blk + thread_id * 2;
          tensor_scale_prepacked.at(dst_idx + 0, col) = tensor_scale.at(src_idx + 0, col);
          tensor_scale_prepacked.at(dst_idx + 1, col) = tensor_scale.at(src_idx + 1, col);
          tensor_scale_prepacked.at(dst_idx + 2, col) = tensor_scale.at(src_idx + 8, col);
          tensor_scale_prepacked.at(dst_idx + 3, col) = tensor_scale.at(src_idx + 9, col);
        }
      }
    }
  } else {
    // In all other cases, we don't prepack scale or offset
    std::copy(tensor_scale.data(), tensor_scale.data() + tensor_scale.shape().product(), tensor_scale_prepacked.data());
  }
}

template <typename Layout, typename QuantBlocking>
void prepack_quant_offsets_ref(
    int in_features,
    int out_features,
    mickey::MatrixRef<uint8_t const, Layout, true> tensor_offset,
    mickey::MatrixRef<uint8_t, Layout, true> tensor_offset_prepacked) {
  if (tensor_offset.shape()[0] != (in_features / QuantBlocking::kRow) || tensor_offset.shape()[1] != (out_features / QuantBlocking::kColumn)) {
    throw std::runtime_error("Unexpected tensor_offset shape! Expected: (" +
                             std::to_string(in_features / QuantBlocking::kRow) + ", " + std::to_string(out_features / QuantBlocking::kColumn) + "), Got: (" +
                             std::to_string(tensor_offset.shape()[0]) + ", " + std::to_string(tensor_offset.shape()[1]) + ").");
  }
  if (tensor_offset_prepacked.shape() != tensor_offset.shape()) {
    throw std::runtime_error("Unexpected tensor_offset_prepacked shape! Expected: (" +
                             std::to_string(in_features / QuantBlocking::kRow) + ", " + std::to_string(out_features / QuantBlocking::kColumn) + "), Got: (" +
                             std::to_string(tensor_offset_prepacked.shape()[0]) + ", " + std::to_string(tensor_offset_prepacked.shape()[1]) + ").");
  }

  // Only prepacking scale and offset tensors for a often used special case:
  //    16b gemm (2 elements per 32b register, operand tile shape 8x8)
  //    2 B operand tiles per mma instruction stacked on k dimension
  //    (1,n) quantization blocking
  if constexpr (QuantBlocking::kRow != 1) {
    std::copy(tensor_offset.data(), tensor_offset.data() + tensor_offset.shape().product(), tensor_offset_prepacked.data());
    return;
  }
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
  // needs two seperate loads for a mma instruction, due to the tile fragment layout shown
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
  if (tensor_offset_prepacked.good()) {
    for (int col = 0; col < tensor_offset.shape()[1]; ++col) {
      for (int row_blk = 0; row_blk < tensor_offset.shape()[0]; row_blk += 16) {
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
      }
    }
  }
}

/*
template <
    int block_size,
    bool column_wise_blocking,
    bool small_m,
    bool has_offsets>
void run_blkq4_gemm(int m, int n, int k);
*/

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
