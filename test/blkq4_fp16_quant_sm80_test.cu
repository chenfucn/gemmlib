/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    blkq4_fp16_gemm_sm80_test.cc
 *
 * Abstract:
 *   Test code for block-wise quantized 4b GEMM kernels.
 *   This part requires gtest header files, which do not play
 *   well with CUTLASS headers.
 */

#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

#include "blk_q4/f16_quant_sm80.cuh"
#include "blkq4_fp16_gemm_sm80.h"

#include "ms_blkq4gemm.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

template <bool ColumnQuantBlocking, bool has_offset = true>
void testPrepack(int rows, int columns) {
  using ElementT = half;
  constexpr int block_size = 32;
  using Base = mickey::BlockwiseQuantization<
      ElementT,
      block_size,
      4,
      ColumnQuantBlocking>;

  using QuantBlocking = typename Base::QuantBlocking;
  using ElementW = typename Base::ElementW;
  using LayoutWPack = typename Base::LayoutWPack;
  using ElementQOffset = typename Base::ElementQOffset;
  using LayoutQmeta = typename Base::LayoutQmeta;

  unsigned int seed = 28571;  // Replace with desired seed value
  std::seed_seq seq{seed};
  std::mt19937 gen(seq);
  std::uniform_int_distribution<> dis(0, 8192);

  const auto q_weight_shape = Base::get_quant_weights_shape(rows, columns);
  const auto meta_shape = Base::get_quant_meta_shape(rows, columns);

  //
  // For testing quantization and dequantization, it is not straight
  // forward to avoid flaky tests due to rounding errors. The way we
  // try to achieve this is to:
  // 1. Generate a set of quantized weights, scales and offsets
  // 2. Dequantize the weights
  // 3. Quantize the dequantized weights
  // 4. Compare the dequantied-and-then-quantized weights with
  //    the original quantized weights
  //
  // Random filling of the initial values are key to get this right.
  // For weights, we must ensure each block gets a full range of
  // values, i.e. must contain 0 and 15. And for scales, they must
  // all be positive.
  //

  std::vector<ElementW> q_weights(q_weight_shape.product());
  mickey::MatrixRef<ElementW, LayoutWPack, true> tensor_q_weight(
      q_weights, cutlass::make_Coord(rows / 2, columns));
  int v = 7;
  for (int c = 0; c < tensor_q_weight.shape()[1]; c++) {
    for (int r = 0; r < tensor_q_weight.shape()[0]; ++r) {
      uint8_t v0 = static_cast<uint8_t>(v);
      v = (v + 5) % 16;
      if (v == 11 || v == 7 || v == 3) {
        // making the cycle 13 instead of 16, avoiding same values in a row
        v = (v + 5) % 16;
      }
      uint8_t v1 = 0;
      if (r + 1 < rows) {
        v1 = static_cast<uint8_t>(v);
        v = (v + 5) % 16;
        if (v == 11 || v == 7 || v == 3) {
          // making the cycle 13 instead of 16, avoiding same values in a row
          v = (v + 5) % 16;
        }
      }

      tensor_q_weight.at(r, c) = ElementW((v1 << 4) | v0);
    }
  }

  std::vector<ElementT> q_scales(meta_shape.product());
  mickey::MatrixRef<ElementT, LayoutQmeta, true> tensor_scale(
      q_scales, meta_shape);

  std::vector<ElementQOffset> q_zp(meta_shape.product());
  mickey::MatrixRef<ElementQOffset, LayoutQmeta, true> tensor_offset(
      q_zp, meta_shape);

  for (int col = 0; col < meta_shape[1]; ++col) {
    for (int row = 0; row < meta_shape[0]; ++row) {
        int f = (((col * v + row + v / 3) % 63) + 1);
        v += 41;
        int m = col * v + row + v * 3;
        float scale =  f / float(1 << (4 + (m % 2)));
        tensor_scale.at(row, col) = ElementT(scale);
        tensor_offset.at(row, col) = ((f + m + v) % 8) + 4;
    }
  }

#if 0  // debug
  // Fill tensor_q_weight with the patterned data, easier to debug with print
  int loop_val = 0;
  int offset = 3;
  for (int col_tile = 0; col_tile < tensor_q_weight.extent().column()/8; ++col_tile) {
    for (int row_tile = 0; row_tile < tensor_q_weight.extent().row()/4; ++row_tile) {
      for (int col = 0; col < 8; ++col) {
        for (int row = 0; row < 4; ++row) {
          auto weight_cord = cutlass::make_Coord(row_tile * 4 + row, col_tile * 8 + col);
          auto val = (loop_val + offset) % 256;
          tensor_q_weight.at(weight_cord) = ElementW(val);
          loop_val++;
          if (loop_val == 256) {
            loop_val = 0;
            offset += 11;
          }
        }
      }
    }
  }
  for (int col = 0; col < tensor_scale.extent().column(); ++col){
    int c =  col * QuantBlocking::kColumn;
    for (int row = 0; row < tensor_scale.extent().row(); ++row){
      int r = row * QuantBlocking::kRow;
      auto weight_cord = cutlass::make_Coord(r/2, c);
      int w = 0;
      if (r % 2 == 0) {
        w = int(tensor_q_weight.at(weight_cord) & 0x0f);
      } else {
        w = int(tensor_q_weight.at(weight_cord) >> 4);
      }
      tensor_scale.at({row, col}) = w;
      tensor_offset.at({row, col}) = ElementQOffset(w);
    }
  }

  int fill_val = -512;
  int factor = 1;
  for (int col = 0; col < tensor_scale.extent().column(); ++col){
    for (int row = 0; row < tensor_scale.extent().row(); ++row){
      tensor_scale.at({row, col}) = ElementQScale((float)fill_val * float(factor));
      fill_val++;
      if (fill_val == 512) {
        fill_val = -512;
        factor += 1;
      }
    }
  }

#endif  // debug

  std::vector<ElementT> dequants(rows * columns);
  mickey::MatrixRef<ElementT, cutlass::layout::ColumnMajor> tensor_dequant(dequants, cutlass::make_Coord(rows, columns));

  // Dequantize weights and save into matrix B for reference
  for (int col = 0; col < tensor_dequant.shape()[1]; ++col) {
    for (int row = 0; row < tensor_dequant.shape()[0]; ++row) {
      auto weight_cord = cutlass::make_Coord(row / 2, col);
      auto scale_cord = cutlass::make_Coord(row / QuantBlocking::kRow, col / QuantBlocking::kColumn);
      const uint8_t offset = has_offset ? tensor_offset.at(scale_cord) : 8;
      int w = 0;
      if (row % 2 == 0) {
        w = int(tensor_q_weight.at(weight_cord) & 0x0f);
      } else {
        w = int(tensor_q_weight.at(weight_cord) >> 4);
      }
      float scale = float(tensor_scale.at(scale_cord));
      float dequant = scale * float(w - offset);
      tensor_dequant.at(row, col) = ElementT(dequant);
      // Prints for help debugging in case of test failure
      // fprintf(stdout, "(%2d,%2d)= %2d, %2d, %f, %f\n", row, col, w, offset, scale, dequant);
    }
  }

  //
  // Run quantization tool:
  //
  ElementW *o_elements_dev_ptr = nullptr;
  EXPECT_EQ(q_weights.size(), q_weight_shape.product());
  cudaMalloc(&o_elements_dev_ptr, q_weight_shape.product() * sizeof(ElementW));

  ElementT *o_scales_dev_ptr = nullptr;
  EXPECT_EQ(q_scales.size(), meta_shape.product());
  cudaMalloc(&o_scales_dev_ptr, meta_shape.product() * sizeof(ElementT));

  uint8_t *o_zp_dev_ptr = nullptr;
  EXPECT_EQ(q_zp.size(), meta_shape.product());
  cudaMalloc(&o_zp_dev_ptr, meta_shape.product() * sizeof(uint8_t));

  ElementT *dequants_dev_ptr = nullptr;
  EXPECT_EQ(dequants.size(), rows * columns);
  cudaMalloc(&dequants_dev_ptr, rows * columns * sizeof(ElementT));
  cudaMemcpy(dequants_dev_ptr, dequants.data(), rows * columns * sizeof(ElementT), cudaMemcpyHostToDevice);

  std::string err_message;
  auto err = blkq4_fp16_quant_sm80(
    block_size,
    ColumnQuantBlocking,
    rows, columns, rows,
    0,
    dequants_dev_ptr, rows * columns * sizeof(ElementT),
    o_elements_dev_ptr, q_weight_shape.product() * sizeof(ElementW),
    o_scales_dev_ptr, meta_shape.product() * sizeof(ElementT),
    has_offset ? o_zp_dev_ptr : nullptr,
    has_offset ? (meta_shape.product() * sizeof(uint8_t)) : 0,
    &err_message); 
  ASSERT_TRUE(err == 0) << "Quantization Failed: " << err_message;

  //
  // Copy results from device to host
  //
  std::vector<ElementW> o_elements(q_weight_shape.product());
  mickey::MatrixRef<ElementW, cutlass::layout::ColumnMajor, true> tensor_o_elements(o_elements, cutlass::make_Coord(rows, columns / 2));

  std::vector<ElementT> packed_scales(meta_shape.product());
  mickey::MatrixRef<ElementT, LayoutQmeta, true> tensor_packed_scales(
      packed_scales, meta_shape);

  std::vector<ElementQOffset> packed_zp(meta_shape.product());
  mickey::MatrixRef<ElementQOffset, LayoutQmeta, true> tensor_packed_zp(
      packed_zp, meta_shape);
  cudaMemcpy(o_elements.data(), o_elements_dev_ptr, q_weight_shape.product() * sizeof(ElementW), cudaMemcpyDeviceToHost);
  cudaMemcpy(packed_scales.data(), o_scales_dev_ptr, meta_shape.product() * sizeof(ElementT), cudaMemcpyDeviceToHost);
  cudaMemcpy(packed_zp.data(), o_zp_dev_ptr, meta_shape.product() * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  // Verify prepacked weights
  std::vector<ElementW> packed_w_ref(q_weight_shape.product());
  mickey::MatrixRef<ElementW, LayoutWPack, true> tensor_packed_w_ref(
      packed_w_ref, cutlass::make_Coord(rows, columns / 2));
  onnxruntime::cuda::test::prepack_weights_ref(rows, columns, tensor_q_weight, tensor_packed_w_ref);
  for (int col = 0; col < tensor_packed_w_ref.shape()[1]; ++col) {
    for (int row = 0; row < tensor_packed_w_ref.shape()[0]; ++row) {
      EXPECT_EQ(tensor_o_elements.at(row, col), tensor_packed_w_ref.at(row, col))
          << "quantized value mismatch at [" << row << "," << col << "]"
          << " shape[" << rows << "," << columns << "]"
          << (ColumnQuantBlocking ? "Column-wise-block" : "Row-wise-block");
    }
  }
  for (size_t i = 0; i < packed_zp.size(); ++i) {
    std::cout << int(packed_zp[i]) << " ";
    if ((i % 9) == 8) std::cout << std::endl;
  }
  std::cout << std::endl;

  // Verify prepacked scales and offsets
  std::vector<ElementT> packed_scales_ref(meta_shape.product());
  mickey::MatrixRef<ElementT, LayoutQmeta, true> tensor_packed_s_ref =
      Base::ShouldRearrangeMeta ? mickey::make_MatrixRef<ElementT, LayoutQmeta, true>(packed_scales_ref, meta_shape)
                                : tensor_scale;
  if (Base::ShouldRearrangeMeta) {
    onnxruntime::cuda::test::prepack_quant_scales_ref<ElementT, LayoutQmeta, QuantBlocking>(
        rows, columns, tensor_scale.const_ref(), tensor_packed_s_ref);
  }
    std::vector<ElementQOffset> packed_zp_ref(meta_shape.product());
    mickey::MatrixRef<ElementQOffset, LayoutQmeta, true> tensor_packed_zp_ref =
        Base::ShouldRearrangeMeta ? mickey::make_MatrixRef<ElementQOffset, LayoutQmeta, true>(packed_zp_ref, meta_shape)
                                  : tensor_offset;
    if (Base::ShouldRearrangeMeta) {
      onnxruntime::cuda::test::prepack_quant_offsets_ref<LayoutQmeta, QuantBlocking>(
          rows, columns, tensor_offset.const_ref(), tensor_packed_zp_ref);
    }
  for (int col = 0; col < meta_shape[1]; ++col) {
    for (int row = 0; row < meta_shape[0]; row += 2) {
      if (has_offset) {
        EXPECT_EQ(tensor_packed_zp_ref.at(row + 0, col), tensor_packed_zp.at(row + 0, col))
            << "quantized offset mismatch at [" << row << "," << col << "]"
            << " expected " << tensor_packed_zp_ref.at(row + 0, col)
            << " got " << tensor_packed_zp.at(row + 0, col)
            << " shape[" << rows << "," << columns << "]"
            << (ColumnQuantBlocking ? "Column-wise-block" : "Row-wise-block");
        if (row + 1 < meta_shape[0]) {
          EXPECT_EQ(tensor_packed_zp_ref.at(row + 1, col), tensor_packed_zp.at(row + 1, col))
              << "quantized offset mismatch at [" << (row + 1) << "," << col << "]"
              << " expected " << tensor_packed_zp_ref.at(row + 1, col)
              << " got " << tensor_packed_zp.at(row + 1, col)
              << " shape[" << rows << "," << columns << "]"
              << (ColumnQuantBlocking ? "Column-wise-block" : "Row-wise-block");
        }
      }

      EXPECT_EQ(tensor_packed_s_ref.at(row + 0, col), tensor_packed_scales.at(row + 0, col))
          << "quantized scale mismatch at [" << row << "," << col << "]"
          << " expected " << static_cast<float>(tensor_packed_s_ref.at(row + 0, col))
          << " got " << static_cast<float>(tensor_packed_scales.at(row + 0, col))
          << " shape[" << rows << "," << columns << "]"
          << (ColumnQuantBlocking ? "Column-wise-block" : "Row-wise-block");
      if (row + 1 < meta_shape[0]) {
        EXPECT_EQ(tensor_packed_s_ref.at(row + 1, col), tensor_packed_scales.at(row + 1, col))
            << "quantized scale mismatch at [" << (row + 1) << "," << col << "]"
            << " expected " << static_cast<float>(tensor_packed_s_ref.at(row + 1, col))
            << " got " << static_cast<float>(tensor_packed_scales.at(row + 1, col))
            << " shape[" << rows << "," << columns << "]"
            << (ColumnQuantBlocking ? "Column-wise-block" : "Row-wise-block");
      }
    }
  }

}

TEST(BlkQ4Fp16GemmPrepack, RowblockSmall) {
  testPrepack<false>(64, 32);
  // testPrepack<false, false>(64, 32);
}

TEST(BlkQ4Fp16GemmPrepack, ColblockSmall) {
  testPrepack<true>(64, 32);
  testPrepack<true, false>(64, 32);
}

TEST(BlkQ4Fp16GemmPrepack, Rowblock) {
  testPrepack<false>(64, 32);
  testPrepack<false>(64, 64);
  testPrepack<false>(64, 128);
  testPrepack<false>(64, 256);
  testPrepack<false>(128, 32);
  testPrepack<false>(256, 32);
  testPrepack<false>(256, 256);
  testPrepack<false, false>(64, 128);
  testPrepack<false, false>(128, 32);
  testPrepack<false, false>(256, 256);
}

TEST(BlkQ4Fp16GemmPrepack, Colblock) {
  testPrepack<true>(64, 32);
  testPrepack<true>(64, 64);
  testPrepack<true>(64, 128);
  testPrepack<true>(64, 256);
  testPrepack<true>(128, 32);
  testPrepack<true>(256, 32);
  testPrepack<true>(256, 256);
  testPrepack<true, false>(64, 128);
  testPrepack<true, false>(128, 32);
  testPrepack<true, false>(256, 256);
}

}  // namespace test
}  // namespace onnxruntime


