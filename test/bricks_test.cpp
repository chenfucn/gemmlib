/**
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file brick_layout_test.cu
 */

#include "int_util.h"
#include "gemm_bricks.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"

#include <iostream>
#include "gtest/gtest.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace onnxruntime {
namespace cuda {
namespace test {

TEST(BrickLayoutTest, TestBrickLayout) {
  const bool print = true;
  int problem_m = 32;
  int problem_n = 4096;
  int problem_k = 4096 - 64;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 64>;
  mickey::BrickMap<ThreadblockShape, 2> brick_map;
  brick_map.init(problem_m, problem_n, problem_k, 108);
  for (int blockIdx_x = 0; blockIdx_x < 108; blockIdx_x++) {
    if (print) printf("TB:%3d", blockIdx_x);
    for (int brick_idx = 0; brick_idx < brick_map.bricks_per_tb; brick_idx++) {
      mickey::BrickPos brick_pos = brick_map.getBrickPosition({blockIdx_x, brick_idx});
      auto pp2 = brick_map.getGridPos(brick_pos);
      EXPECT_EQ(blockIdx_x, pp2.tb_idx);
      EXPECT_EQ(brick_idx, pp2.brick_offset);

      auto column_start = brick_map.getGridPos({brick_pos.m, brick_pos.n, 0});
      auto column_end = brick_map.getGridPos({brick_pos.m, brick_pos.n, brick_map.k_bricks - 1});
      int total_k_split = column_end.tb_idx - column_start.tb_idx + 1;
      int k_split_id = blockIdx_x - column_start.tb_idx;
      if (brick_pos.k == 0) {
        EXPECT_EQ(0, k_split_id);
      }
      if (brick_pos.k == brick_map.k_bricks - 1) {
        EXPECT_EQ(total_k_split - 1, k_split_id);
      }
      
      if (print) {
        printf(" (%2d, %2d, %2d, %2d, %2d)", brick_pos.m, brick_pos.n, brick_pos.k, total_k_split, k_split_id);
      }
    }
    if (print) {
      printf("\n");
    }

    for (int brick_idx = 0; brick_idx < brick_map.bricks_per_tb;) {
      int blk_m_idx;
      int blk_n_idx;
      int blk_k_idx; // debug print only
      int k_start;
      int k_end;
      int total_k_split;

        mickey::BrickPos brick_pos = brick_map.getBrickPosition({int(blockIdx_x), brick_idx});
        if (brick_pos.m >= brick_map.m_bricks) {
          break;
        }
        blk_m_idx = brick_pos.m;
        blk_n_idx = brick_pos.n;
        blk_k_idx = brick_pos.k;
        k_start = brick_pos.k * brick_map.kK;
        k_end = std::min(problem_k, (brick_pos.k + (brick_map.bricks_per_tb - brick_idx)) * brick_map.kK);

        if (brick_idx > 0) {
          EXPECT_EQ(blk_k_idx, 0);
        }

        int num_bricks = mickey::div_up(k_end - k_start, brick_map.kK);
        brick_idx += num_bricks;
        EXPECT_LE(brick_idx, brick_map.bricks_per_tb);

      if (print) {
        printf(" (%2d, %2d, %2d, %2d)", brick_pos.m, brick_pos.n, brick_pos.k, num_bricks);
      }
      
    }

    if (print) {
      printf("\n");
    }
  }
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
