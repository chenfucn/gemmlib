/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    gemm_bricks.h
 *
 * Abstract:
 *   We partition the GEMM computation into 3D bricks,
 *   and try to evenly distribute these bricks into
 *   device SMs.
 */

#pragma once

#include "int_util.h"

namespace mickey {

/**
 * @brief Brick position in the logical 3D space
 *    m, n, k are the brick index in each dimension
 */
struct BrickPos {
  int m;
  int n;
  int k;
};

/**
 * @brief Position in the device SMs space
 *    tb_idx is the thread block index
 *    brick_offset is the offset in the thread block
 */
struct GridPos {
  int tb_idx;
  int brick_offset;
};

/**
 * @brief Map bricks to device SMs
 *    We partition the GEMM computation into 3D bricks,
 *    and try to evenly distribute these bricks into
 *    device SMs.
 */
template <typename ThreadblockShape, int KMultiplier = 2> 
struct BrickMap {
  static constexpr int kM = ThreadblockShape::kM;
  static constexpr int kN = ThreadblockShape::kN;
  static constexpr int kK = ThreadblockShape::kK * KMultiplier;

  int m_bricks;
  int n_bricks;
  int k_bricks;

  int bricks_per_tb; // number of bricks each thread block processes

  /**
   * @brief Initialize the brick map with problem size and number of thread blocks
   * @param m problem size in dimension M
   * @param n problem size in dimension N
   * @param k problem size in dimension K
   * @param num_tbs number of thread blocks
  */
  CUTLASS_HOST_DEVICE
  void init(
    int m,
    int n,
    int k,
    int num_tbs) {
    m_bricks = div_up(m, kM);
    n_bricks = div_up(n, kN);
    k_bricks = div_up(k, kK);
    bricks_per_tb = div_up(m_bricks * n_bricks * k_bricks, num_tbs);
  }

  CUTLASS_HOST_DEVICE
  BrickPos getBrickPosition(const GridPos& pp) {
    int brick_Position = pp.tb_idx * bricks_per_tb + pp.brick_offset;
    int k = brick_Position % k_bricks;
    int mn = brick_Position / k_bricks;
    int n = mn % n_bricks;
    int m = mn / n_bricks;
    return BrickPos{.m = m, .n = n, .k = k};
  }

  CUTLASS_HOST_DEVICE
  GridPos getGridPos(const BrickPos& lp) {
    int position = lp.m * n_bricks * k_bricks + lp.n * k_bricks + lp.k;
    int tb_idx = position / bricks_per_tb;
    int brick_offset = position % bricks_per_tb;
    return GridPos{.tb_idx = tb_idx, .brick_offset = brick_offset};
  }

};

}  // namespace mickey
