/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    int_util.h
 *
 * Abstract:
 *   Utils for integer operations, mostly constants and bit manipulation.
 */

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

namespace mickey {

CUTLASS_HOST_DEVICE
constexpr int div_up(int a, int b) {
  return (a + b - 1) / b;
}

CUTLASS_HOST_DEVICE
constexpr int round_up(int a, int b) {
  return div_up(a, b) * b;
}

}  // namespace mickey
