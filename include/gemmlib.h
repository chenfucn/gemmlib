
#pragma once

#include <string>
#include <driver_types.h>
#include <cuda_fp16.h>

#include <gsl/gsl>


namespace mickey {

std::string blkq4_fp16_gemm_sm80_dispatch(
  int block_size,
  bool column_wise_blocking,
  int m, int n, int k, cudaStream_t stream,
  gsl::span<half const> a,
  gsl::span<uint8_t const> weights,
  gsl::span<half const> scales,
  gsl::span<uint8_t const> offsets,
  gsl::span<half> output);

}  // namespace mickey
