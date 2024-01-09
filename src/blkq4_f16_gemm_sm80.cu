
#include "gemmlib.h"
#include "blk_q4/f16_gemm_sm80.h"

namespace mickey {

/**
 * @brief Helper function to run the GEMM kernel for 4bits quantized gemm on SM80.
 * Only support fp16 for now.
*/
template<
    int block_size,
    bool column_wise_blocking,
    bool small_m,
    bool has_offsets>
std::string blkq4_gemm_sm80(int m, int n, int k, cudaStream_t stream,
                     gsl::span<half const> a,
                     gsl::span<uint8_t const> weights,
                     gsl::span<half const> scales,
                     gsl::span<uint8_t const> offsets,
                     gsl::span<half> output) {

  using ElementDequant = cutlass::half_t;
  using QuantBlocking =
    typename std::conditional<column_wise_blocking,
                     cutlass::MatrixShape<block_size, 1>,
                     cutlass::MatrixShape<1, block_size>>::type;

  using GemmRunner = BlkQ4F16GemmImpl<ElementDequant, QuantBlocking, small_m, has_offsets>;

  using ElementAccumulator = typename GemmRunner::ElementAccumulator;
  using ElementComputeEpilogue = typename GemmRunner::ElementComputeEpilogue;
  using ElementOutput = typename GemmRunner::ElementOutput;
  using ElementW = typename GemmRunner::ElementW;
  using ElementWPack = typename GemmRunner::ElementWPack;
  using ElementQScale = typename GemmRunner::ElementQScale;
  using ElementQOffset = typename GemmRunner::ElementQOffset;

  using LayoutInputA = typename GemmRunner::LayoutInputA;
  using LayoutOutput = typename GemmRunner::LayoutOutput;
  using LayoutInputWPack = typename GemmRunner::LayoutInputWPack;
  using LayoutInputQScale = typename GemmRunner::LayoutInputQScale;

  const cutlass::gemm::GemmCoord problem_size = {m, n, k};

  if (m == 0 || n == 0 || k == 0) {
    return std::string();
  }

  if (a.size_bytes() != m * k * sizeof(ElementDequant)) {
    return "Unexpected activation tensor size: " + std::to_string(a.size_bytes())
           + " expected: " + std::to_string(m * k * sizeof(ElementDequant));
  }
  cutlass::TensorRef<ElementDequant const, LayoutInputA> ref_a(
    reinterpret_cast<ElementDequant const *>(a.data()),
    LayoutInputA::packed({m, k}));

  if (weights.size_bytes() != (k/2) * (n/2) * sizeof(ElementWPack)) {
    return "Unexpected weight tensor size: " + std::to_string(weights.size_bytes())
           + " expected: " + std::to_string(k/2 * n/2 * sizeof(ElementWPack));
  }
  cutlass::TensorRef<ElementWPack const, LayoutInputWPack> ref_W(
    reinterpret_cast<ElementWPack const *>(weights.data()),
    LayoutInputWPack::packed({k/2, n/2}));

  if (scales.size_bytes() != (k/QuantBlocking::kRow) * (n/QuantBlocking::kColumn) * sizeof(ElementQScale)) {
    return "Unexpected scale tensor size: " + std::to_string(scales.size_bytes())
           + " expected: " + std::to_string((k/QuantBlocking::kRow) * (n/QuantBlocking::kColumn) * sizeof(ElementQScale));
  }
  cutlass::TensorRef<ElementQScale const, LayoutInputQScale> ref_scales(
    reinterpret_cast<ElementQScale const *>(scales.data()),
    LayoutInputQScale::packed({k/QuantBlocking::kRow, n/QuantBlocking::kColumn}));

  if (output.size_bytes() != m * n * sizeof(ElementOutput)) {
    return "Unexpected output tensor size: " + std::to_string(output.size_bytes())
           + " expected: " + std::to_string(m * n * sizeof(ElementOutput));
  }

  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_output(
    reinterpret_cast<ElementOutput *>(output.data()),
    LayoutOutput::packed({m, n}));

  // run GEMM
  cutlass::Status status;
  if constexpr (has_offsets) {
    if (offsets.size_bytes() != (k/QuantBlocking::kRow) * (n/QuantBlocking::kColumn) * sizeof(ElementQOffset)) {
      return "Unexpected offsets tensor size: " + std::to_string(offsets.size_bytes())
             + " expected: " + std::to_string((k/QuantBlocking::kRow) * (n/QuantBlocking::kColumn) * sizeof(ElementQOffset));
    }
    cutlass::TensorRef<ElementQOffset const, LayoutInputQScale> ref_offsets(
      reinterpret_cast<ElementQOffset const *>(offsets.data()),
      LayoutInputQScale::packed({k/QuantBlocking::kRow, n/QuantBlocking::kColumn}));
    status = GemmRunner::run(
      stream, problem_size, ref_a, ref_W, ref_scales, ref_offsets,
      ref_output, ref_output);
  } else {
    status = GemmRunner::run(
      stream, problem_size, ref_a, ref_W, ref_scales,
      ref_output, ref_output);
  }
  if (status != cutlass::Status::kSuccess) {
    return "Kernel execution failed: " + std::string(cutlassGetStatusString(status));
  }
  return std::string();
}

std::string blkq4_fp16_gemm_sm80_dispatch(
  int block_size,
  bool column_wise_blocking,
  int m, int n, int k, cudaStream_t stream,
  gsl::span<half const> a,
  gsl::span<uint8_t const> weights,
  gsl::span<half const> scales,
  gsl::span<uint8_t const> offsets,
  gsl::span<half> output) {

  switch (block_size)
  {
  case 16:
    if (column_wise_blocking) {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<16, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<16, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<16, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<16, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<16, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<16, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<16, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<16, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;

  case 32:
    if (column_wise_blocking) {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<32, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<32, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<32, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<32, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<32, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<32, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<32, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<32, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;

  case 64:
    if (column_wise_blocking) {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<64, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<64, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<64, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<64, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<64, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<64, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<64, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<64, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;
  }

  return "Unsupported block size: " + std::to_string(block_size);
}

}  // namespace mickey
