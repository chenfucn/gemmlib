
#include "ms_blkq4gemm.h"
#include "blk_q4/f16_gemm_sm80.h"
#include "blk_q4/f16_quant_sm80.cuh"

#include "cuda_ptr.h"

namespace mickey {

namespace detail {

template <typename T>
gsl::span<T> make_span_from_str(void* ptr, size_t byte_size) {
  return gsl::make_span<T>(reinterpret_cast<T*>(ptr), byte_size / sizeof(T));
}

template <typename T>
gsl::span<const T> make_span_from_str(void const* ptr, size_t byte_size) {
  return gsl::make_span<const T>(reinterpret_cast<T const*>(ptr), byte_size / sizeof(T));
}

} // namespace detail

/**
 * @brief Helper function to run the GEMM kernel for 4bits quantized gemm on SM80.
 * Only support fp16 for now.
*/
template<
    typename ElementDequant,
    int block_size,
    bool column_wise_blocking,
    bool small_m,
    bool has_offsets>
std::string blkq4_gemm_sm80_T(int m, int n, int k, cudaStream_t stream,
                            gsl::span<ElementDequant const> a,
                            gsl::span<uint8_t const> weights,
                            gsl::span<ElementDequant const> scales,
                            gsl::span<uint8_t const> offsets,
                            gsl::span<ElementDequant> output) {
  static_assert(std::is_same<ElementDequant, cutlass::half_t>::value, "Only support fp16 for now");

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
    a.data(), LayoutInputA::packed({m, k}));

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
    scales.data(), LayoutInputQScale::packed({k/QuantBlocking::kRow, n/QuantBlocking::kColumn}));

  if (output.size_bytes() != m * n * sizeof(ElementOutput)) {
    return "Unexpected output tensor size: " + std::to_string(output.size_bytes())
           + " expected: " + std::to_string(m * n * sizeof(ElementOutput));
  }
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_output(
    output.data(), LayoutOutput::packed({m, n}));

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

std::string blkq4_fp16_gemm_sm80(
  int block_size,
  bool column_wise_blocking,
  int m, int n, int k, void* stream_ptr,
  void const* act_ptr, size_t a_size,
  void const* weights_ptr, size_t weights_size,
  void const* scales_ptr, size_t scales_size,
  void const* offsets_ptr, size_t offsets_size,
  void* output_ptr, size_t output_size) {

  using ElementT = cutlass::half_t;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto a = detail::make_span_from_str<ElementT>(act_ptr, a_size);
  auto weights = detail::make_span_from_str<uint8_t>(weights_ptr, weights_size);
  auto scales = detail::make_span_from_str<ElementT>(scales_ptr, scales_size);
  auto offsets = detail::make_span_from_str<uint8_t>(offsets_ptr, offsets_size);
  auto output = detail::make_span_from_str<ElementT>(output_ptr, output_size);

  switch (block_size)
  {
  case 16:
    if (column_wise_blocking) {
      if (m > 16) {
        if (!offsets_ptr || offsets_size == 0)
          return blkq4_gemm_sm80_T<ElementT, 16, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 16, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 16, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 16, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 16, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 16, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 16, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 16, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;

  case 32:
    if (column_wise_blocking) {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 32, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 32, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 32, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 32, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 32, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 32, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 32, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 32, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;

  case 64:
    if (column_wise_blocking) {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 64, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 64, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 64, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 64, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 64, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 64, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80_T<ElementT, 64, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80_T<ElementT, 64, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;
  }

  return "Unsupported block size: " + std::to_string(block_size);
}

template<
    int block_size,
    bool column_wise_blocking>
std::string blkq4_quant_size_sm80_T(
    int rows, int columns,
    int64_t& quant_weights_size,
    int64_t& quant_meta_size)
{
  if (rows == 0 || columns == 0) {
    return std::string();
  }

#ifndef NDEBUG
  constexpr bool ExtraBoundsChecking = true;
#else
  constexpr bool ExtraBoundsChecking = false;
#endif

  using ElementT = half;
  using Base = mickey::BlockwiseQuantization<
      ElementT,
      block_size,
      4,
      column_wise_blocking,
      ExtraBoundsChecking>;

  const auto q_weight_shape = Base::get_quant_weights_shape(rows, columns);
  const auto meta_shape = Base::get_quant_meta_shape(rows, columns);
  if ((rows % 64) != 0 || (rows % meta_shape[0]) != 0) {
    return "Cannot block quantize matrix with rows: " + std::to_string(rows);
  }
  if ((columns % 32) != 0 || (columns % meta_shape[1]) != 0) {
    return "Cannot block quantize matrix with columns: " + std::to_string(columns);
  }

  quant_weights_size = q_weight_shape.product();
  quant_meta_size = meta_shape.product();
  return std::string();
}

std::string blkq4_fp16_quant_size_sm80(
    int block_size,
    bool column_wise_blocking,
    int rows, int columns,
    int64_t& quant_weights_size,
    int64_t& quant_meta_size) {
  quant_weights_size = 0;
  quant_meta_size = 0;

  switch (block_size){
  case 16:
    if (column_wise_blocking)
      return blkq4_quant_size_sm80_T<16, true>(rows, columns, quant_weights_size, quant_meta_size);
    else
      return blkq4_quant_size_sm80_T<16, false>(rows, columns, quant_weights_size, quant_meta_size);
  case 32:
    if (column_wise_blocking)
      return blkq4_quant_size_sm80_T<32, true>(rows, columns, quant_weights_size, quant_meta_size);
    else
      return blkq4_quant_size_sm80_T<32, false>(rows, columns, quant_weights_size, quant_meta_size);
  case 64:
    if (column_wise_blocking)
      return blkq4_quant_size_sm80_T<64, true>(rows, columns, quant_weights_size, quant_meta_size);
    else
      return blkq4_quant_size_sm80_T<64, false>(rows, columns, quant_weights_size, quant_meta_size);
  }
  return "Unsupported block size: " + std::to_string(block_size);
} 


template<
    int block_size,
    bool column_wise_blocking,
    bool has_offsets>
std::string blkq4_quant_sm80_T(
    int rows, int columns, int leadingDimension,
    cudaStream_t stream,
    gsl::span<half const> weights,
    gsl::span<uint8_t> quant_weights,
    gsl::span<half> scales,
    gsl::span<uint8_t> offsets) 
{
  if (rows == 0 || columns == 0) {
    return std::string();
  }

#ifndef NDEBUG
  constexpr bool ExtraBoundsChecking = true;
#else
  constexpr bool ExtraBoundsChecking = false;
#endif

  using ElementT = half;
  using Base = mickey::BlockwiseQuantization<
      ElementT,
      block_size,
      4,
      column_wise_blocking,
      ExtraBoundsChecking>;

  using QuantBlocking = typename Base::QuantBlocking;
  using ElementW = typename Base::ElementW;
  using LayoutWPack = typename Base::LayoutWPack;
  using ElementQOffset = typename Base::ElementQOffset;
  using LayoutQmeta = typename Base::LayoutQmeta;

  int64_t quant_weights_size = 0;
  int64_t quant_meta_size = 0;
  auto err = blkq4_quant_size_sm80_T<block_size, column_wise_blocking>(
    rows, columns, quant_weights_size, quant_meta_size);
  if (!err.empty()) { return err; }

  if (quant_weights.size() < quant_weights_size) {
    return "Unexpected quantized weight tensor size: " + std::to_string(quant_weights.size())
           + " expected: " + std::to_string(quant_weights_size);
  }
  if (scales.size() < quant_meta_size) {
    return "Unexpected scale tensor size: " + std::to_string(scales.size())
           + " expected: " + std::to_string(quant_meta_size);
  }
  if constexpr (has_offsets) {
    if (offsets.size() < quant_meta_size) {
      return "Unexpected offset tensor size: " + std::to_string(offsets.size())
             + " expected: " + std::to_string(quant_meta_size);
    }
  }
  if (weights.size() < columns * leadingDimension) {
    return "Unexpected weight tensor size: " + std::to_string(weights.size())
           + " expected: " + std::to_string(columns * leadingDimension);
  }

  // allocate temp device memory for quantized weights and meta data, for prepacking
  auto quant_weights_dev_ptr = make_cuda_unique<ElementW>(quant_weights_size);
  auto q_weights_buf = gsl::make_span(quant_weights_dev_ptr.get(), quant_weights_size);

  auto scales_dev_ptr = make_cuda_unique<ElementT>(quant_meta_size);
  auto scales_buf = gsl::make_span(scales_dev_ptr.get(), quant_meta_size);

  cuda_unique_ptr<ElementQOffset> quant_offsets_dev_ptr = has_offsets ? make_cuda_unique<ElementQOffset>(quant_meta_size) : make_cuda_unique<ElementQOffset>();
  gsl::span<ElementQOffset> quant_offsets_buf = has_offsets ? gsl::make_span(quant_offsets_dev_ptr.get(), quant_meta_size) : gsl::span<ElementQOffset>();

  Base::block_quantize(q_weights_buf.data(), scales_buf.data(),
                       has_offsets ? quant_offsets_buf.data() : nullptr,
                       weights.data(), rows, columns, leadingDimension, stream);
  err = Base::prepack_weights(rows, columns,
                                   q_weights_buf, quant_weights, stream);
  if (!err.empty()) {
    return err;
  }
  err = Base::prepack_quant_meta(rows, columns,
                                 scales_buf, scales,
                                 quant_offsets_buf,
                                 has_offsets ? offsets : gsl::span<ElementQOffset>(),
                                 stream);

  return err;
}

std::string blkq4_fp16_quant_sm80(
    int block_size,
    bool column_wise_blocking,
    int rows, int columns, int leadingDimension,
    void* stream_ptr,
    void const* weights_ptr, size_t weights_size,
    void* quant_weights_ptr, size_t quant_weights_size,
    void* quant_scales_ptr, size_t quantscales_size,
    void* offsets_ptr, size_t offsets_size) 
{
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto weights = detail::make_span_from_str<half>(weights_ptr, weights_size);
  auto quant_weights = detail::make_span_from_str<uint8_t>(quant_weights_ptr, quant_weights_size);
  auto scales = detail::make_span_from_str<half>(quant_scales_ptr, quantscales_size);
  bool has_offsets = offsets_ptr && offsets_size > 0;
  auto offsets = has_offsets ? detail::make_span_from_str<uint8_t>(offsets_ptr, offsets_size) : gsl::span<uint8_t>();

  switch (block_size)
  {
  case 16:
    if (column_wise_blocking) {
      if (offsets.empty())
        return blkq4_quant_sm80_T<16, true, false>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
      else
        return blkq4_quant_sm80_T<16, true, true>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
    } else {
      if (offsets.empty())
        return blkq4_quant_sm80_T<16, false, false>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
      else
        return blkq4_quant_sm80_T<16, false, true>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
    }
    break;
  case 32:
    if (column_wise_blocking) {
      if (offsets.empty())
        return blkq4_quant_sm80_T<32, true, false>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
      else
        return blkq4_quant_sm80_T<32, true, true>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
    } else {
      if (offsets.empty())
        return blkq4_quant_sm80_T<32, false, false>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
      else
        return blkq4_quant_sm80_T<32, false, true>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
    }
    break;
  case 64:
    if (column_wise_blocking) {
      if (offsets.empty())
        return blkq4_quant_sm80_T<64, true, false>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
      else
        return blkq4_quant_sm80_T<64, true, true>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
    } else {
      if (offsets.empty())
        return blkq4_quant_sm80_T<64, false, false>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
      else
        return blkq4_quant_sm80_T<64, false, true>(rows, columns, leadingDimension, stream, weights, quant_weights, scales, offsets);
    }
    break;  
  default:
    return "Unsupported block size: " + std::to_string(block_size);
  }
}

}  // namespace mickey
