
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "ms_blkq4gemm.h"

constexpr size_t half_t_size = 2; // avoid include 'half' type

std::vector<torch::Tensor> blkq4linear_quant(
  int block_size,
  bool col_wise,
  bool has_offsets,
  torch::Tensor weights) {
  TORCH_CHECK(weights.scalar_type() == torch::kFloat16, "blkq4linear_quant only supports float16 weights");
  const int64_t weights_size = weights.numel();
  if (!weights_size) {
    TORCH_CHECK(false, "blkq4linear_quant only supports non-empty tensors");
  }
  TORCH_CHECK(weights.is_contiguous(), "blkq4linear_quant only supports contiguous tensors");
  const auto shape = weights.sizes();
  TORCH_CHECK(shape.size() == 2, "blkq4linear_quant only supports 2D tensors");
  int64_t out_features = shape[0];
  int64_t in_features = shape[1];

  // Compute the size of the quantized weights and metadata
  int64_t quant_weights_size;
  int64_t quant_meta_size;
  auto err = mickey::blkq4_fp16_quant_size_sm80(
    block_size, col_wise, in_features, out_features,
    quant_weights_size, quant_meta_size);
  TORCH_CHECK(err.empty(), err);

  if (weights.device().type() == at::kCUDA) {
    const int64_t g_idx = weights.device().index();
    const at::cuda::CUDAGuard device_guard(g_idx);

    torch::Tensor q_weights = torch::empty(
      {quant_weights_size}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, g_idx));
    torch::Tensor q_scales = torch::empty(
      {quant_meta_size}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, g_idx));
    torch::Tensor q_zp;
    void* q_zp_ptr = nullptr;
    size_t q_zp_byte_size = 0;
    if (has_offsets) {
      q_zp = torch::empty(
        {quant_meta_size}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, g_idx));
      q_zp_ptr = q_zp.data_ptr();
      q_zp_byte_size = quant_meta_size * sizeof(uint8_t);
    }

    auto err = mickey::blkq4_fp16_quant_sm80(
      int(block_size), bool(col_wise), int(in_features), int(out_features), int(in_features),
      nullptr, // specify stream?
      weights.data_ptr(), weights_size * half_t_size,
      q_weights.data_ptr(), quant_weights_size * sizeof(uint8_t),
      q_scales.data_ptr(), quant_meta_size * half_t_size,
      q_zp_ptr, q_zp_byte_size);

    TORCH_CHECK(err.empty(), err);
    return {q_weights, q_scales, q_zp};
  } else {
    TORCH_CHECK(false, "blkq4linear_quant only supports CUDA tensors");
  }

  return {};
}

torch::Tensor blkq4linear_forward(
  int block_size,
  bool col_wise,
  int in_features,
  int out_features,
  torch::Tensor input,
  torch::Tensor q_weights,
  torch::Tensor q_scales,
  torch::Tensor q_zp) {
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "blkq4linear_forward only supports float16 inputs");
  const auto a_shape = input.sizes();
  TORCH_CHECK(a_shape.size() == 2 && a_shape[1] == in_features, "Activation shape is not compatible with the weights");

  TORCH_CHECK(input.device().type() == torch::kCUDA, "blkq4linear_forward only supports CUDA inputs");
  const int64_t g_idx = input.device().index();
  TORCH_CHECK(q_weights.device().index() == g_idx && q_scales.device().index() == g_idx && q_zp.device().index() == g_idx,
              "blkq4linear_forward requires all tensors on the same device");

  const at::cuda::CUDAGuard device_guard(g_idx);

  auto q_output = torch::empty(
    {a_shape[0], out_features},
    torch::TensorOptions().dtype(torch::kFloat16).device(input.device()));

  const void* q_zp_ptr = nullptr;
  const size_t zp_size = q_zp.numel();
  if (zp_size > 0) {
    q_zp_ptr = q_zp.data_ptr();
  }
  
  auto err = mickey::blkq4_fp16_gemm_sm80(
    block_size, col_wise, int(a_shape[0]), out_features, in_features,
    nullptr, // specify stream?
    input.data_ptr(), input.numel() * half_t_size,
    q_weights.data_ptr(), q_weights.numel() * sizeof(uint8_t),
    q_scales.data_ptr(), q_scales.numel() * half_t_size,
    q_zp_ptr, zp_size * sizeof(uint8_t),
    q_output.data_ptr(), q_output.numel() * half_t_size);
  TORCH_CHECK(err.empty(), err);

  return q_output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant", &blkq4linear_quant, "Quantize float 16 weights");
  m.def("forward", &blkq4linear_forward, "Forward pass for block quantized linear layer");
}