
#pragma once

#include <string>

namespace mickey {

/**
 * @brief Compute fp16 GEMM where the weight tensor is blockwise
 * quantized to int4.
 * 
 * @param[in]  block_size The block size for the blockwise quantization.
 * @param[in]  column_wise_blocking Whether numbers of each quantization blocks come from a single column or row
 * @param[in]  m The number of rows in the input tensor.
 * @param[in]  n The number of output features.
 * @param[in]  k The number of input features.
 * @param[in]  stream The CUDA stream to run the kernel on.
 * @param[in]  act_ptr The input tensor.
 * @param[in]  a_size The byte size of the input tensor.
 * @param[in]  weights_ptr The quantized weight tensor.
 * @param[in]  weights_size The byte size of the quantized weight tensor.
 * @param[in]  scales_ptr The scales for the quantized weight tensor.
 * @param[in]  scales_size The byte size of the scales.
 * @param[in]  offsets_ptr The offsets for the quantized weight tensor.
 * @param[in]  offsets_size The byte size of the offsets.
 * @param[out] output_ptr The output tensor.
 * @param[in]  output_size The byte size of the output tensor.
*/
extern std::string blkq4_fp16_gemm_sm80(
  int block_size,
  bool column_wise_blocking,
  int m, int n, int k, void* stream,
  void const* act_ptr, size_t a_size,
  void const* weights_ptr, size_t weights_size,
  void const* scales_ptr, size_t scales_size,
  void const* offsets_ptr, size_t offsets_size,
  void* output_ptr, size_t output_size);

/**
 * @brief Quantize a fp16 weight tensor to int4 using blockwise quantization.
 * @param[in]  block_size The block size for the blockwise quantization.
 * @param[in]  column_wise_blocking Whether numbers of each quantization blocks come from a single column or row
 * @param[in]  rows The number of input features.
 * @param[in]  columns The number of output features.
 * @param[in]  leadingDimension The leading dimension of the weight tensor.
 * @param[in]  stream The CUDA stream to run the kernel on.
 * @param[in]  weights The fp16 weight tensor.
 * @param[in]  weights_size The byte size of the weight tensor.
 * @param[out] quant_weights The quantized weight tensor.
 * @param[in]  quant_weights_size The byte size of the quantized weight tensor.
 * @param[out] quant_scales The scales for the quantized weight tensor.
 * @param[in]  quant_scales_size The byte size of the scales.
 * @param[out] offsets The offsets for the quantized weight tensor.
 * @param[in]  offsets_size The byte size of the offsets.
 */
extern std::string blkq4_fp16_quant_sm80(
    int block_size,
    bool column_wise_blocking,
    int rows, int columns, int leadingDimension,
    void* stream,
    void const* weights, size_t weights_size,
    void* quant_weights, size_t quant_weights_size,
    void* quant_scales, size_t quant_scales_size,
    void* offsets, size_t offsets_size);

/**
 * @brief Compute the size of the quantized tensors for the blockwise quantization.
 * 
 * @param[in]  block_size The block size for the blockwise quantization.
 * @param[in]  column_wise_blocking Whether numbers of each quantization blocks come from a single column or row
 * @param[in]  rows The number of input features.
 * @param[in]  columns The number of output features.
 * @param[out] quant_weights_size The number of elements in the quantized weight tensor.
 * @param[out] quant_meta_size The number of elements in the quantized meta data tensor.
*/
extern std::string blkq4_fp16_quant_size_sm80(
    int block_size,
    bool column_wise_blocking,
    int rows, int columns,
    int64_t& quant_weights_size,
    int64_t& quant_meta_size);

}  // namespace mickey
