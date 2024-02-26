/**
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file blkq4_fp16_util.h
 * @brief Utility functions for quantization and dequantization of fp16 data
 */

#include "blkq4_fp16_util.h"

namespace onnxruntime {
namespace test{

template <typename QuantBlocking, bool has_offset>
void blkq4_quant_util_test(int rows, int cols) {
    using QuantBaseT = BlkQuantizationRef<QuantBlocking, has_offset>;
    using LayoutQMeta = typename QuantBaseT::LayoutQMeta;

    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> src({rows, cols});
    cutlass::reference::host::TensorFillRandomUniform(src.host_view(), 51, -1.75f, 1.9f);
    cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor> q4_weights;
    cutlass::HostTensor<cutlass::half_t, LayoutQMeta> scales;
    cutlass::HostTensor<uint8_t, LayoutQMeta> offsets;

    QuantBaseT::QuantizeFp16To4Bit(src, q4_weights, scales, offsets);
    QuantBaseT::Dequantize4BitToFp16(src, q4_weights, scales, offsets);

    QuantBaseT::QuantizeFp16To4Bit(src, q4_weights, scales, offsets);
    // std::cout << "-------  src --------\n" << src.host_view() << std::endl;
    // std::cout << "-------  weights --------\n" << q4_weights.host_view() << std::endl;
    // std::cout << "-------  scales --------\n" << scales.host_view() << std::endl;
    // std::cout << "-------  offsets --------\n" << offsets.host_view() << std::endl;

    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> dst;
    QuantBaseT::Dequantize4BitToFp16(dst, q4_weights, scales, offsets);
    // std::cout << "-------  dst --------\n" << dst.host_view() << std::endl;

    bool pass = cutlass::reference::host::TensorEquals(src.host_view(), dst.host_view());
    if (!pass) {
        for (int r = 0; r < src.extent().row(); r++) {
            for (int c = 0; c < src.extent().column(); c++) {
                if (src.at({r, c}) != dst.at({r, c})) {
                    int w = q4_weights.at({r/2, c});
                    if (r % 2 == 0) {
                        w = w & 0x0f;
                    } else {
                        w = (w >> 4) & 0x0f;
                    }
                    std::cout << "Mismatch at " << r << ", " << c << " : " << src.at({r, c}) << " != " << dst.at({r, c}) << " w: " << w << " s: " << scales.at({r/32, c}) << " zp: " << int(offsets.at({r/32, c})) << std::endl;
                }
            }
        }
    }
    ASSERT_TRUE(pass);
}

TEST(QuantTestUtilTest, Quant) {
    blkq4_quant_util_test<cutlass::MatrixShape<32, 1>, true>(64, 16);
    blkq4_quant_util_test<cutlass::MatrixShape<32, 1>, false>(64, 16);
    blkq4_quant_util_test<cutlass::MatrixShape<1, 32>, true>(8, 64);
    blkq4_quant_util_test<cutlass::MatrixShape<1, 32>, false>(8, 64);

}

}  // namespace test
}  // namespace onnxruntime
