/**
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file swizzle_loader_test.cu
 */


#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm_coord.h"

#include "cutlass/util/host_tensor.h"

#include "gemm/warp/swizzle_tile_loader.h"

#include "gtest/gtest.h"


namespace onnxruntime {
namespace cuda {
namespace test {

template <typename Shape_, typename ElementT_, int NumThreads, int SplitKSerial_ = 1>
struct SwizzleLoaderTestKernel {
  using Shape = Shape_;
  using ElementT = ElementT_;
  using SwizzleLoader = mickey::gemm::warp::SwizzleTileLoader<Shape::kN, Shape::kK * sizeof(ElementT), NumThreads>;

  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kThreadCount = SwizzleLoader::kThreads;
  static constexpr int kWarps = kThreadCount / 32;

  struct Params {
    cutlass::gemm::GemmCoord problem_size_;
    cutlass::gemm::GemmCoord grid_tiled_shape_;
    ElementT* ptr_output_;
    const int output_stride_;
    ElementT const * ptr_input_;
    const int input_stride_;
    int gemm_k_size_{0};

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      ElementT* ptr_output,
      const int output_stride,
      ElementT const * ptr_input,
      const int input_stride
    ):
      problem_size_(problem_size),
      ptr_output_(ptr_output),
      output_stride_(output_stride),
      ptr_input_(ptr_input),
      input_stride_(input_stride),
      gemm_k_size_(mickey::round_up(mickey::div_up(problem_size.k(), kSplitK), Shape::kK)),
      grid_tiled_shape_(cutlass::gemm::GemmCoord(
        1, mickey::div_up(problem_size.n(), Shape::kN), kSplitK
      )) { }

  };

  struct SharedStorage {
    ElementT storage[2][Shape::kN * Shape::kK];
  };

  CUTLASS_HOST_DEVICE
  SwizzleLoaderTestKernel() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    const int warp_idx = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int lane_b_k_offset = lane_id % 4;
    const int lane_b_n_offset = lane_id / 4;
    const int n_start = blockIdx.y * Shape::kN;   // TODO! change to thread block shape
    const int n_end = min(params.problem_size_.n(), (blockIdx.y + 1) * Shape::kN);
    const int k_start = blockIdx.z * params.gemm_k_size_;
    const int k_end = min(params.problem_size_.k(), (blockIdx.z + 1) * params.gemm_k_size_);

    SwizzleLoader loader(
        params.ptr_input_,
        params.input_stride_ * sizeof(ElementT),
        n_start, n_end,
        k_start * sizeof(ElementT), k_end * sizeof(ElementT),
        threadIdx.x);
    int double_buffer_i = 1;
    CUTLASS_PRAGMA_UNROLL
    for (int load_k = k_start; load_k < k_end; load_k += Shape::kK) {
      double_buffer_i ^= 1;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < SwizzleLoader::kGloadSplit; ++i) {
        loader.load_to_smem_split(threadIdx.x, shared_storage.storage[double_buffer_i], i);
      }
      ++loader;

      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();

      // if (threadIdx.x == 0){
      //   printf(" ==================== shared_storage == %d ==================", load_k);
      //   for (int i = 0; i < Shape::kN * Shape::kK; ++i) {
      //     if (i % Shape::kK == 0)
      //       printf("\n");
      //     printf("%3d, ", shared_storage.storage[i]);
      //   }
      //   printf("\n ==================== shared_storage ====================\n");
      // }
      if constexpr(Shape::kN == 8 && ((Shape::kK * sizeof(ElementT)) == 64)) {
        // it's a 8 x 64 shape, 4 tiles stacked all on the k dimension
        ElementT frag[8 * 64 / sizeof(ElementT)];
        loader.load_fragment_k64(lane_id, shared_storage.storage[double_buffer_i], 0, frag);

        int byte_stride = params.output_stride_ * sizeof(ElementT);
        unsigned* fragment_b = reinterpret_cast<unsigned*>(frag);

          int n = n_start + lane_b_n_offset;
          int k = load_k * sizeof(ElementT) + lane_b_k_offset * 4;
          uint8_t* dst = reinterpret_cast<uint8_t*>(params.ptr_output_) + n * byte_stride + k;
          unsigned* dst_blk = reinterpret_cast<unsigned*>(dst);
          if (n < n_end && k < k_end * sizeof(ElementT))
            *dst_blk = *fragment_b;
          fragment_b++;

          dst_blk = reinterpret_cast<unsigned*>(dst + 16);
          if (n < n_end && k + 16 < k_end * sizeof(ElementT))
            *dst_blk = *fragment_b;
          fragment_b++;

          dst_blk = reinterpret_cast<unsigned*>(dst + 32);
          if (n < n_end && k + 32 < k_end * sizeof(ElementT))
            *dst_blk = *fragment_b;
          fragment_b++;

          dst_blk = reinterpret_cast<unsigned*>(dst + 48);
          if (n < n_end && k + 48 < k_end * sizeof(ElementT))
            *dst_blk = *fragment_b;
          fragment_b++;
      } else {
        // Shape::kN % 16 == 0
        constexpr int k_depth = 32 / sizeof(ElementT);
        ElementT frag[k_depth * Shape::kN];
        constexpr int kMmaIters = Shape::kK / k_depth;
        // constexpr int kIters = mickey::div_up(kMmaIters, kWarps);
        for (int k_iter = 0; k_iter < kMmaIters; ++k_iter) {
          const int k_mma_iter = k_iter; // * kWarps;
          if (warp_idx != kWarps - 1) {
            break;
          }
          const int k_offset = k_mma_iter * k_depth;
          const int proc_k = load_k + k_offset;
          if (proc_k < k_end) {
            loader.load_fragment_k32(lane_id, shared_storage.storage[double_buffer_i], k_offset * sizeof(ElementT), frag);
            // if (lane_id == 0)
            //   printf(" ==== k_offset: %d, proc_k %d, k_end %d ====\n", k_offset, proc_k, k_end);
            // printf("%3d, %3d, ", frag[0], frag[1]);
            // printf("%3d, %3d, ", frag[2], frag[3]);
            // printf("%3d, %3d, ", frag[4], frag[5]);
            // printf("%3d, %3d, ", frag[6], frag[7]);
            // printf("%3d, %3d, ", frag[8], frag[9]);
            // printf("%3d, %3d, ", frag[10], frag[11]);
            // printf("%3d, %3d, ", frag[12], frag[13]);
            // printf("%3d, %3d, ", frag[14], frag[15]);
            // if (lane_id == 0)
            //   printf("\n ==================== frag ====================\n");

            int byte_stride = params.output_stride_ * sizeof(ElementT);
            unsigned* fragment_b = reinterpret_cast<unsigned*>(frag);
            CUTLASS_PRAGMA_UNROLL
            for (int b_tile_n = 0; b_tile_n < (Shape::kN/16); ++b_tile_n) {
              int n = n_start + b_tile_n * 16 + lane_b_n_offset;
              int k = proc_k * sizeof(ElementT) + lane_b_k_offset * 4;
              uint8_t* dst = reinterpret_cast<uint8_t*>(params.ptr_output_) + n * byte_stride + k;
              unsigned* dst_blk = reinterpret_cast<unsigned*>(dst);
              if (n < n_end && k < k_end * sizeof(ElementT))
                *dst_blk = *fragment_b;
              fragment_b++;

              uint8_t* dst8 = dst + 8 * byte_stride;
              dst_blk = reinterpret_cast<unsigned*>(dst8);
              if (n + 8 < n_end && k < k_end * sizeof(ElementT))
                *dst_blk = *fragment_b;
              fragment_b++;

              dst_blk = reinterpret_cast<unsigned*>(dst + 16);
              if (n < n_end && k + 16 < k_end * sizeof(ElementT))
                *dst_blk = *fragment_b;
              fragment_b++;

              dst_blk = reinterpret_cast<unsigned*>(dst + 8 * byte_stride + 16);
              if (n + 8 < n_end && k + 16 < k_end * sizeof(ElementT))
                *dst_blk = *fragment_b;
              fragment_b++;
            }

          }
        }
      }
    }

  }

};

template <typename Shape_, typename ElementT_, int NumThreads_, int SplitKSerial_ = 1>
void test_swizzle_loader(int m, int n, int k) {
  using Shape = Shape_;
  using ElementT = ElementT_;
  using TestKernel = SwizzleLoaderTestKernel<Shape, ElementT, NumThreads_, SplitKSerial_>;

  std::cout << "Test swizzle loader Shape (";
  std::cout << Shape::kM << ", " << Shape::kN << ", " << Shape::kK << ") (";
  std::cout << m << ", " << n << ", " << k << ")" << std::endl;

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  cutlass::HostTensor<ElementT_, cutlass::layout::ColumnMajor> tensor_b({k + 16, n});
  cutlass::HostTensor<ElementT_, cutlass::layout::ColumnMajor> tensor_output({k + 8, n});
  int v = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < tensor_b.extent()[0]; ++j) {
      uint16_t vv = 65530;
      if (j < k) {
        vv = v;
        v++;
      }
      tensor_b.at({j, i}) = vv;
    }
  }
  // printf("============= tensor_b =============\n");
  // for (int row = 0; row < tensor_b.extent()[0]; ++row) {
  //   for (int col = 0; col < tensor_b.extent()[1]; ++col) {
  //     printf("%3d, ", tensor_b.at({row, col}));
  //   }
  //   printf("\n");
  // }

  tensor_b.sync_device();
  tensor_output.sync_device();
  int output_stride = tensor_output.stride(0);
  int input_stride = tensor_b.stride(0);

  typename TestKernel::Params params(
    problem_size,
    tensor_output.device_data(),
    tensor_output.stride(0),
    tensor_b.device_data(),
    tensor_b.stride(0)
  );
  dim3 grid(params.grid_tiled_shape_.m(), params.grid_tiled_shape_.n(), params.grid_tiled_shape_.k());
  dim3 block(TestKernel::kThreadCount, 1, 1);

  cudaError_t result;
  int smem_size = int(sizeof(typename TestKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    result = cudaFuncSetAttribute(cutlass::Kernel<TestKernel>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem_size);

    ASSERT_EQ(result, cudaSuccess) << "Failed to obtain maximum shared memory size " << smem_size << " for kernel: "
                << cudaGetErrorString(result) << "\n";
  }

  cutlass::Kernel<TestKernel><<<grid, block, smem_size>>>(params);
  cudaDeviceSynchronize();
  tensor_output.sync_host();
  // printf("============= tensor_output =============\n");
  for (int row = 0; row < tensor_output.extent()[0]; ++row) {
    for (int col = 0; col < tensor_output.extent()[1]; ++col) {
      // printf("%3d, ", tensor_output.at({row, col}));
      if (row < k && tensor_output.at({row, col}) != tensor_b.at({row, col})) {
        printf("Error at %d, %d, %d, %d\n", row, col, tensor_output.at({row, col}), tensor_b.at({row, col}));
        ASSERT_EQ(tensor_output.at({row, col}), tensor_b.at({row, col}));
      }
    }
    // printf("\n");
  }
}

TEST(SwizzleLoader, LoadBTest) {
  test_swizzle_loader<cutlass::gemm::GemmShape<1, 32, 64>, uint8_t, 32>(1, 41, 128 + 16);
  test_swizzle_loader<cutlass::gemm::GemmShape<1, 32, 32>, uint16_t, 32>(1, 40, 80);
  test_swizzle_loader<cutlass::gemm::GemmShape<1, 64, 32>, uint16_t, 32>(1, 140, 80);
  test_swizzle_loader<cutlass::gemm::GemmShape<1, 8, 32>, uint16_t, 32>(1, 140, 80);
  test_swizzle_loader<cutlass::gemm::GemmShape<1, 16, 64>, uint16_t, 32>(1, 16, 64 * 9 - 16);
  test_swizzle_loader<cutlass::gemm::GemmShape<1, 64, 64>, uint16_t, 32, 4>(1, 128 - 5, 64 * 15 + 16);
  test_swizzle_loader<cutlass::gemm::GemmShape<1, 64, 128>, uint8_t, 32, 4>(1, 128 - 5, 128 * 17 + 16);

  test_swizzle_loader<cutlass::gemm::GemmShape<1, 16, 64>, uint16_t, 64, 4>(1, 32 + 1, 64 * 16 + 16);
  test_swizzle_loader<cutlass::gemm::GemmShape<1, 32, 64>, uint16_t, 256, 4>(1, 64 - 3, 64 * 16 + 16);

  // test_swizzle_loader<cutlass::gemm::GemmShape<1, 16, 32>, uint8_t>(1, 16, 64);

  test_swizzle_loader<cutlass::gemm::GemmShape<1, 32, 32>, uint8_t, 32>(1, 34, 128 + 16);
  test_swizzle_loader<cutlass::gemm::GemmShape<1, 64, 16>, uint16_t, 32>(1, 60, 64 - 8);

}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
