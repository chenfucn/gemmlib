/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Abstract:
 *   Benchmark to measure the performance of quantized b4 fp16 gemm kernels.
 */

#include <iostream>
#include <variant>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/tensor_fill.h"

// #include "gsl/gsl"
// #include "helper.h"

#include "gemm/kernel/quant_b4_gemm.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};


///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  int batch_count;

  int iterations;
  
  Options():
    help(false),
    problem_size({16, 22016, 4096}),
    batch_count(1),
    iterations(100000) { }

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    
    cmd.get_cmd_line_argument("iterations", iterations);

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "small_blkq4_fp16_gemm_benchmark example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./small_blkq4_fp16_gemm_benchmark --m=1024 --n=512 --k=1024 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product() * batch_count;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

template <
  typename QuantBlocking_,              ///! Shape of the quantization block, either 1xb or bx1
  typename ThreadblockShape_,                  ///! Warp-scoped matrix multiply-accumulate
  int SplitKSerial_ = 1,                ///! How many warps to split the K dimension in the same MxN block
  int Stages_ = 4                       ///! Stages of the pipelined mainloop
>
class QuantB4GemmTestDevKernel {
 public:
  using QuantBlocking = QuantBlocking_;
  using ThreadblockShape = ThreadblockShape_;
  static constexpr int kSplitK = SplitKSerial_;
  static constexpr int kStages = Stages_;

  using TestKernel = mickey::gemm::kernel::QuantB4Gemm<QuantBlocking, false, ThreadblockShape, kSplitK, kStages>;
  using Args = typename TestKernel::Params;

  cutlass::Status run(
    cudaStream_t stream,
    cutlass::gemm::GemmCoord const & problem_size,
    void* ptr_output,
    int output_byte_stride,
    void const *ptr_a,
    int a_byte_stride,
    void const *ptr_packed_b,
    int b_byte_stride,
    void const *ptr_scales,
    int scales_byte_stride) {

    Args args(problem_size, ptr_output, output_byte_stride,
              ptr_a, a_byte_stride, ptr_packed_b, b_byte_stride,
              ptr_scales, scales_byte_stride);
    cutlass::Status status = TestKernel::can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    dim3 grid(args.grid_tiled_shape_.m(), args.grid_tiled_shape_.n(), args.grid_tiled_shape_.k());
    dim3 block(TestKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename TestKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<TestKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        std::cerr << "Failed to obtain maximum shared memory size " << smem_size << " for kernel: "
                  << cudaGetErrorString(result) << "\n";
        return cutlass::Status::kErrorInternal;
      }
    }
   
    cutlass::Kernel<TestKernel><<<grid, block, smem_size, stream>>>(args);

    return cutlass::Status::kSuccess;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////

using QuantBlocking = cutlass::MatrixShape<1,32>;   // <- weights block per scale (1,16/32/64), (16/32/64,1)

// layout of quantization scale and offset tensors
using LayoutQMeta = 
    typename std::conditional<QuantBlocking::kRow == 1,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>::type;

using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 64>;
// Number of pipelines you want to use
constexpr int NumStages = 3;
constexpr int NumSplitK = 8;

using TestKernel = mickey::gemm::kernel::QuantB4Gemm<QuantBlocking, false, ThreadblockShape, NumSplitK, NumStages>;
using Args = typename TestKernel::Params;

int run(Options &options) {

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size = options.problem_size;

  // We don't care about correctness here, so just ensure the shapes are correct.
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_a(problem_size.mk());
  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 174321, 1.5f, -1.125f, 6);

  const int weights_rows = problem_size.k();
  const int weights_cols = problem_size.n();

  // q4 weights 16x16 tile is packed into 128B vector on the k dimension
  const int packed_weights_cols = weights_cols / 16;
  const int packed_weights_rows = weights_rows * (128 / 16);
  const int meta_rows = (weights_rows / QuantBlocking::kRow);
  const int meta_cols = (weights_cols / QuantBlocking::kColumn);

  cutlass::HostTensor<uint8_t, cutlass::layout::ColumnMajor> q4_weights({packed_weights_rows, packed_weights_cols});
  cutlass::reference::host::TensorFillRandomUniform(q4_weights.host_view(), 193456, 255, 0);
  cutlass::HostTensor<cutlass::half_t, LayoutQMeta> scales({meta_rows, meta_cols});

  // Allocate result tensor
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_d(
      problem_size.mn());
  cutlass::reference::host::TensorFill(tensor_d.host_view());

  // Copy data to device
  tensor_a.sync_device();
  q4_weights.sync_device();
  scales.sync_device();
  tensor_d.sync_device();

  // Launch kernel
  cudaStream_t stream = nullptr;

  Args args(problem_size, tensor_d.device_data(), tensor_d.stride(0) * sizeof(cutlass::half_t),
    tensor_a.device_data(), tensor_a.stride(0) * sizeof(cutlass::half_t),
    q4_weights.device_data(), q4_weights.stride(0) * sizeof(uint8_t),
    scales.device_data(), scales.stride(0) * sizeof(cutlass::half_t));
  // Allocate workspace memory
  size_t workspace_size = args.workspace_size();
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  args.set_workspace(workspace.get());

  cutlass::Status status = TestKernel::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Kernel parameter validation failed!" << std::endl;
    return -1;
  }

  dim3 grid(args.grid_tiled_shape_.m(), args.grid_tiled_shape_.n(), args.grid_tiled_shape_.k());
  dim3 block(TestKernel::kThreads, 1, 1);
  std::cout << "Launching kernel with grid " << grid.x << "x" << grid.y << "x" << grid.z << " and block " << block.x << "x" << block.y << "x" << block.z << std::endl;

  cudaError_t cuda_err;
  int smem_size = int(sizeof(typename TestKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
      cuda_err = cudaFuncSetAttribute(cutlass::Kernel<TestKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to obtain maximum shared memory size " << smem_size << " for kernel: "
                  << cudaGetErrorString(cuda_err) << "\n";
        return -1;
      }
  }
   
  cutlass::Kernel<TestKernel><<<grid, block, smem_size, stream>>>(args);
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    std::cerr << "Failed to launch kernel: " << cudaGetErrorString(cuda_err) << "\n";
    return -1;
  }

  // Result structure
  Result result;

  //
  // Construct events
  //

  cudaEvent_t events[2];

  for (auto & event : events) {
    result.error = cudaEventCreate(&event);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
  }

  std::cout << "Running Quantized Gemm...\n";

  // Record an event at the start of a series of GEMMs
  result.error = cudaEventRecord(events[0]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  //
  // Run profiling loop
  //

  for (int iter = 0; iter < options.iterations; ++iter) {
    cutlass::Kernel<TestKernel><<<grid, block, smem_size, stream>>>(args);
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
      std::cerr << "Failed to launch kernel: " << cudaGetErrorString(cuda_err) << "\n";
      return -1;
    }
  }

  //
  // Stop profiling loop
  //

  // Record an event when the GEMMs are complete
  result.error = cudaEventRecord(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Wait for work on the device to complete.
  result.error = cudaEventSynchronize(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Measure elapsed runtime
  float runtime_ms = 0;
  result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Compute average runtime and GFLOPs.
  result.runtime_ms = double(runtime_ms) / double(options.iterations);
  result.gflops = options.gflops(result.runtime_ms / 1000.0);

  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
  std::cout << " GFLOPs: " << result.gflops << std::endl;
  return 0;
}

int main(int argc, const char **argv) {
  
  bool notSupported = false;

  constexpr int deviceid = 0;
  auto err = cudaSetDevice(deviceid);
  if (err != cudaSuccess) {
      std::cerr << "Failed to run on device #" << deviceid << cudaGetErrorString(err) << std::endl;
      return -1;
  }

  // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.0. 
  //
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  std::cout << "Device: " << props.name << " with " << props.multiProcessorCount << " SMs" << std::endl;
  std::cout << "Device compute capability: " << props.major << "." << props.minor << std::endl;

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  printf("%d x %d x %d TF32 tensor op Matrix Multiply\n", \
    options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  return run(options);
}

