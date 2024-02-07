
# GemmLib

This repo contains a blockwise 4-bit quantized gemm kernel optimized for Nvidia Ampere GPUs.
Code of this gemm kernel, together with its quantization kernel is located under directory
`src`

We also provide a pytorch extension so that the kernel can be used in pytorch.

## Usage

To build the library that contains the gemm kernel and quantization code, clone this repo, and under the root directory of the local repo:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

Run `ms_blkq4gemm_test` to test the correctness of the code.

To build the pytorch extension, change to the root directory of the repo:

```
python python/setup.py install --user
```

Python file `python/blkq4linear_test.py` contains usage examples.

