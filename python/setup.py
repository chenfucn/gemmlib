#
# Copyright (c) Microsoft Corporation. All rights reserved.
#

import os
from pathlib import Path
from setuptools import setup, Extension
from torch.utils import cpp_extension

cur_dir = Path(os.path.dirname(os.path.realpath(__file__)))
proj_dir = cur_dir.parent
ext_dir = cur_dir.joinpath('ms_blkq4linear_ext')
lib_build_dir = str(proj_dir.joinpath('build'))

MODULE_NAME = 'ms_blkq4linear_ext'

ext = cpp_extension.CUDAExtension(
    name = MODULE_NAME,
    sources=[str(ext_dir.joinpath('ms_blkq4linear.cpp'))],
    include_dirs=[str(proj_dir.joinpath('include'))],
    library_dirs=[lib_build_dir],
    libraries=["ms_blkq4gemm"]) 

setup(name=MODULE_NAME,
      ext_modules=[ext],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
