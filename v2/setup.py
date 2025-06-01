# setup.py
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the path to the directory containing this script
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='flash_attn_v2_ext',
    ext_modules=[
        CUDAExtension(
            'flash_attn_v2_ext',
            sources=[
                os.path.join(this_dir, 'main.cpp'),
                os.path.join(this_dir, 'flash_attn_v2_kernel.cu'), # Your C++/CUDA source file
            ],
            extra_compile_args={
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode', 'arch=compute_75,code=sm_75', # For Turing (RTX 20-series, T4)
                    '-gencode', 'arch=compute_80,code=sm_80', # For Ampere (RTX 30-series, A100)
                    '-gencode', 'arch=compute_86,code=sm_86', # For Ampere (RTX 30-series, A6000)
                    '-gencode', 'arch=compute_89,code=sm_89', # For Ada Lovelace (RTX 40-series)
                    '-gencode', 'arch=compute_90,code=sm_90', # For Hopper (H100)
                    # Add more archs if needed for broader compatibility
                    '-U__CUDA_NO_HALF_OPERATORS__', # Ensure half operators are enabled
                    '-U__CUDA_NO_HALF_CONVERSIONS__', # Ensure half conversions are enabled
                    # If you want to debug, uncomment -g -G below, but expect performance hit
                    # '-g', '-G'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension # Custom build command for PyTorch extensions
    },
    packages=find_packages(), # Automatically find Python packages (if you have any)
)

