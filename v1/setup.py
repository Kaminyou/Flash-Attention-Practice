# setup.py
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the path to the directory containing this script
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='flash_attn_v1_ext', # Name of your Python package
    ext_modules=[
        CUDAExtension(
            name='flash_attn_v1_ext', # Name of the compiled C++ extension module
            sources=[
                os.path.join(this_dir, 'main.cpp'),
                os.path.join(this_dir, 'flash_attn_v1_kernel.cu'), # Your C++/CUDA source file
            ],
            extra_compile_args={
                'cxx': ['-O3'], # C++ compiler flags
                'nvcc': ['-O3', # CUDA compiler flags
                         '-arch=sm_70', # Target a specific CUDA architecture (e.g., Volta, adjust for your GPU)
                         '-gencode=arch=compute_70,code=sm_70', # This helps generate code for the specified architecture
                         # More architectures can be added, e.g., for Ampere: '-gencode=arch=compute_80,code=sm_80'
                         '--expt-relaxed-constexpr', # Allows more flexibility for constexpr functions
                         '-U__CUDA_NO_HALF_OPERATORS__', # Undefine macros for half precision if needed
                         '-U__CUDA_NO_HALF_CONVERSIONS__'
                         ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension # Custom build command for PyTorch extensions
    },
    packages=find_packages(), # Automatically find Python packages (if you have any)
)
