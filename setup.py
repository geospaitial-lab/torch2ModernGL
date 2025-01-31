import glob
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
include_dirs = [os.path.join(this_dir, "src", "include")]

sources = glob.glob('src/*.cpp') + glob.glob('src/*.cu')

setup(
    name='torch2moderngl',
    version='0.0.1',
    description='Send data from ModernGL to pytorch and back.',
    long_description='Simple utility for transferring data between ModernGL textures and pytorch tensors.',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='torch2moderngl._cuda_functions',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
