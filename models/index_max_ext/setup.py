import setuptools
import torch
import pathlib
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

# setup(name='index_max',
#       ext_modules=[CppExtension('index_max', ['index_max.cpp'])],
#       cmdclass={'build_ext': BuildExtension})

# setuptools.Extension(
#    name='index_max',
#    sources=['index_max.cpp'],
#    include_dirs=torch.utils.cpp_extension.include_paths(),
#    language='c++')

index_max_cpp_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute(), 'index_max.cpp'))
index_max_cuda_path = str(pathlib.Path(pathlib.Path(__file__).parent.absolute(), 'index_max_cuda.cu'))
setup(name='index_max',
      ext_modules=[CUDAExtension('index_max', [index_max_cpp_path, index_max_cuda_path])],
      cmdclass={'build_ext': BuildExtension})