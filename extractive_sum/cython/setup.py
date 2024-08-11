from setuptools import setup
from Cython.Build import cythonize
import numpy

'''
Compile Cython code with: python3 setup.py build_ext --inplace
'''

setup(
    ext_modules=cythonize("extract_sum_mp.pyx"),
    include_dirs=[numpy.get_include()]
)