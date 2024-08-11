from setuptools import setup
from Cython.Build import cythonize

'''
Compile Cython code with: python3 setup.py build_ext --inplace
'''

setup(
    ext_modules=cythonize("extract_sum_mp.pyx")
)