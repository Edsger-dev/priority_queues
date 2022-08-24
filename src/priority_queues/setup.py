# python setup.py build_ext --inplace
from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(ext_modules=cythonize("pq_bin_heap.pyx"), include_dirs=[numpy.get_include()])
