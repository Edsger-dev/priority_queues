"""

python setup.py build_ext --inplace

"""

from distutils.extension import Extension
# from Cython.Build import cythonize
import numpy as np
from setuptools import setup

setup(
    name="graph representation app",
    ext_modules = [
        Extension("graph",
                  ["graph.pyx"],
                  include_dirs=[np.get_include()]),
    ],
    zip_safe=False,
)
