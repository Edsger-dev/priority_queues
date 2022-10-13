"""

python setup.py build_ext --inplace

"""

from Cython.Build import cythonize
from setuptools import setup

setup(
    name="graph representation app",
    ext_modules=cythonize("graph.pyx"),
    zip_safe=False,
)
