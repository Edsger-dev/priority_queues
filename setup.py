import os
import re

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

requirements = ["cython", "numpy", "pandas", "scipy", "psutil"]
setup_requirements = ["cython", "numpy"]
test_requirements = ["pytest"]

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Get the licence
with open("LICENSE") as f:
    license = f.read()

extra_compile_args = ["-Ofast"]

extensions = [
    Extension(
        "priority_queues.commons",
        ["src/priority_queues/commons.pyx"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "priority_queues.pq_bin_heap",
        ["src/priority_queues/pq_bin_heap.pyx"],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "priority_queues.pq_bin_heap_tests",
        ["src/priority_queues/pq_bin_heap_tests.pyx"],
    ),
    Extension(
        "priority_queues.dijkstra",
        ["src/priority_queues/dijkstra.pyx"],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "priority_queues.pq_fib_heap",
        ["src/priority_queues/pq_fib_heap.pyx"],
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="priority_queues",
    version=find_version("src", "priority_queues", "__init__.py"),
    description="Priority queues for path algorithms",
    author="Francois Pacull",
    author_email="pacullfrancois@gmail.com",
    license=license,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "priority_queues.commons": ["src/priority_queues/commons.pxd"],
        "priority_queues.pq_bin_heap": ["src/priority_queues/pq_bin_heap.pxd"],
    },
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        include_path=["src/priority_queues/"],
    ),
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    extras_require={"test": test_requirements},
    include_dirs=[numpy.get_include()],
)
