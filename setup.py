from distutils.core import setup
from setuptools import find_packages

setup(
    name="si_for_nn",
    version="0.0.0",
    description="si_for_nn",
    license="MIT",
    packages=find_packages(where="."),
    install_requires=[
        "numpy",
        "tensorflow",
        "mpmath",
        "matplotlib",
        "scipy",
        "seaborn",
        "statsmodels"
    ]
)
