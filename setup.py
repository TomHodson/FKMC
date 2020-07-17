from setuptools import setup, Extension, find_packages
import numpy
import os



setup(
    name="FKMC",
    version='0.5',
    description='Falikov Kimball simulations',
    author='Tom Hodson',
    author_email='tch14@iac.ac.uk',
    packages=find_packages(),
    include_package_data=True,
    zip_safe = False,
)



#command to build inplace is: python setup.py build_ext --inplace
#command to install is: pip install --editable .
