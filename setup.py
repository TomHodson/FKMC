from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os

os.environ["CC"] = "icc"
os.environ["LDSHARED"] = "icc -shared"


extensions = [
    Extension("FKMC/*", ["FKMC/*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args = ['-ip_no_inlining', '-xhost', '-qopenmp'],
        extra_link_args = [],
        libraries=['mkl_intel_ilp64', 'mkl_intel_thread', 'mkl_core', 'iomp5', 'pthread', 'm', 'dl'],
        #library_dirs=["${MKLROOT}/lib/intel64"], # Doesn't seem necessary for either machine
        ),
]

setup(
    name="FKMC",
    version='0.2',
    description='Falikov Kimball simulations',
    author='Tom Hodson',
    author_email='tch14@iac.ac.uk',
    packages=find_packages(),
    include_package_data=True,
    data_files=[
        ('',['FKMC/CMTH_runscript.sh']),
        ('',['FKMC/CX1_runscript.sh']),
    ],
    zip_safe = False,
    ext_modules=cythonize(extensions, annotate=True, language_level=3),
    install_requires=['Click'],
    entry_points='''
        [console_scripts]
        run_mcmc=FKMC.jobmanager:run_mcmc_command
    ''',
)



#command to build inplace is: python setup.py build_ext --inplace
#command to install is: pip install --editable .
