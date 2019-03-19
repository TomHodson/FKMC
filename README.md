
## Things to do:
- fix bug in the runtime
- investigate the asymetric energy spectra
- finish implementing last_complete_inner_index
- calculte the approx runtime as a function of system size and number of steps
- add a way to print out useful information about a completed job datafile
- change the jobs to save their state at intermediate intervals
- add tempering
- improve the visualisation of which jobs have produced errors, output logs and output results
- Modify setupmcmc and gathermcmc so that the results.hdf5 file made by gather rather than by setup

## Build:
process outline in the build.sh file

## Run a job
- use the python setupscript 
- remember to build
- cd to the log dir and run qsub ../runscript

## Useful commands
### Setup up a custom jupyter server on on CX1 do
```
ssh -f -N -R 8967:localhost:8967 chlorine04 #-f mean background -N means no shell
jupyter notebook 
```
### Mount the RCS home directory on my own do

```
sshfs tch14@login.cx1.hpc.ic.ac.uk: /workspace/tch14/cx1_home
```
unmount it with:
```
fusermount -u /workspace/tch14/cx1_home
```

### Activate my python env on the CMTH system
```
. ./Documents/notebooks/jupyter_env/bin/activate
```

### Setup a local jupyter lab instance
```
cd /workspace/tch14
sshfs tch14@login.cx1.hpc.ic.ac.uk: cx1_home
. jupyterlab_env/bin/activate
jupyter lab
```

### Get CPU info
```
less /proc/cpuinfo
```

### To rename a jupyterlab instance, you have to rebuild it
```
activate the environment
pip install npm nodejs
jupyter lab build --name='CMTH Jupyter'
```

### add a jupyter kernal
```
. env_name/bin/activate
python -m ipykernel install --user --name env_name --display-name "Python3 (intel)"
```

## Notes
### 17th Jan
#### Making a conda env on the cmth machine to run local code

installed 64bit minidonda to /workspace/tch14/miniconda3
configure conda to save envs and packages in the workspace through /home/tch14/.condarc
```
channels:
  - intel
  - conda-forge
  - defaults
envs_dirs:
  - /workspace/tch14/conda-envs
pkgs_dirs:
  - /workspace/tch14/conda-pkgs
```

Then make the intel conda environment with nodejs for jupyterlab and scipy to pull in some scientific packages
I had to make sure CMTH_scipy package was unloaded to get the correct scipy to load
```
conda create -n cmth_intelpython3 intelpython3_core python=3 nodejs scipy
conda activate cmth_intelpython3
conda install -y jupyterlab click hdf5 ipykernel h5py matplotlib cython
jupyter lab build --name='CMTH Jupyter'
python -m ipykernel install --user --name cmth_intelpython3 --display-name "Python3 (conda intel)"
```



### 16th Jan
Trying to set up a local instance of Jupyter Lab to use when CX1 is slow, the problem is that my cython files are all compiled for CX1 with intel compilers, I need to figure out if I can run them locally without recompilling or if I can compile for both targets in situ.


### 5th November
I added some git aliases: https://git-scm.com/book/en/v2/Git-Basics-Git-Aliases
```
$ git config --global alias.co checkout
$ git config --global alias.br branch
$ git config --global alias.ci commit
$ git config --global alias.st status
```

Ran a few thousand jobs, the first 1000 had 100 jobs fail, I need to check the error logs for those jobs and try to figure out if it's a specific machine that can't run my code, maybe it has a different architecture?

The majority of the failed jobs ran on cx1-138-2-3.cx1.hpc.ic.ac.uk though the 2 and the 3 are different each time. The rest seem to have timed out for some reason.
EDIT: The jobs on 138 probably failed because of AVX instructions (added :avx=true)


Questions:
- Can I mount the CX1 drive locally? What's the best way? smb://rds.imperial.ac.uk/RDS/user/tch14/
- How can I redirect the error and output to a specific folder? read docs
- can add :avx=true to get force only running on nodes that have avx
- an alternative to click is docopt
- use the imperial VPN


### trying to get MKL into my cython script
it may be that source $MKL_HOME/bin/mklvars.sh intel64 is a necessary step before compilation
I then added params to setup.py from https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor and it seemed to work
next I need to try to run STEMR to see if it works


### 2nd November
- got everything working on my own machine
- had to add a '-ip_no_inlining' compiler flag to get the intel compiler to work with cythonize
- to buld the cython files use:

```
module load anaconda3/personal intel-suite
python setup.py build_ext --inplace
```

- to install the package and enable the commands like run_mcmc use:
```
pip install --editable ./path/to/project
```

Now having problems with the linker and undefined symbols, I probably need to include some stuff from  $INTEL_HOME though haven't figure out what yet

ok I think the problem was that I wasn't using intel python, I'm now making a conda environment with intel python 3 in it called idp using the instructions from https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda

to avoid adding intel as a channel i used:
```
conda create -c intel -n idp intelpython3_core python=3
```

I then manually added this env as a jupyter kernal as in https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook
```
python -m ipykernel install --user --name idp --display-name "Python3 (intel)"
```

a good reference for using ICC with cython https://software.intel.com/en-us/articles/thread-parallelism-in-cython

Everything seems to work, though I think I need to figure out how to write the data back into the main file using mpi or something



### 30th October
currently working on the setup_mcc, run_mcc, and gather_mcc functions

### 29th October
- Could I use perturbation theory to avoid diagonalising the full matrix? By somehow doing an approximate energy and then refining it?
- Don't forget to think about adding termpering
- Add hd5 support