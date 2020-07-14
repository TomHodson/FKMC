## CX1 code
      
~/FKMC contains the bare minimum code!!!
~/FKMC/batchscripts contains the code to submit jobs and some job code

## PBS and Q commands:

    qsub -h -q v1_debug1 jobscript.sh
starts a job in held mode

    qstat -t 1649928[].pbs

gives some info on it. The `-t` expands array jobs.

    qrls 1649928[].pbs
    
releases it.

    qdel 1649928[].pbs
kills it

## What I want:
To be able to have one task in an array job extend the MCMC chain on from previous ones on from previous ones
also still be able to do honest to god repeats
To be able to start a new job that simply extends the chain of a previous one

Ways to achieve:

1) split the array job into multiple array jobs, making the 'chain extension' axis run over these jobs
then each array job could depend on the next

pros:
should make the dependency logic simpler
should fit well with the hope of also being able to create a job later that extends chains
would help get around the limited max size of array jobs

cons:
have to wait for every job to finish from a previous batch before the next one runs
What do we do if some jobs fails? (probably bail)

## Setting up conda on CX1:
http://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/support/applications/conda/

To regenerate the installation:

    rm -rf /rds/general/user/tch14/home/anaconda3
    module load anaconda3/personal
    anaconda-setup
    
    conda activate base
    conda install numpy scipy
    pip install munch
    cd ~/FKMC && pip install --editable .
    
To activate:

    module load intel-suite anaconda3/personal
    . $MKL_HOME/bin/mklvars.sh intel64
    . ~/anaconda3/etc/profile.d/conda.s$
    conda activate base
    
To install a jupyter kernal for an env:
https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments

    conda activate base #activate the env that you want the kernal in
    conda install ipykernal
    python -m ipykernel install --user --name conda_base --display-name "Anaconda3/base"
    
    


