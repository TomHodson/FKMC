## CX1 code

This is my stripped down repo to run a simulation on CX1. 

NB: I have not set up any kind of syncing across the two systems!!!

I stripped out a lot! Then ran 

    cd ~/FKMC && pip install --editable .
      
~/FKMC contains the bare minimum code!!!
~/FKMC/batchscripts contains the code to submit jobs and some job code

did some testing with

    qsub -h -q v1_debug1 jobscript.sh
starts a job in held mode

    qstat 1649928[].pbs

gives some info on it. 

    qrls 1649928[].pbs
    
releases it.

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


