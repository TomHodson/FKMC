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