#!/bin/bash

#submit checklist:
# check that the ipynb script has the correct values for N_steps, N_burn_in and thin
# check that the ipynb script saves the correct values in the np.savez_compressed call
# check that the jobscript has the correct range 0-(N_jobs-1)
# check that the jobscript has the correct target and time
# restart the kernal and run the script once to test it 

echo 'Script currently in test only mode'
echo 'Is this a new script? Have you checked that the ipynb script saves the correct values in the np.savez_compressed call?'
echo `grep -A 2 'np.savez' reweight_test_batch_script.ipynb`
read

echo 'Have you checked that the ipynb script has the correct values for N_steps, N_burn_in and thin?'
echo `grep -A 2 'N_steps =' reweight_test_batch_script.ipynb`
read

echo 'Have you checked that the jobscript has the correct range 0-(N_jobs-1), target and time?'
read
echo 'Have you restarted the kernal and run the script once to test it'
read
echo "Is $(pwd) the correct working directory?"
read
echo 'Ok going ahead to submit the job'

echo 'Converting the ipynb scripts to py'
jupyter nbconvert --to script *.ipynb

du -sh /data/users/tch14/slurm_runs/*

jobnum=`sbatch --parsable jobscript.sh`

jobdir="/data/users/tch14/slurm_runs/$jobnum"

echo "## data/slurm_runs/$jobnum" >> ~/FKMC/notebooks/run_descriptions.md

echo $jobdir
mkdir -p $jobdir/code

cp ./*.ipynb $jobdir/code/
cp ./*.py $jobdir/code/
cp ./jobscript.sh $jobdir/code/

echo "$jobnum SUBMITTED $(date)" >> /data/users/tch14/jobs_events