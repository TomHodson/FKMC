#!/usr/bin/env bash
#PBS -N debug_array_job
#PBS -lselect=1:ncpus=1:mem=4gb:avx=true
##PBS -lwalltime=24:00:00
#PBS -lwalltime=00:00:60
#PBS -J 1-6

##PBS -o /rds/general/user/tch14/home/HPC_data/${PBS_JOBID}/logs/%{PBS_ARRAY_INDEX}.out
##PBS -e /rds/general/user/tch14/home/HPC_data/${PBS_JOBID}/logs/%{PBS_ARRAY_INDEX}.err

#PBS -o /rds/general/user/tch14/home/HPC_data/
#PBS -e /rds/general/user/tch14/home/HPC_data/

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------

module load intel-suite anaconda3/personal
. /home/tch14/anaconda3/etc/profile.d/conda.sh
conda activate idp


let "JOB_ID = PBS_JOBID" #translate the main unchanging job id number
let "TASK_ID = PBS_ARRAY_INDEX - 1" #translate the index into array jobs

cd $TMPDIR

echo "hello world!" > %{TASK_ID}.npz
#python -u $submit/code/${target}

cp $TMPDIR/%{TASK_ID}.npz /rds/general/user/tch14/home/HPC_data/${PBS_JOBID}/



