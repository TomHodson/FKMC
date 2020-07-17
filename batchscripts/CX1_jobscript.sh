#!/usr/bin/env bash
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
echo PBS: PBS_ARRAY_INDEX = $PBS_ARRAY_INDEX
echo PBS: PBS_JOBID = $PBS_JOBID
echo PBS: SUBMIT_DIR = $SUBMIT_DIR
echo PBS: TMPDIR = $TMPDIR
echo PBS: PYTHON_SCRIPT = $PYTHON_SCRIPT
echo ------------------------------------------------------

#PBS_JOBID has format 1234456[10].pbs so have to parse it out, PBS_ARRAY_INDEX is just a number
export JOB_ID=`echo $PBS_JOBID | awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}'`
export TASK_ID=$PBS_ARRAY_INDEX #translate the index into array jobs

echo JOB_ID = $JOB_ID
echo CHAIN_ID = $CHAIN_ID
echo TASK_ID = $TASK_ID

module load intel-suite anaconda3/personal
. ~/anaconda3/etc/profile.d/conda.sh
conda activate base

cd $TMPDIR
python -u $SUBMIT_DIR/code/$PYTHON_SCRIPT >> $SUBMIT_DIR/logs/${JOB_ID}[${TASK_ID}].pythonlog

cp $TMPDIR/${TASK_ID}.npz $SUBMIT_DIR/data/

