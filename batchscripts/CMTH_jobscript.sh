#!/bin/bash
echo ------------------------------------------------------
echo Job is running on node $(hostname)
echo ------------------------------------------------------
echo SLURM: ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID
echo SLURM: TASK_ID is $SLURM_ARRAY_TASK_ID
echo ------------------------------------------------------

export TMPDIR=/workspace/$USER/scratch/$SLURM_JOB_ID
mkdir -p  $TMPDIR
cd $TMPDIR

export JOB_ID=$SLURM_ARRAY_JOB_ID
export TASK_ID=$SLURM_ARRAY_TASK_ID #translate the index into array jobs

echo JOB_ID = $JOB_ID
echo CHAIN_ID = $CHAIN_ID
echo TASK_ID = $TASK_ID

. /home/tch14/miniconda3/etc/profile.d/conda.sh
conda activate intelpython3.5

nice -n 19 python -u $SUBMIT_DIR/code/$PYTHON_SCRIPT >> $SUBMIT_DIR/logs/${JOB_ID}[${TASK_ID}].log

if cp $TMPDIR/${TASK_ID}.npz $SUBMIT_DIR/data/
then
echo "Copied data over, deleted scratch copy."
rm -r $TMPDIR 
fi