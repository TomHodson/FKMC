#!/bin/bash

#SBATCH --partition tch14
#SBATCH --job-name multi_observable_temp_sweep
#SBATCH --time 0-08:00:00
#SBATCH --mail-type END
#SBATCH --mail-user tch14@imperial.ac.uk

# Create a local directory to run in.
scratch=/workspace/$USER/scratch/$SLURM_JOB_ID
submit=$SLURM_SUBMIT_DIR/data/$SLURM_ARRAY_JOB_ID/

echo ------------------------------------------------------
echo Job is running on node $(hostname)
echo ------------------------------------------------------
echo SLURM: scratch is $scratch
echo SLURM: submit is $submit
echo SLURM: ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID
echo SLURM: TASK_ID is $SLURM_ARRAY_TASK_ID
echo ------------------------------------------------------


mkdir -p $scratch
mkdir -p  $submit

# Copy input file (called in this case input_file) to the directory job will
# run in. Slurm will start in the directory you submit your job from - so be
# sure this is in the home or data directory as workspace isn't shared between
# nodes.
cd $scratch


. /workspace/tch14/miniconda3/etc/profile.d/conda.sh
conda activate cmth_intelpython3_2
python multi_observable_temp_sweep.ipynb > ${SLURM_ARRAY_TASK_ID}.log

# Copy back to submit directory. This is wrapped in an if statement so the
# working copy is only cleaned up if it has been copied back.
if cp scratch/* $submit
then
  # Clean up.
  rm -rf $scratch
else
  echo "Copy failed. Data retained in $scratch on $(hostname)"
fi