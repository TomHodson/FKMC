#!/bin/bash

#SBATCH --job-name {name}
#SBATCH --time 0-10:00:00
#SBATCH --mail-type END
#SBATCH --mail-user tch14@imperial.ac.uk
#SBATCH --partition tch14
#SBATCH --constraint avx
#SBATCH --mem 2G

# Create a local directory to run in.
scratch=/workspace/$USER/scratch/$SLURM_JOB_ID
mkdir -p $scratch

echo Executing in $scratch on $(hostname)

. /workspace/tch14/miniconda3/etc/profile.d/conda.sh
conda activate cmth_intelpython3

cd {working_dir}
let "JOB_ID = PBS_ARRAY_INDEX - 1"
run_mcmc --job-id $JOB_ID --temp-dir $scratch --working-dir ./

# Copy back to submit directory. This is wrapped in an if statement so the
# working copy is only cleaned up if it has been copied back.
if cp output $SLURM_SUBMIT_DIR
then
  # Clean up.
  rm -rf $scratch
else
  echo "Copy failed. Data retained in $scratch on $(hostname)"
fi