#!/bin/bash
#STUFF TO CHECK FOR EVERY JOB
#SBATCH --job-name multi_observable_temp_sweep
#SBATCH --time 0-24:00:00
#SBATCH --array=68-68


#SBATCH --mail-type ALL
#SBATCH --mail-user tch14@imperial.ac.uk

#SBATCH --partition pc
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --nice 19

# Create a local directory to run in.
scratch=/workspace/$USER/scratch/$SLURM_JOB_ID
submit=/data/users/tch14/slurm_runs/$SLURM_ARRAY_JOB_ID
target=phase_transitions

echo ------------------------------------------------------
echo Job is running on node $(hostname)
echo ------------------------------------------------------
echo SLURM: scratch is $scratch
echo SLURM: submit is $submit
echo SLURM: ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID
echo SLURM: TASK_ID is $SLURM_ARRAY_TASK_ID
echo ------------------------------------------------------


if mkdir -p  $submit
then
    echo "$submit created sucessfully"
else
    echo "Failed to create $submit"
fi


if mkdir -p  $scratch
then
    echo "$scratch created sucessfully"
    scratch_present=true
else
    echo "Failed to create $scratch"
    scratch_present=false
fi

#try to cd to a temporary workspace dir but if that doesn't work use the NFS
cd $scratch && cd $submit


. /home/tch14/miniconda3/etc/profile.d/conda.sh
conda activate intelpython3.5

python $SLURM_SUBMIT_DIR/${target}.py # >> ../logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.plog 2>&1

# Copy back to submit directory. This is wrapped in an if statement so the
# working copy is only cleaned up if it has been copied back.
if [ "$scratch_present" = true ] && cp -a $scratch/. $submit
then
  # Clean up.
  rm -rf $scratch
  echo "Copy Successful."
else
  echo "Copy failed. Data retained in $scratch on $(hostname)"
fi

#add a record of where each file is stored so I can rsync it back in later
machinelist=$SLURM_SUBMIT_DIR/logs/${SLURM_ARRAY_JOB_ID}.txt
touch machinelist
echo $(hostname):$scratch >> machinelist