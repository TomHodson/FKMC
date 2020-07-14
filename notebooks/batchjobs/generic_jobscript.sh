#!/bin/bash

# Create a local directory to run in.
scratch=/workspace/$USER/scratch/$SLURM_JOB_ID/

#supposed to be for saving data in the workspace but currently don't have a good way to do the copying
submit=$submit_dir/$SLURM_ARRAY_JOB_ID/
log=$log_dir/$SLURM_ARRAY_JOB_ID
machinelist=$log/machinelist.txt
#$log is where the logs and code are

echo ------------------------------------------------------
echo Job is running on node $(hostname)
echo ------------------------------------------------------
echo SLURM: scratch is $scratch

echo SLURM: submit is $submit
echo SLURM: ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID
echo SLURM: TASK_ID is $SLURM_ARRAY_TASK_ID
echo machinelist is in $machinelist
echo target python scipt is $log/code/${target}
echo ------------------------------------------------------

if mkdir -p  $scratch
then
    echo "$scratch created sucessfully"
else
    echo "Failed to create $scratch, exiting"
    exit 1
fi

cd $scratch
. /home/tch14/miniconda3/etc/profile.d/conda.sh
conda activate intelpython3.5

echo "Running Script"
#the -u forces standard output to flush immediately to disk
#the code is loaded from NFS and logs go there too, only the data is scp'd to workspace
#nice -n 19 python -u $log/code/${target}

#add a record of where each file is stored so I can rsync it back in later
touch machinelist
echo $(hostname):$scratch >> machinelist

echo Done!

# Copy back to submit directory. This is wrapped in an if statement so the
# working copy is only cleaned up if it has been copied back.
#if cp -a $scratch/. $log_dir
#then
  # Clean up.
#  rm -rf $scratch
#  echo "Copy Successful."
#else
#  echo "Copy failed. Data retained in $scratch on $(hostname)"
#fi