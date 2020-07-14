NB ~ and ~/data are backed up. ~/workspace is not but has faster disk access.


    
~/data
    /pickled_data
        the current place I store half processed data
        
    /slurm_runs/{job_id}/
        /code/
            the code that ran that job
        /logs/
            the logs from that job
        {id}.npz -> the actual data files
        
~/workspace/pickled_data
    a place to put pickled data that's not doesn't need to be backed up but does need to have fast file access
    
    /workspace/tch14/pickled_data/{jobid}_processed.pickle
       this contains the results of loading in jobs using get_data_funcmap
        
~/workspace/slurm_runs_copies/
    this contains copies of ~/data/slurm_runs, put they for faster file access.
    
~/workspace/second_wind_pickled_data
    a collection of files from the second wind