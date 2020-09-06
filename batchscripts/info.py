#!/usr/bin/env python3
print('Starting')
from pathlib import Path
import numpy as np
#import os
from munch import *
import argparse
from batch_functions import *
print('Imports done')   

parser = argparse.ArgumentParser(description='Get info about a job given its submission directory')
parser.add_argument('out', help='The output folder name', type = Path)
parser.add_argument('--files', action = 'store_true', default = False, help='Show the datafiles that have been generated.')
parser.add_argument('--logs', action = 'store_true', default = False, help='Show the logs that have been generated.')
args = Munch(vars(parser.parse_args()))

### find the python script
job_folder_name = args.out
if not job_folder_name.is_absolute():
        job_folder_name = (Path.home() / 'HPC_data' / job_folder_name)
job_name = job_folder_name.stem

print(f'Requested Job folder: {rel2home(job_folder_name)}')
if not job_folder_name.exists():
    print('Job folder does not exist.')
    exit()

code_dir = job_folder_name / 'code'
data_dir = job_folder_name / 'data'
logs_dir = job_folder_name / 'logs'
py_script = next(code_dir.glob('*.py'))



### determine if this is CX1 or CMTH
#platform = 'CX1' if 'CX2_SCRATCH' in os.environ else 'CMTH'
#print(f'Platform: {platform}')
    
#print('Executing python Script')
#context = execute_script(py_script)
#batch_params = Munch(context.batch_params)
#print(batch_params)

#Useful data in batch params:
#     chain_exts, 
#     structure_names  #names of each of the dimensions like ['Ts', 'Alphas']
#     structure_dimensions ##the dimensions themselves like [np.linspace(0.1,5,100), np.linspace(0.1,2,100)]


if args.files:
    from collections import defaultdict
    def name2id(n): return tuple(map(int,n.split('_')))
    
    datafiles = dict()
    task_ids = set()
    chain_ids = defaultdict(set)
    for f in data_dir.glob('*.npz'):
        task_id, chain_id = name2id(f.stem)
        datafiles[(task_id, chain_id)] = f
        task_ids.add(task_id)
        chain_ids[task_id].add(chain_id)
    
    #find the maximum chain extension done so far
    N_chains = max(max(c) for c in chain_ids.values()) + 1
    N_jobs = max(task_ids) + 1
    visual_rep = np.full(shape = (N_jobs, N_chains), fill_value = ' ')
    for task_id in task_ids:
        
        #set to '!' for all fields that are expected to be present
        for chain_id in range(max(chain_ids[task_id])):
            visual_rep[task_id, chain_id] = '!'
        
        #overwrite to '#' if they actually are
        for chain_id in chain_ids[task_id]:
            visual_rep[task_id, chain_id] = '#'
    
    print(f'''
    Key:
    ! : no output file present
    # : output file present
    
    N_jobs = {N_jobs}
    <--- N_chains = {N_chains} --->
    ''')
    for line, task_id in zip(visual_rep, range(N_jobs)):
        print(f'{task_id:4}: ' + ''.join(line) + '|')
        
if args.logs:
    pass
    
    
    




    


   



  

