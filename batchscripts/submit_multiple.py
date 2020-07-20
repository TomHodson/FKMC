#!/usr/bin/env python3
from pathlib import Path
import subprocess as sb
import shutil
from sys import exit, argv
from datetime import datetime
import os
from CX1_batch_functions import CX1job
from CMTH_batch_functions import CMTHjob
import time
from munch import Munch
import argparse
import sys
from subprocess import CalledProcessError
import FKMC.import_funcs

def rel2home(p):
    'if a path is relative to home, it prints it as ~/...'
    homes = ['/rdsgpfs/general/user/tch14/home', '/rds/general/user/tch14/home']
    for home in homes:
        try: return '~' / p.relative_to(home)
        except ValueError: pass
    return p
   
parser = argparse.ArgumentParser(description='Submit multiple jobs with dependancies chained together.')
parser.add_argument('script', help='The ipynb script to use.', type = Path)
parser.add_argument('out', help='The output folder name, also used as the job name.', type = Path)
parser.add_argument('--debug', '-d', action = 'store_true', default = False, help='an integer for the accumulator')
args = Munch(vars(parser.parse_args()))
print(args)
    
### find the python script

ipynb_script = args.script.resolve()
job_folder_name = args.out
if not job_folder_name.is_absolute():
        job_folder_name = (Path.home() / 'HPC_data' / job_folder_name)
job_name = job_folder_name.stem
        
py_script = ipynb_script.parent / 'pure_python' / (ipynb_script.stem + '.py')
(ipynb_script.parent / 'pure_python').mkdir(exist_ok = True)

print(f'Python Script: {rel2home(ipynb_script)}') 
print(f'Job folder will be {rel2home(job_folder_name)}')

### determine if this is CX1 or CMTH
platform = 'CX1' if 'CX2_SCRATCH' in os.environ else 'CMTH'
print(f'Platform: {platform}')
JobClass = CMTHjob if platform == 'CMTH' else CX1job
    
### Regenerate the py from the ipynb based on timestamps
if not py_script.exists() or (ipynb_script.stat().st_mtime > py_script.stat().st_mtime):
    print('Regenerating py script from ipynb')
    sb.check_output(['jupyter', 'nbconvert', '--to', 'script', ipynb_script, '--output-dir', py_script.parent]) 

### Execute the script up to a line containing 'batch_params' to get info about the job
context = FKMC.import_funcs.execute_script(py_script)
    
### extract info from batch_params
batch_params = Munch(context.batch_params)

### make the job which gives access to some platform specific info like paths and such
#if args.debug: 
#    print('Debug mode: only doing 1 chain extension')
#    batch_params.chain_exts = batch_params.chain_exts[:2]
                    
jobs = [None for _ in batch_params.chain_exts]
for j,i in enumerate(batch_params.chain_exts):
    indices = (0, batch_params.total_jobs)
    print(f'Starting job with indices={indices} chain={i}')
    this_job_name = f"{job_name[:11]}_{i}"
    jobs[j] = JobClass(py_script, this_job_name, job_folder_name, indices,
                       chain_id = i,
                       startafter = jobs[j - 1] if j > 0 else None,
                       debug = args.debug,
                      )
    
    

### Make the file where the code and data will be saved
code_dir = jobs[0].submit_dir / 'code'
data_dir = jobs[0].submit_dir / 'data'
logs_dir = jobs[0].submit_dir / 'logs'
for d in [code_dir, data_dir, logs_dir]: d.mkdir(parents=True, exist_ok=True)

### copy the code over
shutil.copy(str(py_script), code_dir)
shutil.copy(str(ipynb_script), code_dir)

    
### Add some info about the job to run_descriptions
run_desc = Path.home() / f'FKMC/batchscripts/{platform}_run_descriptions.md' 

with run_desc.open('a') as f:
    from datetime import date
    timestamp = datetime.now().isoformat()
    print(f"\n## data/slurm_runs/{job_folder_name} {timestamp} {batch_params}", file = f)


print(f'See logs with: cat {str(jobs[0].submit_dir)}/logs/*')
#print(f'see erring jobs with find ~/data/slurm_runs/{jobnum}/logs/ -not -empty -name "*.err" | sort')

#do this at the very end so that 
print('Submitting jobs but held')
failed = False
for j in jobs: 
    try: j.submit(held = True)
    except CalledProcessError as e:
        print(f"Tried to submit job with id {j.job_id} but couldn't due to: ")
        print(e.output, e.stderr, e.returncode)
        failed = True
        break

    
if failed or input('Release the above jobs? (n to abort):') == 'n': 
    for j in jobs:
        if not j == None:
            try: j.cancel()
            except CalledProcessError as e:
                print(f"Tried to cancel job with id {j.job_id} but couldn't")
                print(e.output, e.stderr, e.returncode)
    exit()
    
for j in jobs: j.release()
    


   



  

