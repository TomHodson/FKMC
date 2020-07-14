#!/usr/bin/env python3
from pathlib import Path
import subprocess as sb
import shutil
from sys import exit, argv
from datetime import datetime
import os
from jobs import CX1job, CMTHjob
import time
from munch import Munch

### find the python script
debug = False
if len(argv) > 1:
    ipynb_script = Path(argv[1]).resolve()
else:
    ipynb_scripts = Path.cwd().glob('*.ipynb')
    ipynb_script = next(ipynb_scripts)

py_script = ipynb_script.parent / 'pure_python' / (ipynb_script.stem + '.py')
(ipynb_script.parent / 'pure_python').mkdir(exist_ok = True)

### determine if this is CX1 or CMTH
platform = 'CX1' if 'PBS_ENVIRONMENT' in os.environ else 'CMTH'
print(f'Detected platform as {platform}')
JobClass = CMTHjob if platform == 'CMTH' else CX1job
    
### Get the name of the job
job_name = input('Give this job a name: ')
job_folder_name = f'{job_name}_{time.time():.0f}'

### Regenerate the py from the ipynb based on timestamps
if not py_script.exists() or (ipynb_script.stat().st_mtime > py_script.stat().st_mtime):
    print('Regenerating py script from ipynb')
    sb.check_output(['jupyter', 'nbconvert', '--to', 'script', ipynb_script, '--output-dir', py_script.parent]) 

### Execute the script up to a line containing 'batch_params' to get info about the job
contents = list(py_script.open().readlines())
for i, l in enumerate(contents):
    if '#bath_params_end_flag' in l: break
try:
    context = dict()
    code = '\n'.join(contents[:i+1])
    exec(code, globals(), context)
    context = Munch(context)
except indexError:
    print("Didn't find batch_params in script")
   
### extract info from batch_params
batch_params = Munch(context.batch_params)

### make the job which gives access to some platform specific info like paths and such
job = JobClass(py_script, job_name, job_folder_name, batch_params.indices)

### Make the file where the code and data will be saved
code_dir = job.submit_dir / 'code'
data_dir = job.submit_dir / 'data'
logs_dir = job.submit_dir / 'logs'
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


print(f'See logs with: cat {str(job.submit_dir)}/logs/*')
#print(f'see erring jobs with find ~/data/slurm_runs/{jobnum}/logs/ -not -empty -name "*.err" | sort')


job.submit(held = False)

   



  

