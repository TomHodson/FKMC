#!/usr/bin/env python3
from pathlib import Path
import subprocess as sb
import shutil
from sys import exit, argv
from datetime import datetime

debug = False
if len(argv) > 1:
    ipynb_script = Path(argv[1]).resolve()
else:
    ipynb_scripts = Path.cwd().glob('*.ipynb')
    ipynb_script = next(ipynb_scripts)
    
py_script = ipynb_script.parent / (ipynb_script.stem + '.py')

name = input('Give this job a name: ')
if input(f'Is {py_script} the correct script? y/n: ').lower() == 'n': exit()

print(f'Converting the {ipynb_script} to py')
out = sb.check_output(['jupyter', 'nbconvert', '--to', 'script', ipynb_script])

contents = list(py_script.open().readlines())
for i, l in enumerate(contents):
    if 'batch_params' in l: break

try:
    context = dict()
    code = '\n'.join(contents[:i+1])
    exec(code, globals(), context)
except indexError:
    print('Didnt find batch_params in script')
   
job_params = context['batch_params']
N = job_params['total_jobs'] - 1
indices = f'0-{N}'

if input(f'Is {indices} the correct index specification? y/n: ').lower() == 'n': exit()

#du -sh /data/users/tch14/slurm_runs/*
generic_jobscript = Path.home() / 'FKMC/notebooks/batchjobs/generic_jobscript.sh'
sbatch_args = ['sbatch', '--hold', f'--export=target={py_script.name}', f'--job-name={name}', f'--array={indices}', '--parsable','--exclude=dirac', generic_jobscript]

    
jobnum = sb.check_output(sbatch_args, encoding = 'utf8')
print(f'Job number is {jobnum}')

if input('Go ahead? y/n: ').lower() != 'y':
    print('Cancelling job')
    sb.check_output(['scancel', str(jobnum)])
    
run_desc = Path.home() / 'FKMC/notebooks/run_descriptions.md' 
jobnum = int(jobnum)

with run_desc.open('a') as f:
    from datetime import date
    timestamp = datetime.now().isoformat()
    print(f"\n## data/slurm_runs/{jobnum} {name} {timestamp} ", file = f)

jobdir = Path(f"/data/users/tch14/slurm_runs/{jobnum}")
code_dir = (jobdir / 'code')
code_dir.mkdir(parents=True, exist_ok=False)

(jobdir / 'logs').mkdir()

shutil.move(str(py_script), code_dir)
shutil.copy(str(ipynb_script), code_dir)
    
print('Releasing job')
sb.check_output(['scontrol', 'release', str(jobnum)])
print(f'See logs with: cat ~/data/slurm_runs/{jobnum}/logs/*.out')
print(f'see erring jobs with find ~/data/slurm_runs/{jobnum}/logs/ -not -empty -name "*.err" | sort')

