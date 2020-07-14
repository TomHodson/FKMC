#!/usr/bin/env python3
from pathlib import Path
import subprocess as sb
import shutil
from sys import exit, argv
from datetime import datetime

#defaults
debug = False
submit_dir = Path('/workspace/tch14/slurm_runs_scratch')
log_dir = Path('/home/tch14/slurm_runs/') #must be on the NFS so it's accessible to all machines
time_limit = '48:00:00'


if len(argv) > 1:
    ipynb_script = Path(argv[1]).resolve()
else:
    ipynb_scripts = Path.cwd().glob('*.ipynb')
    ipynb_script = next(ipynb_scripts)
    
py_script = ipynb_script.parent / (ipynb_script.stem + '.py')

print(f'Converting the {ipynb_script} to py')
out = sb.check_output(['jupyter', 'nbconvert', '--to', 'script', ipynb_script])

name = input('Give this job a name: ')

if input(f'Is a time limit of {time_limit} ok? y/n: ').lower() == 'n':
    time_limit = input(f'What should it be? (as hh:mm:ss)')

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

generic_jobscript = Path.home() / 'FKMC/notebooks/batchjobs/generic_jobscript.sh'
sbatch_args = ['sbatch', '--hold', #hold the job until the folder has been created
               '--export=' + ",".join([
                   f'target={py_script.name}',
                   f'submit_dir={submit_dir}',
                   f'log_dir={log_dir}',
                   ]),
               f'--job-name={name}',
               f'--array={indices}',
               f'--parsable', #make the output parseable
               f'--exclude=dirac', #dirac always hangs
               f'--time=0-{time_limit}',
               f'--mail-type=ALL',
               f'--mail-user=tch14@imperial.ac.uk',
               f'--partition=pc',
               f'--output={log_dir}/%A/logs/%a.out',
               f'--error={log_dir}/%A/logs/%a.err',
               f'--nice=10', #make the jobs defer to others
               generic_jobscript] #rest of the config is in the jobscript

print(sbatch_args)

    
jobnum = sb.check_output(sbatch_args, encoding = 'utf8')
print(f'Job number is {jobnum}')

if input('Go ahead? y/n: ').lower() != 'y':
    print('Cancelling job')
    sb.check_output(['scancel', str(jobnum)])
    
run_desc = Path.home() / 'FKMC/notebooks/run_descriptions.md' 
jobnum = int(jobnum)

if name != 'test':
    with run_desc.open('a') as f:
        from datetime import date
        timestamp = datetime.now().isoformat()
        print(f"\n## data/slurm_runs/{jobnum} {name} {timestamp} ", file = f)

#make the directory structure
jobdir = Path(f"{submit_dir}/{jobnum}") #for the data, not backed up
joblogdir = Path(f"{log_dir}/{jobnum}") #for the logs and code, backed up

(submit_dir / str(jobnum)).mkdir(parents=True, exist_ok=False)
(log_dir / str(jobnum) / 'logs').mkdir(parents=True, exist_ok=False)
(log_dir / str(jobnum) / 'code').mkdir(parents=True, exist_ok=False)


#copy the code over to the code directory for later debugging
shutil.move(str(py_script), log_dir / str(jobnum) / 'code')
shutil.copy(str(ipynb_script), log_dir / str(jobnum) / 'code')
    
print('Releasing job')
sb.check_output(['scontrol', 'release', str(jobnum)])

print(f'See logs with: cat {log_dir}/{jobnum}/logs/*.out')
print(f'see erring jobs with find {log_dir}/{jobnum}/logs/ -not -empty -name "*.err" | sort')

