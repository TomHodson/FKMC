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
### determine if this is CX1 or CMTH
platform = 'CX1' if 'PBS_ENVIRONMENT' in os.environ else 'CMTH'
if input(f'Is {platform} the correct platform? y/n: ').lower() == 'n': exit()

### Get the name of the job
name = input('Give this job a name: ')
if input(f'Is {py_script} the correct script? y/n: ').lower() == 'n': exit()

### 
print(f'Converting the {ipynb_script} to py')
out = sb.check_output(['jupyter', 'nbconvert', '--to', 'script', ipynb_script])

### Execute the script up to a line containing 'batch_params' to get info about the job
contents = list(py_script.open().readlines())
for i, l in enumerate(contents):
    if 'batch_params' in l: break

try:
    context = dict()
    code = '\n'.join(contents[:i+1])
    exec(code, globals(), context)
except indexError:
    print('Didnt find batch_params in script')
   
### extract 'total_jobs' from batch_params
job_params = context['batch_params']
N = job_params['total_jobs'] - 1
indices = f'0-{N}'

if input(f'Is {indices} the correct index specification? y/n: ').lower() == 'n': exit()

### Submit the job but keep it held, the jobnumber is returned
# set the evironment variables
# target=pythonscript
# that will tell the jobscript to target this python script

if platform == 'CMTH':
    generic_jobscript = Path.home() / 'FKMC/batchscripts/CMTH_jobscript.sh'
    sbatch_args = ['sbatch', '--hold', f'--export=target={py_script.name}', f'--job-name={name}', f'--array={indices}', '--parsable',
                   '--exclude=dirac',
                   CMTH_jobscript]

    
    jobnum = int(sb.check_output(sbatch_args, encoding = 'utf8'))

elif platform == 'CX1':
    #http://docs.adaptivecomputing.com/torque/4-0-2/Content/topics/commands/qsub.htm
    generic_jobscript = Path.home() / 'FKMC/batchscripts/CMTH_jobscript.sh'
    
    qsub_args = ['qsub', '-h', f'-v target={py_script.name}', f'-N {name}', f'-J {indices}',
                   CX1_jobscript]
    
    #Fill me in!
    #
    #
    #
    #
    #
    
print(f'Job number is {jobnum}')

### either cancel or release the job
if input('Go ahead? y/n: ').lower() != 'y':
    print('Cancelling job')
    
    if platform == 'CMTH': sb.check_output(['scancel', str(jobnum)])
    elif platform == 'CX1': 
    
### Add some info about the job to run_descriptions
run_desc = Path.home() / f'FKMC/batchscripts/{platform}_run_descriptions.md' 

with run_desc.open('a') as f:
    from datetime import date
    timestamp = datetime.now().isoformat()
    print(f"\n## data/slurm_runs/{jobnum} {name} {timestamp} {batch_params}", file = f)

if platform == 'CMTH':
    jobdir = Path(f"/data/users/tch14/slurm_runs/{jobnum}")
if platform == 'CX1':
    jobdir = Path(f"/rds/general/user/tch14/home/PBS_runs/{jobnum}")
   
code_dir = (jobdir / 'code')
code_dir.mkdir(parents=True, exist_ok=False)
(jobdir / 'logs').mkdir()

shutil.move(str(py_script), code_dir)
shutil.copy(str(ipynb_script), code_dir)
    
print('Releasing job')
if platform == 'CMTH': 
    sb.check_output(['scontrol', 'release', str(jobnum)])
    print(f'See logs with: cat ~/data/slurm_runs/{jobnum}/logs/*.out')
    print(f'see erring jobs with find ~/data/slurm_runs/{jobnum}/logs/ -not -empty -name "*.err" | sort')

    
if platform == 'CX1': 
   



