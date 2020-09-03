from pathlib import Path
import subprocess as sb
from subprocess import CalledProcessError
import shutil
from sys import exit, argv
from datetime import datetime
from re import match
from munch import *

def rel2home(p):
    'if a path is relative to home, it prints it as ~/...'
    homes = ['/rdsgpfs/general/user/tch14/home', '/rds/general/user/tch14/home']
    for home in homes:
        try: return '~' / p.relative_to(home)
        except ValueError: pass
    return p

def execute_script(py_script):
    contents = list(py_script.open().readlines())
    flag = '#bath_params_end_flag'
    for i, l in enumerate(contents):
        if flag in l: break
    try:
        context = dict()
        code = '\n'.join(contents[:i+1])
        exec(code, globals(), context)
        context = Munch(context)
        return context
    except IndexError:
        print(f"Didn't find {flag} in script")
        raise IndexError

class CX1job(object): 
    job_id = None
    jobscript = Path.home() / 'FKMC/batchscripts/CX1_jobscript.sh'
    running = False
    
    def __init__(self, python_script, job_name, job_folder_name, array_indices, chain_id, startafter = None, debug = False): 
        self.python_script = python_script
        self.job_name = job_name
        self.job_folder_name = job_folder_name
        self.array_indices = array_indices
        self.chain_id = chain_id
        self.submit_dir = Path('/home/tch14/HPC_data') / job_folder_name
        self.startafter = startafter
        self.debug = debug
    def submit(self, held = False):
        #http://docs.adaptivecomputing.com/torque/4-0-2/Content/topics/commands/qsub.htm
        start, stop = self.array_indices
        indices = f'{start}-{stop-1}'
    
        args = ['qsub', 
                     '-v', ', '.join([
                         f'PYTHON_SCRIPT={self.python_script.name}',
                         f'SUBMIT_DIR={self.submit_dir}',
                         f'DEBUG={self.debug}',
                         f'CHAIN_ID={self.chain_id}',
                     ]),
                     '-N', f'{self.job_name}',
                     '-J', f'{indices}',
                     #'-lselect=1:ncpus=10:mem=20gb:avx=true',
                     '-lselect=1:ncpus=8:mem=10gb:cpumodel=24',
                     #'-lselect=1:ncpus=8:mem=10gb:avx2=true',
                     '-lwalltime=24:00:00'if not self.debug else '-lwalltime=00:10:00',
                     '-o', str(self.submit_dir / 'logs'),
                     '-e', str(self.submit_dir / 'logs'),
                    ]
        
        if held: args.append('-h') #hold if necessary
        if self.startafter: args.append(f'-W depend=afterany:{self.startafter.job_id}[].pbs')
            
        args.append(str(self.jobscript)) #the actual script itself
        
        print('qsub command:\n', ' '.join(args))
        
        try:
            #jobid has the form 1649928[].pbs
            jobstring = sb.check_output(args, encoding = 'utf8', stderr=sb.PIPE)
            self.job_id = match(r'(\d+)\[\]\.pbs', jobstring).groups()[0]
        except CalledProcessError as e:
            print(e.output, e.stderr, e.returncode)
            raise e
        
        print(f'Job created with id {self.job_id}')
        return self

    def cancel(self):
        if self.job_id == None: 
            print("Can't cancel, no job_id")
            return
        print(f'Cancelling job with id {self.job_id}')
        try:
            sb.check_output(['qdel', self.job_id + '[]'], encoding = 'utf8', stderr=sb.PIPE)
        except CalledProcessError as e:
            print(e.output, e.stderr, e.returncode)
            raise e
    
    def release(self):
        if self.job_id == None: 
            print("Can't release, no job_id")
            return
        print(f'Releasing job with id {self.job_id}')
        try:
            output = sb.check_output(['qrls', self.job_id + '[]'], encoding = 'utf8', stderr=sb.PIPE)
            self.running = True
        except CalledProcessError as e:
            print(e.output, e.stderr, e.returncode)
            raise e
            
    def status(self):
        try:
            output = sb.check_output(['qstat', self.job_id], encoding = 'utf8', stderr=sb.PIPE)
        except CalledProcessError as e:
            #print(e.output, e.stderr, e.returncode)
            return 'Job finished'
        return output
    
class CMTHjob:
    job_id = None
    jobscript = Path.home() / 'FKMC/batchscripts/CMTH_jobscript.sh'
    running = False
    
    def __init__(self, python_script, job_name, job_folder_name, array_indices, chain_id, startafter = None, debug = False): 
        self.python_script = python_script
        self.job_name = job_name
        self.job_folder_name = job_folder_name
        self.array_indices = array_indices
        self.chain_id = chain_id
        self.submit_dir = Path('/home/tch14/HPC_data') / job_folder_name
        self.startafter = startafter
        self.debug = debug
    
    def submit(self, held = False):
        start, stop = self.array_indices
        indices = f'{start}-{stop-1}%20'
        args = ['sbatch', 
                       '--hold', 
                       '--parsable',
                      f'--export=' + ','.join([
                           f'PYTHON_SCRIPT={self.python_script.name}',
                           f'SUBMIT_DIR={self.submit_dir}',
                           f'DEBUG={self.debug}',
                           f'CHAIN_ID={self.chain_id}',
                      ]),
                      f'--job-name={self.job_name}',
                      f'--array={indices}', 
                       '--exclude=dirac',
                       '--kill-on-invalid-dep=yes',
                       #'--mail-type=ALL',
                       #'--mail-user=tch14@ic.ac.uk',
                      f'--output={self.submit_dir / "logs/%A[%a].log"}',
                      f'--error={self.submit_dir / "logs/%A[%a].log"}',
                       '--open-mode=append',
                      f'--time=0-24:00:00',
                      f'--nice=9', #make the jobs defer to others
                      #f'--partition=tch14', #just my machine 
                      f'--partition=pc', #the general compute queue
                      ]

        if held: args.append('--hold') #hold if necessary
        if self.startafter: args.append(f'--dependency=afterok:{self.startafter.job_id}')
            
        args.append(str(self.jobscript)) #the actual script itself
        print('sbatch command:\n', ' '.join(args))
        
        try:
            #job_id has the form 1649928_[]
            self.job_id = sb.check_output(args, encoding = 'utf8', stderr=sb.PIPE).strip()
        except CalledProcessError as e:
            print(e.output, e.stderr, e.returncode)
            raise e
        
        print(f'Job created with id {self.job_id}')
        return self
    
    def cancel(self):
        if self.job_id == None: return
        return sb.check_output(['scancel', str(self.job_id)])
    
    def release(self):
        if self.job_id == None: return
        sb.check_output(['scontrol', 'release', str(self.job_id)])
        self.runnning = True
            
    def status(self):
        try:
            output = sb.check_output(['qstat', self.job_id], encoding = 'utf8', stderr=sb.PIPE)
        except CalledProcessError as e:
            #print(e.output, e.stderr, e.returncode)
            return 'Job finished'
        return output
        
        