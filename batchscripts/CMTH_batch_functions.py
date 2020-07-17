from pathlib import Path
import subprocess as sb
from subprocess import CalledProcessError
import shutil
from sys import exit, argv
from datetime import datetime
from re import match

class CMTHjob:
    job_id = None
    jobscript = Path.home() / 'FKMC/batchscripts/CMTH_jobscript.sh'
    running = False
    
    def __init__(self, python_script, job_name, job_folder_name, array_indices, startafter = None, debug = False): 
        self.python_script = python_script
        self.job_name = job_name
        self.job_folder_name = job_folder_name
        self.array_indices = array_indices
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

