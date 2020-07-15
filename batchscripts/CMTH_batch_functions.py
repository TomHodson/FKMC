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
        indices = f'{start}-{stop}'
        sbatch_args = ['sbatch', 
                       '--hold', 
                       '--parsable',
                      f'--export=',','.join([
                           f'PYTHON_SCRIPT={self.python_script.name}',
                           f'SUBMIT_DIR={self.submit_dir}',
                           f'DEBUG={self.debug}',
                      ]),
                      f'--job-name={self.job_name}',
                      f'--array={indices}', 
                       '--exclude=dirac',
                       '--kill-on-invalid-dep=no',
                       '--mail-type=ALL',
                       '--mail-user=tch14@ic.ac.uk',
                      f'--output={self.submit_dir / "logs"}',
                      f'--error={self.submit_dir / "logs"}',
                      f'--time=0-24:00:00',
                      f'--nice=10', #make the jobs defer to others
                      ]

        if held: sbatch_args.append('--hold') #hold if necessary
        if self.startafter: sbatch_args.append(f'--dependency=afterok:{self.startafter.job_id}')
            
        sbatch_args.append(str(self.jobscript)) #the actual script itself
        print('sbatch command:\n', ' '.join(qsub_args))
        
        try:
            #jobid has the form 1649928[].pbs
            self.job_id = sb.check_output(sbatch_args, encoding = 'utf8', stderr=sb.PIPE)
        except CalledProcessError as e:
            print(e.output, e.stderr, e.returncode)
            raise e
        
        print(f'Job created with id {self.job_id}')
        return self
    
    def cancel():
        if self.jobid == None: return
        return sb.check_output(['scancel', str(self.jobid)])
    
    def release():
        if self.jobid == None: return
        sb.check_output(['scontrol', 'release', str(self.jobid)])
        self.runnning = True
            
    def status(self):
        try:
            output = sb.check_output(['qstat', self.job_id], encoding = 'utf8', stderr=sb.PIPE)
        except CalledProcessError as e:
            #print(e.output, e.stderr, e.returncode)
            return 'Job finished'
        return output

