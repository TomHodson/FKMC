from pathlib import Path
import subprocess as sb
from subprocess import CalledProcessError
import shutil
from sys import exit, argv
from datetime import datetime
from re import match

class CX1job(object): 
    job_id = None
    jobscript = Path.home() / 'FKMC/batchscripts/CX1_jobscript.sh'
    running = False
    
    def __init__(self, python_script, job_name, job_folder_name, array_indices, startafter = None): 
        self.python_script = python_script
        self.job_name = job_name
        self.job_folder_name = job_folder_name
        self.array_indices = array_indices
        self.submit_dir = Path('/rds/general/user/tch14/home/HPC_data') / job_folder_name
      
    def submit(self, held = False):
        #http://docs.adaptivecomputing.com/torque/4-0-2/Content/topics/commands/qsub.htm
        start, stop = self.array_indices
        indices = f'{start}-{stop-1}'
    
        qsub_args = ['qsub', 
                     '-v', f'PYTHON_SCRIPT={self.python_script.name}, SUBMIT_DIR={self.submit_dir}',
                     '-N', f'{self.job_name}',
                     '-J', f'{indices}',
                     '-lselect=1:ncpus=1:mem=4gb:avx=true',
                     #'-lwalltime=24:00:00',
                     '-lwalltime=5:00:00',
                     '-o', str(self.submit_dir / 'logs'),
                     '-e', str(self.submit_dir / 'logs'),
                    ]
        
        if held: qsub_args.append('-h') #hold if necessary
        if startafter: qsub_args.append(f'-W depend=afterok:{startafter.job_id}')
            
        qsub_args.append(str(self.jobscript)) #the actual script itself
        
        print('qsub command:\n', ' '.join(qsub_args))
        
        try:
            #jobid has the form 1649928[].pbs
            jobstring = sb.check_output(qsub_args, encoding = 'utf8', stderr=sb.PIPE)
            self.job_id = match(r'(\d+)\[\]\.pbs', jobstring).groups()[0]
            if not held: self.release()
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
    jobid = None
    jobscript = Path.home() / 'FKMC/batchscripts/CMTH_jobscript.sh'
    
    def create(python_script, job_name, array_indices):
        start, stop = array_indices
        indices = f'{start}-{stop}'
        sbatch_args = ['sbatch', 
                       '--hold', 
                       '--parsable',
                       f'--export=target={python_script.name}', 
                       f'--job-name={job_name}', f'--array={indices}', 
                       '--exclude=dirac',
                       self.jobscript]


        self.jobid = int(sb.check_output(sbatch_args, encoding = 'utf8'))
    
    def cancel():
        if self.jobid == None: return
        return sb.check_output(['scancel', str(self.jobid)])
    
    def release():
        if self.jobid == None: return
        sb.check_output(['scontrol', 'release', str(self.jobid)])