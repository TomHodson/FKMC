from pathlib import Path
import subprocess as sb
from subprocess import CalledProcessError
import shutil
from sys import exit, argv
from datetime import datetime
from re import match

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

