#!/usr/bin/env python3
print('starting')
from pathlib import Path
print('import Path')
#import os
from munch import *
import munch
print(munch.__file__)
print('imported Munch')
import argparse
print('imported argparse')
print('simple imports done')

from batch_functions import *
print('custom imports done')   


parser = argparse.ArgumentParser(description='Get info about a job given its submission directory')
parser.add_argument('out', help='The output folder name', type = Path)
#parser.add_argument('--debug', '-d', action = 'store_true', default = False, help='')
args = Munch(vars(parser.parse_args()))
print(args)
    
### find the python script
job_folder_name = args.out
if not job_folder_name.is_absolute():
        job_folder_name = (Path.home() / 'HPC_data' / job_folder_name)
job_name = job_folder_name.stem

code_dir = job_folder_name / 'code'
data_dir = job_folder_name / 'data'
logs_dir = job_folder_name / 'logs'
py_script = next(code_dir.glob('*.py'))

print(f'Requested Job folder: {rel2home(job_folder_name)}')

### determine if this is CX1 or CMTH
#platform = 'CX1' if 'CX2_SCRATCH' in os.environ else 'CMTH'
#print(f'Platform: {platform}')
    

context = execute_script(py_script)
batch_params = Munch(context.batch_params)
print(batch_params)

    
    




    


   



  

