#!/usr/bin/env python3
from pathlib import Path
import subprocess as sb
import shutil
from sys import exit, argv
from datetime import datetime

try:
    usage = int(sb.check_output(['du', '-s', '--block-size=1G', '/home/tch14/data/slurm_runs']).split()[0])
    print(f'Data usage {usage}/50G')
    if usage > 40: print('WARNING: DATA IS ALMOST FULL!!!!!')
except Exception as e:
    print(f'Checking disk usage failed with exception {e}')
    
try:
    usage = int(sb.check_output(['du', '-s', '--block-size=1G', '/home/tch14/FKMC']).split()[0])
    print(f'FKMC size is {usage}G')
except Exception as e:
    print(f'Checking disk usage failed with exception {e}')