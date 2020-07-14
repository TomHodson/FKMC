## print('Starting generic python imports')
from pathlib import Path
import numpy as np
import os
from time import time, sleep
import sys
from munch import Munch
from itertools import product, islice
from FKMC.montecarlo import *


jobdir = Path('/workspace/tch14/local_run_data/run_8_T=1.5_U=5')

datafiles = sorted([(int(f.stem), f) for f in jobdir.iterdir() if f.name.endswith('npz') and not f.name == 'parameters.npz'])
jobs = np.array([j_id for j_id, f in datafiles])
last_jobs_number, last_job = datafiles[-1]

d = Munch(np.load(last_job, allow_pickle = True))
Ns = d['Ns']
parameters = d['parameters'][()]
MCMC_params = d['MCMC_params'][()]
logs = d['logs'][()]
previous_states = [logs[i].last_state for i, N in enumerate(Ns)]

print(f'parameters = {parameters}')

logs = np.empty(shape = len(Ns), dtype = object)
t1 = time()

for j in range(last_jobs_number+1,10000):
    print(f"On round j = {j}")
    for i, N in enumerate(Ns):
        print(f'Starting N = {N}')
        MCMC_params['state'] = previous_states[i]

        t0 = time()
        logs[i] = FK_mcmc(**MCMC_params, parameters = parameters)
        logs[i].time = time() - t0
        
        previous_states[i] = logs[i].last_state

        print(f'This N = {N} j ={i} took {time() - t0:.0f} seconds.')

        filepath = jobdir/f'{j}.npz'
        print(f'Saving in {filepath}')
        np.savez_compressed(filepath, 
        parameters = parameters, MCMC_params = MCMC_params, logs = logs, allow_pickle = True, Ns = Ns,
        )

    print(f'Overall: {time() - t1:.0f} seconds.\n\n')




