
# coding: utf-8

# In[11]:


from pathlib import Path
import numpy as np
import os

from FKMC.montecarlo import *

slurm_job_id = int(os.getenv('SLURM_ARRAY_JOB_ID', -1))
slurm_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 126))
max_slurm_task_id = int(os.getenv('SLURM_ARRAY_TASK_MAX', 249))
working_dir = Path(f".")

from time import time, sleep
sleeptime = np.random.random() * (10*60)
if slurm_job_id != -1: sleep(sleeptime)

Ts = np.linspace(0.05, 10, 50)

print(f'len(Ts) = {len(Ts)} max_slurm_task_id = {max_slurm_task_id}')

N = 128

states = [np.arange(N, dtype = np.float64) % 2,
         1 - np.arange(N, dtype = np.float64) % 2,
         np.ones(N, dtype = np.float64),
         np.zeros(N, dtype = np.float64),
         np.random.choice([0.0,1.0], size = N)
        ]

assert((len(Ts)*len(states) - 1) == max_slurm_task_id)
T = Ts[slurm_task_id // len(states)]
state = states[slurm_task_id % len(states)]
print(f'state = {state}')

parameters = dict(t = 1, alpha = 1.5, mu = 0, beta = 1/T, J = 5, U = 5, normalise = True)
n_bins = 1000
MCMC_params = dict(
        state = state,
        N_steps = int(80 * 1000),
        N_burn_in = int(1000), 
        thin = 5,
        logger = Eigenspectrum_IPR_all(bins = n_bins, limit = 10),
        proposal = p_multi_site_uniform_reflect,
        accept_function = perturbation_accept,
        warnings = True,
    )


t0 = time()
log = FK_mcmc(**MCMC_params, parameters = parameters)
log.time = time() - t0

parameters['J_matrix'] = '...'
print(f'params = {parameters} in t = {log.time:.0f} seconds, mc_params = {MCMC_params}')

filename = f'{slurm_task_id}.npz'

print(f'saving in {filename}')

if slurm_job_id != -1: np.savez_compressed(working_dir/filename, 
        state = state, Ts = Ts, parameters = parameters, MCMC_params = MCMC_params, log = log, allow_pickle = True,
        desc = 'large runs as a function of temp and starting state, every 4 or 5 are run at the same temp but with a different starting state'
        )
print('done')


# In[ ]:




