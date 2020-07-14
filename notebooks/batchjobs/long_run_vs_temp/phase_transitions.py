
# coding: utf-8

# In[21]:


from pathlib import Path
import numpy as np
import os

from FKMC.montecarlo import *

slurm_job_id = int(os.getenv('SLURM_ARRAY_JOB_ID', -1))
slurm_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 22))
max_slurm_task_id = int(os.getenv('SLURM_ARRAY_TASK_MAX', 199))
working_dir = Path(f".")

from time import time, sleep
sleeptime = np.random.random() * (5*60)
if slurm_job_id != -1: sleep(sleeptime)

Ts = np.linspace(1.0, 2.0, 200)

print(f'len(Ts) = {len(Ts)} max_slurm_task_id = {max_slurm_task_id}')

Ns = [8,16,32,64,100,128]
T = Ts[slurm_task_id]
logs = np.empty(shape = len(Ns), dtype = object)

for i, N in enumerate(Ns):
    parameters = dict(t = 1, alpha = 1.5, mu = 0, beta = 1/T, J = 5, U = 5, normalise = True)
    MCMC_params = dict(
            state = np.arange(N, dtype = np.float64) % 2,
            N_steps = int(100 * 1000),
            #N_steps = int(0.1 * 1000),
            N_burn_in = int(10 * 1000), 
            thin = 5,
            logger = Eigenspectrum_IPR_all(bins = 1000, limit = 10),
            proposal = p_multi_site_uniform_reflect,
            accept_function = perturbation_accept,
            warnings = True,
        )

    print(f'starting N = {N}')
    t0 = time()
    logs[i] = FK_mcmc(**MCMC_params, parameters = parameters)
    logs[i].time = time() - t0

    parameters['J_matrix'] = '...'
    MCMC_params['state'] = '...'
    print(f'''
    params = {parameters},
    mc_params = {MCMC_params},
    in t = {logs[i].time:.0f} seconds, 
    ''')

filename = f'{slurm_task_id}.npz'

print(f'saving in {filename}')

if slurm_job_id != -1: np.savez_compressed(working_dir/filename, 
        Ns = Ns, Ts = Ts, parameters = parameters, MCMC_params = MCMC_params, logs = logs, allow_pickle = True,
        desc = ''
        )
print('done')


# In[23]:





# In[ ]:




