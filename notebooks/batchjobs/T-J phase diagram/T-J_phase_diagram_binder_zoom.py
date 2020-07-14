
# coding: utf-8

# In[3]:


from pathlib import Path
import numpy as np
import os

from FKMC.montecarlo import *

slurm_job_id = int(os.getenv('SLURM_ARRAY_JOB_ID', -1))
slurm_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 22))
max_slurm_task_id = int(os.getenv('SLURM_ARRAY_TASK_MAX', 559))
working_dir = Path(f".")

from time import time, sleep
sleeptime = np.random.random() * (5*60)
if slurm_job_id != -1: sleep(sleeptime)

Js = np.linspace(0.001, 20.0, 20)
Ns = [4,8,16,32,64,128]
N_Ts = 28

J = Js[slurm_task_id // N_Ts]

T_upper = 3 + 0.5*J
T_lower = max(0, -4 + 0.5*J)
Ts = np.linspace(T_lower, T_upper, N_Ts)

assert(max_slurm_task_id == N_Ts * len(Js) - 1)

T = Ts[slurm_task_id % N_Ts]

logs = np.empty(shape = len(Ns), dtype = object)

thousand = 1000 if slurm_job_id != -1 else 1

for i, N in enumerate(Ns):
    parameters = dict(t = 1, alpha = 1.5, mu = 0, beta = 1/T, J = J, U = 5, normalise = True)
    MCMC_params = dict(
            state = np.arange(N, dtype = np.float64) % 2, #starting from a CDW state
            N_steps = int(100 * thousand),
            N_burn_in = int(10 * thousand), 
            thin = 100,
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
    mc_params = {MCMC_params}
    ''')

filename = f'{slurm_task_id}.npz'

print(f'Total time {time() - t0:.0f}s')

if slurm_job_id != -1: 
    print(f'saving in {filename}')
    np.savez_compressed(working_dir/filename, 
        Js = Js, Ns = Ns, Ts = Ts, parameters = parameters, MCMC_params = MCMC_params, logs = logs, allow_pickle = True,
        desc = ''
        )
print('done')


# In[ ]:


5


# In[ ]:




