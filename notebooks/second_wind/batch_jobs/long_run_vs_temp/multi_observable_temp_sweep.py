
# coding: utf-8

# In[22]:


from pathlib import Path
import numpy as np
import os

from FKMC.montecarlo import *

from time import time

slurm_job_id = os.getenv('SLURM_ARRAY_JOB_ID', 0)
slurm_task_id = os.getenv('SLURM_ARRAY_TASK_ID', 0)
max_slurm_task_id = os.getenv('SLURM_ARRAY_TASK_MAX', 29)

working_dir = Path(f"./data/{slurm_job_id}")
working_dir.mkdir(parents=True, exist_ok=True)

N = 32
Ts = np.linspace(0.01, 10, 30)
assert((len(Ts) - 1) == max_slurm_task_id)
T = Ts[slurm_task_id]

parameters = dict(t = 1, alpha = 1.5, mu = 0, beta = 1/T, J = 5, U = 5, normalise = True)
n_bins = 1000
MCMC_params = dict(
        state = np.arange(N) % 2,
        N_steps = int(0.01 * 1000),
        N_burn_in = int(0.01 * 1000), 
        logger = Eigenspectrum_IPR_all(bins = n_bins, limit = 10, ),
        proposal = p_multi_site_poisson_reflect,
        proposal_args = dict(lam = 1),
        accept_function = perturbation_accept,
        warnings = True,
    )

if slurm_id == 0: np.savez(working_dir / 'parameters.npz', 
            N=N, parameters = parameters, MCMC_params = MCMC_params, allow_pickle = True,
            )


t0 = time()
log = FK_mcmc(**MCMC_params, parameters = parameters)
log.time = time() - t0

print(f'N = {N} in t = {log.time:.0f} seconds')

filename = f'{slurm_task_id:06}.npz'

print(f'saving in {filename}')

np.savez(working_dir/filename, 
        log = log, allow_pickle = True,
        )
print('done')


# In[14]:


N = 1323
f'{N:02}'

