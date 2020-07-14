## print('Starting generic python imports')
from pathlib import Path
import numpy as np
import os
from time import time, sleep
import sys
from munch import Munch
from itertools import product, islice
from FKMC.montecarlo import *

## overall input parameters
T = 2.5
Ns = np.logspace(np.log10(50), np.log10(270), 10, dtype = np.int) // 10 * 10
make_initial_state = lambda N: np.arange(N, dtype = np.float64) % 2 #a CDW initial state
J = 5
U = 5
alpha = 1.25
jobdir = Path('/workspace/tch14/local_run_data/run_8_T=2.5_U=5')
jobdir.mkdir(parents=False, exist_ok=False)

logs = np.empty(shape = len(Ns), dtype = object)
previous_states = [make_initial_state(N) for N in Ns]
t1 = time()

for j in range(10000):
    print(f"On round j = {j}")
    for i, N in enumerate(Ns):
        print(f'Starting N = {N}')
        parameters = dict(t = 1, alpha = alpha, mu = 0, beta = 1/T, J = J, U = U, normalise = True)
        MCMC_params = dict(
                state = previous_states[i],
                N_steps = int(500),
                N_burn_in = int(0),
                thin = 10,
                logger = Eigenspectrum_IPR_all(bins = 10000, limit = 20),
                proposal = p_multi_site_uniform_reflect,
                accept_function = perturbation_accept,
                warnings = True,
            )

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




