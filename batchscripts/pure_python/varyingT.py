
# coding: utf-8

# In[4]:


'''
A script to do a sweep over temperature to make:
    - the binder cumulant plot
    - the IPR and DOS color plots showing the band opening
    
TODO:
put the debug logic back in
allow repeats to start over

a repeat of ## data/slurm_runs/103867 but with 1e5 steps and alpha = 3 to show short range behaviour
For the band opening plot for short ranged. It's a fixed J=5, U=1, alpha = 3, with T from 0.1 to 4 with 200 steps
very fine energy spacing: logger = Eigenspectrum_IPR_all(bins = 2000, limit = 5),
'''
from munch import Munch
import numpy as np

########## Hamiltonian parameters ###########################################################################
Ham_params = Munch(
    t = 1,
    alpha = 1.25,
    mu = 0,
    beta = 'varying',
    J = 5,
    U = 5,
    normalise = True #Whether the long range interaction should be normalised against the CDW or not.
)
print('Ham_params: ', ' '.join(f'{k}={v},' for k,v in Ham_params.items()))

########## Variable Hamiltonian parameters ###########################################################################
chain_exts = np.arange(5) #the number of times to extend the chain maximum is about 50 on CX1
N_steps = int(1e3) #the number of MCMC steps in each individual task
thin = 10
print(f'''
Tasks per chain: {chain_exts.size},
Each doing {N_steps} steps,
{chain_exts.size*N_steps} total chain length,
{chain_exts.size*N_steps // thin} samples,
''')

Ts = np.linspace(0.1, 4, 200)
Ns = np.logspace(np.log10(70), np.log10(270), 10, dtype = np.int) // 10 * 10

structure_names = ['chain_exts', 'Ts'] #Ns is dealt with separately
structure_dimensions = [chain_exts, Ts]

########## Monte Carlo parameters ###########################################################################
from FKMC.montecarlo import Eigenspectrum_IPR_all, p_multi_site_uniform_reflect, perturbation_accept

initial_states = Munch(
    CDW1 = lambda N: np.arange(N, dtype = np.float64) % 2,
    CDW2 = lambda N: (np.arange(N, dtype = np.float64)+1) % 2,
    zeros = lambda N: np.zeros(N, dtype = np.float64),
    ones = lambda N: np.ones(N, dtype = np.float64),
)

state_factory = initial_states.CDW1

MCMC_params = Munch(
        N_steps = N_steps,
        N_burn_in = N_steps,
        thin = thin,
        logger = Eigenspectrum_IPR_all(bins = 2000, limit = 5),
        proposal = p_multi_site_uniform_reflect,
        accept_function = perturbation_accept,
        warnings = False,
    )
print('MCMC_params: ', ' '.join(f'{k}={v},' for k,v in MCMC_params.items()))

########## Batch Job Structure ###########################################################################
from itertools import product as cartesian_product

config_product = cartesian_product(*structure_dimensions)

#give information to the dispatch script
batch_params = Munch(
    total_jobs = len(Ts),
    chain_exts = chain_exts,
    structure_names = structure_names, #names of each of the dimensions like ['Ts', 'Alphas']
    structure_dimensions = structure_dimensions, #the dimensions themselves like [np.linspace(0.1,5,100), np.linspace(0.1,2,100)]
    indices = (0, len(Ts)),
)
#bath_params_end_flag this is here to signal the end of the batch_params variable

########## Parameters particular to this job ################################################################

from itertools import product, islice
from pathlib import Path
import os
from time import time, sleep
import sys
import shutil
from FKMC.import_funcs import timefmt

print('Getting environment variables')
job_id = int(os.getenv('JOB_ID', -1))
debug = (os.getenv('DEBUG', job_id == -1) == 'True')
task_id = int(os.getenv('TASK_ID', 0))
submit_dir = Path(os.getenv('SUBMIT_DIR', '/home/tch14/HPC_data/test/'))
print(f'job_id = {job_id}, task_id = {task_id}, submit_dir = {submit_dir}')

Is_todo = np.arange(len(Ns))
Ns_todo = Ns.copy() 
logs = np.empty(shape = len(Ns), dtype = object)
if not debug and (submit_dir / 'data' / f'{task_id}.npz').exists():
    print(f'Job file {task_id}.npz already exists')
    print(f'Loading {task_id}.npz to retrieve the work')
    d = Munch(np.load(submit_dir / 'data' / f'{task_id}.npz', allow_pickle = True))
    logs = d['logs'][()]
    todo = (logs == None)
    Is_todo = Is_todo[todo]
    Ns_todo = Ns_todo[todo]
    print(f'Ns_todo = {Ns_todo}')
    


(repeat_i, T), = list(islice(config_product, task_id, task_id + 1))
print(f'repeat_i = {repeat_i}, T = {T}')
    
if repeat_i > 0:
    previous_task_id = task_id - len(Ts)
    MCMC_params.burn_in = 0
else:
    previous_task_id = -1

Ham_params.beta = 1 / T

########## Set up debugging and sleep ################################################################

if debug:
    MCMC_params.N_burn_in = 0
    MCMC_params.N_steps = 500
    MCMC_params.thin = 1
    Ham_params.beta = 1 / 2.0 #choose the critical temp where the calculations take longest
    
    

##sleep if necessary
if not debug: 
    sleeptime = np.random.random() * (30)
    print(f'Waiting for {sleeptime:.0f} seconds to randomise the finish time')
    sleep(sleeptime)


########## Load the previous states ################################################################

#load in the last state from the previous run or use initial_states
if previous_task_id != -1:
    print(f'Loading {previous_task_id}.npz to retrieve the last state')
    d = Munch(np.load(submit_dir / 'data' / f'{previous_task_id}.npz', allow_pickle = True))
    logs = d['logs'][()]
    if any(log == None for log in logs):
        print("Previous job didn't finish, exiting")
        raise ValueError
        
    previous_states = [logs[i].last_state for i, N in enumerate(Ns)]
else:
    print('Generating initial state as this is the first run with these params')
    previous_states = [state_factory(N) for N in Ns]

from FKMC.general import tridiagonal_diagonalisation_benchmark
benchmark_time = tridiagonal_diagonalisation_benchmark()
print(f"Diagonalisation benchmark: {benchmark_time:.2f} seconds")

########## The actual simulation code ################################################################
from FKMC.montecarlo import FK_mcmc

Ham_params['J_matrix'] = '...'
MCMC_params['state'] = '...'
t0 = time()

for i, N in zip(Is_todo, Ns_todo):
    if previous_task_id != -1: MCMC_params.N_burn_in = 0
    MCMC_params.state = previous_states[i]

    t0 = time()
    logs[i] = FK_mcmc(**MCMC_params, parameters = Ham_params)
    logs[i].time = time() - t0
    
    np.savez_compressed(f'{task_id}.npz', 
        Ns = Ns, parameters = Ham_params, MCMC_params = MCMC_params, 
        structure_names = structure_names,
        structure_dimensions = structure_dimensions,        
        logs = logs, allow_pickle = True,
        desc = ''
        )
    
    shutil.copy(f'{task_id}.npz', submit_dir / 'data')

t = time() - t0
print(f'{t:.0f} seconds, saving in {Path.cwd()}')

print(f'''
Requested MCMC steps: {N_steps}
Time: {timefmt(t)} 
Estimated task runtime for 1000 steps: {timefmt(t * 1000 / MCMC_params.N_steps)} 
''')
print('Done')


# In[3]:





# In[ ]:




