import numpy as np
from time import time
from operator import mul
from functools import reduce
from itertools import count
from munch import Munch
from itertools import zip_longest
import logging
logger = logging.getLogger(__name__)

import scipy
from FKMC.general import index_histogram_array, sort_IPRs, smooth, shapes, normalise_IPR
from FKMC.stats import binned_error_estimate_multidim, product


#variable classifications
N_dependent_size = set(['IPRs', 'eigenvals', 'state','accept_rates', 'classical_accept_rates', 'last_state', 'proposal_rates'])
per_step = set([ 'Fc', 'Ff', 'Mf_moments', 'Nc', 'Nf', 'eigenval_bins'])
per_run = set(['A', 'N_cumulants','time'])


def allocate(requested_observables, example_datafile, N_jobs, MCMC_slice):
    observables = Munch()
    logs = example_datafile.logs
    Ns = example_datafile.Ns
    observables.total_size = 0

    
    for name in requested_observables:
        src = np.array(getattr(logs[0], name))
        if name in per_step:
            src_shape = src[..., MCMC_slice].shape
            
        elif name in per_run:
            src_shape = src.shape
            
        if name in per_step or name in per_run:
            shape = (len(Ns), N_jobs) + src_shape
            observables[name] = np.full(shape=shape, fill_value = np.nan, dtype = src.dtype)
            shapes = [shape,]

        elif name in N_dependent_size:
            shapes = [(N_jobs,) + getattr(logs[i], name)[MCMC_slice, ...].shape for i, N in enumerate(Ns)]
            observables[name] = [np.full(shape=shape, fill_value = np.nan, dtype = src.dtype)
                                 for shape in shapes]
        else:
            raise ValueError(f'Name {name} not in any of the classifications!')

            
        approx_size = sum(4*product(shape) for shape in shapes) #assumes 64bit floats
        observables.total_size += approx_size
        assert(approx_size < 1e9) #try not to use more than 1Gb per allocation
        logger.info(f"observables['{name}'] = np.array(shape = {shapes}, dtype = {src.dtype}) approx size: {approx_size/1e9:.2f}Gb")
        
    return observables

   
def copy(requested_observables, datafiles, observables, MCMC_slice):
    p = ProgressReporter(len(datafiles))
    for n, (j,file) in enumerate(datafiles):
        p.update(n)
        
        #load this datafile
        try:
            npz_file = np.load(file, allow_pickle = True)['logs']
        except Exception as e:
            logger.warning(f'{e} on {file}')
            continue

        for name in requested_observables:
            for i in range(len(observables.Ns)):
                obs = getattr(npz_file[i], name)
                
                if name in N_dependent_size:
                    observables[name][i][j] = obs[MCMC_slice, ...]
                
                elif name in per_step:
                        observables[name][i, j] = obs[..., MCMC_slice]
                
                elif name in per_run:
                        observables[name][i, j] = obs
                else:
                    raise ValueError(f'Name {name} not in any of the classifications!')

N_dependent_size = set(['IPRs', 'eigenvals', 'state','accept_rates', 'classical_accept_rates', 'last_state', 'proposal_rates'])
per_step = set([ 'Fc', 'Ff', 'Mf_moments', 'Nc', 'Nf', 'eigenval_bins'])
per_run = set(['A', 'N_cumulants','time'])     
def shape_hints(name):
    custom = dict(
        Mf_moments = ('moment', 'MCstep'),
        eigenval_bins = ('bin', 'MCstep'),
        time = (),
                 )
    if name in custom: return custom[name]
    if name in per_step: return ('MCstep',)
    if name in ['IPRs', 'eigenvals', 'state']: return ('MCstep', 'N')
    if name in ['accept_rates', 'classical_accept_rates', 'last_state', 'proposal_rates']:
        return ('MCstep', 'N+1')
    raise ValueError(f'Dont know what to do with {name}')
    
def reshape(structure_dims, requested_observables, observables):
    observables['hints'] = Munch()
    for name in requested_observables:
        if name in N_dependent_size:
            for i in range(len(observables.Ns)):
                o = observables[name][i]
                newshape = structure_dims + o.shape[1:]
                observables[name][i] =  o.reshape(newshape)
        else:
            o = observables[name]
            observables[name] = o.reshape(o.shape[0:1] + structure_dims + o.shape[2:])
    
        observables.hints[name] = ('Ns',) + tuple(observables.structure_names) + shape_hints(name)
                
class ProgressReporter(object):
    def __init__(self, N):
        self.t0 = time()
        self.N = N
        self.dot_batch = N // 50 if N > 50 else 1
    def update(self, j):
        N = self.N
        if j == 0: self.t0 = time()
        if (j == 10) or (j == N//2): 
            dt = time() - self.t0
            logger.info(f'\nTook {dt:.2f}s to do the first {j}, should take {(N-j)*dt/j:.2f}s to do the remaining {N-j}\n')
        if j % self.dot_batch == 0: print(f'{j} ', end='')
            


class extract(object):
    def __init__(self, obsname):
        self.obsname = obsname
    
    def allocate(self, observables, example_datafile, N_jobs):
        logs = example_datafile.logs[0]
        Ns = example_datafile.Ns
        data = np.array(getattr(logs, self.obsname)) #adding the np.array(...) makes it work for bare floats

        shape = (len(Ns), N_jobs) + np.shape(data) #this works even is data is a float
        observables[self.obsname] = np.full(shape=shape, fill_value = np.nan, dtype = data.dtype)
        approx_size = 4*product(shape) #assumes 64bit floats
        assert(approx_size < 1e9) #try not to use more than 1Gb per allocation
        logger.info(f"observables['{self.obsname}'] = np.array(shape = {shape}, dtype = {data.dtype}) approx size: {approx_size/1e9:.2f}Gb")


    def copy(self, observables, j, datafile):
        for i in range(len(observables.Ns)):
            if datafile[i] == None: continue #fail gracefully on partially computed results
            observables[self.obsname][i, j] = getattr(datafile[i], self.obsname)
                
              
    def reshape(self, structure_dims, observables):
        o = observables[self.obsname]
        observables[self.obsname] = o.reshape(o.shape[0:1] + structure_dims + o.shape[2:])
        observables.hints[self.obsname] = ('Ns',) + tuple(observables.structure_names)
 


class mean_over_MCMC(object):
    def __init__(self, obsname):
        self.obsname = obsname
    
    def allocate(self, observables, example_datafile, N_jobs):
        logs = example_datafile.logs[0]
        Ns = example_datafile.Ns
        data = np.array(getattr(logs, self.obsname))

        shape = (len(Ns), N_jobs) + np.shape(data)[:-1] #this works even is data is a float
        self.taking_mean = (len(np.shape(data)) > 0)
        observables[self.obsname] = np.full(shape=shape, fill_value = np.nan, dtype = data.dtype)
        
        if self.taking_mean:
            observables['sigma_' + self.obsname] = np.full(shape=shape, fill_value = np.nan, dtype = data.dtype)
        
        approx_size = 4*product(shape) #assumes 64bit floats
        assert(approx_size < 1e9) #try not to use more than 1Gb per allocation
        logger.debug(f"observables['{self.obsname}'] = np.array(shape = {shape}, dtype = {data.dtype}) approx size: {approx_size/1e9:.2f}Gb")


    def copy(self, observables, j, datafile):
        for i in range(len(observables.Ns)):
            if datafile[i] == None: continue #fail gracefully on partially computed results
            data = getattr(datafile[i], self.obsname)
            if self.taking_mean:
                observables[self.obsname][i, j] = data.mean(axis = -1)
                observables['sigma_' + self.obsname] = scipy.stats.sem(data, axis = -1)
            else:
                observables[self.obsname][i, j] = data
                
                
            
            
    def reshape(self, structure_dims, observables):
        o = observables[self.obsname]
        observables[self.obsname] = o.reshape(o.shape[0:1] + structure_dims + o.shape[2:])

        observables.hints[self.obsname] = ('Ns',) + tuple(observables.structure_names)
        if self.obsname == 'Mf_moments':
            observables.hints[self.obsname] = ('Ns',) + tuple(observables.structure_names) + ('nth moment',)
 

from FKMC.general import compute_IPR_and_DOS_histograms

class IPRandDOS(object):
    def __init__(self, E_bins = np.linspace(-6, 6, 2000 + 1), bootstrap_bins = 10):
        self.E_bins = E_bins
        self.bootstrap_bins = bootstrap_bins
    
    def allocate(self, observables, example_datafile, N_jobs):
        Ns = example_datafile.Ns
        shape = (len(Ns), N_jobs, len(self.E_bins) - 1)
        dtype = np.float64
        
        observables['E_bins'] = self.E_bins
        observables['IPR'] = np.full(shape=shape, fill_value = np.nan, dtype = dtype)
        observables['DOS'] = np.full(shape=shape, fill_value = np.nan, dtype = dtype)
        observables['dIPR'] = np.full(shape=shape, fill_value = np.nan, dtype = dtype)
        observables['dDOS'] = np.full(shape=shape, fill_value = np.nan, dtype = dtype)
    
        
        approx_size = 4*product(shape) #assumes 64bit floats
        assert(approx_size < 1e9) #try not to use more than 1Gb per allocation
        logger.debug(f"observables['IPRs'] = np.array(shape = {shape}, dtype = {dtype}) approx size: {approx_size/1e9:.2f}Gb")
        logger.debug(f"observables['DOS'] = np.array(shape = {shape}, dtype = {dtype}) approx size: {approx_size/1e9:.2f}Gb")


    def copy(self, observables, j, datafile):
        
        for i, N in zip(count(), observables.Ns):
            if datafile[i] == None: continue #fail gracefully on partially computed results
            
            #shape is (MCMC_steps, system_size)
            raw_IPRs = getattr(datafile[i], 'IPRs')
            raw_eigenvals = getattr(datafile[i], 'eigenvals')
            
            DOS, dDOS, IPR, dIPR = compute_IPR_and_DOS_histograms(raw_eigenvals,
                                                                  raw_IPRs, self.E_bins, self.bootstrap_bins)

            observables['DOS'][i, j, :] = DOS
            observables['IPR'][i, j, :] = IPR
            observables['dDOS'][i, j, :] = dDOS
            observables['dIPR'][i, j, :] = dIPR
            
    def reshape(self, structure_dims, observables):
        for name in ['DOS', 'IPR', 'dDOS', 'dIPR']:
            o = observables[name]
            observables[name] = o.reshape(o.shape[0:1] + structure_dims + o.shape[2:])
            observables.hints[name] = ('Ns',) + tuple(observables.structure_names) +  ('energy index',)


                
import re
from path import Path

def update_description(job_id, info):
    descriptions = Path("/home/tch14/FKMC/notebooks/run_descriptions.md").open().read()
    match = re.search(f"(## data/slurm_runs/{job_id} [^#]*)", descriptions, flags = re.MULTILINE + re.DOTALL)
    if match:
        desc = match.group(1).split('\n')
    else:
        desc = ['', f"## Custom run {job_id}",]
    
    path = Path("/home/tch14/FKMC/notebooks/auto_run_descriptions.md")
    auto_descriptions = path.open().read()

    desc.insert(1, info)
    desc = "\n".join(desc)
    
    match = re.search(f"(## data/slurm_runs/{job_id}) (\S*) (\S*)([^#]*)", auto_descriptions, flags = re.DOTALL)
    if match: #if there's already an entry in the file
        s, e = match.span()
        auto_descriptions = auto_descriptions[:s] + desc + auto_descriptions[e:]
    else:
        auto_descriptions = auto_descriptions + desc
    
    path.open('w').write(auto_descriptions)

def timefmt(t):
    if t < 60:
        return f"{int(t)} seconds"
    if t < 60*60:
        return f"{t//60} minutes"
    if t < 60*60*24:
        return f"{t//(60*60)} hours"
    if t < 60*60*24*7:
        return f"{t//(60*60*24)} days"
    return f"{t//(60*60*24*7)} weeks"
    

def get_data_funcmap(this_run,
            functions = [],
            structure_names = ('Ts', 'Repeats'),
            structure_dims = (),
            ):
    
    '''
    This version exposes a list of objects to each datafile and creates the output from that.
    The structure argument has the forms of a tuple containing strings and ints which give the shape of the data,
    strings should refer to labels in the data like Ts whose shape will be used
    and ints create an unamed axis for things like repeats
    '''
    logger.warning(f'looking in {this_run}')
    functions += [extract('time'), mean_over_MCMC('accept_rates'), mean_over_MCMC('proposal_rates')]
    datafiles = sorted([(int(f.stem), f) for f in this_run.iterdir() if f.name.endswith('npz') and not f.name == 'parameters.npz'])
    jobs = np.array([j_id for j_id, f in datafiles])
    if len(jobs) == 0: 
        logger.error("NO DATA FILES FOUND");
        return
    logger.debug(f'job ids range from {min(jobs)} to {max(jobs)}')
    
    #get stuff from an an example datafile
    d = Munch(np.load(datafiles[0][1], allow_pickle = True))
    Ns = d['Ns']
    parameters = d['parameters'][()]
    MCMC_params = d['MCMC_params'][()]
    
    logger.debug(f'structure_dims before inference = {structure_dims}')
    logger.debug(f'Infilling structure_dims from dimensions variables. (len(Ts) etc)')
    #some strucure dims can be inferred from the length of things in the namespace like Ts or Ns or repeats
    def infer(data, dimension_name, dimension_size):
        if dimension_size != None: return dimension_size #when the dimension is just directly in structure_dims
        if dimension_name in d and d[dimension_name].ndim == 0: return d[dimension_name].shape[0] #when the name corresponds to something like Ts or Ns
        if dimension_name in d: return d[dimension_name] #when the struct is just directly in the config as in d[repeats] = 10
        return None
        
    structure_dims = [infer(d, dimension_name, dimension_size) for dimension_size,dimension_name in zip_longest(structure_dims, structure_names, fillvalue=None)]
    
    #the outermost size can be determined from everything else and the number of jobs
    if structure_dims[0] == None:
        inner = product(structure_dims[1:])
        structure_dims[0] = (max(jobs) + 1) // inner
        logger.debug(f'structure_dims[0] was determined as {structure_dims[0]} = {(max(jobs) + 1)} // {inner}')
        
    structure_dims = tuple(structure_dims) #can't remember why I made it a tuple but will leave it for now 
    logger.debug(f'structure_names = {structure_names}')
    logger.debug(f'structure_dims = {structure_dims}')

    #calculate the epected number of jobs
    N_jobs = product(structure_dims)
    logger.debug(f'Expected number of jobs {N_jobs}')
    
    #check if the strucure_dimensions cover enough ground
    if max(jobs) >= N_jobs:
        logger.warning(f"Id of largest job found ({max(jobs)}) is larger than product(structure_dims) = ({N_jobs})")
    
    #look for missing jobs
    missing = set(range(N_jobs)) - set(jobs)
    if missing: 
        logger.warning(f'Missing jobs: {missing}\n')
    
    logger.info(f'Logger keys: {list(d.keys())} \n')
    logger.info(f"MCMC_params keys: {list(MCMC_params.keys())} \n")
    
    original_N_steps = MCMC_params['N_steps']
    thin = MCMC_params['thin']
    N_steps = original_N_steps // thin
    logger.debug(f'MCMC Steps: {original_N_steps} with thinning = {thin} for {N_steps} recorded steps')
    
    logger.debug(list(zip(count(), structure_names, structure_dims)))

    possible_observables = [s for s in dir(d.logs[0]) if not s.startswith("_")]
    logger.info(f'available observables = {possible_observables}')
    
    logger.debug(f'Allocating space for the requested observables:')
    observables = Munch()
    for f in functions: f.allocate(observables, example_datafile = d, N_jobs = N_jobs)
    
    #copy extra info over, note that structure_names might appear as a key in d, but I just overwrite it for now
    observables.update({k : v[()] for k,v in d.items() if k != 'logs'})
    observables.structure_names = structure_names
    observables.structure_dims = structure_dims
    observables['hints'] = Munch() 
    
    p = ProgressReporter(len(datafiles))
    
    file_io_time = 0
    t = time()
    
    for n, (j,file) in enumerate(datafiles):
        
        #stop if j > product(structure_dims) assumes datafiles is ordered by j.
        if j == N_jobs: logger.warn(f'Ignoring datafiles after {j-1}.npz because product(structure_dims) is {N_jobs}')
        if j >= N_jobs: break
        
        p.update(n)
        if n == 10 or n == 100: logger.info(f'After {n} files, {file_io_time*100/(time() - t):.1f}% of the time was file I/O')
        #load this datafile
        try:
            ft = time()
            datafile = np.load(file, allow_pickle = True)['logs']
            file_io_time += time() - ft
        except Exception as e:
            logger.warning(f'{e} on {file}')
            continue
        
        for f in functions: f.copy(observables, j, datafile)
    
    for f in functions:
        f.reshape(structure_dims, observables)
    
   
    logger.info('########################################################################\n')
    logger.info(f'Observables has keys: {observables.keys()}')
    
    o = observables = Munch(observables)
    
    infostring = \
    f"""
    Completed jobs: {len(jobs)}/{N_jobs}
    MCMC Steps: {original_N_steps} with thinning = {thin} for {N_steps} recorded steps
    Burn in: {Munch(MCMC_params).N_burn_in}
    Structure_names: {dict(zip(structure_names, structure_dims))}
    Ns = {Ns}
    Runtimes: 
        Average: {timefmt(np.nanmean(o.time.sum(axis=0)))}
        Min: {timefmt(np.nanmin(o.time.sum(axis=0)))}
        Max: {timefmt(np.nanmax(o.time.sum(axis=0)))}
        Total: {timefmt(np.nansum(o.time))}
    """[1:]
    logger.info(infostring)
    update_description(this_run.stem, infostring)
    
    return observables