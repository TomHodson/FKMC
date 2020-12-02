import numpy as np
from time import time
from operator import mul
from functools import reduce
from itertools import count
from munch import Munch
from itertools import zip_longest
import logging
logger = logging.getLogger(__name__)
import re
from pathlib import Path
import multiprocessing as mp
from termcolor import colored



import scipy
from FKMC.general import index_histogram_array, sort_IPRs, smooth, shapes, normalise_IPR
from FKMC.stats import binned_error_estimate_multidim, product


#variable classifications
N_dependent_size = set(['IPRs', 'eigenvals', 'state','accept_rates', 'classical_accept_rates', 'last_state', 'proposal_rates'])
per_step = set([ 'Fc', 'Ff', 'Mf_moments', 'Nc', 'Nf', ])
per_run = set(['A', 'N_cumulants','time', 'eigenval_bins'])


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
        eigenval_bins = ('bin',),
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
        if (j == 10) or (j == N//2) and (j != 0): 
            dt = time() - self.t0
            logger.info(f'\nTook {dt:.2f}s to do the first {j}, should take {(N-j)*dt/j:.2f}s to do the remaining {N-j}\n')
        if j % self.dot_batch == 0: print(f'{j} ', end='')
            
class extractStates(object):
    def __init__(self):
        self.obsname = 'state'
    
    def allocate(self, observables, example_datafile, N_jobs):
        logs = example_datafile.logs[0]
        Ns = example_datafile.Ns
        data = np.array(getattr(logs, self.obsname))
        data_shape = list(np.shape(data))
        hint = shape_hints(self.obsname)
        #the shape is (observables.max_MC_step, N)
            
        #use a list to store the differently shaped arrays
        shape = (N_jobs, observables.max_MC_step)
        observables.flat[self.obsname] = [
            np.full(shape=(N_jobs, observables.max_MC_step, N), 
                    fill_value = np.nan, 
                    dtype = data.dtype)
                for N in Ns]
        approx_size = 4*product(shape)*len(Ns) #assumes 64bit floats
        assert(approx_size < 1e9) #try not to use more than 1Gb per allocation
        logger.debug(f"observables.flat['{self.obsname}'] = [np.array(shape = (N_jobs, observables.max_MC_step, N), dtype = {data.dtype})] approx size: {approx_size/1e9:.2f}Gb")


    def copy(self, observables, j, datafile):
        for i in range(len(observables.Ns)):
            if datafile[i] == None: continue #fail gracefully on partially computed results
            observables.flat[self.obsname][i][j] = getattr(datafile[i], self.obsname)
                
              
    def reshape(self, structure_dims, observables):
        o = observables.flat[self.obsname]
        
        observables[self.obsname] = [
            o[i].reshape(structure_dims + (observables.max_MC_step, N))
            for i,N in enumerate(observables.Ns)]

        observables.hints[self.obsname] = ('[Ns]',) + tuple(observables.structure_names) + ('MCMC_step','N')

class extract(object):
    def __init__(self, obsname):
        self.obsname = obsname
    
    def allocate(self, observables, example_datafile, N_jobs):
        logs = example_datafile.logs[0]
        Ns = example_datafile.Ns
        data = np.array(getattr(logs, self.obsname)) #adding the np.array(...) makes it work for bare floats
        shape = (len(Ns), N_jobs) + np.shape(data) #this works even is data is a float
        observables.flat[self.obsname] = np.full(shape=shape, fill_value = np.nan, dtype = data.dtype)
        approx_size = 4*product(shape) #assumes 64bit floats
        assert(approx_size < 1e9) #try not to use more than 1Gb per allocation
        logger.debug(f"observables.flat['{self.obsname}'] = np.array(shape = {shape}, dtype = {data.dtype}) approx size: {approx_size/1e9:.2f}Gb")


    def copy(self, observables, j, datafile):
        for i in range(len(observables.Ns)):
            if datafile[i] == None: continue #fail gracefully on partially computed results
            observables.flat[self.obsname][i, j] = getattr(datafile[i], self.obsname)
                
              
    def reshape(self, structure_dims, observables):
        o = observables.flat[self.obsname]
        observables[self.obsname] = o.reshape(o.shape[0:1] + structure_dims + o.shape[2:])

        observables.hints[self.obsname] = ('Ns',) + tuple(observables.structure_names) + ('MCMC_step',)
        if self.obsname == 'Mf_moments':
            observables.hints[self.obsname] = ('Ns',) + tuple(observables.structure_names) + ('nth moment', 'MCMC_step')
 


class mean_over_MCMC(object):
    def __init__(self, obsname, N_error_bins = 1):
        self.obsname = obsname
        self.N_error_bins = N_error_bins
    
    def allocate(self, observables, example_datafile, N_jobs):
        logs = example_datafile.logs[0]
        Ns = example_datafile.Ns
        data = np.array(getattr(logs, self.obsname))

        shape = (len(Ns), N_jobs) + np.shape(data)[:-1] #this works even is data is a float
        self.taking_mean = (len(np.shape(data)) > 0)
        observables.flat[self.obsname] = np.full(shape=shape, fill_value = np.nan, dtype = data.dtype)
        
        if self.taking_mean:
            observables.flat['sigma_' + self.obsname] = np.full(shape=shape, fill_value = np.nan, dtype = data.dtype)
        
        approx_size = 4*product(shape) #assumes 64bit floats
        assert(approx_size < 1e9) #try not to use more than 1Gb per allocation
        logger.debug(f"observables.flat['{self.obsname}'] = np.array(shape = {shape}, dtype = {data.dtype}) approx size: {approx_size/1e9:.2f}Gb")


    def copy(self, observables, j, datafile):
        for i in range(len(observables.Ns)):
            if datafile[i] == None: continue #fail gracefully on partially computed results
            data = getattr(datafile[i], self.obsname)
            self.taking_mean = (len(np.shape(data)) > 0)
            if self.taking_mean:
                observables.flat[self.obsname][i, j] = data.mean(axis = -1)
                #print(i, self.obsname, data.shape, self.N_error_bins)
                observables.flat['sigma_' + self.obsname][i, j] = binned_error_estimate_multidim(data, N_bins = self.N_error_bins, axis = -1)
            else:
                observables.flat[self.obsname][i, j] = data
                
                
            
            
    def reshape(self, structure_dims, observables):
        o = observables.flat[self.obsname]
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
        observables.flat['IPR'] = np.full(shape=shape, fill_value = np.nan, dtype = dtype)
        observables.flat['DOS'] = np.full(shape=shape, fill_value = np.nan, dtype = dtype)
        observables.flat['dIPR'] = np.full(shape=shape, fill_value = np.nan, dtype = dtype)
        observables.flat['dDOS'] = np.full(shape=shape, fill_value = np.nan, dtype = dtype)
    
        
        approx_size = 4*product(shape) #assumes 64bit floats
        assert(approx_size < 1e9) #try not to use more than 1Gb per allocation
        logger.debug(f"observables.flat['IPRs'] = np.array(shape = {shape}, dtype = {dtype}) approx size: {approx_size/1e9:.2f}Gb")
        logger.debug(f"observables.flat['DOS'] = np.array(shape = {shape}, dtype = {dtype}) approx size: {approx_size/1e9:.2f}Gb")


    def copy(self, observables, j, datafile):
        
        for i, N in zip(count(), observables.Ns):
            if datafile[i] == None: continue #fail gracefully on partially computed results
            
            #shape is (MCMC_steps, system_size)
            raw_IPRs = getattr(datafile[i], 'IPRs')
            raw_eigenvals = getattr(datafile[i], 'eigenvals')
            
            DOS, dDOS, IPR, dIPR = compute_IPR_and_DOS_histograms(raw_eigenvals,
                                                                  raw_IPRs, self.E_bins, self.bootstrap_bins)

            observables.flat['DOS'][i, j, :] = DOS
            observables.flat['IPR'][i, j, :] = IPR
            observables.flat['dDOS'][i, j, :] = dDOS
            observables.flat['dIPR'][i, j, :] = dIPR
            
    def reshape(self, structure_dims, observables):
        for name in ['DOS', 'IPR', 'dDOS', 'dIPR']:
            o = observables.flat[name]
            observables[name] = o.reshape(o.shape[0:1] + structure_dims + o.shape[2:])
            observables.hints[name] = ('Ns',) + tuple(observables.structure_names) +  ('energy index',)


            
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
        if dimension_name in d and d[dimension_name].ndim == 1: return d[dimension_name].shape[0] #when the name corresponds to something like Ts or Ns
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
    #copy extra info over, note that structure_names might appear as a key in d, but I just overwrite it for now
    observables.update({k : v[()] for k,v in d.items() if k != 'logs'})
    observables.structure_names = structure_names
    observables.structure_dims = structure_dims
    observables.hints = Munch() 
    observables.flat = Munch()
    
    for f in functions: f.allocate(observables, example_datafile = d, N_jobs = N_jobs)
    

    
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
    #update_description(this_run.stem, infostring)
    
    return observables

def execute_script(py_script):
    contents = list(py_script.open().readlines())
    flag = '#bath_params_end_flag'
    for i, l in enumerate(contents):
        if flag in l: break
    try:
        context = dict()
        code = '\n'.join(contents[:i+1])
        exec(code, globals(), context)
        context = Munch(context)
        return context
    except IndexError:
        print(f"Didn't find {flag} in script")
        raise IndexError
    

from collections import defaultdict


def datafile_load(f):
    #check the file exists
    if not f.exists(): 
        logger.debug(f'{f} is expected but missing!')
        return None
    
    #see if it can be parsed
    try:
        d = np.load(f, allow_pickle = True)['logs']
    except:
        logger.debug(f'{f} exists but cannot be loaded.')
        return None
    
    #check that the final N is present in the file
    if d is None or d[-1] is None:
        logger.debug(f'{f} is only partially finished.')
        return None
    
    return d


def datafile_concat(datafiles, Ns):
    #datafiles is a list of lists where the outer is chain_ext and inner is Ns
    datafile = [Munch() for _ in Ns]
    names = ['IPRs', 'eigenvals', 'Mf_moments', 'eigenval_bins', 'time', 'accept_rates', 'proposal_rates', 'state']
    
    for name in names:
        if not hasattr(datafiles[0][0], name): continue
        shape = shape_hints(name)
        if 'MCstep' in shape:
            axis = shape.index('MCstep')
        for i, N in enumerate(Ns):
            if name == 'time':
                datafile[i][name] = np.sum([getattr(log[i], name) for log in datafiles])
            elif name == 'eigenval_bins':
                log = datafiles[0]
                datafile[i][name] = getattr(log[i], name)
            else:
                datafile[i][name] = np.concatenate([getattr(log[i], name) for log in datafiles], axis = axis)
    return datafile

def _get_data_funcmap_chain_ext_copy_data(o):
    with mp.Pool(18) as p:
        try: todo = set(o.task_id_range)
        except AttributeError: todo = set(range(o.N_tasks))
        todo = sorted(todo - o.processed_task_ids)
        logger.debug(f"todo: {todo}") 
        
        for task_id in todo:
            if task_id in o.processed_task_ids: continue
            
            filename_list = [o.datapath / f'{task_id}_{chain_id}.npz' for chain_id in range(o.N_chains)]
            
            #datafile_load catches three possible errors and returns None:
            #if the file doesn't exist, if it can't be read, or if it is only partial.
            datafile_list = list(p.map(datafile_load, filename_list))
            
                
            #check if there are any problem datafiles
            problems = [d is None for d in datafile_list]
            if any(problems):
                if all(problems): #give up
                    print(colored(task_id, 'red'), end = ' ')
                    continue 
                else: #deal with it and carry on
                    print(colored(task_id, 'yellow'), end = ' ')
                    datafile_list = [d for d in datafile_list if d is not None]
            
            #convert all those datafiles to one
            datafile = datafile_concat(datafile_list, o.Ns)
            
            for f in o.functions: f.copy(o, task_id, datafile)
            
            #should only get to here if everything went well!
            o.processed_task_ids.add(task_id)
            print(task_id, end = ' ')

def get_data_funcmap_chain_ext(this_run,
            functions = [],
            strict_chain_length = True,
            chain_length = None,
            task_id_range = None,
            ):
    
    '''
    '''
    o = Munch()
    o.functions = functions
    o.flat = Munch() #for the flat versions of the copied data
    o.hints = Munch() #for the hints
    o.processed_task_ids = set() #to keep track of what files have been processed
    if task_id_range is not None: o.task_id_range = task_id_range
    
    o.this_run = this_run.expanduser()
    if o.this_run.exists():
        logger.info(f'looking in {o.this_run}')
    else: 
        logger.info(f'{o.this_run} does not exist, quiting.')
        return
        
    o.datapath = o.this_run / 'data'
    o.codepath = o.this_run / 'code'
    
    #get the batch params from the original script
    print(list(o.codepath.glob('*.py')))
    o.py_script = next(o.codepath.glob('*.py'))
    context = execute_script(o.py_script)
    o.batch_params = Munch(context.batch_params)
    o.structure_names = o.batch_params.structure_names
    o.structure_dims = tuple(d.size for d in o.batch_params.structure_dimensions)
    
    logger.debug(f'structure_names = {o.structure_names}')
    logger.debug(f'structure_dims = {o.structure_dims}')
    
    #calculate the expected number of jobs
    def name2id(n): return tuple(map(int,n.split('_')))
    
    datafiles = dict()
    task_ids = set()
    chain_ids = defaultdict(set)
    for f in o.datapath.glob('*.npz'):
        task_id, chain_id = name2id(f.stem)
        datafiles[(task_id, chain_id)] = f
        task_ids.add(task_id)
        chain_ids[task_id].add(chain_id)
    
    
    o.N_tasks = product(o.structure_dims)
    o.chains = [max(c) + 1 for c in chain_ids.values()]
    o.N_chains = max(o.chains)
    
    logger.debug(f'Missing jobs, should all be up to {max(o.chains)-1}')
    logger.debug(f'task_id: chain_ids')
    for task_id, chain_ids in chain_ids.items():
        if len(chain_ids) < o.N_chains: 
            logger.debug(f'{task_id}: {chain_ids}')
    
    logger.info(f'Expected number of tasks {o.N_tasks}')
    logger.info(f'Measured number of tasks {len(task_ids)}')
    logger.info(f'Expected number of chains {chain_length}')
    logger.info(f'Shortest Chain {min(o.chains)}')
    logger.info(f'Longest Chain {max(o.chains)}')
    logger.info(f'Using chain length {o.N_chains}')
    if chain_length is not None: o.N_chains = chain_length
    
    o.functions += [extract('time'), 
                  mean_over_MCMC('accept_rates', N_error_bins = 1),
                  mean_over_MCMC('proposal_rates', N_error_bins = 1)]
    
    if len(datafiles) == 0: 
        logger.error("NO DATA FILES FOUND");
        return
    
    #get stuff from an an example datafile
    d = Munch(np.load(next(iter(datafiles.values())), allow_pickle = True))
    parameters = d['parameters'][()]
    MCMC_params = d['MCMC_params'][()]
    
    logger.info(f'Logger keys: {dir(d.logs[0])} \n')
    logger.info(f"MCMC_params keys: {list(MCMC_params.keys())} \n")
    
    o.original_N_steps = MCMC_params['N_steps']
    o.thin = MCMC_params['thin']
    o.N_steps = o.original_N_steps // o.thin
    o.max_MC_step = o.N_steps * o.N_chains
    
    logger.info(f'Overall steps = {o.N_steps * o.N_chains}')
    
    logger.debug(list(zip(count(), o.structure_names, o.structure_dims)))

    possible_observables = [s for s in dir(d.logs[0]) if not s.startswith("_")]
    logger.debug(f'available observables = {possible_observables}')
    
    logger.debug(f'Allocating space for the requested observables:')
    for f in o.functions: f.allocate(o, example_datafile = d, N_jobs = o.N_tasks)
    
    #copy extra info over ignoring data that alread appears
    [logger.warn(f'data files contain a key {k} which observables already has, not overwriting') for k,v in d.items() if k in o]
    o.update({k : v[()] for k,v in d.items() if k != 'logs' and k not in o})
    
    for name, dim in zip(o.structure_names, o.batch_params.structure_dimensions):
        if name in o: 
            logger.warning(f'{name} is already a key in observables but its a dimension name too!')
            continue
        o[name] = dim
    
    _get_data_funcmap_chain_ext_copy_data(o)
    
    for f in o.functions:
        f.reshape(o.structure_dims, o)
    
   
    logger.info('########################################################################\n')
    logger.info(f'Observables has keys: {o.keys()}')
    
    infostring = \
    f"""
    Completed jobs:?
    MCMC Steps: {o.N_chains} chains of {o.original_N_steps} for {o.original_N_steps*o.N_chains} with thinning = {o.thin} for {o.N_steps*o.N_chains} recorded steps
    Burn in: {Munch(MCMC_params).N_burn_in}
    Structure_names: {dict(zip(o.structure_names, o.structure_dims))}
    Ns = {o.Ns}
    Runtimes: 
        Average: {timefmt(np.nanmean(o.time.sum(axis=0)))}
        Min: {timefmt(np.nanmin(o.time.sum(axis=0)))}
        Max: {timefmt(np.nanmax(o.time.sum(axis=0)))}
        Total: {timefmt(np.nansum(o.time))}
    """[1:]
    logger.info(infostring)
    update_description(o.this_run.stem, infostring)
    
    return o

def incremental_get_data_funcmap_chain_ext(o,
            functions = [],
            strict_chain_length = True,
            task_id_range = None,
            ):
    
    '''
    '''
    o.functions = functions + [extract('time'), 
              mean_over_MCMC('accept_rates', N_error_bins = 1),
              mean_over_MCMC('proposal_rates', N_error_bins = 1)]
    
    if task_id_range is not None: o.task_id_range = task_id_range
    
    #calculate the epected number of jobs
    def name2id(n): return tuple(map(int,n.split('_')))
    
    datafiles = dict()
    task_ids = set()
    chain_ids = defaultdict(set)
    for f in o.datapath.glob('*.npz'):
        task_id, chain_id = name2id(f.stem)
        datafiles[(task_id, chain_id)] = f
        task_ids.add(task_id)
        chain_ids[task_id].add(chain_id)

    N_chains = min(max(c) for c in chain_ids.values()) + 1
    if N_chains > o.N_chains: 
        logger.info("A whole new chain since last time! regenerate the entire thing because new space needs to be allocated")
        return
    o.N_chains = N_chains
    
    _get_data_funcmap_chain_ext_copy_data(o)
    
    #for f in o.functions:
    #    f.reshape(o.structure_dims, o)
    
   
    logger.info('########################################################################\n')
    logger.info(f'Observables has keys: {o.keys()}')
    
    infostring = \
    f"""
    Completed jobs:?
    MCMC Steps: {o.N_chains} chains of {o.original_N_steps} for {o.original_N_steps*o.N_chains} with thinning = {o.thin} for {o.N_steps*o.N_chains} recorded steps
    Burn in: {o.MCMC_params.N_burn_in}
    Structure_names: {dict(zip(o.structure_names, o.structure_dims))}
    Ns = {o.Ns}
    Runtimes: 
        Average: {timefmt(np.nanmean(o.time.sum(axis=0)))}
        Min: {timefmt(np.nanmin(o.time.sum(axis=0)))}
        Max: {timefmt(np.nanmax(o.time.sum(axis=0)))}
        Total: {timefmt(np.nansum(o.time))}
    
    """[1:]
    logger.info(infostring)
    
    return o

import pickle
import shutil

def incremental_load(folder, functions, 
                     force_reload = False, 
                     force_to_use_pickled = False,
                     loglevel = logging.DEBUG):
    folder = Path(folder).expanduser()
    load_filepath = folder / 'loaded_data.pickle'
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    
    if force_reload or not load_filepath.exists():
        o = get_data_funcmap_chain_ext(folder, functions = functions)

    else: #we know the file exists and we're not forcing a reload
        with open(load_filepath, 'rb') as file:
            o = pickle.load(file)
            if force_to_use_pickled: 
                return o

        incremental_get_data_funcmap_chain_ext(o, functions = functions)

    if load_filepath.exists(): shutil.copy(load_filepath, folder / 'loaded_data.pickle.backup')

    #save it whatever process we went through
    try:
        with open(load_filepath, 'wb') as file:
            if 'functions' in o: del o['functions']
            pickle.dump(o, file)
    except:
        pass
        
    return o


from FKMC.general import smooth, spread
from scipy.stats import sem

def interpolate_IPR(E_bins, unsmoothed_DOS, IPR, dIPR):
    newshape = (IPR.size // IPR.shape[-1], IPR.shape[-1])
    _DOS = unsmoothed_DOS.reshape(newshape)
    _IPR = IPR.reshape(newshape)
    _dIPR = dIPR.reshape(newshape)
    
    for i, DOS, I, dI in zip(count(), _DOS, _IPR, _dIPR):
        ei = DOS > 0
        if any(ei):
            _I = I[ei]
            _dI = dI[ei]
            xI = E_bins[1:][ei]

            _IPR[i] = np.interp(E_bins[1:], xI, _I)
            _dIPR[i] = np.interp(E_bins[1:], xI, _dI)
        else:
            _IPR[i] = E_bins[1:] * np.NaN
            _dIPR[i] = E_bins[1:] * np.NaN