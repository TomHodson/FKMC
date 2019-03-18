import numpy as np
from pathlib import Path
import h5py
import click
import time
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from . import quantum_montecarlo



routine_map = {
    #'cython_mcmc' : cython_mcmc,
    'quantum_cython_mcmc' : quantum_montecarlo.quantum_cython_mcmc
              }

def read_config_file(working_dir):
    result_filename = working_dir / "results.hdf5"
    with h5py.File(result_filename, "r") as result_file:
            #you have to conver the hdf5 reference to a python object before closing the file 
            #othewise you get errors because hdf5py uses lazy loading from disk
            config = dict(result_file.attrs)
    return config
    

def config_dimensions(config, key = 'outer_loop'):
    names = config[key]
    vals = np.array([config[v] for v in names])
    lens = np.array([len(config[v]) for v in names])
    dtypes = np.array([config[v].dtype for v in names])
    return names, vals, lens, dtypes

def total_jobs(config):
    _, _, outer_lens, _ = config_dimensions(config, key='outer_loop')
    _, _, inner_lens, _ = config_dimensions(config, key='inner_loop')
    return outer_lens.prod(), inner_lens.prod()

#plan
#make total_jobs return a tuple of internal and external jobs
#get both, need an index into each
#then return the config for that pair
#modify config_dimensions(config) to give the extra dimensions too

#also add some debug information about the total number of jobs, subjobs and insctructions used
#run_mcmc will need an equivalent of gather now
#make sure gather returns the right thing

def loop_keys(config): return set(config['outer_loop']) | set(config['inner_loop'])

def static_config(config):
    """return all the keys from a config that doesn't need to be looped over"""
    return {k:v for k,v in config.items() if k not in loop_keys(config)}

def loop_config(id, loop_name, config):
    keys = config[loop_name]
    
    max_id = np.array([len(config[v]) for v in keys]).prod()
    assert(id < max_id)
    
    #starting from the innermost loop work outwards
    indices = []
    this_config = {}

    id_remainder = id
    for key in keys[::-1]:
        values = config[key]
        id_remainder, i  = divmod(id_remainder, len(values))
        this_config[key] = values[i]
        indices.append(i)

    indices = tuple(indices[::-1])
    this_config[loop_name + '_index'] = id
    this_config[loop_name + '_indices'] = indices
    
    return this_config
    
def outer_loop_shape(config): return np.array([len(config[v]) for v in config['outer_loop']])
def inner_loop_shape(config): return np.array([len(config[v]) for v in config['inner_loop']])

def get_config(outer_index, inner_index, config):
    '''
    Take a job number and a config with multiple values and index into the cartesian product of configurations
    use the value 'loop_over' to decide which values get looped over
    '''
    static = static_config(config)
    outer = loop_config(outer_index, 'outer_loop', config)
    inner = loop_config(inner_index, 'inner_loop', config)
    
    this_config = {**static, **outer, **inner}
    return this_config

def setup_mcmc(config, working_dir = Path('./'), overwrite = False):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'Working in: {working_dir}')
    
    if 'loop_over' in config:
        logger.warning(f"'Loop_over' shouldn't be in the config anymore")
        return
    
    #check that we're not trying to loop over parameters that affect the shape of any of the outputs
    bad_combos = dict(
        output_history=['N_steps'],
        output_correlator=['N_system'],
        output_state=['N_steps','N_system'],
    )
    loop_k = loop_keys(config)
    bad = False
    for flag, bad_params in bad_combos.items():
        for param in bad_params:
            if config[flag] and param in loop_k:
                logger.warning(f"Can't have both {flag} = True and loop over {param}")
                bad = True
    if bad: return

    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / 'jobs').mkdir(parents=True, exist_ok=True)
    (working_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    result_filename = working_dir / "results.hdf5"
    N_jobs = total_jobs(config)
    
    if result_filename.exists() and overwrite == False:
        logger.info(f'Directory exists, skipping setup step')
    else:
        #create the results file and metadata from scratch
        first_config = get_config(outer_index=0, inner_index=0, config=config)
        mcmc_routine = routine_map[config['mcmc_routine']]
        results = mcmc_routine(**first_config, sample_output = True)
        results['runtime'] = np.array([0,])
        
        logger.info(f'Sample results:')
        for k,v in results.items():
            description = f'array(shape={v.shape}, dtype={v.dtype})' if type(v)==np.ndarray else v
            logger.info(f'{k}: {description}')
        
        looped_shape = np.append(outer_loop_shape(config), inner_loop_shape(config))
        with h5py.File(result_filename, "w") as result_file:
            result_file.attrs.update(config)
            for name, val in results.items():
                data_shape = tuple(np.append(looped_shape, val.shape))
                result_file.create_dataset(name, data = np.zeros(data_shape)*np.nan, shape = data_shape, dtype = val.dtype)

    #update or create the script because the number of jobs to do might have changed
    #cx1_wdir = '$HOME' / working_dir.resolve().relative_to('/workspace/tch14/cx1_home/')
    cx1_wdir = working_dir.resolve()
    
    import pkg_resources
    logger.info(pkg_resources.resource_listdir(__name__, "."))
    cx1 = pkg_resources.resource_string(__name__, "CX1_runscript.sh").decode()
    cmth = pkg_resources.resource_string(__name__, "CMTH_runscript.sh").decode()
    
    
    script = cx1.format(working_dir=cx1_wdir, N_jobs=outer_loop_shape(config).prod(), name = working_dir.stem)
    with open(working_dir / 'CX1_runscript.sh', 'w') as f:
        f.write(script)
        print(script)
        
    script = cmth.format(working_dir=cx1_wdir, N_jobs=outer_loop_shape(config).prod(), name = working_dir.stem)
    with open(working_dir / 'CMTH_runscript.sh', 'w') as f:
        f.write(script)
        print(script)

    


@click.command()
@click.option('--job-id', default=1, help='which job to run')
@click.option('--working-dir', default='./', help='where to look for the config files')
@click.option('--temp-dir', default='./', help='where to store temporary files')
def run_mcmc_command(*args, **kwargs):
    return run_mcmc(*args, **kwargs)

def run_mcmc(job_id, 
             working_dir = Path('./'),
             temp_dir = Path('./'),
             overwrite = False):
    '''
    Does the work that a single thread is expected to do.
    '''
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'job_id: {job_id}')
    
    ##get a handle on the results file
    working_dir = Path(working_dir)
    result_file = working_dir / "results.hdf5"
    
    if not result_file.exists():
        logger.info(f'No result file found')
        return
    
    ##load up the config into memory
    with h5py.File(result_file, "r") as f:
        config = dict(f.attrs)
    logger.debug(f'Loaded config')
    
    ##get the routine warmed up
    routine_name = config['mcmc_routine']
    logger.debug(f'Executing routine {routine_name}')
    mcmc_routine = routine_map[routine_name]
    
    ##get the job file
    (working_dir / 'jobs').mkdir(exist_ok = True)
    job_file_path = working_dir / 'jobs' / f"job_{job_id}.hdf5"
    if job_file_path.exists() and overwrite == False:
        logger.info(f'Job File already exists, not overwriting it')
        return

    ##figure out the outer_config
    static = static_config(config)
    outer_config = loop_config(job_id, 'outer_loop', config)

    logger.info(f'This jobs outer_config is {outer_config}')
    logger.info(f'Starting MCMC routine {mcmc_routine} inner loop')
    starttime = time.time()
    
    #open the file and create the datasets needed
    with h5py.File(job_file_path, "w") as job_file:
        job_file.attrs.update(outer_config)
        
        #do the inner loop
        for inner_index in range(inner_loop_shape(config).prod()):
            inner_time = time.time()
            inner_config = loop_config(inner_index, 'inner_loop', config)
            this_config = {**static, **outer_config, **inner_config}
            results = mcmc_routine(**this_config)
            idx = this_config['inner_loop_indices']
            
            
            #the first time, initialise the file
            if inner_index == 0:
                logger.info(f"Since it's the first one, creating the datasets:")
                for name, val in results.items():
                    data_shape = tuple(np.append(inner_loop_shape(config), val.shape))
                    job_file.create_dataset(name, data = np.zeros(data_shape)*np.nan, shape = data_shape, dtype = val.dtype)
                    logger.debug(f"Dataset: name: {name}, data.shape {data_shape}, dtype: {val.dtype}")
                
                
            #the rest of the time, just copy the data in
            for name, result in results.items():
                job_file[name][idx] = result
            
            logger.info(f"Done: Inner Job: {inner_index} indices: {idx} runtime: {time.time() - inner_time:.2f} seconds")
            
    runtime = np.array([time.time() - starttime,])
    logger.info(f"MCMC routine finished after {runtime[0]:.2f} seconds")

def job_completion(working_dir):
    result_filename = working_dir / "results.hdf5"
    job_dir =  working_dir / "jobs"

    missing = []
    with h5py.File(result_filename, "r+") as result_file:
        config = dict(result_file.attrs)

        for job_id in range(outer_loop_shape(config).prod()):
            job_filename = job_dir / f"job_{job_id}.hdf5"
            if not job_filename.exists():
                missing.append(job_id)
                
        return np.array(missing)
   
from ipywidgets import IntProgress
from IPython.display import display

def gather_mcmc(working_dir, do_all = False):
    '''
    copied_in: if any of the results need to be overwritten (if a job erred but still produced a jobresult) then set this to true
    '''
    
    logger = logging.getLogger(__name__)
    result_filename = working_dir / "results.hdf5"
    job_dir =  working_dir / "jobs"
    
    #logger.info(f'Number of jobs that have finished (for any reason): {}')
    #logger.info(f'Number of jobs that have emitted errors (for any reason): {}')

    missing = []
    with h5py.File(result_filename, "r+") as result_file:
        config = result_file.attrs
        if 'copied_in' not in config:
            logger.info(f"copied_in wasn't in config, initialising it")
            config['copied_in'] = np.zeros(outer_loop_shape(config).prod(), dtype=np.int8)
        
        #jobs to copy is the list of results that we're going to try to find, this could be all of them or just a subset
        if do_all: jobs_to_copy = np.arange(outer_loop_shape(config).prod())
        else: jobs_to_copy = np.where(config['copied_in'] == 0)[0]
        
        logger.debug(f'Config: {dict(config)}')
        logger.info(f'Number of Jobs to copy in: {len(jobs_to_copy)}')
        logger.info(f'Job IDs: {jobs_to_copy}...')
        
        #bar = IntProgress(max=len(jobs_to_copy),description='Progress:')
        #display(bar)
        for job_id in jobs_to_copy:
            #bar.value += 1
            job_filename = job_dir / f"job_{job_id}.hdf5"
            if not job_filename.exists():
                logger.debug(f"Job ID {job_id} results file doesn't exist")
                missing.append(job_id)
                continue
            
            logger.debug(f'Starting Job ID: {job_id}')
            
            try:
                with h5py.File(job_filename, 'r') as job_file:
                    #loop over the datasets, ie energy, magnetisation etc
                    for index, (dataset_name, val) in enumerate(job_file.items()):
                        dataset = result_file[dataset_name]

                        indices = tuple(job_file.attrs['outer_loop_indices'])

                        #label each axis of the dataset
                        for dim,name in zip(dataset.dims,np.append(config['outer_loop'],config['inner_loop'])):
                            dim.label = name
                            #dataset.dims.create_scale(result_file.attrs[name], name)
                            #dim.attach_scale(result_file.attrs[name])

                        logger.debug(f'{dataset_name}, indices:{indices}, val.shape {val.shape}, dataset[indices].shape: {dataset[indices].shape}')
                        dataset[indices] = val

                    #indicate that this data has been copied into the result file sucessfully
                    #by putting a 1 in the right place
                    #have to be careful because hdf5py attribute variables cannot be modified by slicing
                    mask = np.arange(len(config['copied_in'])) == job_id
                    config['copied_in'] = np.logical_or(config['copied_in'], mask)
            except OSError:
                logger.info(f"Couldn't open {job_filename}")
                
        jobs_remaining_to_copy = np.where(config['copied_in'] == 0)[0]
        logger.info(f'missing : {missing}')
        logger.info(f'Jobs attempted this time: {len(jobs_to_copy)}')
        logger.info(f'Overall completion: {total_jobs(config)[0] - len(jobs_remaining_to_copy)} / {total_jobs(config)}')
        logger.info(f'File size: {result_filename.stat().st_size / 10**9:.2f}Gb')
        logger.debug(f'Config: {dict(result_file.attrs)}')
        
        return missing
