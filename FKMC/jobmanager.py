import numpy as np
from pathlib import Path
import h5py
import click
import time
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from . import quantum_montecarlo



routine_map = {
    #'cython_mcmc' : cython_mcmc,
    'quantum_cython_mcmc' : quantum_montecarlo.quantum_cython_mcmc
              }

def read_config_file(working_dir):
    result_filename = working_dir / "results.hdf5"
    with h5py.File(result_filename, "r") as result_file:
            #you have to conver the hdf5 reference to a python object before closing the file 
            #othewise you get errors because hdf5py doesn't actually read any data until you ask
            config = dict(result_file.attrs)
    return config
    

def config_dimensions(config):
    names = config['loop_over']
    vals = np.array([config[v] for v in names])
    lens = np.array([len(config[v]) for v in names])
    dtypes = np.array([config[v].dtype for v in names])
    return names, vals, lens, dtypes

def total_jobs(config):
    names, vals, lens, dtypes = config_dimensions(config)
    return lens.prod()

def get_config(job_id, configs):
    '''
    Take a job number and a config with multiple values and index into the cartesian product of configurations
    use the value 'loop_over' to decide which values get looped over
    '''
    assert(job_id < total_jobs(configs))
    #get all the keys that aren't looped over
    single_config= {k:v for k,v in configs.items() if k not in set(configs['loop_over'])}

    #starting from the innermost loop work outwards
    indices = []
    job_id_remainder = job_id
    for key in configs['loop_over'][::-1]:
        values = configs[key]
        job_id_remainder, i  = divmod(job_id_remainder, len(values))
        single_config[key] = values[i]
        indices.append(i)

    indices = tuple(indices[::-1])

    single_config['job_id'] = job_id
    single_config['indices'] = indices

    return single_config

def setup_mcmc(config, working_dir = Path('./'), overwrite = False):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'Working in: {working_dir}')

    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / 'jobs').mkdir(parents=True, exist_ok=True)
    (working_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    result_filename = working_dir / "results.hdf5"
    N_jobs = total_jobs(config)
    
    if result_filename.exists() and overwrite == False:
        logger.info(f'Directory exists, skipping setup step')
    else:
        #create the results file and metadata from scratch
        names, vals, lens, dtypes = config_dimensions(config)
        
        first_config = get_config(job_id=0, configs=config)
        mcmc_routine = routine_map[config['mcmc_routine']]
        results = mcmc_routine(**first_config, sample_output = True)
        results['runtime'] = np.array([0,])
        
        logger.info(f'Sample results:')
        for k,v in results.items():
            description = f'array(shape={v.shape}, dtype={v.dtype})' if type(v)==np.ndarray else v
            logger.info(f'{k}: {description}')
        

        with h5py.File(result_filename, "w") as result_file:
            result_file.attrs.update(config)
            for name, val in results.items():
                data_shape = tuple(np.append(lens, val.shape))
                result_file.create_dataset(name, data = np.zeros(data_shape)*np.nan, shape = data_shape, dtype = val.dtype)

    #update or create the script because the number of jobs to do might have changed
    #cx1_wdir = '$HOME' / working_dir.resolve().relative_to('/workspace/tch14/cx1_home/')
    cx1_wdir = working_dir.resolve()
    
    script = open('../sample_runscript.sh').read().format(working_dir=cx1_wdir, N_jobs=N_jobs, name = working_dir.stem)
    with open(working_dir / 'runscript.sh', 'w') as f:
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
    
    working_dir = Path(working_dir)
    result_file = working_dir / "results.hdf5"
    
    if not result_file.exists():
        logger.info(f'No result file found')
        return
    
    with h5py.File(result_file, "r") as f:
        config = dict(f.attrs)
    logger.info(f'Loaded config')
    
    routine_name = config['mcmc_routine']
    logger.info(f'Executing routine {routine_name}')
    mcmc_routine = routine_map[routine_name]
    
    (working_dir / 'jobs').mkdir(exist_ok = True)
    job_file = working_dir / 'jobs' / f"job_{job_id}.hdf5"
    if job_file.exists() and overwrite == False:
        logger.info(f'Job File already exists, not overwriting it')
        return

    starttime = time.time()
    this_config = get_config(job_id, config)
    
    logger.info(f'This jobs config is {this_config}')
    logger.info(f'Starting MCMC routine {mcmc_routine}')
    
    results = mcmc_routine(**this_config)
    runtime = np.array([time.time() - starttime,])
    results['runtime'] = runtime
    logger.info(f"MCMC routine finished after {runtime[0]:.2f} seconds")

    logger.info(f'Copying the results into the h5py file')
    with h5py.File(job_file, "w") as f:
        f.attrs.update(this_config)
            
        for name, result in results.items():
            result_data = f.create_dataset(name, data=result)

def job_completion(working_dir):
    result_filename = working_dir / "results.hdf5"
    job_dir =  working_dir / "jobs"

    missing = []
    with h5py.File(result_filename, "r+") as result_file:
        config = dict(result_file.attrs)

        for job_id in range(total_jobs(config)):
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
            config['copied_in'] = np.zeros(total_jobs(config), dtype=np.int)
        
        #jobs to copy is the list of results that we're going to try to find, this could be all of them or just a subset
        if do_all: jobs_to_copy = np.arange(total_jobs(config))
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
                    for dataset_name, val in job_file.items():
                        dataset = result_file[dataset_name]

                        indices = tuple(job_file.attrs['indices'])

                        #label each axis of the dataset
                        for dim,name in zip(dataset.dims,config['loop_over']):
                            dim.label = name

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
        logger.info(f'Overall completion: {total_jobs(config) - len(jobs_remaining_to_copy)} / {total_jobs(config)}')
        logger.info(f'File size: {result_filename.stat().st_size / 10**9:.2f}Gb')
        logger.debug(f'Config: {dict(result_file.attrs)}')
        
        return missing
