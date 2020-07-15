def get_data(this_run,
            observable_names = ['Ff', 'Fc', 'Mf_moments', 'time'],
            temp_slice = slice(None,None,1),
            MCMC_slice = slice(None,None,1),
            ):
    print(f'looking in {this_run}')
    datafiles = sorted([(int(f.stem), f) for f in this_run.iterdir() if f.name.endswith('npz') and not f.name == 'parameters.npz'])
    jobs = np.array([j_id for j_id, f in datafiles])
    print(f'job ids range from {min(jobs)} to {max(jobs)}')
    print(f'missing jobs: {set(range(max(jobs))) - set(jobs)}\n')
    
    d = np.load(datafiles[0][1], allow_pickle = True)
    Ts = d['Ts']
    Ns = d['Ns']
    parameters = d['parameters'][()]
    MCMC_params = d['MCMC_params'][()]
    
    
    print('keys: ', list(d.keys()), '\n')
    print('MCMC_params: ', MCMC_params, '\n')
    
    original_N_steps = MCMC_params['N_steps']
    thin = MCMC_params['thin']
    N_steps = original_N_steps // thin
    print(f'MCMC Steps: {original_N_steps} with thinning = {thin} for {N_steps} recorded steps')
    print(f'Slicing this with MCMC_slice = {MCMC_slice}\n')
    
    N_Ts = len(Ts)
    T = Ts[temp_slice]
    
    print(f'T has (min, max, len) = {[f(Ts) for f in [min, max, len]]}, slicing by {temp_slice} to get {[f(T) for f in [min, max, len]]}\n')

    possible_observables = [s for s in dir(d["logs"][0]) if not s.startswith("_")]
    print(f'dir(d["logs"][0]) = {possible_observables}')
    print(f'Allocating space for the requested observables {observable_names}:')
    observables = Munch()
    for name in observable_names:
        src = getattr(d['logs'][0], name)
        if type(src) == np.ndarray: 
            shape = (len(Ns), len(T),) + src[..., MCMC_slice].shape
        elif type(src) == float:
            src = np.array(src)
            shape = (len(Ns), len(T),)
        else:
            raise ValueError(f'Not sure what to do with {src}')
            
        
        print(name, shape)
        approx_size = 4*product(shape) #assumes 64bit floats
        assert(approx_size < 1e9) #try not to use more than 1Gb per allocation
        observables[name] = np.zeros(shape, dtype = src.dtype)
        print(f"observables['{name}'] = np.array(shape = {shape}, dtype = {src.dtype}) approx size: {approx_size/1e9:.2f}Gb")

    print('Copying the data')
    t0 = time()
    dot_batch = len(T) // 50 if len(T) > 50 else 1
    for j, (n,file) in enumerate(datafiles[temp_slice]):
        npz_file = np.load(file, allow_pickle = True)['logs']
        if (j == 10) or (j == len(T)//2): 
            dt = time() - t0
            print(f'\nTook {dt:.2f}s to do the first {j}, should take {(len(T)-j)*dt/j:.2f}s to do the remaining {len(T) - j}\n')
        if j % dot_batch == 0: print(f'{n} ', end='')
        for name in observable_names:
            for i in range(len(Ns)):
                obs = getattr(npz_file[i], name)
                if type(obs) == np.ndarray:
                    observables[name][i, j] = obs[..., MCMC_slice]
                elif type(obs) == float:
                    observables[name][i, j] = obs
                else:
                    raise ValueError(f'Not sure what to do with {obs}')
    print('')
    print('########################################################################\n')
    observables['T'] = T
    observables.update({k : v[()] for k,v in d.items() if k != 'logs'})
    print(f'Observables has keys: {observables.keys()}')
    
    return Munch(observables)
   
    
def get_data_eigenvals(this_run,
            observable_names = ['Ff', 'Fc', 'Mf_moments', 'time'],
            temp_slice = slice(None,None,1),
            MCMC_slice = slice(None,None,1),
            ):
    
    'This version has support for getting the eigenvalues too, it also pads out the data'
    print(f'looking in {this_run}')
    datafiles = sorted([(int(f.stem), f) for f in this_run.iterdir() if f.name.endswith('npz') and not f.name == 'parameters.npz'])
    jobs = np.array([j_id for j_id, f in datafiles])
    print(f'job ids range from {min(jobs)} to {max(jobs)}')
    print(f'missing jobs: {set(range(max(jobs))) - set(jobs)}\n')
    
    d = np.load(datafiles[0][1], allow_pickle = True)
    
    Ts = d['Ts']
    Ns = d['Ns']
    parameters = d['parameters'][()]
    MCMC_params = d['MCMC_params'][()]
    
    
    print('keys: ', list(d.keys()), '\n')
    print('MCMC_params: ', MCMC_params, '\n')
    
    original_N_steps = MCMC_params['N_steps']
    thin = MCMC_params['thin']
    N_steps = original_N_steps // thin
    print(f'MCMC Steps: {original_N_steps} with thinning = {thin} for {N_steps} recorded steps')
    print(f'Slicing this with MCMC_slice = {MCMC_slice}\n')
    
    N_Ts = len(Ts)
    T = Ts[temp_slice]
    
    print(f'T has (min, max, len) = {[f(Ts) for f in [min, max, len]]}, slicing by {temp_slice} to get {[f(T) for f in [min, max, len]]}\n')

    possible_observables = [s for s in dir(d["logs"][0]) if not s.startswith("_")]
    print(f'dir(d["logs"][0]) = {possible_observables}')
    print(f'Allocating space for the requested observables {observable_names}:')
    observables = Munch()
    for name in observable_names:
        src = getattr(d['logs'][0], name)
        
        #if it's something with a dimension that depends on N it can't be in a simple np array
        if name in ['IPRs', 'eigenvals', 'state']:
            shapes = [(len(T),) + getattr(d['logs'][i], name)[..., MCMC_slice].shape for i, N in enumerate(Ns)]
            observables[name] = [np.full(shape=shape, fill_value = np.nan, dtype = src.dtype)
                                 for shape in shapes]
            print(f"observables['{name}'] = python list of np arrays with shapes {shapes}")

        else:
            #if it's a single observable measured for every MCMC step
            if type(src) == np.ndarray: 
                shape = (len(Ns), len(T),) + src[..., MCMC_slice].shape
                observables[name] = np.full(shape=shape, fill_value = np.nan, dtype = src.dtype)
            
            #if it's something measured over the whole MCMC run
            elif type(src) == float:
                src = np.array(src)
                shape = (len(Ns), len(T),)
                observables[name] = np.full(shape=shape, fill_value = np.nan, dtype = src.dtype)
            
            approx_size = 4*product(shape) #assumes 64bit floats
            assert(approx_size < 1e9) #try not to use more than 1Gb per allocation
            print(f"observables['{name}'] = np.array(shape = {shape}, dtype = {src.dtype}) approx size: {approx_size/1e9:.2f}Gb")

                                 
                                 
        if name not in observables:
            raise ValueError(f'Not sure what to do with {src}')
            
        

    print('Copying the data')
    t0 = time()
    dot_batch = len(T) // 50 if len(T) > 50 else 1
    for j, (n,file) in enumerate(datafiles[temp_slice]):
        npz_file = np.load(file, allow_pickle = True)['logs']
        if (j == 10) or (j == len(T)//2): 
            dt = time() - t0
            print(f'\nTook {dt:.2f}s to do the first {j}, should take {(len(T)-j)*dt/j:.2f}s to do the remaining {len(T) - j}\n')
        if j % dot_batch == 0: print(f'{n} ', end='')
        for name in observable_names:
            for i in range(len(Ns)):
                obs = getattr(npz_file[i], name)
                
                if name in ['IPRs', 'eigenvals', 'state']:
                    observables[name][i][j] = obs[:]
                else:
                    if type(obs) == np.ndarray:
                        observables[name][i, j] = obs[..., MCMC_slice]
                    elif type(obs) == float:
                        observables[name][i, j] = obs
                    else:
                        raise ValueError(f'Not sure what to do with {obs}')
    print('')
    print('########################################################################\n')
    observables['T'] = T
    observables.update({k : v[()] for k,v in d.items() if k != 'logs'})
    print(f'Observables has keys: {observables.keys()}')
    
    return Munch(observables)

def get_data_structured(this_run,
            requested_observables = ['Ff', 'Fc', 'Mf_moments', 'time'],
            MCMC_slice = slice(None,None,1),
            structure_names = ('Ts', 'Repeats'),
            structure_dims = (),
            ):
    
    '''
    This version has support for getting the eigenvalues too, it also pads out the data
    The structure argument has the forms of a tuple containing strings and ints which give the shape of the data,
    strings should refer to labels in the data like Ts whose shape will be used
    and ints create an unamed axis for things like repeats
    '''
    logger.critical(f'looking in {this_run}')
    datafiles = sorted([(int(f.stem), f) for f in this_run.iterdir() if f.name.endswith('npz') and not f.name == 'parameters.npz'])
    jobs = np.array([j_id for j_id, f in datafiles])
    if len(jobs) == 0: 
        logger.error("NO DATA FILES FOUND")
        return
    logger.info(f'job ids range from {min(jobs)} to {max(jobs)}')
    
    #an example datafile
    d = Munch(np.load(datafiles[0][1], allow_pickle = True))
    
    Ts = d['Ts']
    Ns = d['Ns']
    parameters = d['parameters'][()]
    MCMC_params = d['MCMC_params'][()]
    
    structure_dims = tuple(i if i != None else d[name].shape[0] if d[name].ndim > 0 else d[name] for i,name in zip_longest(structure_dims, structure_names, fillvalue=None))
    logger.info(f'structure_names = {structure_names}')
    logger.info(f'structure_dims = {structure_dims}')

    N_jobs = product(structure_dims)
    logger.info(f'Expected number of jobs {N_jobs}')
    missing = set(range(N_jobs)) - set(jobs)
    if missing: logger.warning(f'Missing jobs: {missing}\n')
    
    logger.info(f'Logger keys: {list(d.keys())} \n')
    logger.info(f'MCMC_params: {MCMC_params} \n')
    
    original_N_steps = MCMC_params['N_steps']
    thin = MCMC_params['thin']
    N_steps = original_N_steps // thin
    logger.info(f'MCMC Steps: {original_N_steps} with thinning = {thin} for {N_steps} recorded steps')
    logger.info(f'Slicing this with MCMC_slice = {MCMC_slice}\n')
    
    N_Ts = len(Ts)
    T = Ts
    
    logger.info(list(zip(count(), structure_names, structure_dims)))

    possible_observables = [s for s in dir(d.logs[0]) if not s.startswith("_")]
    logger.info(f'available observables = {possible_observables}')
    logger.info(f'requested observables = {requested_observables}')
    
    logger.info(f'Allocating space for the requested observables:')
    observables = allocate(requested_observables, d, N_jobs, MCMC_slice)

    #copy extra info over, note that structure_names might appear as a key in d, but I just overwrite it for now
    observables.update({k : v[()] for k,v in d.items() if k != 'logs'})
    observables.structure_names = structure_names
    observables.structure_dims = structure_dims
    
    logger.info(f'Copying the data approx size: {observables.total_size/1e9:.2f}Gb')
    copy(requested_observables, datafiles, observables, MCMC_slice)
    reshape(structure_dims, requested_observables, observables)
    
   
    logger.info('########################################################################\n')
    observables['T'] = T
    logger.info(f'Observables has keys: {observables.keys()}')
    
    return Munch(observables)