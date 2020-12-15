import numpy as np
from .general import interaction_matrix, solve_H
from numpy.random import default_rng
from time import time

def FK_disorder_model(
    states,
    parameters = dict(mu=0, beta=1, alpha=1.5, J=1, U=1, t=1, normalise = True),            
    logger = None,
    warnings = True,
    info = False,
    rng = None,
    **kwargs,
    ):
    
    if isinstance(states, np.ndarray):
        N_steps, N_sites = states.shape
    
    if rng is None: rng = default_rng()
        
    t0 = time()
    
    parameters.update(J_matrix = interaction_matrix(N_sites, dtype = np.float64, **parameters))

    if logger == None: logger = DataLogger()
    logger.start(N_steps, N_sites)
    
    cache = dict()
    
    update_batch = max(1 ,(N_steps) // 10)
    
    for i in range(N_steps):
        if (i%update_batch == 0): print(f"N = {N_sites}: {100*i/(N_steps):.0f}% through after {(time() - t0)/60:.2f}m")
                
        current_Ff, current_Fc, evals, evecs = solve_H(states[i], **parameters)
        logger.update(i, current_Ff, current_Fc, states[i], evals, evecs, **parameters)
    
    params_sans_matrix = parameters.copy()
    params_sans_matrix.update(J_matrix = 'suppressed for brevity')
    
    if info:
        print(f"""
        Number of disorder realisations = {N_steps}
        logger = {logger}
        parameters = {parameters}
        """)

    return logger.return_vals()