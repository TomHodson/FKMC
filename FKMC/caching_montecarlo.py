from math import exp
import numpy as np
from scipy.linalg import eigh_tridiagonal, LinAlgError, circulant
import scipy
from time import time
from collections import Counter
from munch import Munch
from numpy.random import default_rng

from .general import interaction_matrix, solve_H, convert_to_central_moments, index_histogram
from .montecarlo import perturbation_accept, p_multi_site_uniform_reflect, Ff

def diagonalise(state, Ff, A, powers, U, mu, t, beta, **kwargs): 
    d = Munch()
    d.evals, d.evecs = eigh_tridiagonal(d = U*(state - 1/2) - mu, e =-t*np.ones(state.shape[0] - 1), lapack_driver = 'stev')
    d.Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * d.evals)))
    
    d.Nf = np.sum(state) / state.size
    d.Nc = np.sum(1/(1 + np.exp(beta * d.evals))) / state.size
    d.Mf_moments = np.sum(2*(state - 1/2) * A / state.size)**powers
    d.state = state
    d.IPRs = ((d.evecs * np.conj(d.evecs))**2).sum(axis = 0)
    
    return d

### Actual Algorithm #######################################################################################################################################################
def FK_mcmc(
    state = None, proposal = None, proposal_args = dict(), accept_function = None, parameters = dict(mu=0, beta=1, alpha=1.5, J=1, U=1, t=1, normalise = True),            
    N_steps = 100, N_burn_in = 10, thin = 1, logger = None, warnings = True, info = False, rng = None, raw_steps = False, **kwargs,
    ):
    assert(N_steps % thin == 0)
    if isinstance(state, np.ndarray):
        N_sites = state.shape[0]
    elif 'N_sites' in parameters:
        N_sites = parameters['N_sites']
        state = np.arange(N_sites, dtype = np.float64) % 2
        
    if N_sites % 2 == 1:
        print("Odd system sizes don't have CDW phases!!!! Exiting")
        return
    
    if rng is None:
        rng = default_rng()
        
    t0 = time()
    parameters.update(J_matrix = interaction_matrix(N_sites, dtype = np.float64, **parameters))
    
    #first run
    current_Ff, current_Fc, evals, evecs = solve_H(state=state, **parameters)
    logger = Munch()
    logger.accept_rates, logger.proposal_rates, logger.classical_accept_rates = np.zeros(shape = (N_sites+1,3)).T
    logger.Ff_cache_hits, logger.Ff_cache_misses, logger.full_cache_hits, logger.full_cache_misses = 0, 0, 0, 0
    
    A = 2*(np.arange(state.size) % 2) - 1
    powers = np.arange(5)
    
    Ffs = dict()
    counts = Counter()
    full_cache = dict()
    
    actual_steps = (N_steps + N_burn_in) * max(1, N_sites*N_sites // 100)
    if raw_steps: actual_steps = (N_steps + N_burn_in)
    states = np.zeros((actual_steps,N_sites), dtype = np.float64)
    update_batch = max(1, actual_steps // 10)
    for i in range(actual_steps):
        if (i%update_batch == 0):
            c_r = (np.sum(logger.classical_accept_rates) / max(1, np.sum(logger.proposal_rates)))
            q_r = np.sum(logger.accept_rates) / max(1, np.sum(logger.classical_accept_rates))
            o_r = np.sum(logger.accept_rates) / max(1, np.sum(logger.proposal_rates))
            print(f"N = {N_sites}: {100*i/actual_steps:.0f}% through after {(time() - t0)/60:.2f}m \
            acceptance rate: classical = {c_r*100:.2g}% quantum = {q_r*100:.2g}% overall = {o_r*100:.2g}% \
            Ff cache {100*logger.Ff_cache_hits/(logger.Ff_cache_hits+logger.Ff_cache_misses+1):.2g} full: {100*logger.full_cache_hits/(logger.full_cache_misses+logger.full_cache_hits+1):.2g}")
        
        sites = proposal(i, N_sites, rng, **proposal_args)
        logger.proposal_rates[len(sites)] += 1
        state[sites] = 1 - state[sites]
        state_int = hash(tuple(state))
        
        #decide whether to accept this new state

        #Look up the classical energy in the cache
        if state_int in Ffs: 
            logger.Ff_cache_hits += 1
            new_Ff = Ffs[state_int]
        else:
            logger.Ff_cache_misses += 1
            new_Ff = Ff(state, **parameters)
            Ffs[state_int] = new_Ff

        dFf = new_Ff - current_Ff
        beta = parameters['beta']
        accepted = False
        
        if dFf < 0 or exp(- beta * dFf) > rng.random():
            logger.classical_accept_rates[len(sites)] += 1

            if state_int in full_cache:
                logger.full_cache_hits += 1
                new = full_cache[state_int]
            else:
                logger.full_cache_misses += 1
                new = diagonalise(state, new_Ff, A, powers, **parameters)
                full_cache[state_int] = new

            dFc = new.Fc - current_Fc
            accepted =  dFc < 0 or exp(- beta * dFc) > rng.random()

        if accepted:    
            current = new
            current_Ff = new_Ff
            current_Fc = new.Fc
            logger.accept_rates[len(sites)] += 1
            if i >= N_burn_in: counts[state_int] += 1
        else:
            state[sites] = 1 - state[sites]
            
        states[i] = state
    
    p_acc = sum(logger.accept_rates) / sum(logger.proposal_rates)
    params_sans_matrix = parameters.copy()
    params_sans_matrix.update(J_matrix = 'suppressed for brevity')
    if warnings:
        if p_acc < 0.2 or p_acc > 0.5: print(f"Warning, p_acc = {p_acc}, {params_sans_matrix}")
    if info:
        p_propose = logger.proposal_rates / sum(logger.proposal_rates)
        p_classical_accept = logger.classical_accept_rates / sum(logger.classical_accept_rates) / p_propose
        p_accept = logger.accept_rates / sum(logger.accept_rates) / p_propose
        prop = ' '.join(f'{p:.0f}%' for p in 100 * p_propose)
        c_acc = ' '.join(f'{p:.0f}%' for p in 100 * p_classical_accept)
        acc = ' '.join(f'{p:.0f}%' for p in 100 * p_accept)
        pert_saving = 100 * (1 - sum(logger.classical_accept_rates) / sum(logger.proposal_rates))
        print(f"""
        Number of burn in steps = {N_burn_in}
        Number of MCMC steps = {N_steps}
        Thinning = {thin}
        Proposal function = {proposal.__name__}
        Acceptance function = {accept_function.__name__}
        logger = {logger}
        parameters = {parameters}
        Chance of proposing an N spin flip: {prop}
        Chance of classically accepting an N spin flip: {c_acc} 
        Chance of accepting an N spin flip: {acc}
        Percentage of the time the matrix is not diagonalised: {pert_saving:.0f}%
        """)
        
    #print(f'acceptance probability: {accepted / N_sites / (N_steps + N_burn_in)}, mu = {mu}')
    return counts, full_cache, state, states

### Datalogger #######################################################################################################################################################

#a catch all datalogger which I use for everything
class Eigenspectrum_IPR_all(object):
    def __init__(self, bins = 70, limit = 5, N_cumulants = 5):
        self.N_cumulants = N_cumulants
        self.eigenval_bins = np.linspace(-limit, limit, bins + 1)
    
        
    def start(self, N_steps, N_sites):
        self.N_sites = N_sites
        self.N_steps = N_steps
        self.A = 2*(np.arange(N_sites) % 2) - 1
        self.Ff, self.Fc, self.Nf, self.Nc = np.zeros((N_steps,4), dtype = np.float64).T
        self.state = np.zeros((N_steps,N_sites), dtype = np.float64)
        self.powers = np.arange(self.N_cumulants)
        self.Mf_moments = np.zeros((self.N_cumulants, N_steps), dtype = np.float64)
        self.state, self.eigenvals, self.IPRs = np.zeros((3,N_steps,N_sites), dtype = np.float64)
    
    def update(self, j, Ff, Fc, state, evals, evecs, mu, beta, J_matrix, **kwargs):
        self.Ff[j] = Ff
        self.Fc[j] = Fc
        self.Nf[j] = np.sum(state) / self.N_sites
        self.Nc[j] = np.sum(1/(1 + np.exp(beta * evals))) / self.N_sites
        self.Mf_moments[:, j] = np.sum(2*(state - 1/2) * self.A / self.N_sites)**self.powers
        self.state[j] = state
        
        self.eigenvals[j] = evals
        self.IPRs[j] = ((evecs * np.conj(evecs))**2).sum(axis = 0)

    def return_vals(self):
        return self
    
