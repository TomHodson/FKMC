from math import exp
import numpy as np
from .general import interaction_matrix, solve_H, convert_to_central_moments, index_histogram
from scipy.linalg import eigh_tridiagonal, LinAlgError, circulant
import scipy
from time import time
import cachetools
from numpy.random import default_rng


### proposal functions #######################################################################################################################################################
def p_single_typewriter(j, N_sites, **kwargs): return [j%N_sites,]
def p_single_random_site(j, N_sites, **kwargs): return [np.random.randint(N_sites),]
def p_multi_site_fixed(multi, **kwargs): return lambda k, N_sites:  np.random.randint(N_sites, size = multi)
def p_multi_site_uniform(j, N_sites, **kwargs): return np.random.randint(N_sites, size = np.random.randint(1,N_sites))
def p_uncorrelated_proposal(j, N_sites, **kwargs): return np.nonzero(np.random.choice([0,1], size = N_sites))[0]

def p_one_or_two_reflect(j, N_sites, **kwargs):
    p_reflect = 1/N_sites
    reflect = p_reflect > np.random.rand() #whether or not we're gonna reflect the whole state this time
    if reflect:
        return np.arange(N_sites) #flip all the sites
    else:
        n_sites = np.random.randint(1,3)
        return np.random.randint(low=0, high=N_sites, size=n_sites, dtype=np.int)

def p_multi_site_uniform_reflect(j, N_sites, rng, **kwargs):
    p_reflect = 1/N_sites
    reflect = p_reflect > rng.random() #whether or not we're gonna reflect the whole state this time
    if reflect:
        return np.arange(N_sites) #flip all the sites
    else:
        n_sites = rng.integers(1, N_sites)
        return rng.integers(0, N_sites, size=n_sites, dtype=np.int)


def p_multi_site_poisson_reflect(j, N_sites, rng, lam=1, **kwargs):
    p_reflect = 1/N_sites
    reflect = p_reflect > rng.random() #whether or not we're gonna reflect the whole state this time
    if reflect:
        return np.arange(N_sites) #flip all the sites
    else:
        n_sites = min(rng.poisson(lam = lam), N_sites-1)
        return rng.integers(0, N_sites, size=n_sites, dtype=np.int)

def p_multi_site_variable_reflect_exponential(j, N_sites, rng, scale = 1, **kwargs):
    p_reflect = 1/N_sites
    reflect = p_reflect > rng.random() #whether or not we're gonna reflect the whole state this time
    if reflect:
        return np.arange(N_sites) #flip all the sites
    else:
        n_sites = N_sites
        while n_sites >= N_sites:
            n_sites = int(rng.exponential(scale))
        return rng.integers(0, N_sites, size=n_sites, dtype=np.int)
    

### Acceptance functions #######################################################################################################################################################
def simple_accept(state, sites, logger, current_Ff, current_Fc, parameters, rng):
    new_Ff, new_Fc, evals, evecs = solve_H(state=state, **parameters)
    dF = (new_Ff + new_Fc) - (current_Ff + current_Fc)
    
    beta = parameters['beta']
    if dF < 0 or exp(- beta * dF) > rng.random():
        return True, new_Ff, new_Fc
    else:
        return False, current_Ff, current_Fc

def FKhash(state, **params):
    return hash((state.tobytes(), state.size, (v for k,v in params.items() if k in ['t', 'mu', 'alpha', 'U', 'J', 'beta'])))
 
#implements perturbation mcmc staving off having to calculate the determinant every time
@cachetools.cached(cachetools.LFUCache(maxsize=int(1e3)), key = FKhash)
def Ff(state, U, mu, J_matrix, **kwargs): 
    return - U/2*np.sum(state - 1/2) - mu*np.sum(state) + (state - 1/2).T @ J_matrix @ (state - 1/2)

@cachetools.cached(cachetools.LFUCache(maxsize=int(1e3)), key = FKhash)
def solve_H(state, mu, beta, U, J_matrix, t, J, alpha, **kwargs):
    state = np.array(state)
    muf = muc = mu
    
    #This is already cached, add J and alpha as arguments because they define J_matrix for caching purposes
    _Ff = Ff(state, U=U, mu=mu, J_matrix=J_matrix, J=J, alpha=alpha)
    evals, evecs = eigh_tridiagonal(d = U*(state - 1/2) - mu, e =-t*np.ones(state.shape[0] - 1), lapack_driver = 'stev')
    Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * evals)))
    return _Ff, Fc, evals, evecs


def perturbation_accept(state, sites, logger, current_Ff, current_Fc, parameters, rng, cache):
    
    new_Ff = Ff(state, **parameters)
    dFf = new_Ff - current_Ff

    beta = parameters['beta']
    #print(f"{exp(- beta * dFf):.3g}")
    if dFf < 0 or exp(- beta * dFf) > rng.random():
        logger.classical_accept_rates[len(sites)] += 1
        try: 
           evals, evecs = cache[tuple(state)]
        except KeyError:
            evals, evecs = diagonalise(state, **parameters)
            cache[tuple(state)] = (evals, evecs)
            
        new_Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * evals)))
        dFc = new_Fc - current_Fc
        
        if dFc < 0 or exp(- beta * dFc) > rng.random():
            return True, new_Ff, new_Fc
    
    return False, current_Ff, current_Fc

from collections import Counter

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
    current_Ff, current_Fc, evals, evecs = solve_H(state=state, **parameters)
    if logger == None: logger = DataLogger()
    logger.start(N_steps // thin, N_sites)
    logger.accept_rates, logger.proposal_rates, logger.classical_accept_rates = np.zeros(shape = (N_sites+1,3)).T
    cache = dict()
    
    update_batch = max(1 ,(N_steps + N_burn_in) // 10)
    for i in range(N_steps + N_burn_in):
        if (i%update_batch == 0):
            c_r = (np.sum(logger.classical_accept_rates) / max(1, np.sum(logger.proposal_rates)))
            q_r = np.sum(logger.accept_rates) / max(1, np.sum(logger.classical_accept_rates))
            o_r = np.sum(logger.accept_rates) / max(1, np.sum(logger.proposal_rates))
            print(f"N = {N_sites}: {100*i/(N_steps+N_burn_in):.0f}% through after {(time() - t0)/60:.2f}m \
            acceptance rates: classical = {c_r*100:.2g}% quantum = {q_r*100:.2g}% overall = {o_r*100:.2g}%")
        
        #I realised late that all the autocorrelation times seem to scale with N_sites**2, so I divide by 100
        #so that jobs of size 100 will do the number of iterations as before
        substeps = 1 if raw_steps else max(1, N_sites*N_sites // 100)
        for j in range(substeps): 
            sites = proposal(j, N_sites, rng, **proposal_args)
            logger.proposal_rates[len(sites)] += 1
            state[sites] = 1 - state[sites]
            accepted, current_Ff, current_Fc = accept_function(state, sites, logger, current_Ff, current_Fc, parameters, rng, cache)

            if accepted:
                logger.accept_rates[len(sites)] += 1
            if not accepted:
                state[sites] = 1 - state[sites]
                
        if i >= N_burn_in and i % thin == 0:
            current_Ff, current_Fc, evals, evecs = solve_H(state, **parameters)
            j = (i - N_burn_in) // thin
            logger.update(j, current_Ff, current_Fc, state, evals, evecs, **parameters)
    
    logger.last_state = state
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
    return logger.return_vals()

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
    
class Mf_moments(object):
    def __init__(self, N_cumulants = 5):
        self.N_cumulants = N_cumulants
    
    def start(self, N_steps, N_sites):
        self.N_sites = N_sites
        self.N_steps = N_steps
        self.A = 2*(np.arange(N_sites) % 2) - 1
        self.powers = np.arange(self.N_cumulants)
        self.Mf_moments = np.zeros((self.N_cumulants, N_steps), dtype = np.float64)
    
    def update(self, j, Ff, Fc, state, evals, evecs, mu, beta, J_matrix, **kwargs):
        self.Mf_moments[:, j] = np.sum(2*(state - 1/2) * self.A / self.N_sites)**self.powers

    def return_vals(self):
        return self

'''
class Eigenspectrum_IPR(object):
    def __init__(self, bins = 70, limit = 5):
        self.eigenval_bins = np.linspace(-limit, limit, bins + 1)
    
    def start(self, N_steps, N_sites):
        self.N_steps = N_steps
        self.eigenval_histogram = np.zeros((N_steps,self.eigenval_bins.shape[0]-1), dtype = np.float64)
        self.IPR_histogram = np.zeros((N_steps,self.eigenval_bins.shape[0]-1), dtype = np.float64)

    def update(self, j, Ff, Fc, state, evals, evecs, mu, beta, J_matrix, **kwargs):
        IPRs = ((evecs * np.conj(evecs))**2).sum(axis = 0)
        self.eigenval_histogram[j], _, indices = index_histogram(self.eigenval_bins, evals)
        self.IPR_histogram[j] = np.bincount(indices, weights=IPRs, minlength = self.eigenval_bins.shape[0] + 1)[1:-1]

    
    def return_vals(self):
        E_histogram = np.mean(self.eigenval_histogram, axis = 0)
        normalisation_factor = np.sum(E_histogram)
        
        E_histogram = E_histogram / normalisation_factor 
        dE_histogram = scipy.stats.sem(self.eigenval_histogram, axis = 0) / normalisation_factor
        
        IPR_histogram, dIPR_histogram = np.mean(self.IPR_histogram, axis = 0), scipy.stats.sem(self.IPR_histogram, axis = 0)
        
        return self.eigenval_bins, E_histogram, dE_histogram, IPR_histogram, dIPR_histogram
    
def FK_mcmc_2(*args, **kwargs): return FK_mcmc(*args, **kwargs)

#a catch all datalogger
class DataLogger(object):
    def __init__(self, N_cumulants = 5):
        self.N_cumulants = N_cumulants
        
    def start(self, N_steps, N_sites):
        self.N_sites = N_sites
        self.N_steps = N_steps
        self.A = 2*(np.arange(N_sites) % 2) - 1
        self.Ff, self.Fc, self.Nf, self.Nc = np.zeros((N_steps,4), dtype = np.float64).T
        self.state, self.eigenvals = np.zeros((2,N_steps,N_sites), dtype = np.float64)
        self.eigenvecs = np.zeros((N_steps,N_sites,N_sites), dtype = np.float64)
        self.powers = np.arange(self.N_cumulants)
        self.Mf_moments = np.zeros((self.N_cumulants, N_steps), dtype = np.float64)
    
    def update(self, j, Ff, Fc, state, evals, evecs, mu, beta, J_matrix, **kwargs):
        self.Ff[j] = Ff
        self.Fc[j] = Fc
        self.Nf[j] = np.sum(state) / self.N_sites
        self.Nc[j] = np.sum(1/(1 + np.exp(beta * evals))) / self.N_sites
        self.Mf_moments[:, j] = np.sum(2*(state - 1/2) * self.A / self.N_sites)**self.powers
        self.state[j] = state
        self.eigenvals[j] = evals

    
    def return_vals(self):
        #self.Mf, self.dMf = np.mean(self.Mf_moments, axis = 0), scipy.stats.sem(self.Mf_moments, axis = 0)
        self.cMf_moments = (self.Mf_moments[1] - np.mean(self.Mf_moments[1]))[None, :] ** self.powers[:, None]
        return self
    #a catch all datalogger
    
class Observables(object):
    def __init__(self, N_cumulants = 5):
        self.N_cumulants = N_cumulants
        
    def start(self, N_steps, N_sites):
        self.N_sites = N_sites
        self.N_steps = N_steps
        self.A = 2*(np.arange(N_sites) % 2) - 1
        self.Ff, self.Fc, self.Nf, self.Nc = np.zeros((N_steps,4), dtype = np.float64).T
        self.powers = np.arange(self.N_cumulants)
        self.Mf_moments = np.zeros((self.N_cumulants, N_steps), dtype = np.float64)
    
    def update(self, j, Ff, Fc, state, evals, evecs, mu, beta, J_matrix, **kwargs):
        self.Ff[j] = Ff
        self.Fc[j] = Fc
        self.Nf[j] = np.sum(state) / self.N_sites
        self.Nc[j] = np.sum(1/(1 + np.exp(beta * evals))) / self.N_sites
        self.Mf_moments[:, j] = np.sum(2*(state - 1/2) * self.A / self.N_sites)**self.powers

    
    def return_vals(self):
        #self.Mf, self.dMf = np.mean(self.Mf_moments, axis = 0), scipy.stats.sem(self.Mf_moments, axis = 0)
        self.cMf_moments = (self.Mf_moments[1] - np.mean(self.Mf_moments[1]))[None, :] ** self.powers[:, None]
        return self

##a datalogger to compute the cumulants of magnetisation
class Magnetisation_cumulants(object):
    def __init__(self, N_cumulants = 5):
        self.N_cumulants = N_cumulants
    
    def start(self, N_steps, N_sites):
        self.N_sites = N_sites
        self.A = 2*(np.arange(N_sites) % 2) - 1
        self.powers = np.arange(self.N_cumulants)
        self.Mf_moments = np.zeros((N_steps, self.N_cumulants), dtype = np.float64)
    
    def update(self, j, Ff, Fc, state, evals, evecs, mu, beta, J_matrix, **kwargs):
        self.Mf_moments[j] = np.sum(2*(state - 1/2) * self.A / self.N_sites)**self.powers
    
    def return_vals(self):
        moments, dmoments = np.mean(self.Mf_moments, axis = 0), scipy.stats.sem(self.Mf_moments, axis = 0)
        central_moments = convert_to_central_moments(moments)
        return moments, central_moments

class Magnetisation_squared(object):
    def start(self, N_steps, N_sites):
        self.A = 2*(np.arange(system_size) % 2) - 1
        self.M2f = np.zeros((N_steps), dtype = np.float64).T
    
    def update(self, j, Ff, Fc, state, evals, evecs, mu, beta, J_matrix, **kwargs):
        self.M2f[j] = np.sum((state - 1/2) * self.A)**2
    
    def return_vals(self):
        return np.mean(self.M2f), scipy.stats.sem(self.M2f)
    
class Density(object):
    def start(self, N_steps, N_sites):
        self.N = np.zeros((N_steps), dtype = np.float64)
    
    def update(self, j, Ff, Fc, state, evals, evecs, mu, beta, J_matrix, **kwargs):
        self.N[j] = np.sum(state) + np.sum(1/(1 + np.exp(beta * evals)))
    
    def return_vals(self):
        return np.mean(self.N), scipy.stats.sem(self.N)
    
class NfNc(object):
    def start(self, N_steps, N_sites):
        self.N_sites = N_sites
        self.Nf = np.zeros((N_steps), dtype = np.float64)
        self.Nc = np.zeros((N_steps), dtype = np.float64)
    
    def update(self, j, Ff, Fc, state, evals, evecs, mu, beta, J_matrix, **kwargs):
        self.Nf[j] = np.sum(state) / self.N_sites
        self.Nc[j] = np.sum(1/(1 + np.exp(beta * evals))) / self.N_sites
    
    def return_vals(self):
        Nf, Nc, dNf, dNc = np.mean(self.Nf), np.mean(self.Nc), scipy.stats.sem(self.Nf), scipy.stats.sem(self.Nc)
        return Nf, Nc, dNf, dNc

'''