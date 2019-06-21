from math import exp
import numpy as np
from general import interaction_matrix, solve_H, convert_to_central_moments
from scipy.linalg import eigh_tridiagonal, LinAlgError, circulant
import scipy

def FK_mcmc(
    state,            
    N_steps = 100, N_burn_in = 10,
    mu = 0, beta = 0.1, J=1, alpha=1.5, U = 1, t=1,
    logger = None, normalise = True, **kwargs,
    ):
    
    N_sites = state.shape[0]
    random_numbers = np.random.rand(N_steps + N_burn_in, N_sites)
    J_matrix = interaction_matrix(N_sites, alpha=alpha, J=J, normalise = normalise, dtype = np.float64)
    Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t)
    old_F = Ff + Fc
    if logger == None: logger = DataLogger()
    logger.start(N_steps, N_sites)
    accepted = 0
    
    for i in range(N_steps + N_burn_in):
        for site in range(N_sites):
            state[site] = 1 - state[site]
            Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t=t)
            dF = (Ff+Fc) - old_F

            #if the move is rejected flip the site back and pretend nothing happened
            if dF > 0 and exp(- beta * dF) < random_numbers[i, site]:
                state[site] = 1 - state[site]
            else:
                old_F = Ff+Fc
                accepted += 1
    
        if i >= N_burn_in:
            Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t=t)
            j = i - N_burn_in
            logger.update(j, Ff, Fc, state, evals, evecs,  mu, beta, J_matrix)

    p_acc = accepted / N_sites / (N_steps + N_burn_in)
    if p_acc < 0.2 or p_acc > 0.5: print(f'Warning, p_acc = {p_acc}, mu = {mu}, beta = {beta}, J = {J}')
    logger.p_acc = p_acc
    #print(f'acceptance probability: {accepted / N_sites / (N_steps + N_burn_in)}, mu = {mu}')
    return logger.return_vals()

### proposal functions
def p_single_typewriter(j, N_sites, **kwargs): return [j,]
def p_single_random_site(j, N_sites, **kwargs): return [np.random.randint(N_sites),]
def p_multi_site_fixed(multi, **kwargs): return lambda k, N_sites:  np.random.randint(N_sites, size = multi)
def p_multi_site_uniform(j, N_sites, **kwargs): return np.random.randint(N_sites, size = np.random.randint(1,N_sites))
def p_uncorrelated_proposal(j, N_sites, **kwargs): return np.nonzero(np.random.choice([0,1], size = N_sites))[0]

def p_multi_site_uniform_reflect(j, N_sites, **kwargs):
    p_reflect = 1/N_sites
    reflect = p_reflect > np.random.rand() #whether or not we're gonna reflect the whole state this time
    if reflect:
        return np.arange(N_sites) #flip all the sites
    else:
        n_sites = np.random.randint(1,N_sites)
        return np.random.randint(low=0, high=N_sites, size=n_sites, dtype=np.int)


def p_multi_site_poisson_reflect(j, N_sites, lam=1, **kwargs):
    p_reflect = 1/N_sites
    reflect = p_reflect > np.random.rand() #whether or not we're gonna reflect the whole state this time
    if reflect:
        return np.arange(N_sites) #flip all the sites
    else:
        n_sites = min(np.random.poisson(lam = lam), N_sites-1)
        return np.random.randint(low=0, high=N_sites, size=n_sites, dtype=np.int)

def p_multi_site_variable_reflect_exponential(j, N_sites, scale = 1, **kwargs):
    p_reflect = 1/N_sites
    reflect = p_reflect > np.random.rand() #whether or not we're gonna reflect the whole state this time
    if reflect:
        return np.arange(N_sites) #flip all the sites
    else:
        n_sites = N_sites
        while n_sites >= N_sites:
            n_sites = int(np.random.exponential(scale))
        return np.random.randint(low=0, high=N_sites, size=n_sites, dtype=np.int)
    

### Acceptance functions
def simple_accept(state, logger, current_Ff, current_Fc, parameters):
    new_Ff, new_Fc, evals, evecs = solve_H(state=state, **parameters)
    dF = (new_Ff + new_Fc) - (current_Ff + current_Fc)
    
    beta = parameters['beta']
    if dF < 0 or exp(- beta * dF) > np.random.rand():
        return True, new_Ff, new_Fc
    else:
        return False, current_Ff, current_Fc

## FIXME!

#implements perturbation mcmc staving off having to calculat the determinant every time
def perturbation_accept(state, logger, current_Ff, current_Fc, parameters):
    locals().update(**parameters)
    new_Ff = - U/2*np.sum(state - 1/2) - mu*np.sum(state) + (state - 1/2).T @ J_matrix @ (state - 1/2)
    dFf = new_Ff - current_Ff

    if dFf < 0 or exp(- beta * dFf) > np.random.rand():
        evals, evecs = eigh_tridiagonal(d = U*(state - 1/2) - mu, e =-t*np.ones(state.shape[0] - 1), lapack_driver = 'stev')
        new_Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * evals)))
        dFc = new_Fc - current_Fc
        
        if dFc < 0 or exp(- beta * dFc) > np.random.rand():
            return True, new_Ff, new_Fc
    
    return False, current_Ff, current_Fc

from collections import Counter
def FK_mcmc_2(
    state = None, proposal = None, proposal_args = dict(), accept_function = None, parameters = dict(mu=0, beta=1, alpha=1.5, J=1, U=1, t=1, normalise = True),            
    N_steps = 100, N_burn_in = 10, logger = None, **kwargs,
    ):
    
    N_sites = state.shape[0]
    parameters.update(J_matrix = interaction_matrix(N_sites, dtype = np.float64, **parameters))
    current_Ff, current_Fc, evals, evecs = solve_H(state=state, **parameters)
    if logger == None: logger = DataLogger()
    logger.start(N_steps, N_sites)
    logger.accept_rates, logger.proposal_rates  = np.zeros(shape = (N_sites+1,2)).T

    for i in range(N_steps + N_burn_in):
        for j in range(N_sites):
            sites = proposal(j, N_sites, **proposal_args)
            logger.proposal_rates[len(sites)] += 1
            state[sites] = 1 - state[sites]
            
            accepted, current_Ff, current_Fc = accept_function(state, logger, current_Ff, current_Fc, parameters)
            
            if accepted:
                logger.accept_rates[len(sites)] += 1
            if not accepted:
                state[sites] = 1 - state[sites]
                
    
        if i >= N_burn_in:
            current_Ff, current_Fc, evals, evecs = solve_H(state=state, **parameters)
            j = i - N_burn_in
            logger.update(j, current_Ff, current_Fc, state, evals, evecs, **parameters)
    
    logger.last_state = state
    p_acc = sum(logger.accept_rates) / sum(logger.proposal_rates)
    params_sans_matrix = parameters.copy()
    params_sans_matrix.update(J_matrix = 'suppressed for brevity')
    if p_acc < 0.2 or p_acc > 0.5: print(f"Warning, p_acc = {p_acc}, {params_sans_matrix}")
        
    #print(f'acceptance probability: {accepted / N_sites / (N_steps + N_burn_in)}, mu = {mu}')
    return logger.return_vals()


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