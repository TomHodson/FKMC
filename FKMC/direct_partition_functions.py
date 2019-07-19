import numpy as np
from .general import interaction_matrix, solve_H
from .general import convert_to_central_moments
import functools
from itertools import product

def states_to_numbers(states):
    return np.int64(np.sum(states * 2**np.arange(states.shape[1])[None, ::-1], axis = 1))
def state_to_number(state):
    return np.int64(np.sum(state * 2**np.arange(states.shape[0])))
def number_to_state(number):
    return np.unpackbits(np.array([number], dtype = np.uint8))

def enumerate_states(system_size, mu, beta, U, J, t = 1, alpha = 1.5, normalise = True, n_cumulants = 5):
    Z = 0
    Nf = Nc = Mf = M2f = 0
    muf = muc = mu
    A = 2*(np.arange(system_size) % 2) - 1
    J_matrix = interaction_matrix(system_size, alpha=alpha, J=J, normalise = normalise, dtype = np.float64)
    powers = np.arange(n_cumulants)
    
    states = np.array(list(product((0,1), repeat = system_size)), dtype = np.float64)
    state_labels = states_to_numbers(states)
    
    Fcs, Ffs, Nfs, Ncs = np.zeros(shape=(2**system_size,4), dtype = np.float64).T
    Mf_moments = np.zeros(shape=(2**system_size,n_cumulants), dtype = np.float64)
    
    for i, state in zip(state_labels, states):
        Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t)
        Ffs[i], Fcs[i] = Ff, Fc
        Nfs[i], Ncs[i] = np.sum(state) / system_size, np.sum(1 / (1 + np.exp(beta * evals))) / system_size
        Mf_moments[i] = (np.sum(2*(state - 1/2) * A)/system_size)**powers
    
    ix = np.argsort(Fcs + Ffs)
    return ix, state_labels[ix], states[ix], Ffs[ix], Fcs[ix], Nfs[ix], Ncs[ix], Mf_moments[ix, :].T

def direct_compute_observables(system_size, mu, beta, U, J_matrix, t, max_BF, A, state):
    muf = muc = mu
    if system_size > 1:
        Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t)
    elif system_size == 1:
        evals = np.array([U*(state - 1/2)]) - muc
        Ff = - U/2*np.sum(state - 1/2) - muf*np.sum(state)
        Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * evals)))    

    Boltz_factor = np.exp(- beta * (Ff + Fc))
    Nf = np.sum(state) * Boltz_factor / system_size
    Nc = np.sum(1 / (1 + np.exp(beta * evals))) * Boltz_factor / system_size
    Mf = np.sum(2*(state - 1/2) * A) * Boltz_factor / system_size
    M2f = np.sum(2*(state - 1/2) * A / system_size)**2 * Boltz_factor
    
    return Boltz_factor, Nf, Nc, Mf, M2f

def direct_parrallel(system_size, mu, beta, U, J, t = 1, alpha = 1.5, pool = None, normalise = True):
    J_matrix = interaction_matrix(system_size, alpha, J, normalise = normalise, dtype = np.float64)
    A = 2*(np.arange(system_size) % 2) - 1
    
    ##check the ground state energy
    state = (np.arange(system_size) % 2)
    muf = muc = mu
    if system_size > 1:
        Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t)
    elif system_size == 1:
        evals = np.array([U*(state - 1/2)]) - muc
        Ff = - U/2*np.sum(state - 1/2) - muf*np.sum(state)
        Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * evals)))  
    max_BF = - beta * (Ff + Fc)
    #print(f'max_BF = {max_BF}')
    
    compute = functools.partial(direct_compute_observables, system_size, mu, beta, U, J_matrix, t, max_BF, A)
    Z, Nf, Nc, Mf, M2f = np.sum([compute(state) for state in product((0,1), repeat = system_size)], axis = 0)
    Nf, Nc, Mf, M2f =  Nf/Z, Nc/Z, Mf/Z, M2f/Z

    return Nf/Z, Nc/Z, Mf/Z, M2f/Z
    

def direct(system_size, mu, beta, U, J, t = 1, alpha = 1.5, normalise = True, n_cumulants = 5):
    Z = Nf = Nc = Ff_bar = Fc_bar = 0
    powers = np.arange(n_cumulants, dtype = np.float64)
    Mf_moments = np.zeros_like(powers)
    A = 2*(np.arange(system_size) % 2) - 1
    J_matrix = interaction_matrix(system_size, alpha, J, normalise = normalise, dtype = np.float64)
    
    for state in product((0,1), repeat = system_size):
        state = np.array(state)
        Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t)
        Boltz_factor = np.exp(- beta * (Ff + Fc))
        
        Z += Boltz_factor
        Ff_bar += Ff * Boltz_factor
        Fc_bar += Fc * Boltz_factor
        Nf += np.sum(state) * Boltz_factor / system_size
        Nc += np.sum(1 / (1 + np.exp(beta * evals))) * Boltz_factor / system_size
        Mf_moments += (np.sum(2*(state - 1/2) * A)/system_size)**powers * Boltz_factor

    return np.concatenate([[Ff_bar / Z,], [Fc_bar / Z,], [Nf / Z,], [Nc / Z,], Mf_moments / Z])

def binder(system_size, mu, beta, U, J, t = 1, alpha = 1.5, n_cumulants = 5, normalise = True):
    Z = 0
    Nf = Nc = Mf = M2f = 0
    muf = muc = mu
    A = 2*(np.arange(system_size) % 2) - 1
    J_matrix = interaction_matrix(system_size, alpha, J, normalise = normalise, dtype = np.float64)
    
    powers = np.arange(n_cumulants)
    Mf_moments = np.zeros(powers.shape[0], dtype = np.float64)

    ##check the ground state energy
    state = (np.arange(system_size) % 2)
    Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t)
    max_BF = - beta * (Ff + Fc)
    #print(f'max_BF = {max_BF}')
    
    for i,state in enumerate(product((0,1), repeat = system_size)):
        state = np.array(state)
        Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t)

        lnP = - beta * (Ff + Fc) - max_BF + 650
        if(lnP > 700): print(f'- beta * (Ff + Fc) too large! : {- beta * (Ff + Fc)}')
        Boltz_factor = np.exp(lnP)
        Z += Boltz_factor
        Mf_moments += Boltz_factor * (np.sum(2*(state - 1/2) * A)/system_size)**powers

    #print(Mf_moments, Z)
    Mf_moments = Mf_moments / Z
    Mf_central_moments = convert_to_central_moments(Mf_moments)
    return Mf_moments, Mf_central_moments

def compute_observables(system_size, mu, beta, U, J_matrix, t, state):
    eigenval_bins = np.linspace(-5, 5, 70 + 1)
    state = np.array(state)
    muf = muc = mu
    if system_size > 1:
        Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t)
    elif system_size == 1:
        evals = np.array([U*(state - 1/2)]) - muc
        Ff = - U/2*np.sum(state - 1/2) - muf*np.sum(state)
        Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * evals)))    

    Boltz_factor = np.exp(- beta * (Ff + Fc))
    Z = Boltz_factor
    Nf = np.sum(state) * Boltz_factor
    Nc = np.sum(1 / (1 + np.exp(beta * evals))) * Boltz_factor

    IPRs = ((evecs * np.conj(evecs))**2).sum(axis = 0)

    #these are the indices corresponding to the energy bin that each energy and IPR should go into
    hist, _, indices = index_histogram(eigenval_bins, evals)
    this_IPR_hist = np.bincount(indices, weights=IPRs, minlength = eigenval_bins.shape[0] + 1)[1:-1]

    eigenval_hist = hist * Boltz_factor
    IPR_hist = this_IPR_hist * Boltz_factor #/ np.where(hist != 0, hist, 1)

    return np.concatenate([[Z,], [Nf,], [Nc,], eigenval_hist, IPR_hist])

def direct_spectrum(system_size, mu, beta, U, J, t = 1, pool = None):
    eigenval_bins = np.linspace(-5, 5, 70 + 1)
    J_matrix = interaction_matrix(system_size, alpha, J, normalise = True, dtype = np.float64)
    
    compute = functools.partial(compute_observables, system_size, mu, beta, U, J_matrix, t)
    
    if pool:
            packed_result = np.sum(pool.map(compute, product((0,1), repeat = system_size)), axis = 0)
    else:
        packed_result = np.sum(pool.map(compute, product((0,1), repeat = system_size)), axis = 0)
    
    Z, Nf, Nc, *rest = packed_result
    eigenval_hist, IPR_hist = rest[:eigenval_bins.shape[0]-1], rest[eigenval_bins.shape[0]-1:] 
        
    eigenval_hist = eigenval_hist / np.sum(eigenval_hist)
    return Nf / Z, Nc / Z, eigenval_hist, eigenval_bins, IPR_hist / Z / np.where(eigenval_hist != 0, eigenval_hist, 1)

def binder_compute_observables(system_size, mu, beta, U, J_matrix, t, max_BF, A, powers, state):
        state = np.array(state)
        Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t)
        lnP = - beta * (Ff + Fc) - max_BF + 650
        if(lnP > 700): print(f'- beta * (Ff + Fc) too large! : {- beta * (Ff + Fc)}')
        
        Boltz_factor = np.exp(lnP)
        Mf_moments = Boltz_factor * (np.sum(2*(state - 1/2) * A)/system_size)**powers
        return np.concatenate([[Boltz_factor,], Mf_moments])

def binder_parrallel(system_size, mu, beta, U, J, t = 1, alpha = 1.5, pool = None, normalise = True, n_cumulants = 5):
    J_matrix = interaction_matrix(system_size, alpha, J, normalise = normalise, dtype = np.float64)
    A = 2*(np.arange(system_size) % 2) - 1
    powers = np.arange(n_cumulants)
    
    ##check the ground state energy
    state = (np.arange(system_size) % 2)
    Ff, Fc, evals, evecs = solve_H(state, mu, beta, U, J_matrix, t)
    max_BF = - beta * (Ff + Fc)
    #print(f'max_BF = {max_BF}')
    
    compute = functools.partial(binder_compute_observables, system_size, mu, beta, U, J_matrix, t, max_BF, A, powers)
    
    if pool != None:
            packed_result = np.sum(pool.map(compute, product((0,1), repeat = system_size), chunksize = 4), axis = 0)
    else:
        packed_result = np.sum([compute(state) for state in product((0,1), repeat = system_size)], axis = 0)

    Z, *Mf_moments = packed_result
    Mf_moments = Mf_moments / Z
    Mf_central_moments = convert_to_central_moments(Mf_moments)
    return Mf_moments, Mf_central_moments