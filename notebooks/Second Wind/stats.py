import numpy as np
import math

def binned_error_estimate(O, M):
    '''Take an array of measurements O, split it into M bins
        and return an estimate for the std dev of <O> from the 
        errors of those M bins. M must evenly divide the length of O'''
    return np.sqrt(np.mean(np.split(O, M), axis = -1).var(ddof = 1) / M)

def moments(O, order):
    '''Take an array of measurements O, and calculate the moments
    <O**1>,<O**2>,<O**3>...<O**(order-1)>'''
    return np.array([(O**i).mean() for i in range(1, order)])

def moment_errors(O, order, M):
    '''Take an array of measurements O, and calculate the error in the moments
    <O**1>,<O**2>,<O**3>...<O**(order-1)> using binned_error_estimate'''
    return np.array([binned_error_estimate(O**i, M) for i in range(1, order)])

def autocorrelation(O):
    '''Take an array of measurements O, calculate the lag d autocorrelations which are the averages over i of
    <O'_i O'_i+d> where O' = (O - O.mean())/O.std()
    '''
    Ob = (O - O.mean()) / O.std()
    return np.correlate(Ob, Ob, mode = 'full') / len(O)

#Note, I've also implemented this is shared_mcmc_functions in cython, with slightly different call signature.
def spin_spin_correlation(state):
    '''
    take a state, where the last axis labels spins and compute the correlation function
    for half the system size because it's periodic
    the last axis becomes the correlation function instead
    S(k) = <s_i s_{i+k}> averaged over i
    '''

    #determine the new shape of the data
    output_shape = np.array(state.shape)
    output_shape[-1] = math.floor(state.shape[-1]/2) + 1
    
    output = np.zeros(shape = output_shape, dtype = state.dtype)
    for shift in range(output_shape[-1]):
        output[..., shift] = np.mean(np.roll(state, shift, axis=-1) * state, axis=-1)
    
    output = output - (state[..., np.newaxis].mean(axis=-2)**2)
    
    return output

