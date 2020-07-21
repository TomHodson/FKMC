import numpy as np
import math
import scipy.stats

from functools import reduce
def product(iterable): return reduce(mul, iterable, 1)

def binned_error_estimate(O, M):
    '''Take an array of measurements O, split it into M bins
        and return an estimate for the std dev of <O> from the 
        errors of those M bins. M must evenly divide the length of O'''
    return np.sqrt(np.mean(np.split(O, M), axis = -1).var(ddof = 1) / M)

from scipy.stats import sem
def binned_error_estimate_multidim(O, N_bins, axis = -1, debug = False):
    '''Take an array of measurements O shape [..., M], split it 
        into N_bins bins with shape [..., M/N_bins]
        return an estimate for the std dev of <O> from the 
        errors of those bins. N_bins must evenly divide M'''
    if N_bins == 1: #fall back to the simpler case
        return sem(O, axis=axis, ddof=1, nan_policy='omit')
    
    #split a [..., N, ...] array into a list of B arrays with shape [..., N/B, ...]
    if debug: print('O.shape' , O.shape)
    splits = np.split(O, N_bins, axis = axis)
    if debug: print('splits[0].shape' ,splits[0].shape)

    #take the mean over the N/B axis for each of those givingshape [B, ..., 1, ...]
    means = np.array([np.nanmean(b, axis = axis) for b in splits])
    if debug: print('means.shape', means.shape)

    #take the variance over the bins
    var = means.var(ddof = 1, axis = 0)
    if debug: print('var.shape', var.shape)
    
    #estimate the standard error in the means with shape [..., ...]
    return np.sqrt(var / N_bins)

def moments(O, order):
    '''Take an array of measurements O, and calculate the moments
    <O**1>,<O**2>,<O**3>...<O**(order-1)>'''
    return np.array([(O**i).mean() for i in range(1, order)])

def moment_errors(O, order, M):
    '''Take an array of measurements O, and calculate the error in the moments
    <O**1>,<O**2>,<O**3>...<O**(order-1)> using binned_error_estimate'''
    return np.array([binned_error_estimate(O**i, M) for i in range(1, order)])

#def autocorrelation(O):
 #   '''Take an array of measurements O, calculate the lag d autocorrelations which are the averages over i of
 #   <O'_i O'_i+d> where O' = (O - O.mean())/O.std()
 #   '''
 #   Ob = (O - O.mean()) / O.std()
 #   return np.correlate(Ob, Ob, mode = 'full') / len(O)

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


###Correlation related functions
def autocorrelation(X):
    N = X.shape[0]
    lagged = scipy.signal.correlate(X, X, mode = 'full') / (N - np.abs(np.arange(1,2*N)-N))
    full = (lagged - np.mean(X)**2 ) / (np.mean(X**2) - np.mean(X)**2)
    return full[N:]

def bin_std(X, binsize):
    '''
    Estimate the standard error in the mean of X by looking at the distribution of the means of bins of size binsize
    Operates on axis = -1
    '''
    binnumber = X.shape[-1] // binsize
    N = binnumber * binsize
    newshape = X.shape[:-1] + (binnumber, binsize)
    binmeans = X[..., :N].reshape(newshape).mean(axis = -1)
    bin_std = scipy.stats.sem(binmeans, axis = -1)
    return bin_std

def bin_estimate_tau(ys, binsize = 10, max_its = 10, binsize_multiplier = 5):
    sigma_naive = ys.std(axis = -1)
    for i in range(max_its):
        taus = ys.shape[-1] / 2 * (bin_std(ys, binsize) / sigma_naive)**2
        binsize = int(binsize_multiplier*np.nanmax(taus))
    return taus


from numpy.fft import rfft, irfft, rfftfreq, hfft, ihfft

def correlated_noise(correlation_function, replications = 100, loc = 0, scale = 1):
    'Generate noise whose two point correlator looks like correlation_function (which is real symmetric and only the positive half is supplied)'
    N = len(correlation_function)*2 - 1
    U = np.random.normal(size = (replications, N), scale = scale, loc = loc)
    Uq = rfft(U) #output is hermitian because U is real
    
    #use hfft which assumes the input is hermitian (ie symmetric in this case)
    #it returns a symmetric answer, ordered in the fft way, and then take on the positive frequency part of that
    Sq = hfft(correlation_function)[:Uq.shape[-1]]
    
    Nq = np.sqrt(Sq) * Uq
    
    correlated_noise = irfft(Nq)
    return correlated_noise, U, Uq, Sq, Nq