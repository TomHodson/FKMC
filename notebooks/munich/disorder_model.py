import numpy as np
from numpy.fft import rfft, irfft, rfftfreq, hfft, ihfft

def correlated_noise(correlation_function, replications, scale = 1, loc = 0):
    'Generate noise whose two point correlator looks like correlation_function (which is real symmetric and only the positive half is supplied)'
    N = len(correlation_function)*2 - 1
    #U = np.random.rand(*(replications, N))
    U = np.random.normal(size = (replications, N), scale = scale, loc = 0)
    #U = np.random.choice([-1,1], size = (replications, N))
    Uq = rfft(U) #output is hermitian because U is real
    
    #use hfft which assumes the input is hermitian (ie symmetric in this case)
    #it returns a symmetric answer, ordered in the fft way, and then take on the positive frequency part of that
    Sq = hfft(correlation_function)[:Uq.shape[-1]]
    
    Nq = np.sqrt(Sq) * Uq
    
    correlated_noise = irfft(Nq)
    return correlated_noise, U, Uq, Sq, Nq