from scipy.linalg import eigh_tridiagonal, LinAlgError, circulant
import scipy
import numpy as np
import scipy.signal

def interaction_matrix(N, alpha, J, normalise = True, dtype = np.float64, **kwargs):
    alternating_signs = 2*(np.arange(1,N) % 2) - 1
    row0 = alternating_signs * np.abs((N/np.pi * np.sin(np.pi * np.arange(1,N, dtype = dtype)/N)) ** (-alpha))
    row0 = np.concatenate([[0,], row0])# put the first 0 in by hand
    if normalise and (N > 1): row0 = row0 / np.sum(np.abs(row0))
    row0 = J * row0
    return circulant(row0)

def solve_H(state, mu, beta, U, J_matrix, t, **kwargs):
    state = np.array(state)
    muf = muc = mu
    if state.size == 1:
        evals = np.array([np.sum(U*(state - 1/2) - muc)])
        evecs = np.array([1,])
        Ff = np.sum(- U/2*(state - 1/2) - muf*state)
        Fc = - 1/beta * np.log(1 + np.exp(- beta * evals[0]))
    else:
        evals, evecs = eigh_tridiagonal(d = U*(state - 1/2) - muc, e =-t*np.ones(state.shape[0] - 1), lapack_driver = 'stev')
        Ff = - U/2*np.sum(state - 1/2) - muf*np.sum(state) + (state - 1/2).T @ J_matrix @ (state - 1/2)
        Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * evals)))
    
    return Ff, Fc, evals, evecs

def moments_about_zero(data, n):
    powers = np.arange(0,5)
    N = data.shape[0]
    return (data[None, :]**powers[:, None]).sum(axis = -1) / N
    
    
def moments_about_mean(data, n):
    powers = np.arange(0,5)
    mean = np.average(data)
    N = data.shape[0]
    z = (data[None, :]-mean)
    return (z**powers[:, None]).sum(axis = -1) / N

def convert_to_central_moments(non_central_moments):
    mean = non_central_moments[1]
    N = non_central_moments.shape[0]
    n = np.arange(N)[:, None] * np.ones((N,N)) # n changes on the 0th axis
    i = np.arange(N)[None, :] * np.ones((N,N))# i changes on the 1st axis

    m = np.where((n-i) >= 0, n-i, 1)
    return (scipy.special.binom(n, i) * non_central_moments[None, :] * (-mean)**m).sum(axis = 1)

def index_histogram(bin_edges, data):
    "perform a similar function to np.histogram except also return the indices that sort the data into bins"
    
    indices = np.searchsorted(bin_edges, data)
    hist = np.bincount(indices, minlength = bin_edges.shape[0] + 1)
    return hist[1:-1], bin_edges, indices

def running_mean(quantity):
    'return an array where the ith element is the mean of the first i values of the given array'
    return np.cumsum(quantity, axis = 0) / np.arange(1,quantity.shape[0]+1)

def smooth(s, scale = 1):
    x = np.linspace(-100, 100, s.shape[0])
    kernal = scipy.stats.cauchy.pdf(x, scale = scale)
    kernal /= np.sum(kernal)
    return scipy.signal.convolve(s, kernal, mode = 'same')