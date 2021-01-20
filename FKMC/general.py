from scipy.linalg import eigh_tridiagonal, LinAlgError, circulant
import scipy
import numpy as np
import scipy.signal
from itertools import count
from scipy.linalg.lapack import dstev
from time import time

from FKMC.stats import binned_error_estimate_multidim

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
        evals, evecs, info =  dstev(d = U*(state - 1/2) - muc, e =-t*np.ones(state.shape[0] - 1), compute_v = True)
        Ff = - U/2*np.sum(state - 1/2) - muf*np.sum(state) + (state - 1/2).T @ J_matrix @ (state - 1/2)
        Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * evals)))
    
    return Ff, Fc, evals, evecs

def solve_H_vectorised(states, mu, beta, U, J_matrix, t, **kwargs):
    assert(((states == 0) | (states == 1)).all()) #I alsways forget that states should be 0s and 1s
    R, N = states.shape# = (disorder realisations, system size)
    
    muf = muc = mu
    evals = np.empty(shape = states.shape, dtype = np.float64)
    evecs = np.empty(shape = states.shape + (states.shape[-1],), dtype = np.float64)
    e = -t*np.ones(states.shape[-1] - 1)
    ds = U*(states - 1/2) - muc
    
    for i in range(R):
        evals[i], evecs[i] = eigh_tridiagonal(d = ds[i], e = e, lapack_driver = 'stev')
    
    #the below is a vectorised version of:
    #Ff = - U/2*np.sum(state - 1/2) - muf*np.sum(state) + (state - 1/2).T @ J_matrix @ (state - 1/2)
    #Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * evals)))
    
    S = np.sum(states, axis = -1)
    t = (states - 1/2)
    #shapes(t, J, t) = [(R, N), (N, N), (R, N)] giving contraction ij,ik,ik -> i
    inner_prod = np.einsum('ij,jk,ik -> i', t, J_matrix, t, optimize = 'greedy')
    
    Ff = - U/2*(S - N/2) - muf*S + inner_prod
    Fc = - 1/beta * np.sum(np.log(1 + np.exp(- beta * evals)), axis = -1)
    
    #shapes(Ff, Fc, evals, evecs) = [(R,), (R,), (R, N), (R, N, N)]
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

## stuff to do with binning energies and IPRs
##Generally the operations that need doing:
#
#0) Calculate the IPRs from the eigenvectors, this looks like
#   IPRs = ((evecs * np.conj(evecs))**2).sum(axis = 0)
#   making sure to sume over the correct axis!
#
# 1) transform from lists of energies and IPRs to histograms binned by energy
#    this is achieved using index histogram and sort_IPRs
#       E_bins = np.linspace(-6, 6, 500 + 1)
#       E_hist, _, indices = index_histogram_array(E_bins, E_vals)
#       IPR_hist = sort_IPRs(indices, IPRs, E_bins)
#
# 2) divide IPR by DOS using DOS, IPR = normalise_IPR(DOS, IPR)
#
# 3) smooth out the spikieness of the resulting histogram using
#    DOS = smooth(DOS, scale = 1, axis = -1)
#    IPR = smooth(IPR, scale = 1, axis = -1)

# Step 0 happens in the montecarlo routine itself
# There is currently a slight mismatch in that get_data_structured does step 1 while 
# get_data_func does steps 1 and 2. Beware of this!!!!
# Examples usage below:
"""
from FKMC.general import interaction_matrix, solve_H_vectorised, shapes

M = int(100) # average over this many replications
N = 256 # system size
states = np.random.choice([1,0], size = (M,N)) #take 50-50 uncorrelated coin flips to simulate infinite temp limit.

#alternating = 2*(np.arange(N) % 2) - 1
#alternating_power_law_noise = power_law_noise * alternating

params = Munch(mu=0.0, beta = np.nan, U = 5.0, t = 1.0, N = N, alpha = 1.25, J = 1.0)
params.J_matrix = interaction_matrix(**params)

#NB the states argument should have only 1s and 0s
#Fc is infinite at infinite temperature
Ff, Fc, eigenvals, eigenvecs = solve_H_vectorised(states, **params)

from FKMC.general import index_histogram_array, sort_IPRs, normalise_IPR, smooth

#step 0) calculate the IPR measure
IPR_raw_values = ((eigenvecs * np.conj(eigenvecs))**2).sum(axis = -2)

energy_histogram_bins = np.linspace(-6, 6, 500 + 1)

#step 1) take lists of IPR and eigenvalues and bin them into histograms
DOS, _, sorting_indices = index_histogram_array(energy_histogram_bins, eigenvals)
IPR = sort_IPRs(sorting_indices, IPR_raw_values, energy_histogram_bins)

# step 1.5) average over disorder realisations and take binned errors
DOS, dDOS = DOS.mean(axis = 0), binned_error_estimate_multidim(DOS, N_bins = 100, axis = 0)
IPR, dIPR = IPR.mean(axis = 0), binned_error_estimate_multidim(IPR, N_bins = 100, axis = 0)

# step 2) divide IPR by DOS without incurring divide by errors where DOS = 0, set IPR = 0 there too
DOS, IPR = normalise_IPR(DOS, IPR)

#smooth out the spikey IPR values
DOS = smooth(DOS, scale = 64 / N, axis = -1)
IPR = smooth(IPR, scale = 64 / N, axis = -1)
"""

def normalise_IPR(DOS_raw, IPR_raw):
    'take raw histogram values of DOS and IPR, do IPR = IPR/DOS'
    #prepare a version of E_hist with no zeros, and a version of IPR_hist with no infinities
    DOS_nonzero = np.where(DOS_raw > 0, DOS_raw, 1)
    
    DOS = DOS_raw #doesn't actually need an processing of this form
    IPR = np.where(DOS_raw > 0, IPR_raw/DOS_nonzero, 0)
    
    return DOS, IPR

def smooth(s, scale = 1, axis = -1):
    if scale == 0: return s
    'Applies a 1D cauchy smoothing kernal with adjustable scale onto the last axis by default'
    x = np.linspace(-100, 100, s.shape[-1])
    kernal = scipy.stats.cauchy.pdf(x, scale = scale)
    kernal /= np.sum(kernal)
    #return scipy.signal.convolve(s, kernal, mode = 'same') #the non vectorised version
    return np.apply_along_axis(scipy.signal.convolve,
                              arr = s,
                              in2 = kernal,
                              axis = axis,
                              mode = 'same',
                              )

def index_histogram(bin_edges, data):
    "perform a similar function to np.histogram except also return the indices that sort the data into bins"
    indices = np.searchsorted(bin_edges, data)
    hist = np.bincount(indices, minlength = bin_edges.shape[0] + 1)
    hist[1] += hist[0]
    hist[-2] += hist[-1]
    hist = hist[1:-1]
    return hist, bin_edges, indices

## Vectorised versions of the above
def index_histogram_array(bin_edges, data):
    """
    A vectorised version of the above that takes data with shape (..., N) and gives out bin indices with shape (..., len(bin_edges)-1)
    
    usage:
    
    E_bins = np.linspace(-6, 6, 500 + 1)
    E_hist, _, indices = index_histogram_array(E_bins, E_vals)
    IPR_hist = sort_IPRs(indices, IPRs, E_bins)

                   bins:   0 1 2 3     len(a) - 1
                          _|_|_|_|    _|_ 
    searchsorted values:  0 1 2 3 ...   len(a)
    
    With the default side = 'left' this returns i such that a[i-1] < v <= a[i]
    where a are the bin_edges and v the data
    hence 0 means the data is below the minimum bin and len(a) means the data is above the maximum bin
    
    the maximum value returned by searchsorted is len(bin_edges)
    bincount returns an array with an entry for the number of 
    occurances of all the integers from 0 to len(bin_edges)
    hence the length of hist is a max len(bin_edges) + 1
    """
    indices = np.searchsorted(bin_edges, data, side = 'left')
    hist = np.apply_along_axis(np.bincount, axis=-1, arr=indices, minlength = bin_edges.shape[0] + 1)
    hist[..., 1] += hist[..., 0]
    hist[..., -2] += hist[..., -1]
    hist = hist[..., 1:-1]
    return hist, bin_edges, indices

def sort_IPRs(indices, IPRs, E_bins):
    """See above for usage"""
    
    o_shape = IPRs.shape
    IPRs = IPRs.reshape(-1, o_shape[-1])
    indices = indices.reshape(-1, o_shape[-1])
    IPR_hist = np.full(fill_value = np.nan, shape = (IPRs.shape[0], E_bins.shape[0] - 1))
    
    for i, IPR, index in zip(count(), IPRs, indices):
        res = np.bincount(index, weights=IPR, minlength = E_bins.shape[0] + 1)[1:-1]
        res[1] += res[0]
        res[-2] += res[-1]
        IPR_hist[i] = res
    IPR_hist = IPR_hist.reshape(o_shape[:-1] + (E_bins.shape[0]-1,))

    return IPR_hist



def compute_IPR_and_DOS_histograms(raw_eigenvals, raw_IPRs, E_bins, bootstrap_bins = 1):
    '''
    Start with observations of states with energy raw_eigenvals[i,j] and inverse participation ratio raw_IPRs[i,j]
    They have shape M,N where M is the number of observed systems and N is the system size (hence number of states observed)
    i labels system conigurations and j labels individual states
    E_bins are the bins to sort into
    bootstrap_bins is how many bins to use to estimate the error when autocorrelation is present.

    #In the MCMC routine itself
    0) Calculate the IPRs from the eigenvectors, this looks like
       IPRs = ((evecs * np.conj(evecs))**2).sum(axis = 0)
       making sure to sum over the correct axis!

    #In this function
    1) transform from lists of energies and IPR observations to sum of:
        sum_DOS_ik: the number of observations of energy in each bin k at each sys config i
        sum_IPR_ik: the sum of IPRs for states in energy bin k at each sys config i
        
    2) estimate errors:
        DOS by simply binning
        IPR by estimating IPR = sum_IPR / sum_DOS at each sys config and binning
        
    3) take means of sum_DOS and sum_IPR

    4) calculate 
        IPR = <sum_IPR> / <sum_DOS>
        DOS = sum_DOS / energy_bin_width / system_size
    
    5) smooth out the spikieness of the resulting histogram using smooth
    
    Crucially the mean is taken before the ratio, doing it the other way doesn't seem to work.
    '''
    #take lists of IPR and eigenvalues and bin them into histograms
    sum_DOS, _, sorting_indices = index_histogram_array(E_bins, raw_eigenvals)
    sum_IPR = sort_IPRs(sorting_indices, raw_IPRs, E_bins)

    #the below method splits into bins to deal with autocorrelation, set N_bins = 1 to ignore autocorellation
    sum_dDOS = binned_error_estimate_multidim(sum_DOS, N_bins = bootstrap_bins, axis = 0)
    
    #NB you can't caculate the error in sum_IPR because that is highly correlated with how many states appear in a bin!!!
    _, non_meaned_IPR_ratios = normalise_IPR(sum_DOS, sum_IPR)
    dIPR = binned_error_estimate_multidim(non_meaned_IPR_ratios, N_bins = bootstrap_bins, axis = 0)
    
    #now take the means and then take the ratio again!
    sum_DOS = sum_DOS.mean(axis = 0)
    sum_IPR = sum_IPR.mean(axis = 0)
    
    #divide <IPR> by <DOS> without incurring divide by errors where DOS = 0, set IPR = 0 there too
    _, IPR = normalise_IPR(sum_DOS, sum_IPR)
    
    M, N = raw_eigenvals.shape
    bin_width = E_bins[1] - E_bins[0]
    a = bin_width * N
    DOS, dDOS = sum_DOS / a, sum_dDOS / a
    
    #shapes(DOS=DOS, dDOS=dDOS, IPR=IPR, dIPR=dIPR)
    return DOS, dDOS, IPR, dIPR


def running_mean(quantity, axis = -1):
    'return an array where the ith element is the mean of the first i values of the given array'
    return np.cumsum(quantity, axis = axis) / np.arange(1,quantity.shape[axis]+1)

def running_sem(quantity, axis = -1):
    'return an array where the ith element is the standard error of the mean of the first i values of the      given array'
    std = np.sqrt(running_mean(quantity ** 2) - running_mean(quantity) ** 2)
    sem = std / np.sqrt(np.arange(1,quantity.shape[axis]+1))
    return sem

def shapes(*args, **kwargs):
    'print out the shapes of multiple dats structures'
    out = []
    for a in args:
        try: 
            out.append(str(a.shape))
        except AttributeError: 
            out.append(str(type(a)))
    for name, value in kwargs.items():
        try: 
            out.append(f'{name}.shape = {value.shape}')
        except AttributeError: 
            out.append(f'type({name}) = {type(value)}')
    print(', '.join(out))
    
def find_zero_crossings(f):
    'given a 1d array, finds the first index where f has a sign change or is 0'
    s = np.sign(f)
    z = np.where(s == 0)[0]
    if z.size > 0:
        return z
    else:
        s = s[0:-1] + s[1:]
        z = np.where(s == 0)[0]
        return z

def interp_x_position(f,x,z):
    'given two arrays representing a function f evaluated at x and the index of a zero crossing z, interpolate to the expected value of x'
    m = (f[z+1] - f[z]) / (x[z+1] - x[z])
    return x[z] - f[z]/m

def diag2column(arr):
    'take a square matrix and shift the rows so that the diagonal becomes the first column'
    N, M = arr.shape
    assert(N==M)
    out = np.empty_like(arr)
    for i in range(N):
        out[i] = np.roll(arr[i], -i)
    return out

def spread(ax, X, Y, dY, alpha = 0.3, **kwargs): 
    print('Warning! spread is now defined in FKMC.plotting not FKMC.general!')
    from FKMC.plotting import spread as s
    return s(*args, **kwargs)

def get_nearby_index(sorted_list, value):
    'take sorted_list, value, return the closest value in the list and its index'
    i = np.searchsorted(sorted_list, value)
    return sorted_list[i], i


def scaling_dimension(Ns, IPR, dIPR, use_true_errors = True):
    Y = np.log(IPR)
    dY = np.max(dIPR / IPR, axis = -1) #take the maximum error across the energy spectrum because we can't do it individually
    X = np.log(Ns)

    if use_true_errors:
        (m, c), cov = np.ma.polyfit(X, Y, deg = 1, cov=True, w = 1 / dY)
    else:
        (m, c), cov = np.ma.polyfit(X, Y, deg = 1, cov=True)
    dm, dc = np.sqrt(np.einsum('iik -> ik', cov))
    
    return m, c, dm, dc


def tridiagonal_diagonalisation_benchmark(M = 100, N = 250):
    '''
    diagonalise a system of size 250, 100 times and report how long it took.
    gives 1.6s on cx1
    '''
    t = time()
    states = np.random.choice([0,1], size = [M, N])
    e = -np.ones(N - 1)
    ds = 5*(states - 1/2)
    
    evals = np.zeros(shape = [M,N])
    evecs = np.zeros(shape = [M,N,N])
    for i in range(M):
        evals[i], evecs[i] = eigh_tridiagonal(d = ds[i], e = e, lapack_driver = 'stev')
        
    return time() - t

def interpolate_IPR(E_bins, unsmoothed_DOS, IPR, dIPR):
    newshape = (IPR.size // IPR.shape[-1], IPR.shape[-1])
    _DOS = unsmoothed_DOS.reshape(newshape)
    _IPR = IPR.reshape(newshape)
    _dIPR = dIPR.reshape(newshape)
    
    for i, DOS, I, dI in zip(count(), _DOS, _IPR, _dIPR):
        ei = DOS > 0
        if any(ei):
            _I = I[ei]
            _dI = dI[ei]
            xI = E_bins[1:][ei]

            _IPR[i] = np.interp(E_bins[1:], xI, _I)
            _dIPR[i] = np.interp(E_bins[1:], xI, _dI)
        else:
            _IPR[i] = E_bins[1:] * np.NaN
            _dIPR[i] = E_bins[1:] * np.NaN

"""
Retired code left here just in case I find a reference to it and have forgotten what it did.

#I haven't used this but I think it could be a simpler definition of a rotationally invariant interaction
def interaction_matrix_2(N, alpha, J, normalise = True, dtype = np.float64, **kwargs):
    M = 100
    i = np.arange(1,N,dtype = dtype)
    row0 = i**(-alpha) + np.abs(N-i)**(-alpha)
    row0 = np.concatenate([[0,], row0])# put the first 0 in by hand
    row0 = row0 / np.sum(np.abs(row0))
    row0 = J * row0
    return row0
    
    
def compute_IPR_and_DOS_histograms_ratio_then_mean(raw_eigenvals, raw_IPRs, E_bins, bootstrap_bins = 1):
    #create sums of observations, for DOS just count the number of states seen in each bin 
    M, N = raw_eigenvals.shape
    sum_DOS, _, indices = index_histogram_array(E_bins, raw_eigenvals)
    sum_IPR = sort_IPRs(indices, raw_IPRs, E_bins)
    
    #normalise the IPR by the number of observations, but set to NaN when there are none.
    _, IPR = normalise_IPR(sum_DOS, sum_IPR)
    
    #normalise by bin_width so that different binning doesn't affect the value
    #normalise by the system size so that plotting different ones together works
    #the outcome is that sum(DOS * bin_width) == 1
    bin_width = E_bins[1] - E_bins[0]
    DOS = sum_DOS / (N * bin_width)
    
    #the below method splits into bins to deal with autocorrelation, set N_bins = 1 to ignore autocorellation
    dDOS = binned_error_estimate_multidim(DOS, N_bins = bootstrap_bins, axis = 0)
    dIPR = binned_error_estimate_multidim(IPR, N_bins = bootstrap_bins, axis = 0)
    
    DOS = DOS.mean(axis = 0)
    IPR = IPR.mean(axis = 0)

    return DOS, dDOS, IPR, dIPR

def compute_IPR_and_DOS_histograms_1D(raw_eigenvals, raw_IPRs, E_bins, bootstrap_bins = 1):
    raw_eigenvals = raw_eigenvals.flatten()
    raw_IPRs = raw_IPRs.flatten()
    N, = raw_eigenvals.shape
    
    sum_DOS, _, indices = index_histogram_array(E_bins, raw_eigenvals)
    sum_IPR = sort_IPRs(indices, raw_IPRs, E_bins)
    sum_squared_IPR = sort_IPRs(indices, raw_IPRs**2, E_bins)
    
    #normalise the IPR by the number of observations, but set to NaN when there are none.
    _, IPR = normalise_IPR(sum_DOS, sum_IPR)
    _, IPR_squared = normalise_IPR(sum_DOS, sum_squared_IPR)
    
    #assume poissonian errors because it's just counting
    #error of a poisson is sqrt(N)
    #error in the mean of a poisson in 1 / sqrt(N)
    dsum_DOS = 1 / np.sqrt(sum_DOS) 
    #use standard error in the mean = sqrt(<x**2> - <x>**2) / sqrt(N)
    dIPR = np.sqrt(IPR_squared - IPR**2) / np.sqrt(N) 
    
    #normalise by bin_width so that different binning doesn't affect the value
    #normalise by the system size so that plotting different ones together works
    #the outcome is that sum(DOS * bin_width) == 1
    bin_width = E_bins[1] - E_bins[0]
    DOS = sum_DOS / (N * bin_width)
    dDOS = dsum_DOS / (N * bin_width)
    
    return DOS, dDOS, IPR, dIPR


"""