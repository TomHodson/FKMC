import numpy as np
import scipy as scipy
from FKMC.general import interaction_matrix, solve_H_vectorised
from FKMC.general import index_histogram_array, sort_IPRs, normalise_IPR, smooth, get_nearby_index,  compute_IPR_and_DOS_histograms

def solve_systems(states, params, energy_histogram_bins, scale = 1):
    '''
    suggested usage:
    states = np.random.choice([1,0], size = (M,N)) #take 50-50 uncorrelated coin flips to simulate infinite temp limit.
    params = Munch(mu=0.0, beta = np.nan, U = U, t = t, N = N, alpha = 1.25, J = 1.0)
    energy_histogram_bins = np.linspace(-3*U, 3*U, 1000 + 1)
    
    #alternating = 2*(np.arange(N) % 2) - 1
    #alternating_power_law_noise = power_law_noise * alternating

    DOS, IPR, dDOS, dIPR = solve(states, params, energy_histogram_bins, scale = 10 / N)
    '''
    N = states.shape[-1]
    params.J_matrix = interaction_matrix(**params)

    #NB the states argument should have only 1s and 0s
    #Fc is infinite at infinite temperature
    Ff, Fc, eigenvals, eigenvecs = solve_H_vectorised(states, **params)

    #calculate the IPR measure
    raw_IPRs = ((eigenvecs * np.conj(eigenvecs))**2).sum(axis = -2)

    DOS, dDOS, IPR, dIPR = compute_IPR_and_DOS_histograms(eigenvals, raw_IPRs, energy_histogram_bins, bootstrap_bins = 1)

    #smooth out the spikey IPR values
    DOS = smooth(DOS, scale = scale, axis = -1)
    IPR = smooth(IPR, scale = scale, axis = -1)
    
    dDOS = smooth(dDOS, scale = scale, axis = -1)
    dIPR = smooth(dIPR, scale = scale, axis = -1)
    
    return DOS, IPR, dDOS, dIPR
