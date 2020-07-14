#cython: boundscheck=False, wraparound=False, infer_types=True, initializedcheck=False, cdivision=True

cimport cython
from libc.math cimport exp, log
import numpy as np
cimport numpy as np
import scipy as sp
from scipy import linalg

#TODO change the code to just rotate a single row of this matrix when needed rather than generating it all the time
def interaction_matrix(N, alpha, V, normalise = True, dtype = np.float64):
    row0 = np.abs((N/np.pi * np.sin(np.pi * np.arange(1,N, dtype = dtype)/N)) ** (-alpha) )
    row0 = np.concatenate([[0,], row0])# put the first 0 in by hand
    if normalise: row0 = row0 / np.sum(row0)
    row0 = V * row0
    return linalg.circulant(row0)

cpdef initialise_state_representations(double [::1] state, double [:, ::1] interaction_matrix):
    'initialise useful representations of the state'
    cdef int N = state.shape[0]

    alternating_signs = np.ones(N, dtype = np.double)
    ut = np.ones(N, dtype = np.double)
    t = np.ones(N, dtype = np.double)


    cdef double [::1] alt_v = alternating_signs
    cdef double [::1] ut_v = ut
    cdef double[::1]  t_v = t
    cdef double[::1] st_v = state


    cdef int s = 1
    cdef int i, j
    for i in range(N):
        s = -s
        alt_v[i] = s
        ut_v[i] = (2*st_v[i] - 1)
        t_v[i] = s * ut_v[i]

    cdef double [:] background = (interaction_matrix @ t)
    return alternating_signs, ut, t, background

cpdef double c_classical_energy(
                                    double mu,
                                    double[::1] state,
                                    double[::1] t,
                                    double [::1] background) nogil:
    'compute the energy of the f electrons'
    cdef int N = state.shape[0]
    cdef double F = 0
    cdef int i
    for i in range(N):
        F += - mu * state[i] + t[i] * background[i]

    return F

cpdef void invert_site_inplace(
                        long i,
                        double [::1] alternating_signs,
                        double [::1] state,
                        double [::1] ut,
                        double [::1] t,
                        double [::1] background,
                        double [:, ::1] interaction_matrix,
                       ) nogil:
    'invert site i and update the useful state representations in place'
    cdef long N = state.shape[0]

    state[i] = 1 - state[i]
    ut[i] = -ut[i]
    t[i] = -t[i]

    #these expresions only work if we're already flipped the spin
    cdef double dni = ut[i]
    cdef double dti = 2 * t[i]

    cdef int j
    for j in range(N):
        background[j] += interaction_matrix[i, j] * dti

cpdef double incremental_energy_difference(long i,
                                    double mu,
                                    double[::1] ut,
                                    double[::1] t,
                                    double [::1] background) nogil:
    'compute the energy difference for the site i WHICH HAS BEEN FLIPPED ALREADY'
    cdef double dni = ut[i] #the changes are simply related to the value after flipping too
    cdef double dti = 2*t[i]
    cdef double dF_f =  - mu * dni + 2 * dti * background[i]
    return dF_f

cdef void spin_spin_correlation(double[::1] correlation, double[::1] state) nogil:
    '''
    take a 1D state and compute the 1D correlation function for the whole system size
    S(k) = <s_i s_{i+k}> averaged over i
    NOTE: does not overwrite input 'correlation' just adds the result to it, enabling running averages.
    returns the first len(correlation) corelates.
    '''

    #determine the new shape of the data
    cdef int N = state.shape[0]
    cdef double mean = 0 #will be used to subtract the mean squared <s>**2 at the end
    cdef double corr #this is used to save the intermediate <s_i s_i+k> correlation to high precision before adding it to the running total.
    cdef int k, i 
    for i in range(N):
        mean += state[i]
    mean /= N
    
    for k in range(correlation.shape[0]): 
        corr = 0
        #after this loop correlation[k] will contain N * <s_i s_{i+k}>
        
        #for values of i where (i+k)%N == N
        for i in range(N-k):
            corr += (state[i+k]-mean) * (state[i] - mean)
        
        #at this point i+k wraps and (i+k)%N == (i+k) - N
        for i in range(N-k,N):
            corr += (state[i+k-N]-mean) * (state[i] - mean)
        
        correlation[k] += corr/N