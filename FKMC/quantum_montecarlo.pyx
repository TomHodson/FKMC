#cython: boundscheck=False, wraparound=False, infer_types=True, initializedcheck=False, cdivision=True

cimport cython
from libc.math cimport exp, log, floor
import numpy as np
cimport numpy as np
import scipy as sp
from scipy import linalg

import math

from .shared_mcmc_routines cimport initialise_state_representations, c_classical_energy, invert_site_inplace, incremental_energy_difference, spin_spin_correlation
from .stats import moments, binned_error_estimate, moment_errors
from .shared_mcmc_routines import interaction_matrix
from .wrapped_C_functions cimport diagonalise_scipy

cpdef void quantum_cython_mcmc_helper(
                    #outputs
                    double [::1] classical_energies,
                    double [::1] quantum_energies,
                    double [::1] numbers,
                    double [::1] magnetisations,
                    double [:, ::1] states,
                    double [: ,::1] ts,
                    double [:, ::1] eigenvalue_hist,
                    double [:, :, ::1] eigenvector_hist,
                    double [::1] correlator,
                    double [::1] q_energy_histogram,
                    double [::1] IPR_histogram,

                    #inputs
                    double [::1] state,
                    double [::1] alternating_signs,
                    double [::1] ut,
                    double [::1] t,
                    double [::1] eigenvalues,
                    double [:, ::1] eigenvectors,
                    double [::1] new_eigenvalues,
                    double [:, ::1] new_eigenvectors,
                    double [::1] background,
                    double [:, ::1] interaction_matrix,
                    double [::1] diags,
                    double [::1] offdiags,
                    double [:, ::1] random_numbers,

                    #parameters
                    long N_steps,
                    long N_burn_in,
                    long N_system,
                    double mu,
                    double beta,
                    double V,
                    double alpha,
                    double U,
                    double energy_min,
                    double energy_max, #the bounds of the energy histogram
                   ) nogil:
    
            
    cdef double classical_energy, quantum_energy, number, magnetisation, new_quantum_energy 
    
    with gil:
        #diagonalise H and put the answers into eigenvalues and eigenvectors
        if not eigenvalues is None:
            update_matrix(U, state, diags)
            diagonalise_scipy(diags, offdiags, eigenvalues, eigenvectors)
            quantum_energy = average_quantum_energy(beta, eigenvalues)
        
        classical_energy = c_classical_energy(mu, state, t, background)
        number = np.sum(state)
        magnetisation = np.sum(t)

    #variables to track changes in the above until the move is either accepted or rejected
    cdef double quantum_dF, classical_dF, dn, dt

    cdef long site #the site we're considering flipping
    cdef int i;

    for i in range(N_steps + N_burn_in):
        for site in range(N_system):
            #flip the site
            invert_site_inplace(site, alternating_signs, state, ut, t, background, interaction_matrix)


            #Do quantum specific calculations
            if not eigenvalues is None:
                with gil:
                    update_matrix(U, state, diags)
                    diagonalise_scipy(diags, offdiags, new_eigenvalues, new_eigenvectors)
            
                #calculate all the changes, quantum_dF is the only one that can't be done incrementally
                new_quantum_energy = average_quantum_energy(beta, new_eigenvalues)    
                quantum_dF = new_quantum_energy - quantum_energy
            else:
                quantum_df = 0
                
            classical_dF = incremental_energy_difference(site, mu, ut, t, background)
            dn = ut[site]
            dt = 2 * t[site]

            dF = classical_dF + quantum_dF

            #if we must reject this move
            if dF > 0 and exp(- beta * dF) < random_numbers[i, site]:
                #NB: It's super important that this rejected move resets the variables like the eigenvalues and vectors
                #otherwise the results will contain values from rejected moves!!
                #change the site back and don't change the variables
                invert_site_inplace(site, alternating_signs, state, ut, t, background, interaction_matrix)
                
            else:
                #keep the site as it is and update the variables
                classical_energy += classical_dF
                
                quantum_energy = new_quantum_energy #different because this calculation isn't incremental
                eigenvalues[:] = new_eigenvalues[:]
                eigenvectors[:, :] = new_eigenvectors[:, :]
                
                number += dn
                magnetisation += dt

        #choose to store the arrays in a normalised manner
        if i >= N_burn_in:
            j = i - N_burn_in
            classical_energies[j] = classical_energy / N_system
            if not quantum_energies is None: quantum_energies[j] = quantum_energy / N_system
            numbers[j] = number / N_system
            magnetisations[j] = magnetisation / N_system

            if not q_energy_histogram is None: update_bins(eigenvalues, eigenvectors, q_energy_histogram, IPR_histogram, energy_min, energy_max)
            if not states is None: states[j] = state
            if not ts is None: ts[j] = t
            if not eigenvalue_hist is None: eigenvalue_hist[j] = eigenvalues
            if not eigenvector_hist is None: eigenvector_hist[j] = eigenvectors
            if not correlator is None: spin_spin_correlation(correlator, t)

def quantum_cython_mcmc(
                N_steps = 10**4,
                N_burn_in = 10*2,
                N_system = 1000,
                mu = 0,
                beta = 0.1,
                V=1,
                alpha=1.5,
                U=1,
                        
                sample_output = False, #whether to actually do the computation or just output data with the right shape.
                job_id = 0, #job_id, also used to set the seed
                
                bins= 20, #how many bins to use for subsampled error estimates
                N_moments = 10, #how many moments of the data to calculate
                        
                N_energy_bins= 500, #How many bins to use for energy histograms
                energy_min = -5,
                energy_max = 5, #the bounds of the energy histogram
                        
                output_state = False,
                output_correlator = False,
                output_history = False,
                quantum = True,
                        
                **kwargs,
               ):

    #initialise the random number generator
    np.random.seed(job_id)

    state = np.arange(N_system, dtype = np.float64) % 2
    M = interaction_matrix(N=N_system, alpha=alpha, V=V)
    alternating_signs, ut, t, background = initialise_state_representations(state, interaction_matrix=M)
    
    inputs = dict(
        state=state,
        alternating_signs=alternating_signs,
        ut=ut,
        t=t,
        eigenvalues = np.zeros(N_system, dtype = np.double) if quantum else None,
        eigenvectors = np.zeros((N_system, N_system), dtype = np.double) if quantum else None,
        new_eigenvalues = np.zeros(N_system, dtype = np.double) if quantum else None,
        new_eigenvectors = np.zeros((N_system, N_system), dtype = np.double) if quantum else None,
        background=background,
        interaction_matrix=M,
        diags = np.ones(N_system, dtype = np.double),
        offdiags = -1.0 * np.ones(N_system, dtype = np.double),
        random_numbers=np.random.random_sample((N_steps+N_burn_in, N_system)),
        )

    #arrays to hold the running observables, must be amenable to moment calculations
    observables = dict(
        classical_energies = np.zeros(shape = N_steps, dtype = np.float64),
        numbers = np.zeros(shape = N_steps, dtype = np.float64),
        magnetisations = np.zeros(shape = N_steps, dtype = np.float64),

        quantum_energies = np.zeros(shape = N_steps, dtype = np.float64) if quantum else None,
    )
    
    
    #observables that aren't stored per step but are averaged on the fly
    averages = dict(
        correlator = np.zeros(shape = math.floor(N_system/2)+1, dtype = np.float64) if output_correlator else None,
        q_energy_histogram = np.zeros(shape = N_energy_bins, dtype = np.float64) if quantum else None,
        IPR_histogram = -1*np.ones(shape = N_energy_bins, dtype = np.float64) if quantum else None,
    )
    
    #arrays to hold much larger objects that probably won't be output
    state_arrays = dict(
        states = np.zeros((N_steps,N_system), dtype = np.float64) if output_state else None,
        ts = np.zeros((N_steps,N_system), dtype = np.float64) if output_state else None,
        eigenvalue_hist = np.zeros((N_steps,N_system), dtype = np.float64) if output_state and quantum else None,
        eigenvector_hist = np.zeros((N_steps,N_system,N_system), dtype = np.float64) if output_state and quantum else None,
    )
    
    #static_memory_usage = sum(a.nbytes for a in [])

    #if sample_output is true, we just want to get the shape of the data
    if not sample_output:
        quantum_cython_mcmc_helper(
                    #outputs
                    **observables,
                    **state_arrays,
                    **averages,

                    #inputs
                    **inputs,

                    #parameters
                    N_steps=N_steps,
                    N_burn_in = N_burn_in,
                    N_system=N_system,

                    mu=mu,
                    beta=beta,
                    V=V,
                    alpha=alpha,
                    U=U,
                    energy_min=energy_min,
                    energy_max=energy_max, #the bounds of the energy histogram
                   )  
    
    return_vals = dict()
    return_vals['flag_6'] = np.array([1,2])
    
    #calculate the first N_moments moments <x^n> of the observables
    #also estimate the errors using a bining scheme
    moments_dict = {name+'_moments':moments(val, N_moments) for name,val in observables.items() if not val is None}
    moment_errs_dict = {name+'_moments_err':moment_errors(val, N_moments, bins) for name,val in observables.items() if not val is None}
    return_vals.update(moments_dict)
    return_vals.update(moment_errs_dict)

    normed_averages = {name: val for name,val in averages.items() if not val is None}
    return_vals.update(normed_averages)


    if quantum: return_vals.update(dict(
        q_energy_histogram_bins = np.linspace(energy_min, energy_max, N_energy_bins + 1),
        q_energy_bounds = np.array([energy_min, energy_max]),
    ))
        
    if output_history: return_vals.update({name:val for name,val in observables.items() if not val is None})
    if output_state: return_vals.update({name:val for name,val in state_arrays.items() if not val is None})
    
    return return_vals

cdef double average_quantum_energy(double beta, double[::1] eigenvalues) nogil:
    cdef long N = eigenvalues.shape[0]
    cdef double energy = 0
    cdef int i
    for i in range(N):
        energy += log(1 + exp(-beta * eigenvalues[i]))

    return 1/beta * energy

cpdef void update_bins(double [::1] energies, double[:, ::1] eigenvectors, double [::1] energy_histogram, double [::1] IPR_histogram, double binmin, double binmax) nogil:
    '''
    bincounts is a histogram representing values from binmin to binmax, this functions adds the numbers in inputs into it.
    the number is mapped to the bin_index with floor(bincounts.shape[0] * (values[i] - binmin) / (binmax - binmin))
    The first bin includes also includes all values below binmin and the the last all values above binmax
    '''
    cdef int i, N, Nbins, bin_index
    cdef double I2, I4 = 0
    N = energies.shape[0]
    Nbins = energy_histogram.shape[0]
    cdef double scale = Nbins / (binmax - binmin)

    for i in range(N):
        bin_index = int(floor(scale * (energies[i] - binmin)))
        if bin_index > (Nbins-1): bin_index = Nbins - 1
        elif bin_index < 0: bin_index = 0
        energy_histogram[bin_index] += 1
        
        I2 = 0; I4 = 0
        for j in range(N):
            I2 += eigenvectors[j, i]*eigenvectors[j, i]
            I4 += eigenvectors[j, i]*eigenvectors[j, i]*eigenvectors[j, i]*eigenvectors[j, i]
        
        IPR_histogram[bin_index] += I4 / (I2*I2)
    
cdef void update_matrix(double U, double[::1] state, double[::1] diags) nogil:
    cdef int N
    N = state.shape[0]
    for i in range(N):
        diags[i] = U * (state[i] - 1/2)
        
    