# FKMC/notebooks/Summer 2020/April2020.ipynb
- IPR, DOS, Correlation functions as a function of temp animation
- Moving around the UT phase diagram animation

# FKMC/notebooks/Summer 2020/April2020-2_Infinite_temperature_limit.ipynb
- IPR and DOS at infinite temperature
- 

# FKMC/notebooks/Summer 2020/May2020.ipynb

# FKMC/notebooks/Summer 2020/May2020-2.ipynb
- A nice plot of the IPR scaling at infinite temperature as U changes, shows that at low U you get weak localisation where the localisation length is comparable to system size. 
- attempt at looking at the same thing but for real MCMC data but it didn't look great, probably wasn't converged.
- started only plotting IPR when DOS > a threshold

# FKMC/notebooks/Summer 2020/May2020-3.ipynb
- explored the question of mean of ratios or ratio of means. went with ratio of means
- 
# FKMC/notebooks/Summer 2020/June2020-1.ipynb
- wrote local_run.py for faster results at a single UJT tuple
- confirmed that at U = 10 there are both delocalised states and localisated states below Tc
- started using Ns = np.logspace(np.log10(50), np.log10(270), 10, dtype = np.int) // 10 * 10

# FKMC/notebooks/Summer 2020/June2020-2.ipynb
- made a nice plot of T = 1.5, U =5, J = 5, showing that there are still both delocalised states and localisated states below Tc

# FKMC/notebooks/Summer 2020/June2020-3.ipynb
- disorder modelling, random, perfect CDW and dirty CDW
- plot of energy penalty vs random defect density

# FKMC/notebooks/Summer 2020/July2020-1.ipynb
- reading in the new batchscript output format
- start preparing final figures

# FKMC/notebooks/Summer 2020/August2020-1.ipynb
- an attempt to implement correlated noise generation by generating correlated probabilities, doesn't work so far.


# Figures for the short paper
data_location: ~/HPC_data/pickled_data
figure_location: ~/git/FK_short_paper/figs on my mac which sync with overleaf through git

## Fig 1
Figure name: phase_diagram2.eps

### Top left: M squared plot vs T
    Created in: ?
    Data saved in: binder_data.pickle
    N steps = ?

### Top right: Binder plot
    Created in: ?
    Data saved in: binder_data.pickle
    N steps = ?

### Bottom left: TU phase diagram
    Created in: ?
    Data saved in: TU_phase_data.pickle

    Simulation Parameters
    N_sites : 128, t : 1, alpha : 1.5, mu : 0, beta : 10.0, J : 5, U : 0.0, normalise : True
    MCMC Parameters
    state : None, N_steps : 100000, N_burn_in : 10000, thin : 100, logger : <FKMC.montecarlo.Eigenspectrum_IPR_all object at 0x7ff73d536750>, proposal : <function         p_multi_site_uniform_reflect at 0x7ff7609ccb00>, accept_function : <function perturbation_accept at 0x7ff7609cce60>, warnings : True

### Bottom right: TJ phase diagram
    Created in: ?
    Data saved in: TJ_phase_data.pickle
    Simulation Parameters
    N_sites : 128, t : 1, alpha : 1.25, mu : 0, beta : 10.0, J : 0.0, U : 1, normalise : True
    MCMC Parameters
    state : None, N_steps : 100000, N_burn_in : 10000, thin : 100, logger : <FKMC.montecarlo.Eigenspectrum_IPR_all object at 0x7ff74ce7f7d0>, proposal : <function     
    p_multi_site_uniform_reflect at 0x7ff7609ccb00>, accept_function : <function perturbation_accept at 0x7ff7609cce60>, warnings : True
    
## Fig 2
Figure name: band_opening.eps
Data in: Long_range_IPR_and_DOS.pickle

### DOS vs T
### IPR vs T


