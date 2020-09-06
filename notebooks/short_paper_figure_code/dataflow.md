High level:
1) Generate data on CX1 -> ~/HPC_data/*
2) Reduce data ~/HPC_data/* -> process script -> ~/HPC_data/pickled_data
3) plot figures ~/HPC_data/pickled_data/* -> final_figures.ipynb -> figures

## Binder Plots 


    varyingT_binder_zoom.ipynb --> HPC_data/Tsweep3_binder ----
                                                               |
                                                                --> process_binder_data.ipynb --> HPC_data/pickled_data/binder_data.pickle ---> final_figures.ipynb
                                                               |
           varyingT_full.ipynb --> HPC_data/Tsweep3_full   ----        


~/HPC_data/Tsweep3_full
    #with the new definition of steps as where the overall number of trials is N_steps*N_sites*N_sites // 100
    #rather than N_steps*N_sites as it was before
    MCMC Steps: 7 chains of 5000 for 35000 with thinning = 10 for 3500 recorded steps
    Burn in: 0
    Structure_names: {'Rs': 10, 'Ts': 25}
    Ns = [ 10  20  30  50  70 110 160 250]
    Runtimes: 
        Average: 6.0 hours
        Min: 52.0 minutes
        Max: 20.0 hours
        Total: 9.0 weeks

~/HPC_data/Tsweep3_binder
    MCMC Steps: 20 chains of 5000 for 100000 with thinning = 10 for 10000 recorded steps
    Burn in: 0
    Structure_names: {'Rs': 10, 'Ts': 25}
    Ns = [ 10  20  30  50  70 110 160 250]
    Runtimes: 
        Average: 9.0 hours
        Min: 2.0 hours
        Max: 1.0 days
        Total: 14.0 weeks
  
## 2D Phase diagrams
        
    /data/users/tch14/slurm_runs/117969 --                                                 --> pickled_data/TJ_phase_data.pickle ---> final_figures.ipynb
                                          |--->(on CMTH) process_2D_phase_diagrams.ipynb -|                                         |          
    /data/users/tch14/slurm_runs/117734 --                                                 --> pickled_data/TU_phase_data.pickle ---
                                          |                                                  |       
                                          |        
                                          |--> gap_opening_TU_plot.ipynb ---> ?


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
