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

