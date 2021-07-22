## Markov Chain Montecarlo for Quantum Tight Binding Models with Classical Sectors.
[![DOI](https://zenodo.org/badge/174118363.svg)](https://zenodo.org/badge/latestdoi/174118363)

This repo contains code to perform [Markov Chain Monte Carlo][mcmc] over [the Falikov-Kimball model][fk] for the publication "One-dimensional long-range Falikov-Kimball model: Thermal phase transition and disorder-free localization" available in [PRB][prb] or on the [arXiv][arxiv]. I provide the code here as a record of what was done, it's likely not reproducable without extra help but if you wish to know more please get in touch.

[prb]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.045116
[arxiv]: https://arxiv.org/abs/2103.11735

---

### Applicability and Optimisations 

The code exploits two special properties of this model and would work well on other models with either property:

<p align="center">
  <img width="460" src="https://user-images.githubusercontent.com/2063944/111747025-17724e00-888f-11eb-928a-a98f2f65d70f.png">
</p>

1. The model contains a classical degree of freedom (in this case S_i) and a quantum degree of freedom (c_i) such that when the classical degree of freedom is fixed the model is quadratic in c_i or otherwise easy to solve.
2. The model contains entirely classical terms like J_ij S_i S_j

Property 1 allows the MCMC to define a walk over the classical states of the system and solve the quantum system at each point, effectively factoring the classical subspace out of the full Hilbert space. 

---

Property 2 allows for an optimisation where the standard Metropolis-Hastings alogorithm is modified to compute the classical terms first, perform a probabalistic accept/reject step, and only on acceptance compute the quantum terms, saving upto 90% computation time in our model.

<p align="left">
  <img height="300" src="https://user-images.githubusercontent.com/2063944/111749795-8f8e4300-8892-11eb-9d0a-afab6a83d964.png">
</p>

Also includes code for submission to cluster systems (PBS and SLURM) and analysis of the results, such as computing energy-resolved Inverse Participation Ratios.

[mcmc]: https://arxiv.org/abs/cond-mat/9612186
[fk]: https://arxiv.org/abs/math-ph/0502041

---
A phase diagram for the above model showing the Charge Denisity Wave phase where the spins order antiferromagnetically, the gapped Mott Insulator phase and the gapless Anderson Insulator phase.
<p align="left">
  <img height="300" src="https://user-images.githubusercontent.com/2063944/111750304-5bffe880-8893-11eb-8e4d-73276c1240fb.png">
</p>

---

### Installation and Use
1. Clone the repo
    ```sh
    git clone https://github.com/TomHodson/FKMC
    cd FKMC
    ```
1. setup an environment using conda or venv
    ```sh
    conda env create -f environment.yml
    conda activate FKMC
    ```
or
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate.bat` if you're using Windows
   python -m pip install -U pip
   python -m pip install -U -r requirements.txt
   ```
1.
Install the FKMC package
    ```sh
    pip install --editable .
    ```
   
1. (Optional) Install a git filter to prevent cell output from showing up underversion control
    ```ssh
    nbstripout --install
    ```
    
## Use
A typical use would be to:
1. Create a notebook for a simulation e.g batchscripts/gap_opening_U-5_logarithmic.ipynb
1. Submit that simulation to the cluster using the helper script at batchscripts/submit.py This would likely require modification for other cluster setups. The script will create a directory for the simulation at ~/HPC_data with subdirectories containing the logs and data. e.g
```sh
./batchscripts/submit.py 
```
