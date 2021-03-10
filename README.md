# Long-Range Falikov Kimball Model in One Dimension: Phase Transition, Disorder-Free Localisation and Delocalisation

[![DOI](https://zenodo.org/badge/174118363.svg)](https://zenodo.org/badge/latestdoi/174118363)

This repository contains code to perform Markov Chain Monte Carlo (MCMC) simulations of a 1D tight binding model with a mixture of classical and quantum degrees of freedom. The strategy used is to perform an MCMC walk over the classical configurations of the system, diagonalising the quantum hamiltionian at each step. Includes code for submission to cluster systems and analysis of the results, such as computing an energy-resolved Inverse Participation Ratio.

## Installation
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
1.
