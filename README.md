# Long-Range Falikov Kimball Model in One Dimension: Phase Transition, Disorder-Free Localisation and Delocalisation

This repository contains code to perform Markov Chain Monte Carlo (MCMC) simulations of a 1D tight binding model with a mixture of classical and quantum degrees of freedom. The strategy used is to perform an MCMC walk over the classical configurations of the system, diagonalising the quantum hamiltionian at each step. 

To install:
    ```sh
    # Create a new conda environment call FKMC
    conda env create -f environment.yml
    conda activate FKMC
    
   # Or using venv:
   python3 -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate.bat` if you're using Windows
   python -m pip install -U pip
   python -m pip install -U -r requirements.txt

   # Install the package
   pip install --editable .
   
    
    ```