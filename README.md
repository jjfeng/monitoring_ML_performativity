# Companion code for the manuscript "Monitoring machine learning (ML)-based risk prediction algorithms in the presence of confounding medical interventions"

## Installation instructions
* This code is designed for running with pip. Packages are listed in requirements.txt.
* To run the Bayesian changepoint monitoring procedure, install cmdstanpy and stan.

## Reproducing simulations
Code for each simulation is organized in a folder with the prefix `simulation_`. Use `scons` to run these simulations, e.g. `scons simulation_null_naive`.
