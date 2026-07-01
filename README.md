[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14639677.svg)](https://zenodo.org/doi/10.5281/zenodo.14639677)

## Spike Synchrony Reliably Disentangles Stimulus Saliency and Familiarity
This is a repository for the paper **When Firing Rate Falls Short: Spike Synchrony Reliably Disentangles Stimulus Saliency and Familiarity**. 
This repository contains code to run experiments and analyze spiking neural network models that encode known (familiar) patterns as recurrent excitatory ensembles, and use firing rate and spike synchrony to decode familiar versus new inputs from firing activity.

### Models
- V1 LIF network with Pyr and SOM neurons; recurrent connectivity statistics based on biological observations. Every neuron forms both excitatory and inhibitory connections (no Dale’s law).
- Associative memory LIF network (AMN) with within-ensemble excitatory and cross-ensemble inhibitory connections. Every neuron forms both excitatory and inhibitory connections (no Dale’s law).

### Familiarity detection
1) The model receives familiar (matching excitatory connectivity) and new input stimuli and runs for 500 ms.
2) Output firing activity is used to perform pattern identification (detect which ensemble receives the external input) and familiarity detection (whether the pattern is familiar or new).
3) Familiarity is decoded from population spike trains from either their synchrony (Rsync) or spike count, using binary logistic regression.

### Installation
Clone the repository and all libraries from `requirements.txt`.

### The code base
Brief description of files. 

The **src** folder contains the source code for the model:
- `model.py` contains a class for a Leaky Integrate-and-Fire (LIF) spiking model.
- `network_v1.py` contains the functions for constructing a V1 network.
- `network_abstract.py` contains the functions for constructing an AMN.
- `experiment.py` defines the main class for running simulations of the associative memory experiments.
- `measure.py` defines metrics for spike trains (currently firing rate and Rsync).
- `config.py` defines simulation hyperparameters.
- `utils.py` contains additional helper functions.

The main folder has scripts for running all simulations and analyses:
- `run_v1.py` runs the simulation for main experiments with familiarity detection under changing input saliency by the V1 model; saves the results as *csv* files.
- `plot_v1_main_results.py` plots the familiarity detection performance of the V1 model under different input saliency variability.
- `plot_v1_connectivity.py` plots the example V1 model connectivity.
- `plot_v1_activity.py` plots spike trains for four input regimes: high saliency+high familiarity, high saliency+low familiarity, low saliency+high familiarity, low saliency+low familiarity.
- `run_amn.py` runs the simulation for main experiments with familiarity detection under changing input saliency by the AMN with different connectivity parameters (the sparsity of within-pattern excitatory and cross-pattern inhibitory connections); saves the results as *csv* files.
- `plot_amn_main_results.py` plots the familiarity detection performance of the AMN under different input saliency variability.
- `plot_amn_activity.py` plots spike trains for four input regimes: high saliency+high familiarity, high saliency+low familiarity, low saliency+high familiarity, low saliency+low familiarity.
- `plot_amn_performance.py` plots the familiarity detection and pattern classification performance of the AMN under highest input saliency variability, for different connectivity parameters (the sparsity of within-pattern excitatory and cross-pattern inhibitory connections) and the number of familiar patterns.
