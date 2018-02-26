This repo contains code to automatically optimize parameters of map matcher pipelines, using Bayesian optimization.
The initial code was created as part of my master's thesis and will hopefully get more polished in the future.

Dependencies:
* python3
* numpy
* scipy
* matplotlib
* scikit-learn
* bayesian-optimization (https://github.com/fmfn/BayesianOptimization)

Python3 should be installable with your favourite package manager.
All other dependencies can be installed via a python package manager, e.g. with pip install [pkg-name].

Setup:
To optimize the parameters of your specific system, you need to implement the `create_objective_function_sample` function.
The file in which you define this function needs to be set in the `objective_function.py` as `INTERFACE_MODULE` variable.
The Bayesian Map Matcher Optimizer calls this function whenever it wants to make a new observation (i.e. create a new sample).
The implementation for the map matcher of DLR RM can be used as a reference of what it could do. Have a look at the file `dlr_map_matcher_interface_tools.py`.

Usage:
The main script is experiment\_coordinator.py, but please use wrapper.sh to run it.
It's a simple wrapper script which sets PYTHONHASHSEED to a fixed value.
Without it, the SampleDatabase won't be able to recognize previously generated Samples.
