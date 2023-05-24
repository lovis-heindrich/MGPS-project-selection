# jas-mdp

## Experiments

Analysis code for the benchmark simulations can be found in ```analysis-pouct.ipynb``` and ```analysis_simulations.ipynb```.

Analysis code for the human experiment can be found in ```process_experiment.ipynb``` and ```analysis-experiment.ipynb```.

R scripts containing the statistical analysis can be found in ```src/evals```.

## Code structure

The main source of this project can be found in the ```src``` repository. To run a strategy discovery algorithm, the following steps are required: 

1. Create an environment. The environment is implemented in ```src/utils/mouselab_jas```, helper functions to simplify initializing new environments can be found in ```src/utils/env_creation.py```.

2. Initialize a policy. Generic policies and a few baselines can be found in ```src/policy/jas_policy.py```, the implementation of the myopic policy can be found in ```src/policy/jas_voc_policy.py```. 

3. Run the policy. This can be done with the ```run_episode``` and ```run_simulation``` functions in ```simulations.py```. 

## Task

The project selection task is adapted from Khalili-Damghani and Sadi-Nezhad (2013). An estimation of inferred environment parameters can be found in ```khalili_env_params.ipynb```. The full environment is stored in ```src/utils/khalili_env.py```.
> Khalili-Damghani, K., & Sadi-Nezhad, S. (2013). A hybrid fuzzy multiple criteria group decision making approach for sustainable project selection. Applied Soft Computing, 13(1), 339–352.

## Environment

The environment makes use of the [Mouselab-MDP](https://github.com/RationalityEnhancementGroup/mouselab-mdp-tools) framework and is implemented under ```scr/utils/mouselab_jas.py```.

## MGPS

The MGOS algorithm is implemented in ```src/policy/jas_voc_policy.py```. The VOC calculation builds upon Heindrich et al. (2023), a strategy discovery method for partially observable metalevel MDPs.

> Heindrich, L., Consul, S., & Lieder, F. (2023). Leveraging AI to improve human planning in large partially observable environments. arXiv preprint arXiv:2302.02785.
