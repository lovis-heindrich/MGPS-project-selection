# Project Selection metareasoning model

## Code structure

The main source of this project can be found in the ```src``` repository. To run a strategy discovery algorithm, the following steps are required: 

1. Create an environment. The environment is implemented in ```src/utils/mouselab_jas```, helper functions to simplify initializing new environments can be found in ```src/utils/env_creation.py```.

2. Initialize a policy. Generic policies and a few baselines can be found in ```src/policy/jas_policy.py```, the implementation of the myopic policy can be found in ```src/policy/jas_voc_policy.py```. 

3. Run the policy. This can be done with the ```run_episode``` and ```run_simulation``` functions in ```simulations.py```. 

## Task

The project selection task is adapted from Khalili-Damghani and Sadi-Nezhad (2013). An estimation of inferred environment parameters can be found in ```khalili_env_params.ipynb```.
> Khalili-Damghani, K., & Sadi-Nezhad, S. (2013). A hybrid fuzzy multiple criteria group decision making approach for sustainable project selection. Applied Soft Computing, 13(1), 339–352.

## Environment

The environment is based on the [Mouselab-MDP](https://github.com/RationalityEnhancementGroup/mouselab-mdp-tools). The Mouselab-MDP environment has been adjusted for the project selection task in multiple ways:
1. Observations and ground truths are generated from a Normal distribution. Observations are rounded to a discrete interval of expert guesses (e.g. 1 to 5).
2. Multiple distinct meta-level actions are available for each node in the belief state, which represents asking different experts for their estimate of the same node.
3. Each project's utility is determined by multiple criteria which can have different weights that express their importance.

## VOC calculation

The VOC algorithm is based on Heindrich et al. (2023), a strategy discovery method for partially observable metalevel MDPs. 

> Heindrich, L., Consul, S., & Lieder, F. (2023). Leveraging AI to improve human planning in large partially observable environments. arXiv preprint arXiv:2302.02785.
