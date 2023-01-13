# jas-mdp

## Code structure

The main source of this project can be found in the ```src``` repository. To run a strategy discovery algorithm, the following steps are required: 

1. Create an environment. The environment is implemented in ```src/utils/mouselab_jas```, helper functions to simplify initializing new environments can be found in ```src/utils/env_creation.py```.

2. Initialize a policy. Generic policies and a few baselines can be found in ```src/policy/jas_policy.py```, the implementation of the myopic policy can be found in ```src/policy/jas_voc_policy.py```. 

3. Run the policy. This can be done with the ```run_episode``` and ```run_simulation``` functions in ```simulations.py```. 