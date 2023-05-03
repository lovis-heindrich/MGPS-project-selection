# Optimization script for the myopic voc
import argparse
from src.utils.khalili_env import get_env
from simulation import run_simulation
from src.policy.jas_voc_policy import JAS_voc_policy


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    env, config = get_env()

    n_eval = 5000
    start_seed = 5000
    cost_weight = 0.5798921379230035

    voc_policy = JAS_voc_policy(discrete_observations=True, cost_weight=cost_weight)
    res, _ = run_simulation(env, voc_policy, start_seed=start_seed, n=n_eval)

    res.to_csv(f"./data/simulation_results/eval_voc.json")