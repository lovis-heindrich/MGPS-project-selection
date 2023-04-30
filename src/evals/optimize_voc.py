# Optimization script for the myopic voc
import GPyOpt
import time
import numpy as np
import argparse
import json
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

    n_eval = 100
    restarts = 10
    steps = 50

    def blackbox(W):
        voc_policy = JAS_voc_policy(discrete_observations=True, cost_weight=W[0,0])
        res, _ = run_simulation(env, voc_policy, start_seed=5000, n=n_eval)
        print(W, res["expected_reward"].mean(), res["true_reward"].mean())
        return - (res["expected_reward"].mean())
        
    np.random.seed(123456)

    space = [{'name': 'cost_weight', 'type': 'continuous', 'domain': (0,1)}]
    feasible_region = GPyOpt.Design_space(space = space)
    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, restarts)
    objective = GPyOpt.core.task.SingleObjective(blackbox)
    model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=restarts,verbose=False)
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

    # --- Stop conditions
    max_time  = None
    tolerance = 1e-6     # distance between two consecutive observations        

    # Run the optimization
    max_iter  = steps
    time_start = time.time()
    train_tic = time_start
    bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True)

    W = np.array([bo.x_opt])[0,0]
    train_toc = time.time()

    print("\nSeconds:", train_toc-train_tic)
    print("Weights:", W)

    voc_policy = JAS_voc_policy(discrete_observations=True, cost_weight=W)
    res, _ = run_simulation(env, voc_policy, start_seed=0, n=n_eval)

    optimization_res = {
        "W": W,
        "true_reward": res["true_reward"].mean(),
        "expected_reward": res["expected_reward"].mean(),
        "runtime": res["runtime"].mean(),
        "actions": res["actions"].mean()
    }

    with open(f"./data/simulation_results/optimized_voc.json", 'w') as out_f:
        json.dump(optimization_res, out_f)