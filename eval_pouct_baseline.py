### CLuster script for running PO-UCT
# Requires a res/po_uct folder to store results
import argparse
import json
from src.utils.khalili_env import get_env
from simulation import run_simulation
from src.policy.po_uct import POUCT_policy
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_eval', type=int, default=1, help='Number of evaluated environments')
    parser.add_argument('steps', type=int, default=1000, help='MCTS evaluation steps per action')
    parser.add_argument('rollout_depth', type=int, default=3, help='MCTS rollout policy depth')
    parser.add_argument('exploration_coeff', type=float, default=1, help='UCB exploration coefficient')
    parser.add_argument('seed', type=int, default=0, help='Seed of the first environment')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    print(args.rollout_depth, type(args.rollout_depth))
    env, config = get_env(5)
    policy = POUCT_policy(steps=args.steps, rollout_depth=args.rollout_depth, exploration_coeff=args.exploration_coeff)

    res, _ = run_simulation(env, policy, start_seed=args.seed, n=args.n_eval)

    optimization_res = {
        "true_reward": res["true_reward"].mean(),
        "expected_reward": res["expected_reward"].mean(),
        "runtime": res["runtime"].mean(),
        "actions": res["actions"].mean(),
        "seed": res["seed"].mean(),
        "steps": args.steps,
        "exploration_coeff": args.exploration_coeff,
        "rollout_depth": args.rollout_depth
    }

    with open(f"./data/pouct_eval/pouct_{args.steps}_{int(args.exploration_coeff*10)}_{args.rollout_depth}_{args.seed}.json", 'w') as out_f:
        json.dump(optimization_res, out_f)

    res["steps"] = args.steps
    res["exploration_coeff"] = args.exploration_coeff
    res["rollout_depth"] = args.rollout_depth

    res.to_csv(f"./data/pouct_eval/pouct_{args.steps}_{int(args.exploration_coeff*10)}_{args.rollout_depth}_{args.seed}.csv")