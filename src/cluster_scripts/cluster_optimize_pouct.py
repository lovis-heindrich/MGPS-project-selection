from condor_utils import create_sub_file, submit_sub_file

if __name__ == "__main__":
    bid = 15 # The bid that you want to place
    script = "eval_pouct_baseline.py" # The file that you want to run on the cluster.

    #call_args = []
    #print("Submitting", call_args)
    #call_args = [str(arg) for arg in call_args]

    # parser.add_argument('n_eval', type=int, default=1, help='Number of evaluated environments')
    # parser.add_argument('steps', type=int, default=1000, help='MCTS evaluation steps per action')
    # parser.add_argument('rollout_depth', type=int, default=3, help='MCTS rollout policy depth')
    # parser.add_argument('exploration_coeff', type=float, default=1, help='UCB exploration coefficient')
    # parser.add_argument('seed', type=int, default=0, help='Seed of the first environment')
    # n_eval = 500
    # steps = [10, 100, 1000, 5000]
    # rollout_depths = [0, 3]
    # exploration_coeffs: list[float] = [0.5, 1, 5, 10, 50, 100]
    # seed = 0

    # for step in steps:
    #     for rollout_depth in rollout_depths:
    #         for exploration_coeff in exploration_coeffs:
    #             #n_eval, steps, rollout_depth, exploration_coeff, seed
    #             args = [n_eval, step, rollout_depth, exploration_coeff, seed]

    #             call_args: list[str] = [str(arg) for arg in args]
    #             print("Submitting", call_args, len(call_args))
    #             sub_file = create_sub_file(script, call_args, num_runs=1, num_cpus=1, req_mem = 8192, outputs = True, errs = True)
    #             submit_sub_file(sub_file, bid)


    call_args = [500, 5000, 0, 10, 0]
    print("Submitting", call_args, len(call_args))
    sub_file = create_sub_file(script, call_args, num_runs=1, num_cpus=1, req_mem = 8192, outputs = True, errs = True)
    submit_sub_file(sub_file, bid)