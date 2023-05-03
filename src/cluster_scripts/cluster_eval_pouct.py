from condor_utils import create_sub_file, submit_sub_file

if __name__ == "__main__":
    bid = 30 # The bid that you want to place
    script = "eval_pouct_baseline.py" # The file that you want to run on the cluster.


    # 10 steps: c=0.5, depth=0 (score=3.4378383647496817)
    # 100 steps: c=0.5, depth=0 (score=3.6508409110380886)
    # 1000 steps: c=10, depth=0 (score=3.6951088300956654)
    # 5000 steps: c=10, depth=0 (score=3.704249016068181)
    
    #n_eval, steps, rollout_depth, exploration_coeff, seed
    
    n_eval = 1000
    for start_seed in [5000, 6000, 7000, 8000, 9000]:
        call_args = [n_eval, 10, 0, 0.5, start_seed]
        print("Submitting", call_args, len(call_args))
        sub_file = create_sub_file(script, call_args, num_runs=1, num_cpus=1, req_mem = 8192, outputs = True, errs = True)
        submit_sub_file(sub_file, bid)

        call_args = [n_eval, 100, 0, 0.5, start_seed]
        print("Submitting", call_args, len(call_args))
        sub_file = create_sub_file(script, call_args, num_runs=1, num_cpus=1, req_mem = 8192, outputs = True, errs = True)
        submit_sub_file(sub_file, bid)

        call_args = [n_eval, 1000, 0, 10, start_seed]
        print("Submitting", call_args, len(call_args))
        sub_file = create_sub_file(script, call_args, num_runs=1, num_cpus=1, req_mem = 8192, outputs = True, errs = True)
        submit_sub_file(sub_file, bid)

        call_args = [n_eval, 5000, 0, 10, start_seed]
        print("Submitting", call_args, len(call_args))
        sub_file = create_sub_file(script, call_args, num_runs=1, num_cpus=1, req_mem = 8192, outputs = True, errs = True)
        submit_sub_file(sub_file, bid)