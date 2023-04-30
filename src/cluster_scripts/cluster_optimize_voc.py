from condor_utils import create_sub_file, submit_sub_file

if __name__ == "__main__":
    bid = 15 # The bid that you want to place
    script = "optimize_voc.py" # The file that you want to run on the cluster.

    #call_args = []
    #print("Submitting", call_args)
    #call_args = [str(arg) for arg in call_args]
    sub_file = create_sub_file(script, [], num_runs=1, num_cpus=1, req_mem = 8192, outputs = True, errs = True)
    submit_sub_file(sub_file, bid)