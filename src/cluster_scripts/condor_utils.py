### Made with the help of submit_jobs.py file by tnestmeyer on thw MPI-IS wiki
### Place this file in the cluster in the same folder as the script file you want to submit ###
import os
import sys

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_sub_file(script_name, args, process_arg = False, num_cpus = 1, num_gpus = 0,
                    req_mem = 2000, num_runs = 1, py_version = 3, logs = True, 
                    errs = False, outputs = False):
    """
        Create condor sub file named temp_submission.sub with python as executable.
        
        Arguments:
            script_name  -- Name of the python script
            args         -- List of arguments (args will be converted to string)
            process_arg  -- If process_arg is True, the first argument that the script takes
                            has to be the job number
            num_cpus     -- Number of CPUs to request
            num_gpus     -- Number of GPUs to request
            req_mem      -- Number of MBs of memory to request
            num_runs     -- Number of jobs to run
            py_version   -- The version of python to be run (i.e 2 or 3)
            logs         -- If logs are to be create for each run
            errs         -- If err files are to be created for each run
            outputs      -- If output files are to be created for each run
    """
    script = script_name.split(".")[0]
    sub_file_name = "temp_submission.sub"
    py_path = "/usr/bin/python"
    args = [str(arg) for arg in args]
    script_args = ""
    if process_arg:
        script_args += " $(Process)"
    for arg in args:
        script_args += f" {arg}"
    combined_script_args = "_".join(args)
    if py_version == 3:
        py_path += "3"
    with open(sub_file_name, "w") as sub_file:
        sub_file.write(f"executable = {py_path}\n")
        sub_file.write(f"arguments = {script_name}{script_args}\n")
        if logs:
            make_dir("logs")
            sub_file.write(f"log = logs/{combined_script_args}_$(Process).log\n")
        if errs:
            make_dir("errs")
            sub_file.write(f"error = errs/{combined_script_args}_$(Process).err\n")
        if outputs:
            make_dir("outputs")
            sub_file.write(f"output = outputs/{combined_script_args}_$(Process).out\n")
        sub_file.write("getenv = True\n")
        sub_file.write(f"request_memory = {req_mem}\n")
        sub_file.write(f"request_cpus = {num_cpus}\n")
        if num_gpus:
            sub_file.write(f"request_gpus = {num_gpus}\n")
        sub_file.write(f"queue {num_runs}\n")
    return sub_file_name

def submit_sub_file(sub_file, bid = 3, remove_on_submission = False):
    """
    Submit jobs with the specified bid
    Arguments:
        sub_file             -- Name of the submission file
        bid                  -- Bid for the condor system
        remove_on_submission -- Whether to remove the submit file after submission
    """
    command = f"condor_submit_bid {bid} {sub_file}"
    os.system(command)
    if remove_on_submission:
        os.remove(sub_file)

if __name__ == "__main__":
    """
        Example of a submission file creation
    """
    sub_file = create_sub_file("control_model_running.py", [1,"lvoc", "likelihood", "false", "false"], outputs = True)
    submit_sub_file(sub_file, bid = 100)