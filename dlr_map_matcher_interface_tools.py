import datetime
import sys
import rosparam
import os
import pickle
import subprocess
import time
import yaml
import ast

###########################################
# Helper functions, not directly used by the experiment_coordinator
#########################
def _run_evaluation(command):
    """
    Helper method; Starts the evaluation program as subprocess, since it only works with python2 (thanks, ROS...)
    Yields strings to describe the subprocess' status as it runs.
    """
    indicator_length = 0
    eval_process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(eval_process.stdout.readline, ""):
        if stdout_line.startswith('{'): # Is the output a python dict's string representation?
            info_dict = ast.literal_eval(stdout_line)
            # Print info string by reading from info dict
            info_string = "[" + str(info_dict['completed_jobs']) + "/" + str(info_dict['total_jobs']) + "] "
            for hostname, state_list in info_dict['remotes'].items():
                info_string += "[" + hostname
                for state in state_list: # Print state for each RemoteWorker
                    info_string += "|" + state
                info_string +=  "] "
            info_string += "[avg. " + info_dict['avg_job_time'] + "] "
            info_string += "[eta " + info_dict['estimated_finish_time'] + "]"
            status_string = "\t\t" + info_string + indicator_length * '.' + '                  \r'
            indicator_length = (indicator_length + 1 if indicator_length < 3 else 0)
        else: # otherwise just yield the string
            status_string = "\t\t" + stdout_line 
        yield status_string
    eval_process.stdout.close()
    return_code = eval_process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

###########################################
# Actual interface functions, those need to be implemented.
#########################
def create_evaluation_function_sample(toplevel_directory, sample):
    """
    DLR-map-matcher-evaluation-specific code to get an evaluation_function Sample from a finished map matcher run.

    :param toplevel_directory: The directory from which the Sample's can be generated.
    :param sample: Sample object which should be filled with data
    """
    sample.origin = toplevel_directory
    sample.translation_errors = []
    sample.rotation_errors = []
    sample.time = datetime.timedelta(0)
    for root, dirs, files in os.walk(toplevel_directory, topdown=False, followlinks=True):
        for file_name in files:
            if file_name == "statistics.pkl":
                # Check if data generation actually finished, using the map matcher log file
                with open(os.path.join(root, "map_matcher.log"), 'r') as log_file:
                    completed = False
                    for line in log_file.readlines():
                        if "no more submaps. stop." in line:
                            completed = True
                            break
                    if not completed:
                        raise IOError("Something went wrong while generating this part of the sample,\
                                       couldn't find ", os.path.join(root, "MAP_MATCHER_JOB_COMPLETED"))
                # Sample seems ok, find and load pickle...
                pickle_path = os.path.join(root, file_name)
                print("\tLoading pickle", pickle_path)
                eval_result_data = pickle.load(open(pickle_path, 'rb'), encoding='latin1')
                sample.translation_errors.extend(eval_result_data['results']['hough3d_to_ground_truth']['translation'].values())
                sample.rotation_errors.extend(eval_result_data['results']['hough3d_to_ground_truth']['rotation'].values())
                sample.time += datetime.timedelta(seconds=float(eval_result_data['timings']['MapMatcherNode::runBatchMode total']['total']))
    
    # Check if there any statistics.pkl were found!
    if len(sample.translation_errors) == 0:
        raise IOError("No statistics.pkl files were found in the toplevel_directory's filetree.")

    for root, dirs, files in os.walk(toplevel_directory, topdown=False, followlinks=True):
        for file_name in files:
            if file_name == "params.yaml":
                # As soon as one params.yaml is found somewhere, we're done.
                # (because all other dirs have exactly the same parameters)
                sample.parameters = rosparam.load_file(os.path.join(root, file_name))[0][0]
                return

def generate_sample(rosparams, sample_generator_config):
    """
    Generates data for a new Sample and returns the path at which the generated data lies.

    :param rosparams: A dict of the complete set of parameters.
    :param sample_generator_config: A dict with other parameters needed for generating the Sample's data.
                                    Make sure params in this dict don't mess up the validity of the SampleDatabase.
                                    The experiment_coordinator only manages the rosparams.
    """

    # Create the path to the folder in which all information for this map matcher run is stored
    env_dir = os.path.join(sample_generator_config['environment'], time.strftime("%Y-%m-%d_%H:%M:%S"))
    if os.path.exists(env_dir):
        raise IOError(env_dir, "already exists, can't generate sample here!")
    os.makedirs(env_dir) # Create the folder on disk
    results_path = os.path.join(env_dir, "results") # Path for map matcher's output
    yaml_path = os.path.join(env_dir, "evaluation.yaml") # Path for map matcher's config
    rosparams_path = os.path.join(env_dir, "params.yaml") # Path for the rosparams
    eval_log_path = os.path.join(env_dir, "evaluation.log") # Path for the rosparams
    # Add the paths to the config, so the map_matcher_evaluation script finds it
    sample_generator_config['parameters'] = rosparams_path
    sample_generator_config['results'] = results_path
    # Create *the* map matcher evaluation script config.
    # sample_generator_config may contain some unnecessary fields, but that doesn't hurt.
    mm_yaml_config = {'MapMatcherJobs': {'gpr_auto_entry': sample_generator_config}}
    with open(yaml_path, 'w') as yaml_file:
        # Store the yaml file on disk. The evaluator will load it again.
        yaml.dump(mm_yaml_config, yaml_file)
    with open(rosparams_path, 'w') as rosparams_file:
        yaml.dump(rosparams, rosparams_file)

    evaluation_command = [sample_generator_config['evaluator_executable'], yaml_path, '-l', eval_log_path]
    for arg_string in sample_generator_config['extra_arguments']:
        evaluation_command.append(arg_string)
    print("\t\tStarting evaluation process in directory", env_dir)
    # Run the evaluation and print its output
    for output in _run_evaluation(evaluation_command):
        print(output, end="")
        sys.stdout.flush()
    return results_path
