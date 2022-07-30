import os
import pathlib
import pickle
import numpy as np
import pandas as pd
results_folder_name = "ARCHCOMP2022_UR"
list_benchmark_name = "AT2_budget_2000_10_reps_UR"

base_path = pathlib.Path()
result_directory = base_path.joinpath(results_folder_name)#.joinpath(f"{benchmark_name}")
print(result_directory)
dirs = os.listdir(result_directory)
benchmark_names = []

for direc in dirs:
    
    if os.path.isdir(result_directory.joinpath(direc)):
        benchmark_names.append(result_directory.joinpath(direc))

benchmark = benchmark_names[0]
files = [benchmark.joinpath(f) for f in os.listdir(benchmark)] 

system_list = []
property_list = []
simulations_list = []
time_list = []
robustness_list = []
falsified_list = []
input_list = []

for run in files:
    with open(run, "rb") as f:
        result_dictionary = pickle.load(f)

    points_history = np.array(result_dictionary["point_history"], dtype = object)
    fals_indices = np.where(points_history[:,-1]<0)[0]

    if fals_indices.shape[0] != 0:
        point = points_history[fals_indices[0], :]
        first_fals = fals_indices[0] + 1
    else:    
        first_fals = np.argmin(points_history[:,-1]) + 1
        point = points_history[np.argmin(points_history[:,-1]), :]

    rob = point[-1]
    

    if "AT" in str(benchmark):
        system_list.append("AT")
        AT_prop_list = ["AT1","AT2","AT51", "AT52", "AT53", "AT54", "AT6a", "AT6b", "AT6c", "AT6abc"]
        property_list.append(AT_prop_list[[i for i, x in enumerate([prop in str(benchmark) for prop in AT_prop_list]) if x][0]])
    elif "CC" in str(benchmark):
        system_list.append("CC")


    simulations_list.append(first_fals)

    time_list.append(result_dictionary["total_time_elapsed"])

    robustness_list.append(point[-1])

    falsified_list.append(result_dictionary["falsified"])

    input_list.append(point[1])
    print(result_dictionary["first_fals"])
    
submission_df = pd.DataFrame({
        "system" : system_list,
        "property": property_list,
        "simulations" : simulations_list,
        "time" : time_list,
        "robustness" : robustness_list,
        "falsified" : falsified_list,
        "input": input_list
    })
print(submission_df)    
