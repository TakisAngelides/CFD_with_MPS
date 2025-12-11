import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from numpy import pi
from matplotlib import rc 
from matplotlib.animation import FuncAnimation, PillowWriter
# rc('text', usetex=True)

# username
username = 'tangelides'

# Project number of dag
project_number = os.getcwd().strip().split('/')[-1]
path_to_project_number = f'/lustre/fs24/group/cqta/{username}/CFD_MPS/Exponential_Integrator/DAGS/{project_number}'

def write_dag():
    
    # Name of dag
    name_of_dag = 'run.dag'
    
    # Open text file to write the dag instructions
    f_dag = open(name_of_dag, 'w')
    
    # This will contain DAGMAN_USE_DIRECT_SUBMIT = False to avoid obscure bugs of authentication
    f_dag.write(f'CONFIG /lustre/fs24/group/cqta/{username}/CFD_MPS/dagman.config\n')
        
    # Path to submission file to run from dag
    path_to_sub = f'/lustre/fs24/group/cqta/{username}/CFD_MPS/Exponential_Integrator/run.sub'
    
    # Where to find the inputs
    path_to_inputs_h5 = path_to_project_number + '/inputs.h5'
    f_h5 = h5py.File(path_to_inputs_h5, 'w')
        
    # Create relevant folders if needed
    if not os.path.exists(f'{path_to_project_number}/Plots'):
        os.makedirs(f'{path_to_project_number}/Plots')
    if not os.path.exists(f'{path_to_project_number}/HDF5'):
        os.makedirs(f'{path_to_project_number}/HDF5')        
    if not os.path.exists(f'{path_to_project_number}/HDF5_b'):
        os.makedirs(f'{path_to_project_number}/HDF5_b')   
    if not os.path.exists(f'{path_to_project_number}/Logs/Error'):
        os.makedirs(f'{path_to_project_number}/Logs/Error')        
    if not os.path.exists(f'{path_to_project_number}/Logs/Output'):
        os.makedirs(f'{path_to_project_number}/Logs/Output')        
    if not os.path.exists(f'{path_to_project_number}/Logs/Log'):
        os.makedirs(f'{path_to_project_number}/Logs/Log')
                                                                           
    # Writing jobs
    job_id = 1 # job counter
    iterations_list = [2000]
    dt_list = [0.1]
    n_qubits_list = [8]
    max_bond_ttc_u0 = 100
    tol_ttc_u0 = 1e-11 # tolerance for tt cross for the initial state
    qft_mpo_cutoff = 1e-14 # cutoff to build qft mpo
    cutoff_list = [1e-14] # general cutoff for apply of mpo to mps
    qft_sign_convention = -1 
    max_bond_ttc_kx = 100 # maximum bond dimension for the tt cross of the kx, prop mpo
    tol_ttc_kx = 1e-11 # tolerance for the tt cross of the kx, prop mpo
    kappa = 1.0
    delta_val = 1.0
    scheme_list = ["euler"] # "euler", "heun"
    initial_state_name_list = ["sin"] # "sin_tt", "-sin", "sin", "exp_tt", "analytical_sol_tt", "cos_sin" (only last one is for no_tn)
    memory = 4 # in gigabytes
    cpu = 1 # how many cpus to use
    days = 1 # maximum runtime (max allowed is 6.99)
    day_to_backup = 7
    max_retries = 1 # if a node in the DAG crashes it gets retried
    save_real_space_statevector = "true"
    save_uhat = "false" # save the uhat state for every step in steps_to_save
    save_D_t = "true" # for the case of tn_flag_str = "no_tn" this variable just means save_t which means to save the time per iteration 
    save_all_steps = "true" # whether to make a save at every step of the observables we are tracking
    save_all_steps_flag = save_all_steps.lower() == "true"
    tn_flag_str = "tn" # "tn", "no_tn"
    which_equation_list = ["B", "KS"] # "B", "KS"
    coeff_set_dict = {"B" : [[1.0, 1.0], [1.0, 0.01], [0.01, 1.0]], "KS" : [[1.0, 1.0, 1.0], [1.0, 0.01, 0.01], [0.01, 1.0, 0.5]]} # For B this is nonlinear_coeff, kx2_coeff and for KS this is nonlinear_coeff, kx2_coeff, kx4_coeff
    
    for n_qubits in n_qubits_list:
        for cutoff in cutoff_list:
            for dt_idx, dt in enumerate(dt_list):
                iterations = iterations_list[dt_idx]
                steps_to_backup = []
                if not save_all_steps_flag:
                    num_elements = 1200
                    steps_to_save = np.linspace(1, iterations, num_elements, dtype=int)
                for scheme in scheme_list:
                    for initial_state_name in initial_state_name_list:
                        for which_equation in which_equation_list:
                            for coeff_set in coeff_set_dict[which_equation]:
                                
                                if which_equation == "B":
                                    nonlinear_coeff, kx2_coeff = coeff_set
                                    kx4_coeff = 0
                                elif which_equation == "KS":
                                    nonlinear_coeff, kx2_coeff, kx4_coeff = coeff_set
                                    
                                if initial_state_name == "exp_tt":
                                    x_start = -5
                                    x_end = 5
                                elif initial_state_name == "analytical_sol":
                                    x_start = 0.0
                                    x_end = 0.08
                                elif initial_state_name == "sin":
                                    x_start = 0
                                    x_end = 32*pi
                                else:
                                    x_start = 0
                                    x_end = 32*pi
            
                                g = f_h5.create_group(f'{job_id}')   
                                g.attrs["which_equation"] = which_equation
                                g.attrs["kx2_coeff"] = kx2_coeff
                                g.attrs["kx4_coeff"] = kx4_coeff
                                g.attrs["nonlinear_coeff"] = nonlinear_coeff
                                g.attrs["x_start"] = x_start
                                g.attrs["x_end"] = x_end
                                g.attrs["n_qubits"] = n_qubits
                                g.attrs["max_bond_ttc_u0"] = max_bond_ttc_u0
                                g.attrs["tol_ttc_u0"] = tol_ttc_u0
                                g.attrs["qft_mpo_cutoff"] = qft_mpo_cutoff
                                g.attrs["iterations"] = iterations
                                g.attrs["qft_sign_convention"] = qft_sign_convention
                                g.attrs["cutoff"] = cutoff
                                g.attrs["max_bond_ttc_kx"] = max_bond_ttc_kx
                                g.attrs["tol_ttc_kx"] = tol_ttc_kx
                                g.attrs["dt"] = dt
                                g.attrs["kappa"] = kappa
                                g.attrs["delta_val"] = delta_val
                                g.attrs["scheme"] = scheme
                                g.attrs["initial_state_name"] = initial_state_name
                                g.attrs["steps_to_backup"] = steps_to_backup
                                g.attrs["day_to_backup"] = day_to_backup
                                g.attrs["save_real_space_statevector"] = save_real_space_statevector
                                g.attrs["save_uhat"] = save_uhat
                                g.attrs["save_D_t"] = save_D_t
                                g.attrs["save_all_steps"] = save_all_steps
                                if not save_all_steps_flag:
                                    g.attrs["steps_to_save"] = steps_to_save

                                # Write job to dag
                                job_name = f'{job_id}'
                                f_dag.write(f'JOB ' + job_name + f' {path_to_sub}\n')
                                f_dag.write(f'VARS ' + job_name + f' job_id="{job_id}" path_to_project_number="{path_to_project_number}" cpu="{cpu}" mem="{memory}" days="{days}" max_retries="{max_retries}" username="{username}" tn_flag_str="{tn_flag_str}"\n')
                                f_dag.write('RETRY ' + job_name + f' {max_retries}\n')
                                job_id += 1 # increment job counter by 1 (jobs need to have unique names as nodes in a DAG)
            
    # Close the dag file and the h5 input file
    f_dag.close() 
    f_h5.close()
    print(f'Total number of jobs in the dag is {job_id-1}')                                        

def get_initial_state(initial_state_name, x):

    if initial_state_name == "cos_sin":
        
        return np.cos(x / 16) * (1 + np.sin(x / 16))

def analytical_solution(x, t, initial_state_name):
    
    def f_implicit(x):
        return get_initial_state(initial_state_name, x)

    def u(x, t):
        return f_implicit(x - t*f_implicit(x))
    
    return u(x, t)

def plot_heatmap_time_evolution():
    
    if not os.path.exists(f'{path_to_project_number}/Plots/Heatmap_time_evolution'):
        os.makedirs(f'{path_to_project_number}/Plots/Heatmap_time_evolution')
        
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    counter = 0

    for file in os.listdir(path_to_HDF5):
        
        try:
        
            g = f_inputs[file[:-3]]
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items()}
            iterations = attributes_dict['iterations']
            f = h5py.File(f'{path_to_HDF5}/{file}', 'r')
            
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items() if attr_name != ''}
            scheme = attributes_dict['scheme']
            dt = attributes_dict['dt']
            n = attributes_dict['n_qubits']
            initial_state_name = attributes_dict['initial_state_name']
            cutoff = attributes_dict['cutoff']
            nonlinear_coeff = attributes_dict['nonlinear_coeff']
            kx2_coeff = attributes_dict['kx2_coeff']
            kx4_coeff = attributes_dict['kx4_coeff']
            
            states = []
            keys_to_plot = [element for element in f.keys() if "s_" in element]
            keys_to_plot = sorted(keys_to_plot, key=lambda x: int(x.split("_")[1]))
            x_start = attributes_dict["x_start"] 
            x_end = attributes_dict["x_end"]
            N = 2**n
            dx = (x_end - x_start)/N
            x_values = x_start + dx * range(N)
            t = dt
            for i in keys_to_plot:
                i_num = int(i.strip().split("_")[-1])
                if i_num == -1:
                    break
                states.append(np.abs(np.array(f[f"{i}"])))
                # states.append(np.abs(analytical_solution(x_values, t)))
                t += dt
        
            plt.figure(figsize=(14, 12))
            plt.imshow(np.transpose(states), cmap = 'magma')
            plt.colorbar(fraction = 0.03)
            # plt.xticks([])
            plt.yticks([])
            plt.title(f'{scheme}, dt = {dt}, n = {n}, u0 = {initial_state_name}, cutoff = {cutoff}, nonlinear_coeff = {nonlinear_coeff}, kx2_coeff = {kx2_coeff}, kx4_coeff = {kx4_coeff}')
            plt.savefig(f'Plots/Heatmap_time_evolution/{file[:-3]}.png', bbox_inches = 'tight')
            plt.close()
                                     
        except:
            
            counter += 1
            
    if counter > 0:
        
        print(f'From plotting the heatmap time evolution {counter} jobs failed.')        
        
def plot_real_vs_abs_comparison():
    
    if not os.path.exists(f'{path_to_project_number}/Plots/Real_vs_abs_comparison'):
        os.makedirs(f'{path_to_project_number}/Plots/Real_vs_abs_comparison')
        
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    counter = 0

    for file in os.listdir(path_to_HDF5):
        
        try:
        
            g = f_inputs[file[:-3]]
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items()}
            iterations = attributes_dict['iterations']
            f = h5py.File(f'{path_to_HDF5}/{file}', 'r')
            
            states = []
            keys_to_plot = [element for element in f.keys() if "s_" in element]
            keys_to_plot = sorted(keys_to_plot, key=lambda x: int(x.split("_")[1]))
            for i in keys_to_plot:
                states.append(np.sum(np.abs(np.abs(np.real(np.array(f[f"{i}"]))) - np.abs(np.array(f[f"{i}"])))))
            
            # plt.figure(figsize=(14, 12))
            # plt.imshow(np.transpose(states), cmap = 'magma')
            # plt.xticks([])
            # plt.yticks([])
            plt.plot(states)
            plt.ylabel('Sum over spatial sites of |abs(real()) - abs()|')
            plt.xlabel('Iteration number')
            plt.savefig(f'Plots/Real_vs_abs_comparison/{file[:-3]}.png', bbox_inches = 'tight')
            plt.close()
                            
        except:
            
            counter += 1
            
    if counter > 0:
        
        print(f'From plotting the heatmap time evolution {counter} jobs failed.')        

def plot_scheme_comparison():
    # Create output directory for the plots
    output_dir = f'{path_to_project_number}/Plots/Scheme_Comparison'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    # Build the dictionary to group files by (dt, n_qubits, initial_state_name)
    scheme_files = {}
    for file in os.listdir(path_to_HDF5):
        try:
            g = f_inputs[file[:-3]]
            attrs = g.attrs
            key = (attrs['dt'], attrs['n_qubits'], attrs['initial_state_name'])
            scheme = attrs['scheme']
            if key not in scheme_files:
                scheme_files[key] = {}
            scheme_files[key][scheme] = file
        except:
            continue

    # Sort the dictionary keys by decreasing dt
    sorted_keys = sorted(scheme_files.keys(), key=lambda x: x[0], reverse=True)

    # Iterate over the sorted dictionary to plot comparisons
    counter = 0
    for key in sorted_keys:
        schemes = scheme_files[key]
        try:
            if 'euler' in schemes and 'heun' in schemes:
                euler_file = schemes['euler']
                heun_file = schemes['heun']
                
                # Load Euler and Heun data
                f_euler = h5py.File(f'{path_to_HDF5}/{euler_file}', 'r')
                f_heun = h5py.File(f'{path_to_HDF5}/{heun_file}', 'r')
                                
                euler_states = [
                    np.abs(np.array(f_euler[key]))
                    for key in sorted(f_euler.keys(), key=lambda x: int(x.split("_")[1]) if "s_" in x and "_" in x else float('inf'))
                    if "s_" in key and "_" in key and key.split("_")[1].isdigit()
                ]

                heun_states = [
                    np.abs(np.array(f_heun[key]))
                    for key in sorted(f_heun.keys(), key=lambda x: int(x.split("_")[1]) if "s_" in x and "_" in x else float('inf'))
                    if "s_" in key and "_" in key and key.split("_")[1].isdigit()
                ]
                
                # Compute the difference summed over spatial points for each time step
                differences = [
                    np.sum(np.abs(euler - heun))
                    for euler, heun in zip(euler_states, heun_states)
                ]
                
                # Plot the differences
                dt, n_qubits, initial_state_name = key
                plt.plot(np.linspace(0, len(differences)*dt, len(differences)), differences, label=f'dt = {dt}')
                plt.xlabel("Time")
                plt.ylabel("Difference (Summed Over Spatial Points)")
                plt.title(f"Comparison: Euler vs. Heun, n = 6, u0 = exp_tt")
                plt.legend()
                
        except Exception as e:
            print(f"Failed to process {key}: {e}")
            counter += 1
            
    output_file = os.path.join(output_dir, f"n_{n_qubits}_u0_{initial_state_name}.png")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
            
    if counter > 0:
        print(f"{counter} jobs failed during Euler-Heun scheme comparison.")
            
def plot_error_vs_time():
    
    if not os.path.exists(f'{path_to_project_number}/Plots/Error_vs_time'):
        os.makedirs(f'{path_to_project_number}/Plots/Error_vs_time')
        
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    counter = 0

    for file in os.listdir(path_to_HDF5):
        
        # try:
        
        g = f_inputs[file[:-3]]
        attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items()}
        iterations = attributes_dict['iterations']
        f = h5py.File(f'{path_to_HDF5}/{file}', 'r')
        
        attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items() if attr_name != ''}
        scheme = attributes_dict['scheme']
        dt = attributes_dict['dt']
        n = attributes_dict['n_qubits']
        initial_state_name = attributes_dict['initial_state_name']
        
        states = []
        keys_to_plot = [element for element in f.keys() if "s_" in element]
        keys_to_plot = sorted(keys_to_plot, key=lambda x: int(x.split("_")[1]))
        t = dt
        x_start = attributes_dict["x_start"] 
        x_end = attributes_dict["x_end"]
        N = 2**n
        dx = (x_end - x_start)/N
        x_values = x_start + dx * range(N)
        for idx, i in enumerate(keys_to_plot):
            if idx > 400:
                break
            states.append(np.sum(np.abs(np.abs(np.array(f[f"{i}"])) - np.abs(analytical_solution(x_values, t)))))
            t += dt
        
        plt.figure(figsize=(14, 12))
        plt.plot(np.transpose(states))
        plt.title(f'{scheme}, dt = {dt}, n = {n}, u0 = {initial_state_name}')
        plt.savefig(f'Plots/Error_vs_time/{file[:-3]}.png', bbox_inches = 'tight')
                                     
        # except:
            
        #     counter += 1
            
    # if counter > 0:
        
    #     print(f'From plotting the heatmap time evolution {counter} jobs failed.')        

def plot_iteration_time():
    
    if not os.path.exists(f'{path_to_project_number}/Plots/Iteration_time'):
        os.makedirs(f'{path_to_project_number}/Plots/Iteration_time')
        
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    counter = 0

    for file in os.listdir(path_to_HDF5):
        
        try:
        
            g = f_inputs[file[:-3]]
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items()}
            iterations = attributes_dict['iterations']
            f = h5py.File(f'{path_to_HDF5}/{file}', 'r')
            
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items() if attr_name != ''}
            scheme = attributes_dict['scheme']
            dt = attributes_dict['dt']
            n = attributes_dict['n_qubits']
            initial_state_name = attributes_dict['initial_state_name']
            cutoff = attributes_dict['cutoff']
            
            keys_to_plot = [element for element in f.keys() if "t_" in element]
            keys_to_plot = sorted(keys_to_plot, key=lambda x: int(x.split("_")[1]))
            x_start = attributes_dict["x_start"] 
            x_end = attributes_dict["x_end"]
            N = 2**n
            dx = (x_end - x_start)/N
            x_values = x_start + dx * range(N)
            iteration_times = []
            for i in keys_to_plot:
                iteration_times.append(np.float64(f[i]))
            
            plt.figure(figsize=(14, 12))
            plt.plot(iteration_times[1:])
            plt.title(f'{scheme}, dt = {dt}, n = {n}, u0 = {initial_state_name}, cutoff = {cutoff}')
            plt.savefig(f'Plots/Iteration_time/{file[:-3]}.png', bbox_inches = 'tight')
            plt.close()
                                     
        except:
            
            counter += 1
            
    if counter > 0:
        
        print(f'From plotting the heatmap time evolution {counter} jobs failed.')        

def plot_D_vs_time():
    
    if not os.path.exists(f'{path_to_project_number}/Plots/Bond_dimension'):
        os.makedirs(f'{path_to_project_number}/Plots/Bond_dimension')
        
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    counter = 0

    for file in os.listdir(path_to_HDF5):
        
        try:
        
            g = f_inputs[file[:-3]]
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items()}
            iterations = attributes_dict['iterations']
            f = h5py.File(f'{path_to_HDF5}/{file}', 'r')
            
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items() if attr_name != ''}
            scheme = attributes_dict['scheme']
            dt = attributes_dict['dt']
            n = attributes_dict['n_qubits']
            initial_state_name = attributes_dict['initial_state_name']
            cutoff = attributes_dict['cutoff']
            
            keys_to_plot = [element for element in f.keys() if "D_" in element]
            keys_to_plot = sorted(keys_to_plot, key=lambda x: int(x.split("_")[1]))
            x_start = attributes_dict["x_start"] 
            x_end = attributes_dict["x_end"]
            N = 2**n
            dx = (x_end - x_start)/N
            x_values = x_start + dx * range(N)
            avg_D = []
            for i in keys_to_plot:
                avg_D.append(np.average(f[i]))
            
            plt.figure(figsize=(14, 12))
            plt.plot(avg_D)
            plt.title(f'{scheme}, dt = {dt}, n = {n}, u0 = {initial_state_name}, cutoff = {cutoff}')
            plt.savefig(f'Plots/Bond_dimension/{file[:-3]}.png', bbox_inches = 'tight')
            plt.close()
                                     
        except:
            
            counter += 1
            
    if counter > 0:
        
        print(f'From plotting the heatmap time evolution {counter} jobs failed.')      

def plot_average_iteration_time_vs_n_qubits():
    
    # Ensure the plots directory exists
    output_dir = f'{path_to_project_number}/Plots/Average_Iteration_Time_vs_n_qubits'
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    # Dictionary to store iteration times grouped by n_qubits and fixed parameters
    data = {}

    # Iterate over files in the HDF5 folder
    for file in os.listdir(path_to_HDF5):
        try:
            # Load attributes from the inputs file
            g = f_inputs[file[:-3]]
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items()}
            
            # Extract relevant parameters
            n_qubits = attributes_dict['n_qubits']
            scheme = attributes_dict['scheme']
            dt = attributes_dict['dt']
            initial_state_name = attributes_dict['initial_state_name']
            cutoff = attributes_dict['cutoff']
            
            # Skip specific initial_state_name
            if initial_state_name == "exp_tt" or scheme == "heun" or dt != 0.1:
                continue
            
            # Load the HDF5 file for iteration times
            f = h5py.File(f'{path_to_HDF5}/{file}', 'r')
            keys_to_plot = sorted([key for key in f.keys() if "t_" in key], key=lambda x: int(x.split("_")[1]))
            iteration_times = [np.float64(f[key]) for key in keys_to_plot]

            # Calculate average iteration time
            avg_iteration_time = np.mean(iteration_times[1:])  # Skip the first value as per original code

            # Group data by n_qubits with fixed parameters as legend key
            legend_key = f"{scheme}, dt={dt}, cutoff={cutoff}, u0={initial_state_name}"
            if legend_key not in data:
                data[legend_key] = {}
            if n_qubits not in data[legend_key]:
                data[legend_key][n_qubits] = []
            data[legend_key][n_qubits].append(avg_iteration_time)
        
        except Exception as e:
            print(f"Failed to process file {file}: {e}")
            continue

    # Plot all data in a single figure
    plt.figure(figsize=(14, 12))
    for legend_key, qubits_data in data.items():
        n_qubits_list = sorted(qubits_data.keys())
        avg_times = [np.mean(qubits_data[n]) for n in n_qubits_list]
        plt.plot(n_qubits_list, avg_times, label=legend_key, marker='o')

    # Finalize plot
    plt.title("Average Iteration Time vs n_qubits", fontsize=16)
    plt.xlabel("n_qubits", fontsize=14)
    plt.ylabel("Average Iteration Time", fontsize=14)
    plt.legend(title="Parameters", fontsize=12)
    plt.savefig(f'{output_dir}/average_iteration_time_vs_n_qubits.png', bbox_inches='tight')
    plt.show()

def plot_average_D_vs_n_qubits():
    
    # Ensure the plots directory exists
    output_dir = f'{path_to_project_number}/Plots/Average_D_vs_n_qubits'
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    # Dictionary to store iteration times grouped by n_qubits and fixed parameters
    data = {}

    # Iterate over files in the HDF5 folder
    for file in os.listdir(path_to_HDF5):
        try:
            # Load attributes from the inputs file
            g = f_inputs[file[:-3]]
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items()}
            
            # Extract relevant parameters
            n_qubits = attributes_dict['n_qubits']
            scheme = attributes_dict['scheme']
            dt = attributes_dict['dt']
            initial_state_name = attributes_dict['initial_state_name']
            cutoff = attributes_dict['cutoff']
            
            # Skip specific initial_state_name
            if initial_state_name == "exp_tt" or scheme == "heun" or dt != 0.1:
                continue
            
            # Load the HDF5 file for iteration times
            f = h5py.File(f'{path_to_HDF5}/{file}', 'r')
            keys_to_plot = sorted([key for key in f.keys() if "D_" in key], key=lambda x: int(x.split("_")[1]))
            iteration_times = [np.float64(f[key]) for key in keys_to_plot]

            # Calculate average iteration time
            avg_iteration_time = np.mean(iteration_times)  # Skip the first value as per original code

            # Group data by n_qubits with fixed parameters as legend key
            legend_key = f"{scheme}, dt={dt}, cutoff={cutoff}, u0={initial_state_name}"
            if legend_key not in data:
                data[legend_key] = {}
            if n_qubits not in data[legend_key]:
                data[legend_key][n_qubits] = []
            data[legend_key][n_qubits].append(avg_iteration_time)
        
        except Exception as e:
            print(f"Failed to process file {file}: {e}")
            continue

    # Plot all data in a single figure
    plt.figure(figsize=(14, 12))
    for legend_key, qubits_data in data.items():
        n_qubits_list = sorted(qubits_data.keys())
        avg_times = [np.mean(qubits_data[n]) for n in n_qubits_list]
        plt.plot(n_qubits_list, avg_times, label=legend_key, marker='o')

    # Finalize plot
    plt.title("Average D vs n_qubits", fontsize=16)
    plt.xlabel("n_qubits", fontsize=14)
    plt.ylabel("Average D", fontsize=14)
    plt.legend(title="Parameters", fontsize=12)
    plt.savefig(f'{output_dir}/average_iteration_time_vs_n_qubits.png', bbox_inches='tight')
    plt.show()

def plot_average_D_vs_n_qubits():
    
    # Ensure the plots directory exists
    output_dir = f'{path_to_project_number}/Plots/Max_D_vs_n_qubits'
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    # Dictionary to store iteration times grouped by n_qubits and fixed parameters
    data = {}

    # Iterate over files in the HDF5 folder
    for file in os.listdir(path_to_HDF5):
        try:
            # Load attributes from the inputs file
            g = f_inputs[file[:-3]]
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items()}
            
            # Extract relevant parameters
            n_qubits = attributes_dict['n_qubits']
            scheme = attributes_dict['scheme']
            dt = attributes_dict['dt']
            initial_state_name = attributes_dict['initial_state_name']
            cutoff = attributes_dict['cutoff']
            
            # Skip specific initial_state_name
            if initial_state_name == "exp_tt" or scheme == "heun" or dt != 0.1:
                continue
            
            # Load the HDF5 file for iteration times
            f = h5py.File(f'{path_to_HDF5}/{file}', 'r')
            keys_to_plot = sorted([key for key in f.keys() if "D_" in key], key=lambda x: int(x.split("_")[1]))
            iteration_times = [np.float64(f[key]) for key in keys_to_plot]

            # Calculate average iteration time
            avg_iteration_time = np.max(iteration_times)  # Skip the first value as per original code

            # Group data by n_qubits with fixed parameters as legend key
            legend_key = f"{scheme}, dt={dt}, cutoff={cutoff}, u0={initial_state_name}"
            if legend_key not in data:
                data[legend_key] = {}
            if n_qubits not in data[legend_key]:
                data[legend_key][n_qubits] = []
            data[legend_key][n_qubits].append(avg_iteration_time)
        
        except Exception as e:
            print(f"Failed to process file {file}: {e}")
            continue

    # Plot all data in a single figure
    plt.figure(figsize=(14, 12))
    for legend_key, qubits_data in data.items():
        n_qubits_list = sorted(qubits_data.keys())
        avg_times = [np.mean(qubits_data[n]) for n in n_qubits_list]
        plt.plot(n_qubits_list, avg_times, label=legend_key, marker='o')

    # Finalize plot
    plt.title("Max D vs n_qubits", fontsize=16)
    plt.xlabel("n_qubits", fontsize=14)
    plt.ylabel("Max D", fontsize=14)
    plt.legend(title="Parameters", fontsize=12)
    plt.savefig(f'{output_dir}/average_iteration_time_vs_n_qubits.png', bbox_inches='tight')
    plt.show()

def plot_imag_part_vs_time():
    
    if not os.path.exists(f'{path_to_project_number}/Plots/Imag_part_vs_time'):
        os.makedirs(f'{path_to_project_number}/Plots/Imag_part_vs_time')
        
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    counter = 0

    for file in os.listdir(path_to_HDF5):
        
        try:
        
            g = f_inputs[file[:-3]]
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items()}
            iterations = attributes_dict['iterations']
            f = h5py.File(f'{path_to_HDF5}/{file}', 'r')
            
            states = []
            keys_to_plot = [element for element in f.keys() if "s_" in element]
            keys_to_plot = sorted(keys_to_plot, key=lambda x: int(x.split("_")[1]))
            for i in keys_to_plot:
                states.append(np.sum(np.imag(np.array(f[f"{i}"]))))
            
            # plt.figure(figsize=(14, 12))
            # plt.imshow(np.transpose(states), cmap = 'magma')
            # plt.xticks([])
            # plt.yticks([])
            plt.plot(states)
            plt.ylabel('Imag part vs time')
            plt.xlabel('Iteration number')
            plt.savefig(f'Plots/Imag_part_vs_time/{file[:-3]}.png', bbox_inches = 'tight')
            plt.close()
                            
        except:
            
            counter += 1
            
    if counter > 0:
        
        print(f'From plotting the heatmap time evolution {counter} jobs failed.')

def create_animation_gif():
    
    if not os.path.exists(f'{path_to_project_number}/Plots/Animation_gif'):
        os.makedirs(f'{path_to_project_number}/Plots/Animation_gif')
        
    path_to_HDF5 = f'{path_to_project_number}/HDF5'
    path_to_inputs = f'{path_to_project_number}/inputs.h5'
    f_inputs = h5py.File(path_to_inputs, 'r')

    counter = 0

    for file in os.listdir(path_to_HDF5):
        try:
            g = f_inputs[file[:-3]]
            attributes_dict = {attr_name: attr_value for attr_name, attr_value in g.attrs.items()}
            iterations = attributes_dict['iterations']
            f = h5py.File(f'{path_to_HDF5}/{file}', 'r')
            
            scheme = attributes_dict['scheme']
            dt = attributes_dict['dt']
            n = attributes_dict['n_qubits']
            initial_state_name = attributes_dict['initial_state_name']
            cutoff = attributes_dict['cutoff']
            nonlinear_coeff = attributes_dict['nonlinear_coeff']
            kx2_coeff = attributes_dict['kx2_coeff']
            kx4_coeff = attributes_dict['kx4_coeff']
            
            states = []
            keys_to_plot = [element for element in f.keys() if "s_" in element]
            keys_to_plot = sorted(keys_to_plot, key=lambda x: int(x.split("_")[1]))
            x_start = attributes_dict["x_start"] 
            x_end = attributes_dict["x_end"]
            N = 2**n
            dx = (x_end - x_start) / N
            x_values = x_start + dx * np.arange(N)
            t_values = np.arange(0, dt * len(keys_to_plot), dt)

            for key in keys_to_plot:
                states.append(np.abs(np.array(f[key])))
            
            states = np.array(states)  # Shape: (time_steps, space_points)

            # Create Animation
            fig, ax = plt.subplots(figsize=(10, 6))
            line, = ax.plot([], [], lw=2)
            ax.set_xlim(x_start, x_end)
            ax.set_ylim(np.min(states), np.max(states))
            ax.set_xlabel('Space')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{scheme}, dt={dt}, n={n}, u0={initial_state_name}\n'
                         f'cutoff={cutoff}, nonlinear_coeff={nonlinear_coeff}, kx2_coeff={kx2_coeff}, kx4_coeff={kx4_coeff}')
            
            def update(frame):
                line.set_data(x_values, states[frame])
                ax.set_title(f'Time: {t_values[frame]:.2f}, '
                             f'{scheme}, dt={dt}, n={n}, u0={initial_state_name}\n'
                             f'cutoff={cutoff}, nonlinear_coeff={nonlinear_coeff}, kx2_coeff={kx2_coeff}, kx4_coeff={kx4_coeff}')
                return line,

            ani = FuncAnimation(fig, update, frames=len(states), blit=True)
            gif_path = f'{path_to_project_number}/Plots/Animation_gif/{file[:-3]}.gif'
            ani.save(gif_path, writer=PillowWriter(fps=10))
            plt.close(fig)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            counter += 1
            
    if counter > 0:
        print(f'From creating animation GIFs, {counter} jobs failed.')    
        
# write_dag()

plot_heatmap_time_evolution()

# plot_real_vs_abs_comparison()

# plot_scheme_comparison()

# plot_error_vs_time()

# plot_iteration_time()

# plot_D_vs_time()

# plot_average_iteration_time()

# plot_average_D_vs_n_qubits()

# plot_imag_part_vs_time()

# create_animation_gif()
