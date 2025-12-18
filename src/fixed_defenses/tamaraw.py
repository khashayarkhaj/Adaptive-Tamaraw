# The Implementation of Tamaraw. mainly jiajun and Taos code


#Anoa consists of two components:
#1. Send packets at some packet rate until data is done.
#2. Pad to cover total transmission size.
#The main logic decides how to send the next packet. 
#Resultant anonymity is measured in ambiguity sizes.
#Resultant overhead is in size and time.
#Maximizing anonymity while minimizing overhead is what we want. 
import math
import random
import utils.config_utils as cm
from time import strftime
import argparse
import numpy as np
from utils.visualization_utils import visualize_tam
from utils.trace_dataset import TraceDataset
from utils.parser_utils import str2bool
from tqdm import tqdm
from utils.dataset_utils import to_integrated_cell_format, to_seperated_cell_format, to_tam
from utils.overhead import total_data_overhead, total_time_overhead
from multiprocessing import Pool
from functools import partial
import gc
from utils.file_operations import save_file, load_file
from experiments.clustering.clustering_utils import tamaraw_overhead_vector

import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
import copy

#params I commented them and turned them into arguments
# DATASIZE = 1
# DUMMYCODE = 1
# PadL = 50

# MON_SITE_NUM = 100
# MON_INST_NUM = 100
# UNMON_SITE_NUM = 1


#MON_SITE_NUM = 1
#MON_INST_NUM = 2
#UNMON_SITE_NUM = 0


# tardist = [[], []]
# defpackets = []
##    parameters = [100] #padL
##    AnoaPad(list2, lengths, times, parameters)

import sys
import os
# for x in sys.argv[2:]:
#    parameters.append(float(x))
#    print(parameters)

def fsign(num):
    if num > 0:
        return 0
    else:
        return 1

def rsign(num):
    if num == 0:
        return 1
    else:
        return abs(num)/num

def AnoaTime(parameters, ro_out = 0.04, ro_in = 0.012):
    """
    Determines timing intervals for packet transmission
    parameters[0]: direction (0=outgoing, 1=incoming)
    parameters[1]: method (currently only supports method 0)
    
    Returns:
    - ro_out (default 0.04s) for outgoing packets
    - ro_in (default 0.012s) for incoming packets
    """
    direction = parameters[0] #0 out, 1 in
    method = parameters[1]
    if (method == 0):
        if direction == 0:
            return ro_out
        if direction == 1:
            return ro_in  




        

def AnoaPad(list1, list2, padL, method, data_size = 1, dummy_packet_size = 1,
            ro_out = 0.04, ro_in = 0.012, randomized_extension = True, logger = None): # I guess list1 is like [[t1, d1], [t2, d2], ...]
    """
    Implements padding mechanism: ( I think this extends a given trace to reach a coefficient of L)
    1. Counts packets in each direction
    2. Adds dummy packets until reaching target length
    3. Uses geometric distribution for padding length
    """

    if logger is None:
        logger = cm.global_logger
    lengths = [0, 0]
    times = [0, 0]
    for x in list1:
        if (x[1] > 0):
            lengths[0] += 1
            times[0] = x[0]
        else:
            lengths[1] += 1
            times[1] = x[0]
        list2.append(x)

    paddings = []

    for j in range(0, 2):
        curtime = times[j]
        if randomized_extension:
            topad = -int(math.log(random.uniform(0.00001, 1), 2) - 1) #1/2 1, 1/4 2, 1/8 3, ... #check this - This number is between 1 nad 17
        else:
            topad = 1
        if (method == 0):
            if padL == 0:
                topad = 0
            else:
                topad = (lengths[j]//padL + topad) * padL # they don't pad the sequence to the next multiple of L, but a random multiple
            
        #logger.info("Need to pad to %d packets."%topad)
        while (lengths[j] < topad):
            curtime += AnoaTime([j, 0], ro_out= ro_out, ro_in= ro_in)
            if j == 0:
                paddings.append([curtime, dummy_packet_size * data_size])
            else:
                paddings.append([curtime, -dummy_packet_size* data_size])
            lengths[j] += 1
    paddings = sorted(paddings, key = lambda x: x[0]) # sorting based on time?
    list2.extend(paddings)

def Anoa(list1, list2, parameters, data_size = 1, ro_out = 0.04, ro_in = 0.012, second_ro_in = 0.012,
         second_ro_out = 0.04, dummy_packet = 1, switch_percentage = None): #inputpacket, outputpacket, parameters
    #Does NOT do padding, because ambiguity set analysis. 
    #list1 WILL be modified! if necessary rewrite to tempify list1.
    # normal packet number is assumed to be 1 (data_size)
    # second ros are given so that we switch two them once a specific percentage of the original trace is covered
    starttime = list1[0][0]
    times = [starttime, starttime] #lastpostime, lastnegtime
    curtime = starttime
    lengths = [0, 0]
    datasize = data_size
    method = 0
    if (method == 0):
        parameters[0] = "Constant packet rate: " + str(AnoaTime([0, 0], ro_out= ro_out, ro_in= ro_in)) + ", " + str(AnoaTime([1, 0], ro_out= ro_out, ro_in= ro_in)) + ". "
        parameters[0] += "Data size: " + str(datasize) + ". "
    if (method == 1):
        parameters[0] = "Time-split varying bandwidth, split by 0.1 seconds. "
        parameters[0] += "Tolerance: 2x."



    listind = 1 #marks the next packet to send
    switching_index = None
    if switch_percentage is not None:
        switching_index = int(len(list1) * switch_percentage)
        #print(f'switch index is {switching_index}')
        if switching_index == 0:
            ro_in = second_ro_in
            ro_out = second_ro_out
    
    while (listind < len(list1)):
        #decide which packet to send
        if times[0] + AnoaTime([0, method, times[0]-starttime], ro_out= ro_out, ro_in= ro_in) < times[1] + AnoaTime([1, method, times[1]-starttime], ro_out= ro_out, ro_in= ro_in):
            cursign = 0
        else:
            cursign = 1
        times[cursign] += AnoaTime([cursign, method, times[cursign]-starttime], ro_out= ro_out, ro_in= ro_in)
        curtime = times[cursign]
        
        tosend = datasize
        while (list1[listind][0] <= curtime and fsign(list1[listind][1]) == cursign and tosend > 0):
            if (tosend >= abs(list1[listind][1])):
                tosend -= abs(list1[listind][1])
                listind += 1
                # checking if we should change the params
                if switching_index is not None:
                    if listind >= switching_index and switching_index != 0:
                        ro_in = second_ro_in
                        ro_out = second_ro_out
            else:
                list1[listind][1] = (abs(list1[listind][1]) - tosend) * rsign(list1[listind][1])
                tosend = 0
            if (listind >= len(list1)):
                break
        if cursign == 0:
            list2.append([curtime, datasize])
        else:
            list2.append([curtime, -datasize])
        lengths[cursign] += 1
        


def perform_tamaraw_on_trace(
    trace,                    # Input trace to be morphed: list of [time, direction] or [direction, time] pairs
    cell_mode = 'dir_time',   # Format of input trace: 'dir_time' ([[d1, t1], [d2, t2], ...]) or 'time_dir' ([t1,d1], [t2,d2], ...])
    data_size = 1,            # Size of normal packets after morphing (constant size all packets will be normalized to)
    dummy_packet_size = 1,    # Size multiplier for dummy packets used in padding
    pad_length = 50,          # Base padding length (L) - traces will be extended to multiples of this value
    ro_out = 0.04,           # Time interval between outgoing packets (default 0.04 seconds)
    ro_in = 0.012,           # Time interval between incoming packets (default 0.012 seconds)
    randomized_extension = True, # If True, extends trace length to random multiple of L between 1-17
                                  # If False, extends to next immediate multiple of L
    logger = None,
    second_ro_in = 0.04,
    second_ro_out = 0.012,
    switch_percentage = None):  
    """
    Main function to apply Tamaraw defense to a network trace:
    1. Normalizes input format if needed (cell_mode handling)
    2. Applies constant-rate packet morphing (Anoa)
    3. Extends trace with padding (AnoaPad)
    4. Returns transformed trace in original format
    
    The function works in two phases:
    - First applies basic Tamaraw morphing (packet timing, sizes)
    - Then extends the trace length through padding
    """

    if trace is None or len(trace) == 0:
        return []
    if cell_mode == 'dir_time':
        # Convert from [direction, time] to [time, direction] format
        trace = [[cell[1], cell[0]] for cell in trace]

    tamaraw_trace = [trace[0]]  # Initialize with first packet
    parameters = [""]  # Required by Anoa function
    
    # Apply basic Tamaraw morphing (constant rates, normalized sizes)
    # Anoa might change the trace! so I pass a copy of my input trace to it
    Anoa(copy.deepcopy(trace), tamaraw_trace, parameters, data_size=data_size, ro_out=ro_out, ro_in=ro_in,
         second_ro_in= second_ro_in, second_ro_out= second_ro_out, switch_percentage= switch_percentage)
    
    tamaraw_trace = sorted(tamaraw_trace, key=lambda tamaraw_trace: tamaraw_trace[0])
    
    # Apply padding to extend trace length
    tamaraw_trace_extended = []  # Will contain final padded trace
    
    if switch_percentage is None or switch_percentage == 1:
        AnoaPad(tamaraw_trace, tamaraw_trace_extended, padL=pad_length, method=0, 
                data_size=data_size, dummy_packet_size=dummy_packet_size,
                ro_out=ro_out, ro_in=ro_in, randomized_extension= randomized_extension, logger= logger)
    else:
        AnoaPad(tamaraw_trace, tamaraw_trace_extended, padL=pad_length, method=0, 
                data_size=data_size, dummy_packet_size=dummy_packet_size,
                ro_out = second_ro_out, ro_in= second_ro_in, randomized_extension= randomized_extension, logger= logger)

    # Convert back to original format if needed
    if cell_mode == 'dir_time':
        tamaraw_trace_extended = [[cell[1], cell[0]] for cell in tamaraw_trace_extended]

    return tamaraw_trace_extended

def _process_single_trace(undefended_trace, ro_in, ro_out, L, random_extend, cell_mode, return_trace = False,
                          dummy_packet_size = 1, time_step_list = None):
    """Helper function to process a single trace with Tamaraw"""
    tamaraw_trace = perform_tamaraw_on_trace(
        trace=undefended_trace,
        pad_length=L,
        ro_in=ro_in,
        ro_out=ro_out,
        randomized_extension=random_extend,
        cell_mode=cell_mode,
        dummy_packet_size= dummy_packet_size
    )

    if time_step_list is None:
        bandwidth_undefended = sum([abs(d[1]) for d in undefended_trace])
        bandwidth_defeneded = sum([abs(d[1]) for d in tamaraw_trace])
        time_undefended = undefended_trace[-1][0]
        time_defeneded = tamaraw_trace[-1][0]
        

        if return_trace is False:
            del tamaraw_trace
            gc.collect()
            tamaraw_trace = None

        
        return tamaraw_trace, bandwidth_undefended, bandwidth_defeneded, time_undefended, time_defeneded
    else:
        # we need to know which packets are dummies (dummy_packet_size should be a different number)
        # we will iterate the undefended and defended traces. at each time step, we will compute how much overhead we have
        bandwidth_undefended, bandwidth_defeneded, time_undefended, time_defeneded = {}, {}, {}, {}
        for time_step in time_step_list:
            pass

def perform_tamaraw_on_a_dataset_parallel(undefended_traces, ro_in=0.012, 
                                        ro_out=0.04, L=50, random_extend=True, 
                                        cell_mode='time_dir', n_cores=None, logger = None, return_dataset = False,
                                        num_visualiztion = 0, save_dir = None):
    """
    Apply Tamaraw defense to a dataset using parallel processing when beneficial
    
    Args:
        undefended_traces: List of network traces
        ro_in: Time interval for incoming packets
        ro_out: Time interval for outgoing packets
        L: Padding length parameter
        random_extend: Whether to use random extension
        cell_mode: Format of trace data
        n_cores: Number of CPU cores to use (None = all available cores)
        return_dataset: wether we want to return the defended traces
        num_visualiztion: randomly visualize the tams of this count traces
        save_dir: dir for saving the visualizations
    """
    # If n_cores not specified, use all available cores minus 1
    if n_cores is None:
        n_cores = max(1, os.cpu_count() - 1)
    
    if logger is None:
        logger = cm.global_logger
    logger.info(f'tamaraw will be performed with {n_cores} cores in paralell')
    logger.info(f"Number of CPUs (multiprocessing.cpu_count()): {multiprocessing.cpu_count()}")
    
    total_bandwidth_undefended = 0
    total_bandwidth_defeneded = 0
    total_time_undefended = 0
    total_time_defeneded = 0
    defended_traces = []

    if save_dir is None:
        save_dir = os.path.join(cm.BASE_DIR, 'results', 'fixed_defenses', 'tamaraw', cm.data_set_folder, 'visualization', 'random_tams')
    # Use sequential processing if n_cores = 1

    # Presample the indexes to visualize

    visualization_indexes = random.sample( range(len(undefended_traces)), min(num_visualiztion, len(undefended_traces)))
    print('these samples will be visualized')
    print(visualization_indexes)
    if n_cores == 1:
        for trace_idx, undefended_trace in enumerate(tqdm(undefended_traces, desc='Performing Tamaraw (sequential)')):
            tamaraw_trace = perform_tamaraw_on_trace(
                trace=undefended_trace,
                pad_length=L,
                ro_in=ro_in,
                ro_out=ro_out,
                randomized_extension=random_extend,
                cell_mode=cell_mode
            )

            

            # Visualize if this index was selected
            if trace_idx in visualization_indexes:
                logger.info(f'The visualization of trace {trace_idx} will be saved')
                times, directions = zip(*[np.array(tamaraw_trace).T])
                # directions = directions[0]
                # times = times[0]
                tams = to_tam(directions= directions, times= times)
                for idx, tam in enumerate(tams):
                    visualize_tam(tams= [tam], trace_num= trace_idx, save_dir= save_dir)
            
            
            #computing overheads
            total_bandwidth_undefended += sum([abs(d[1]) for d in undefended_trace])
            total_bandwidth_defeneded += sum([abs(d[1]) for d in tamaraw_trace])
            total_time_undefended += undefended_trace[-1][0]
            total_time_defeneded += tamaraw_trace[-1][0]

            if return_dataset:
                defended_traces.append(tamaraw_trace)
            
            # else: # it seems this part actually slows down the process
            #     if trace_idx > 0 and trace_idx % 10000 == 0:
            #     # call gc every 10000 traces?
            #     # del tamaraw_trace
            #         gc.collect()
        
    else: # todo modify this part to return overheads instead of traces and also add visualization
        # Use parallel processing for n_cores > 1

        # If chunk_size not specified, calculate optimal chunk size
        chunk_size = max(1, len(undefended_traces) // (n_cores * 4))
        process_func = partial(_process_single_trace, 
                            ro_in=ro_in,
                            ro_out=ro_out,
                            L=L,
                            random_extend=random_extend,
                            cell_mode=cell_mode,
                            return_trace = return_dataset)
        
        with Pool(n_cores) as pool:
            results = list(tqdm(
                pool.imap(process_func, undefended_traces, chunksize= chunk_size),
                total=len(undefended_traces),
                desc=f'Performing Tamaraw (using {n_cores} cores)'
            ))

        # Unpack results
        if return_dataset:
            defended_traces = [r[0] for r in results] # i think this should be reordered
        total_bandwidth_undefended = sum(r[1] for r in results)
        total_bandwidth_defeneded = sum(r[2] for r in results)
        total_time_undefended = sum(r[3] for r in results)
        total_time_defeneded = sum(r[4] for r in results)
    
    
    


    bw_oh = (total_bandwidth_defeneded - total_bandwidth_undefended)/ total_bandwidth_undefended
    time_oh = (total_time_defeneded - total_time_undefended)/ total_time_undefended

    return defended_traces, bw_oh, time_oh

# Example usage:
# For sequential processing:
# defended_traces = perform_tamaraw_on_a_dataset_parallel(undefended_traces, n_cores=1)
#
# For parallel processing:
# defended_traces = perform_tamaraw_on_a_dataset_parallel(undefended_traces, n_cores=4)
#
# For automatic core selection:
# defended_traces = perform_tamaraw_on_a_dataset_parallel(undefended_traces)




def grid_search_tamaraw_params(undefended_traces, 
                             ro_in_range=np.arange(0.004, 0.024, 0.004),
                             ro_out_range=np.arange(0.02, 0.10, 0.02),
                             L=50, n_cores=None, random_extend = True, 
                             save_dir = None,
                             trace_oh_dict = None,
                             cluster_indices = None):
    """
    Perform grid search over ro_in and ro_out parameters to find optimal trade-offs
    
    Args:
        undefended_traces: List of network traces
        ro_in_range: Array of ro_in values to test
        ro_out_range: Array of ro_out values to test
        L: Padding length parameter
        n_cores: Number of CPU cores to use
        trace_oh_dict: in case the defense has been previously applied to the traces, use those results instead of applying it again
        cluster_indices: in case we want to use trace_oh_dict, we need the elements of the elements in the cluster
    
    Returns:
        DataFrame with results and plots
    """
    results = []
    
    # Test all combinations of parameters
    param_combinations = list(product(ro_in_range, ro_out_range))
    desired_inidces = cluster_indices
    if desired_inidces is None:
        desired_inidces = [i for i in range(len(undefended_traces))]
    combination_counter = 0
    for ro_in, ro_out in tqdm(param_combinations, desc="Testing parameter combinations"):
        # Run Tamaraw with current parameters
        if trace_oh_dict is None:
            defended_traces, bw_oh, time_oh = perform_tamaraw_on_a_dataset_parallel(
                undefended_traces=undefended_traces,
                ro_in=ro_in,
                ro_out=ro_out,
                L=L,
                n_cores=n_cores,
                random_extend= random_extend,
            )

            del defended_traces
            gc.collect()

        else:
            overhead_vector = tamaraw_overhead_vector(trace_oh_dict= trace_oh_dict,
                                                      trace_indices= desired_inidces,
                                                      num_combinations= len(param_combinations),
                                                      desired_combination= combination_counter)
            bw_oh = overhead_vector[0]
            time_oh = overhead_vector[1]

        results.append({
            'ro_in': ro_in,
            'ro_out': ro_out,
            'bandwidth_overhead': bw_oh,
            'time_overhead': time_oh
        })
        combination_counter += 1
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if save_dir is not None:
        save_file(content=results_df, file_name='all_params.csv', dir_path=save_dir)
    
    pareto_df = obtain_pareto_points(results_df= results_df)

    # Calculate normalized distance and find best parameters
    scaler = MinMaxScaler()
    normalized_metrics = scaler.fit_transform(pareto_df[['bandwidth_overhead', 'time_overhead']])
    pareto_df['normalized_distance'] = np.sqrt(normalized_metrics[:, 0]**2 + normalized_metrics[:, 1]**2)
    pareto_df_sorted = pareto_df.sort_values('normalized_distance')
    best_params = pareto_df_sorted.iloc[0]
    
    # First plot: All configurations and Pareto frontier
    plt.figure(figsize=(12, 8))
    
    # Scatter plot of all points
    plt.scatter(results_df['bandwidth_overhead'], 
               results_df['time_overhead'], 
               alpha=0.5, 
               label='All configurations')
    
    # Highlight Pareto frontier
    plt.scatter(pareto_df['bandwidth_overhead'], 
               pareto_df['time_overhead'], 
               color='red', 
               s=100, 
               label='Pareto frontier')
    
    # Highlight best parameters
    plt.scatter(best_params['bandwidth_overhead'],
               best_params['time_overhead'],
               color='gold',
               s=200,
               label='Best configuration',
               zorder=5,
               edgecolor='black',
               linewidth=2)
    
    # Add labels for Pareto points
    for _, point in pareto_df.iterrows():
        if point.equals(best_params):
            plt.annotate(f"BEST\nro_in={point['ro_in']:.3f}\nro_out={point['ro_out']:.3f}", 
                        (point['bandwidth_overhead'], point['time_overhead']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(facecolor='gold', alpha=0.5))
        else:
            plt.annotate(f"ro_in={point['ro_in']:.3f}\nro_out={point['ro_out']:.3f}", 
                        (point['bandwidth_overhead'], point['time_overhead']),
                        xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('Bandwidth Overhead (%)')
    plt.ylabel('Time Overhead (%)')
    plt.title('Tamaraw Parameter Optimization')
    plt.legend()
    plt.grid(True)
    
    if save_dir is None:
        plt.show()
    else:
        save_file(file_name='trade_offs.png', dir_path=save_dir)
    
    # Second plot: Curve plot
    plt.figure(figsize=(12, 8))
    
    # Sort Pareto points by bandwidth overhead for smooth curve
    pareto_sorted = pareto_df.sort_values('bandwidth_overhead')
    
    # Scatter plot of Pareto points
    plt.scatter(pareto_sorted['bandwidth_overhead'], 
                pareto_sorted['time_overhead'], 
                color='red', 
                s=100, 
                label='Pareto frontier points')
    
    # Highlight best parameters
    plt.scatter(best_params['bandwidth_overhead'],
               best_params['time_overhead'],
               color='gold',
               s=200,
               label='Best configuration',
               zorder=5,
               edgecolor='black',
               linewidth=2)
    
    # Add curve connecting Pareto points
    plt.plot(pareto_sorted['bandwidth_overhead'], 
             pareto_sorted['time_overhead'], 
             'r--', 
             linewidth=2, 
             label='Pareto frontier curve')
    
    # Add points labels
    for _, point in pareto_sorted.iterrows():
        if point.equals(best_params):
            plt.annotate(f"BEST\nro_in={point['ro_in']:.3f}\nro_out={point['ro_out']:.3f}", 
                        (point['bandwidth_overhead'], point['time_overhead']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(facecolor='gold', alpha=0.5))
        else:
            plt.annotate(f"ro_in={point['ro_in']:.3f}\nro_out={point['ro_out']:.3f}", 
                        (point['bandwidth_overhead'], point['time_overhead']),
                        xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('Bandwidth Overhead (%)')
    plt.ylabel('Time Overhead (%)')
    plt.title('Pareto Frontier Curve')
    plt.legend()
    plt.grid(True)
    
    if save_dir is None:
        plt.show()
    else:
        save_file(file_name='trade_offs_curve.png', dir_path=save_dir)
        
    if save_dir is not None:
        # Create a dictionary with the best parameters and their performance
        best_params_dict = {
            'best_ro_in': best_params['ro_in'],
            'best_ro_out': best_params['ro_out'],
            'bandwidth_overhead': best_params['bandwidth_overhead'],
            'time_overhead': best_params['time_overhead'],
            'normalized_distance': best_params['normalized_distance']
        }
        
        # Save as JSON
        save_file(dir_path=save_dir, file_name='best_params.json', content=best_params_dict)
    
    return results_df, pareto_df_sorted
    
def obtain_pareto_points(results_df = None, load_dir = None, file_name = None):
    # Find Pareto frontier
    pareto_points = []
    if load_dir is not None:
        results_df = load_file(dir_path= load_dir, file_name= file_name)
    
    for idx, row in results_df.iterrows():
        is_pareto = True
        for idx2, row2 in results_df.iterrows():
            if idx != idx2:
                if (row2['bandwidth_overhead'] <= row['bandwidth_overhead'] and 
                    row2['time_overhead'] <= row['time_overhead'] and
                    (row2['bandwidth_overhead'] < row['bandwidth_overhead'] or 
                     row2['time_overhead'] < row['time_overhead'])):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_points.append(row)
    
    pareto_df = pd.DataFrame(pareto_points)
    return pareto_df
    
    
# Example usage:
"""
# Define parameter ranges to test
ro_in_range = np.arange(0.004, 0.024, 0.004)  # From 0.004 to 0.02 in steps of 0.004
ro_out_range = np.arange(0.02, 0.10, 0.02)    # From 0.02 to 0.08 in steps of 0.02

# Run optimization
results_df, pareto_df = grid_search_tamaraw_params(
    undefended_traces=traces,
    ro_in_range=ro_in_range,
    ro_out_range=ro_out_range,
    L=50,
    n_cores=4
)

# Print top 3 balanced configurations
print("\nTop 3 balanced configurations:")
print(pareto_df[['ro_in', 'ro_out', 'bandwidth_overhead', 'time_overhead']].head(3))
"""



if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser(description='Visualizing Tams')
    parser.add_argument('-e',  '--extract_ds', type=str2bool, nargs='?', const=True, default=False, 
                         help='should we extract the dataset or is it already stored')
        
    parser.add_argument('-conf', '--config', help='which config file to use', default = 'Tik_Tok')

    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to save the computed elements')
    
    parser.add_argument('-preload', type = str2bool, nargs='?', const=True, default=False,
                         help='Whether to load the pre-computed elements')
    
    parser.add_argument('-vizualize_tams', type = int,
                    help='hom many defended traces to visualize', default= 10)
    
    parser.add_argument('-ro_out', type = float,
                         help='the p_out param for tamaraw', default = 0.04)
    
    parser.add_argument('-ro_in', type = float,
                         help='the p_in param for tamaraw', default = 0.012)
    
    parser.add_argument('-L', type = int,
                         help='the L param for tamaraw', default = 50)
    
    parser.add_argument('-n_cores', type = int,
                         help='number of cores to use for tamaraw', default = 1) # None means use all, 1 means don't do parallelism
    
    parser.add_argument('-random_extend', type = str2bool, nargs='?', const=True, default=True,
                         help='Whether to randomly extend the tamaraw trace to a factor of L')
    
    

    logger = cm.init_logger(name = 'Performing Tamaraw')
    args = parser.parse_args()

    cm.initialize_common_params(args.config)

    

    dataset = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'cell', keep_original_trace= True)
    undefended_traces = dataset.original_traces # each original trace is in the form [[t1, d1], [t2, d2] , ...]

    

    
    # performing tamaraw on every trace

    
    if not args.preload:
        
        defended_traces, bw_oh_percentage, time_oh_percentage = perform_tamaraw_on_a_dataset_parallel(undefended_traces= undefended_traces,
                                                                ro_in= args.ro_in,
                                                                ro_out = args.ro_out,
                                                                L = args.L,
                                                                random_extend= args.random_extend,
                                                                cell_mode= 'time_dir',
                                                                n_cores = args.n_cores,
                                                                logger = logger,
                                                                num_visualiztion= args.vizualize_tams)

       
    

# python3 -m fixed_defenses.tamaraw 
        

