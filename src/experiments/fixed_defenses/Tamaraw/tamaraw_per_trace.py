# perofrm tamaraw with different configs on each trace and save the results for each trace
# finding the best parameters of tamaraw for a given dataset
import utils.config_utils as cm
import argparse
import numpy as np

from utils.trace_dataset import TraceDataset
from utils.parser_utils import str2bool
from tqdm import tqdm
import os
from itertools import product
from fixed_defenses.tamaraw import perform_tamaraw_on_a_dataset_parallel
import gc
from utils.file_operations import save_file


def partition_traces(start_trace, end_trace, num_partitions):
    """
    Divide traces between start_trace and end_trace into equal partitions.
    
    Parameters:
    -----------
    start_trace : int
        Starting index
    end_trace : int
        Ending index
    num_partitions : int
        Number of desired partitions
        
    Returns:
    --------
    partition_indices : list of tuples
        List of (start_idx, end_idx) for each partition
    """
    total_traces = end_trace - start_trace
    base_size = total_traces // num_partitions
    
    partition_indices = []
    current_idx = start_trace
    
    for i in range(num_partitions):
        # For all partitions except the last one
        if i < num_partitions - 1:
            next_idx = current_idx + base_size
        # For the last partition, go up to end_trace
        else:
            next_idx = end_trace
            
        partition_indices.append((current_idx, next_idx))
        current_idx = next_idx
        
    return partition_indices
if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser(description='Visualizing Tams')

    
    parser.add_argument('-e',  '--extract_ds', type=str2bool, nargs='?', const=True, default=False, 
                         help='should we extract the dataset or is it already stored')
        
    parser.add_argument('-conf', '--config', help='which config file to use', default = 'Tik_Tok')

    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to save the computed elements')
    
    
    
    
    
    
    
    parser.add_argument('-n_cores', type = int,
                         help='number of cores to use for tamaraw', default = 1) # None means use all, 1 means don't do parallelism
    
    
    parser.add_argument('-partitions', type = int,
                         help='since the computation will be heavy we will do it for partitions of the traces', default = 1)
    parser.add_argument('-start_trace', type = int , default= 0 , help = 'for perfromance reasons, I might want to start from a different trace rather than 0')
    parser.add_argument('-end_trace', type = int , default= None , help = 'for perfromance reasons, I might want to end in a different trace rather than the last')
    

    logger = cm.init_logger(name = 'Performing Tamaraw')
    args = parser.parse_args()

    cm.initialize_common_params(args.config)

    

    dataset = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'cell', keep_original_trace= True)
    undefended_traces = dataset.original_traces # each original trace is in the form [[t1, d1], [t2, d2] , ...]

    


    # do what tao does
    rho_in_initial = 0.012
    rho_out_initial = 0.04
    # n ranges from -10 to +10 (inclusive), giving us 11 values
    n_values = np.arange(-7, 7)  # [-10, -9, ..., 9, 10]

    # Calculate the exponential factor for each n
    factors = np.power(7, n_values / 7)  # factor = 10^(n/10)

    # Generate the rho_in and rho_out values
    ro_in_range = rho_in_initial * factors
    ro_out_range = rho_out_initial * factors

    start_trace = args.start_trace
    end_trace = args.end_trace

    if end_trace is None:
        end_trace = len(undefended_traces)
    

    
    

    

    

    Ls = [100, 500, 1000]


    for L in Ls:
        print(f'Computing Traces for L = {L} and start_trace = {start_trace}')
        
        save_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'tamaraw_params', f'L_{L}')
        param_combinations = list(product(ro_in_range, ro_out_range))
        combination_counter = 0
        trace_oh_dict = {} # dictionary, key is a trace idx. value is a dictionary. in the second dictionary, the key is index of (rho_in, rho_out) in param combinations, and value is a list: [len(defended), time(defended)]
        
        for trace_number in range(start_trace, end_trace):
            trace_oh_dict[trace_number] = {}
            trace_oh_dict[trace_number]['bw'] = len(undefended_traces[trace_number])
            trace_oh_dict[trace_number]['time'] = undefended_traces[trace_number][-1][0]
        
        partition_indices = partition_traces(start_trace, end_trace, args.partitions)


        for ro_in, ro_out in tqdm(param_combinations, desc="Testing parameter combinations"):
            # Run Tamaraw with current parameters

            for partition_num, (start_idx, end_idx) in enumerate(partition_indices):
                print(f'partition {partition_num + 1}/{len(partition_indices)}, indexes {start_idx} to {end_idx}')
                current_partition = undefended_traces[start_idx:end_idx]

                defended_traces, bw_oh, time_oh = perform_tamaraw_on_a_dataset_parallel(
                    undefended_traces=current_partition,
                    ro_in=ro_in,
                    ro_out=ro_out,
                    L= L,
                    n_cores=args.n_cores,
                    random_extend= False,
                    return_dataset= True
                )

            
                for trace_number in range(start_idx, end_idx):
                    
                    actual_idx = trace_number - start_idx
                    trace_oh_dict[trace_number][combination_counter] = [len(defended_traces[actual_idx]), defended_traces[actual_idx][-1][0]] # added length and added time

                del defended_traces
                gc.collect()
            
            combination_counter += 1

            
        
        if args.end_trace is not None:
            save_file(dir_path= save_dir, content= trace_oh_dict, file_name= f'trace_params_{start_trace}_{end_trace}.json')
        else:
            save_file(dir_path= save_dir, content= trace_oh_dict, file_name= f'trace_params.json')


    


# python3 -m experiments.fixed_defenses.Tamaraw.tamaraw_per_trace -n_cores 1