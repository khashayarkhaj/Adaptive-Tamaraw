# ths file contains different functions that facilitate working with different datasets.
import os
import glob
import pickle
import utils.config_utils as cm
import math
import numpy as np
from tqdm import tqdm
from utils.file_operations import load_file
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import h5py

def load_DF18(split = 'train', logger = None):
    """
    loads the DF18 undefended traces.
    Args:
         split: Can be one of train, val, test, indicating the specific file we want to extract.
    Returns:
        the directions (list of ndarrays) and their labels. note the directions are at the cell level.
    """

    if not logger:
        logger = cm.global_logger
    
    logger.info('Loading the pkl file containing the traces')
    data_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, cm.dataset_category)

    if cm.compute_canada:
        relative_path = os.path.relpath(data_dir, cm.BASE_DIR) # part of the dir_path that doesn't have BASE_DIR
        data_dir = os.path.join(cm.compute_canada_datadir, relative_path)
    if split not in ['train', 'val', 'test']:
        raise ValueError("split must be 'train', 'val', or 'test'")

    # Use glob to find the traces and labels files dynamically
    traces_files = glob.glob(os.path.join(data_dir, f'*X*{split}*.pkl'))
    labels_files = glob.glob(os.path.join(data_dir, f'*y*{split}*.pkl'))

    if not traces_files or not labels_files:
        print(f'this address was not found: {data_dir}')
        raise FileNotFoundError(f"No matching files found for split '{split}'")

    # Assuming the first match is the correct file (make sure this assumption is valid)
    traces_path = traces_files[0]
    labels_path = labels_files[0]

    traces_directory_path = os.path.dirname(traces_path) #This function returns the directory part of the file path. 
    traces_file_name = os.path.basename(traces_path) #This function returns the base name (i.e., the file name) of the file path. 
    directions = load_file(dir_path= traces_directory_path,
                           file_name= traces_file_name, encoding = 'latin1')
    

    labels_directory_path = os.path.dirname(labels_path) 
    labels_file_name = os.path.basename(labels_path) 
    labels = load_file(dir_path= labels_directory_path, file_name= labels_file_name)
   


    logger.info('loading complete')
    
    return directions, labels



def to_dt(directions, times, time_zero = 1e-6, padding_length = None, verbose = False):
    """
    Returns the directional timing (dt) for a given trace, similar to tik tok

    Parameters
    ----------
    directions: list of traces in form of +1 -1.

    times: the corresponding timing of each trace

    Returns
    -------
    list of traces each in dt format. each of the traces will be a list of numbers (nparray), each number is direction * time
    """
    number_of_traces = len(directions)
    dts = []
    number_of_all_zeros = 0 # number of traces that result in an all zero array
    for i in tqdm(range(number_of_traces), desc= 'Extracting DTs', disable= not verbose):
        dirs = directions[i]
        time = times[i]
        assert len(dirs) == len(time), f'trace number {i} has {len(dir)} directions but {len(time)} timings!'
        dt = [dirs[idx] * time[idx] for idx in range(len(dirs))]
        # the first timestamp is 0, so we will lose the first packet!
        # I checked and they do this in tik tok source code as well:
        # sequence = [sequence[0][i]*sequence[1][i] for i in range(len(sequence[0]))] 
        # update: in holmes, they have 1e-6 as the start time, so I will do the same
        # TODO is this correct? or should time be inter packet time?
        dt[0] = dirs[0] * time_zero
        
        
        if not np.count_nonzero(dt) > 0:
            number_of_all_zeros += 1
        
        if padding_length is not None:
            if len(dt) > padding_length:
                dt = dt[:padding_length]
            elif len(dt)< padding_length:
                difference = padding_length - len(dt)
                dt += [0 for diff in range(difference)]
        dt = np.array(dt)
        dts.append(dt) 

    if verbose:
        percentage_of_all_zeros = (number_of_all_zeros/number_of_traces) * 100
        print(f'Number of traces that are all zero is {number_of_all_zeros}, which is {percentage_of_all_zeros:.2f}% of the dataset')
    return dts



def to_tam(directions, times, verbose = False):

    
    """
    Produces the traffic aggregation matrix (tam).

    Parameters
    ----------
         directions: list of traces in form of +1 -1.

         times: the corresponding timing of each trace
    Returns
    -------
        tams: ndarray with shape of [num_traces,2,N], containing all the tams.
    """
    if not cm.Max_tam_matrix_len:
        number_of_slots = math.ceil(cm.Maximum_Load_Time/ cm.Time_Slot)
    else:
        number_of_slots = cm.Max_tam_matrix_len
    number_of_traces = len(directions)
    tams = np.zeros([number_of_traces, 2, number_of_slots])
    number_of_traces_exceeding = 0
    for i in tqdm(range(number_of_traces), desc= 'Extracting TAMs', disable= not verbose):
        trace = directions[i]
        time = times[i]

        for t, d in zip(time, trace):
            if t > cm.Maximum_Load_Time: 
                #break 
                # # Initially, I thought we should end the process if we exceed the max load time
                # but looking at the RF code (packets_per_slot.py), you can see that they add the exceeding packets to the last time slot
                # this is not consistent with their paper. on page 611 in the second column, they mention that they discard the packet in this case.
                time_slot = -1
            else:
                time_slot = math.floor(t / cm.Time_Slot) # note that this should be floor, not ceil, because our first bin is zero
                if time_slot >= cm.Max_tam_matrix_len:
                    time_slot = -1
            if d > 0: # initially, this was if d == 1. but apparently, if the traces have positive or negative size, we just add one to their bin 
                tams[i,0,time_slot] += 1 #outgoing
            elif d < 0:
                tams[i,1,time_slot] += 1 #incomming
        if time[-1] > cm.Maximum_Load_Time:
            number_of_traces_exceeding += 1
        
    if verbose:
        print(f'number of traces exceeding {cm.Maximum_Load_Time}s is: {number_of_traces_exceeding}/{number_of_traces}')
    return tams


def extract_cells():
    pass

def load_extracted_cells():
    pass

def convert_ndarray_to_list(data):
    # util function for converting ndarray to list
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert ndarray to list
    else:
        return data  # Return as is if it's not an ndarray

def ensure_numpy_array(data):
    # util function for converting list to nparray
    if not isinstance(data, np.ndarray):
        return np.array(data)
    return data
    


# the next three functions are from the early stage attack paper, and are used to construct tafs
# the original functions can be found https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/blob/master/WFlib/tools/data_processor.py

def fast_count_burst(arr):
    """
    Returns the burst sizes of a given sequence

    Parameters
    ----------
    arr : np.array, one dimensional array containing the directions 1 -1 

    Returns
    -------
    an 1 dimensional np.array containing the burst size of each burst with the sign of its direction (like [2, -5, 3, -7,...])
    """

    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0] #indices - 1 where 1 turns to -1 or vice verca - np.nonzero gives the indices of nonzero elements
    segment_starts = np.insert(change_indices + 1, 0, 0) # start of each burst np.insert inserts the element to that specific indice
    segment_ends = np.append(change_indices, len(arr) - 1) # end of each burst np.append adds to the end
    segment_lengths = segment_ends - segment_starts + 1 # length of each burst
    segment_signs = np.sign(arr[segment_starts]) # sign of each burst
    adjusted_lengths = segment_lengths * segment_signs

    return adjusted_lengths

def agg_interval(packets):
    """
    Returns the aggregated features of a given list of packets

    Parameters
    ----------
    packets : np.array, one dimensional array containing the packets in dt format

    Returns
    -------
    an list containing three lists, each having two features:
    - the number of incoming and outgoing packets. 
    - the number of incoming and outgoing bursts. 
    - the average size of incoming and outgoing bursts
    """
    packets = ensure_numpy_array(packets)
    features = []
    features.append([np.sum(packets>0), np.sum(packets<0)])

    dirs = np.sign(packets)
    assert not np.any(dir == 0), "Array contains zero!"
    bursts = fast_count_burst(dirs)
    features.append([np.sum(bursts>0), np.sum(bursts<0)])

    pos_bursts = bursts[bursts>0]
    neg_bursts = np.abs(bursts[bursts<0])
    vals = []
    if len(pos_bursts) == 0:
        vals.append(0)
    else:
        vals.append(np.mean(pos_bursts))
    if len(neg_bursts) == 0:
        vals.append(0)
    else:
        vals.append(np.mean(neg_bursts))
    features.append(vals)

    return np.array(features, dtype=np.float32)

def process_TAF(index, trace):
    """
    Returns the TAF of a given trace

    Parameters
    ----------
    trace : np.array, one dimensional array containing the packets in dt format
    index: index of the trace in our list of traces

    Returns
    -------
    The TAF of the given trace
    """
    if not cm.Max_tam_matrix_len:
        max_len = math.ceil(cm.Maximum_Load_Time/ cm.Time_Slot)
    else:
        max_len = cm.Max_tam_matrix_len

    packets = np.trim_zeros(trace, "fb")
    abs_packets = np.abs(packets)
    
    TAF = np.zeros((3, 2, max_len))

    if len(abs_packets) > 0: # some traces turned out to be empty after trimming!
        st_time = abs_packets[0]
        st_pos = 0
        for interval_idx in range(max_len):
            ed_time = (interval_idx + 1) * cm.Time_Slot
            if interval_idx == max_len - 1:
                ed_pos = abs_packets.shape[0]
            else:
                ed_pos = np.searchsorted(abs_packets, st_time + ed_time)

            assert ed_pos >= st_pos, f"{index}: st:{st_pos} -> ed:{ed_pos}"
            if st_pos < ed_pos:
                cur_packets = packets[st_pos:ed_pos]
                TAF[:, :, interval_idx] = agg_interval(cur_packets)
            st_pos = ed_pos
    
    return TAF

def to_taf(directions, times):
    
    """
    Produces the traffic aggregation features (taf) for a given set of traces

    Parameters
    ----------
         directions: list of traces in form of +1 -1.
         
         times: the corresponding timing of each trace
    Returns
    -------
        tafs: ndarray with shape of [num_traces,3, 2, N], containing all the tafs.
    """

    if not cm.Max_tam_matrix_len:
        number_of_slots = math.ceil(cm.Maximum_Load_Time/ cm.Time_Slot)
    else:
        number_of_slots = cm.Max_tam_matrix_len
    number_of_traces = len(directions)
    tafs = np.zeros([number_of_traces, 3, 2, number_of_slots])
    dt_traces = to_dt(directions= directions, times= times)
    for i in tqdm(range(number_of_traces), desc= 'Extracting TAFs'):
        
        dt_trace = dt_traces[i]

        tafs[i] = process_TAF(index = i, trace= dt_trace)
    
    return tafs

def to_seperated_cell_format(traces):
    # given a list of traces, each like [[d1, t1], [d2, t2], ...], convert the trace to a list of directions and times

    directions = []
    times = []

    for trace in tqdm(traces, desc= 'seperating the cells to time and directions'):
        directions_in_this_trace = [cell[0] for cell in trace]
        timings_in_this_trace = [cell[1] for cell in trace]

        directions.append(directions_in_this_trace)
        times.append(timings_in_this_trace)

    return directions, times

def to_integrated_cell_format(directions, times):
    # given a set of traces, each having a direction list and timestamp list, we want to convert them to a single array [[d1, t1], [d2, t2], ...]

    traces = []

    for i in tqdm(range(len(directions)), desc= 'converting directions and timings into one trace'):
        directions_of_this_trace = directions[i]
        timings_of_this_trace = times[i]
        trace = [[directions_of_this_trace[cell], timings_of_this_trace[cell]] for cell in range(len(directions_of_this_trace))]

        traces.append(trace)
        del directions_of_this_trace
        del timings_of_this_trace
        gc.collect()
    
    return traces
    




def analyze_class_distribution(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Analyze and visualize class distribution across train, validation and test sets.
    
    Parameters:
    -----------
    X_train, X_val, X_test : array-like
        Feature sets
    y_train, y_val, y_test : array-like
        Target variables containing class labels
    
    Returns:
    --------
    dict
        Dictionary containing class distribution statistics
    """
    # Get class counts for each dataset
    train_counts = pd.Series(y_train).value_counts().sort_index()
    val_counts = pd.Series(y_val).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    # Calculate percentages
    train_percentages = (train_counts / len(y_train) * 100).round(2)
    val_percentages = (val_counts / len(y_val) * 100).round(2)
    test_percentages = (test_counts / len(y_test) * 100).round(2)
    
    # Create a DataFrame for easy comparison
    distribution_df = pd.DataFrame({
        'Train Count': train_counts,
        'Train %': train_percentages,
        'Val Count': val_counts,
        'Val %': val_percentages,
        'Test Count': test_counts,
        'Test %': test_percentages
    })
    
    # Calculate basic statistics
    stats = {
        'total_samples': {
            'train': len(y_train),
            'val': len(y_val),
            'test': len(y_test)
        },
        'n_classes': len(np.unique(y_train)),
        'class_distribution': distribution_df
    }
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar plot
    x = np.arange(len(train_counts))
    width = 0.25
    
    plt.bar(x - width, train_counts, width, label='Train')
    plt.bar(x, val_counts, width, label='Validation')
    plt.bar(x + width, test_counts, width, label='Test')
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution Across Datasets')
    plt.xticks(x, train_counts.index)
    plt.legend()
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Total samples - Train: {stats['total_samples']['train']}, "
          f"Validation: {stats['total_samples']['val']}, "
          f"Test: {stats['total_samples']['test']}")
    print(f"\nNumber of classes: {stats['n_classes']}")
    print("\nClass Distribution:")
    print(distribution_df)
    
    plt.show()
    
    return stats
def replace_dummies(trace, dummy_code=888, real_code=1):
    """
    Replace dummy codes with real codes in a trace.
    
    :param trace: List of [timestamp, direction] pairs
    :param dummy_code: Code indicating dummy packets (default: 888)
    :param real_code: Code to replace dummy packets with (default: 1)
    :return: New trace with dummy codes replaced
    """
    def sign(x):
        return (x > 0) - (x < 0)
    # Create a new list to avoid modifying the original trace
    new_trace = []
    
    # Iterate through each [timestamp, direction] pair
    for timestamp, direction in trace:
        # If the direction matches dummy_code, replace it with real_code
        if abs(direction) == abs(dummy_code):
            new_trace.append([timestamp, sign(direction) * real_code])
        else:
            new_trace.append([timestamp, direction])
            
    return new_trace

def last_packet_stats(trace, dummy_code=888, real_code=1):
    """
    Find indices of the last dummy and real packets in the (defended) trace.
    
    :param trace: List of [timestamp, direction] pairs
    :param dummy_code: Code indicating dummy packets (default: 888)
    :param real_code: Code indicating real packets (default: 1)
    :return: Tuple of (last_dummy_outgoing_idx, last_dummy_incoming_idx, 
                       last_real_outgoing_idx, last_real_incoming_idx)
    """
    last_dummy_outgoing_idx = -1
    last_dummy_incoming_idx = -1
    last_real_outgoing_idx = -1
    last_real_incoming_idx = -1
    
    for i, (_, direction) in enumerate(trace):
        # Determine if it's a dummy or real packet based on the code
        packet_type = dummy_code if abs(direction) == dummy_code else real_code
        
        if packet_type == dummy_code:
            if direction > 0:
                last_dummy_outgoing_idx = i
            elif direction < 0:
                last_dummy_incoming_idx = i
        elif packet_type == real_code:
            if direction > 0:
                last_real_outgoing_idx = i
            elif direction < 0:
                last_real_incoming_idx = i
    
    return (last_dummy_outgoing_idx, 
            last_dummy_incoming_idx, 
            last_real_outgoing_idx, 
            last_real_incoming_idx)



def efficient_store_traces(traces, trim_length=None):
    """
    Efficiently store variable-length traces using two arrays:
    1. A data array containing all values concatenated
    2. A metadata array containing the start indices and lengths
    
    Parameters:
    traces: list of lists of (time, direction) tuples
    trim_length: if provided, trim= each trace to this length before storing (no padding because we want to keep all traces)
    
    Returns:
    data_array: ndarray containing all times and directions concatenated
    metadata: ndarray containing (start_idx, length) for each trace
    """
    
    # If trim_length is provided, process traces first
    if trim_length is not None:
        processed_traces = []
        
        for trace in tqdm(traces, desc=f'trimming/padding traces to length {trim_length}'):
            # Convert to numpy array for consistent handling
            if len(trace) == 0:
                # Handle empty traces
                processed_trace = np.zeros((trim_length, 2))
            else:
                trace_array = np.array(trace)
                
                if len(trace_array) > trim_length:
                    # Trim excess length
                    processed_trace = trace_array[:trim_length]
                # elif len(trace_array) < trim_length:
                    
                    # Pad shorter traces
                    # pad_length = trim_length - len(trace_array)
                    # last_time = trace_array[-1, 0] if len(trace_array) > 0 else 0
                    
                    # # Create padding array with repeated last time and zero directions
                    # padding = np.full((pad_length, 2), [last_time, 0])
                    # processed_trace = np.vstack([trace_array, padding])
                else:
                    # Length is exactly trim_length
                    processed_trace = trace_array
                
            processed_traces.append(processed_trace)
        
        
        del traces
        gc.collect()
        traces = processed_traces
    
    # Calculate total length and create metadata array
    lengths = [len(trace) for trace in traces]
    total_length = sum(lengths)
    
    # Create metadata array with start indices and lengths
    start_indices = np.cumsum([0] + lengths[:-1])
    metadata = np.column_stack((start_indices, lengths))
    
    # Pre-allocate the data array
    data_array = np.zeros((total_length, 2))  # 2 columns for time and direction
    
    # Fill the data array
    current_idx = 0
    for trace in tqdm(traces, desc='efficiently storing traces'):
        if len(trace) > 0:  # Check length instead of truthiness
            # Ensure trace is a numpy array
            if not isinstance(trace, np.ndarray):
                trace_array = np.array(trace)
            else:
                trace_array = trace
                
            length = len(trace_array)
            data_array[current_idx:current_idx + length] = trace_array
            current_idx += length
            
    return data_array, metadata





def reconstruct_efficient(data_array, metadata):
    """
    Optimized reconstruction of traces from the efficient storage format.
    Uses array views and minimizes list operations for better performance.
    
    Parameters:
    data_array: ndarray containing all times and directions concatenated
    metadata: ndarray containing (start_idx, length) for each trace
    
    Returns:
    times: ndarray of time arrays (using views)
    directions: ndarray of direction arrays (using views)
    """
    n_traces = len(metadata)
    
    # Create arrays to store views
    times = np.zeros(n_traces, dtype=object)
    directions = np.zeros(n_traces, dtype=object)
    
    # Create views for each trace
    for i, (start_idx, length) in enumerate(tqdm(metadata.astype(int), desc='reconstrcting traces efficiently')):
        # Get views of the data
        trace_data = data_array[start_idx:start_idx + length]
        times[i] = trace_data[:, 0]  # view of times
        directions[i] = trace_data[:, 1]  # view of directions
    
    return times, directions


import random

def get_removed_websites(remove_websites, remove_seed, site_start_ind, site_end_ind):
    """
    Generate a list of website indices to remove.
    
    Args:
        remove_websites (int): Number of websites to remove
        remove_seed (int): Seed for random selection
        site_start_ind (int): Starting index of websites
        site_end_ind (int): Ending index of websites (exclusive)
        
    Returns:
        list: List of website indices to remove
    """
    if remove_websites <= 0:
        return []
    
    # Get total number of websites
    total_sites = site_end_ind - site_start_ind
    
    # Set random seed for reproducibility
    random.seed(remove_seed)
    
    # Get random indices (capped at total number of sites)
    num_to_remove = min(remove_websites, total_sites)
    removed_indices = random.sample(range(site_start_ind, site_end_ind), num_to_remove)
    
    return sorted(removed_indices)