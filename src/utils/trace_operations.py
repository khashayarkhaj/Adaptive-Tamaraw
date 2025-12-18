import numpy as np
import utils.config_utils as cm
from .mathematical_ops import sign
import pandas as pd
import multiprocessing as mp
from .file_operations import extract_label
from tqdm import tqdm


def flatten_trace(t):
    """
    takes a trace with a shape like [1,l] or [l] and returns the trace with shape [l]
    Args:
        t(np.ndarray): traces which is at the cell level.

    Returns:
        flattened trace
    """
    t = np.asarray(t) # change t to np.ndarray if it is a list

    if t.ndim == 2:
        try:  
            if t.shape[0] == 1:
                t = t.flatten()
            else:
                raise ValueError(f"Expected the first dimension of trace to be 1, but the shape of the trace is {t.shape}")
        except ValueError as e:
            cm.global_logger.error(f"Error processing array: {e}")
            raise  # Optionally re-raise the exception if you want it to propagate
    return t

def flatten_inputs(func):
    """
    decorator that flattens the input traces
    """
    def wrapper(*args):
        #args should contain traces
        #flatenning the traces if they are not flattened
        flattened_traces = [flatten_trace(t) for t in args]
        

        return func(*flattened_traces)
    return wrapper




def to_cell_format(*traces):
    """
    takes list of traces, each in burst format, and returns their corresponding trace in cell format.
    Args:
        *traces: a list of traces each in burst format

    Returns:
        list of traces in cell format
    """
    cell_traces = []

    for burst_sequence in traces:
        cell_sequence = []
        for burst in burst_sequence:
            burst_size = sign(burst) * burst # to handle negative and positive numbers
            cell_sequence.extend([sign(burst)] * burst_size)

        cell_traces.append(np.asarray(cell_sequence))
     
    return cell_traces

def to_burst_format(*traces):
    """
    takes list of traces, each in cell format, and returns their corresponding trace in burst format.
    Args:
        *traces: a list of traces each in cell format

    Returns:
        list of traces in cell format
    """
    burst_traces = []

    for cell_sequence in traces:
        burst_sequence = []
        cell_cnt = 0
        while cell_cnt < len(cell_sequence):
            current_direction = sign(cell_sequence[cell_cnt])
            current_burst_size = 0
            while cell_cnt < len(cell_sequence) and sign(cell_sequence[cell_cnt]) == current_direction:
                cell_cnt += 1
                current_burst_size += 1
            
            burst_sequence.append(current_direction * current_burst_size)
    
        burst_traces.append(np.asarray(burst_sequence))
     
    return burst_traces



def cell_inputs(flatten = True):
    """
    decorator that converts input traces to cell format
    if flatten = true, the traces are flattened in the beginning
    """
    def decorator(func):
        def wrapper(*traces, burst_mode = False):
            #args should contain traces
            #flatenning the traces if they are not flattened
            if flatten:
                traces = [flatten_trace(trace) for trace in traces]

            if burst_mode:
                traces = to_burst_format(*traces)
            return func(*traces)
            
        return wrapper
    return decorator

@cell_inputs(flatten = True)
def compute_super_sequence(t1, t2):
    """
    takes traces t1 and t2, and returns their supersequence, as mentioned in Walkie-Talkie
    Args:
        t1, t2 (np.ndarray): traces which are at the cell level

    Returns:
        supersequence of both traces
    """


    n = t1.shape[-1]
    m = t2.shape[-1]

    super_sequence = []
    t1_cursor = 0
    t2_cursor = 0
    
    while t1_cursor < n and t2_cursor < m:
        
        direction = sign(t1[t1_cursor]) # +1 outgoing, -1 incoming

        # burst in t1
        start_of_burst1 = t1_cursor
        end_of_burst1 = t1_cursor
        while(end_of_burst1 < n and sign(t1[end_of_burst1]) == direction ):
            end_of_burst1 += 1
        burst_size_1 = end_of_burst1 - start_of_burst1

        #burst in t2
        start_of_burst2 = t2_cursor
        end_of_burst2 = t2_cursor
        while(end_of_burst2 < m and sign(t2[end_of_burst2]) == direction):
            end_of_burst2 += 1
        burst_size_2 = end_of_burst2 - start_of_burst2


        smaller_burst = min(burst_size_1, burst_size_2)
        bigger_burst = max(burst_size_1, burst_size_2)

        #adding the max to the supersequence
        super_sequence += [direction for i in range(bigger_burst)]


        t1_cursor = end_of_burst1
        t2_cursor = end_of_burst2


   
    
    if t1_cursor != n:
        super_sequence += t1[t1_cursor : ].tolist()
        #TODO cost should be clarified
    elif t2_cursor!= m:
        super_sequence += t2[t2_cursor : ].tolist()
    
    return np.asarray(super_sequence)

# t1 = [1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1]
# t2 = [1,1,-1,-1,1,1,1,1,-1,-1]
# super_sequence  = compute_super_sequence(t1, t2)
# print(super_sequence)

@cell_inputs(flatten = True)
def scs(t1, t2):
    """
    takes traces t1 and t2, and returns their shortest common supersequence, as mentioned in effective attacks and provable defenses.
    The goal is to check if this is the same when we are molding the two sequences.
    Args:
        t1, t2 (np.ndarray): traces which are at the cell level
        
    Returns:
        shortest common supersequence of both traces
    """
    n = len(t1)
    m = len(t2)
    def lcs(seq1, seq2):
        # takes to sequences, and computes their longest common subsequence

        n = len(seq1)
        m = len(seq2)

        dp = np.zeros([n + 1, m + 1])
        operations = np.zeros([n + 1, m + 1])

        # if the last elements are the same, then one option is two extend the lsc fro i-1,j-1
        # else, the only options are the lsc in i-1,j and i, j-1
        for i in tqdm(range(1, n + 1), desc= 'lcs computation'):
            for j in range(1, m + 1):
                length_adding_both = np.NINF
                if seq1[i - 1] == seq2[j - 1]:
                    length_adding_both = dp[i-1][j-1] + 1
                
                chosen_operation, max_length = max(enumerate([dp[i-1,j], dp[i, j-1], length_adding_both]), key= lambda x: x[1])
                dp[i][j] = max_length
                operations[i][j] = chosen_operation # 0: i-1,j 1: i, j-1, 2: i-1, j-1
        
        # getting the lsc
        lsc_list = []
        seq1_index = n 
        seq2_index = m 
        while seq1_index != 0 and seq2_index != 0:
            operation = operations[seq1_index][seq2_index]
            if operation == 2:
                lsc_list.append(seq1[seq1_index - 1]) 
                seq1_index -= 1
                seq2_index -= 1
            elif operation == 1:
                seq2_index -= 1
            else:
                seq1_index -= 1
            
        
        lsc_list.reverse()

        return lsc_list, dp
    
    lcs_list, dp = lcs(t1, t2)
    
    seq1_index = n 
    seq2_index = m 
    scs_list = []
    while seq1_index > 0 and seq2_index > 0:

        if t1[seq1_index - 1] == t2[seq2_index - 1]:
            scs_list.append(t1[seq1_index - 1])
            seq1_index -= 1
            seq2_index -= 1
        elif dp[seq1_index - 1][seq2_index] > dp[seq1_index][seq2_index - 1]:
            scs_list.append(t1[seq1_index - 1])
            seq1_index -= 1
        else:
            scs_list.append(t2[seq2_index - 1])
            seq2_index -= 1
          
    # Add remaining trace if any    

    while seq1_index > 0:
        scs_list.append(t1[seq1_index - 1])
        seq1_index -= 1

    while seq2_index:
        scs_list.append(t2[seq2_index - 1])
        seq2_index -= 1
    
    scs_list.reverse()

    scs_list = np.asarray(scs_list)

    return scs_list



            

def load_single_trace(fdir, return_label = True):
    """
    loads a trace from a given address
    Args:
        fdir: path to the trace file.
        return_label: whether to return the label as well
    Returns:
        trace, an np array with form of [[t1,d1], [t2,d2],...], label, which is the website number
    """

    
    with open(fdir, 'r') as f:
        tmp = f.readlines()

    # Try to detect the delimiter
    if tmp and '\t' in tmp[0]:
        delimiter = '\t'
    else:
        delimiter = None  # This will split on any whitespace

    trace = np.array(pd.Series(tmp).str.strip().str.split(delimiter, expand=True).astype(float))
    
    # .slice(0, -1): This method slices each string in the Series from the start (index 0) to the second-last character (index -1).
    #  str.split() method is used to split each string in a Series or DataFrame by a specified delimiter, turning each string into a list of substrings. 
    #print('loading trace with path ',fdir)
    if cm.normalize_bursts:
        # in case we feed in defended traces that
        # use +-888 to represent a dummy packet
        # I guess for datasets like Tik Tok this will be necessary
        trace[:, 1] = np.sign(trace[:, 1])

    #truncating the trace and removing the outliers
    trace = truncate_trace(trace)
    if cm.TIME_THRESHOLD_TRACE: # remove the packets that exceed our timing threshold
        mask = trace[:, 0] <= cm.TIME_THRESHOLD_TRACE
        trace = trace[mask]
    if return_label:
        label = extract_label(fdir)
        return trace, label
    else:
        return trace

def load_all_traces(flist, n_jobs = 10, time_threshold = None):
    """
    loads a all traces given a list of addresses, using multiprocessing
    Args:
        flist: list of paths to trace files.
        
    Returns:
        a list of (trace, label), each trace, an np array with form of [[t1,d1], [t2,d2],...]
    """
    
    if n_jobs == 1:
        # Sequential execution with tqdm
        results = []
        for trace_file in tqdm(flist, desc='loading the traces from the list of address'):
            results.append(load_single_trace(trace_file))
    else:
        # Parallel execution with multiprocessing
        with mp.Pool(n_jobs) as p:
            results = list(tqdm(p.imap(load_single_trace, flist), total=len(flist), desc='loading the traces from the list of address'))
    

    return results



def truncate_trace(trace, fdir = None):
    """
    cuts of the begining or end of the trace if it has outliers greater than cut_off or starts with incomming packets.
    Args:
        trace : an ndarray with form of [[t1,d1], [t2,d2],...]
        fdir: the adress to the cell, used for debugging
    Returns:
        a truncated trace with the form of [[t1,d1], [t2,d2],...]
    """
    # first check whether there are some outlier pkts based on the CUT_OFF_THRESHOLD
    # If the outlier index is within 50, cut off the head
    # else, cut off the tail
    start, end = 0, len(trace)
    ipt_burst = np.diff(trace[:, 0]) # this gives an array [diff1 diff2 diff3 ...] where diffi = ti+1 - ti
    ipt_outlier_inds = np.where(ipt_burst > cm.CUT_OFF_THRESHOLD)[0]
    # returns the indices where the time difference is greater than threshold. note that [0] still gives you the entire list
    
    original_length = len(trace)
    outliers = []
    
    
    if len(ipt_outlier_inds) > 0:
        outlier_ind_first = ipt_outlier_inds[0]
        if outlier_ind_first < 50:
            start = outlier_ind_first + 1
            outliers.append(outlier_ind_first)
        outlier_ind_last = ipt_outlier_inds[-1]
        if outlier_ind_last > 50:
            end = outlier_ind_last + 1
            outliers.append(outlier_ind_last)

    if (start != 0 or end != len(trace)) and fdir:
        print("File {} with length {} had outliers {} in trace has been truncated from {} to {}".format(fdir,original_length, outliers, start, end))

    trace = trace[start:end].copy()
    
    # remove the first few lines that are incoming packets
    start = -1
    for time, size in trace:
        start += 1
        if size > 0:
            break
    
    trace = trace[start:].copy()
    trace[:, 0] -= trace[0, 0]
    assert trace[0, 0] == 0

    
    return trace

                
def glove_cost_function(*traces, **kwargs):
    """
    takes list of traces, each in cell format, and returns the cost function described in glove: |ci|(max(reqj) + max(resj))
    Args:
        * traces: a list of traces each in cell format
        ** kwargs: if max_requests, max_responses, and indices are given, the compution will be conducted faster.
    Returns:
        list of traces in cell format
    """

    if kwargs:

        assert ('max_requests' in kwargs), 'max_requests argument required'
        assert ('max_responses' in kwargs), 'max_responses argument required'
        assert ('indices' in kwargs), 'max_responses argument required'
        max_requests = kwargs['max_requests']
        max_responses = kwargs['max_responses']
        indices = kwargs['indices']

        
    #for glove cost function
    else:
        max_requests = {i : np.sum(traces[i] == 1) for i in range(len(traces))}
        max_responses = {i : np.sum(traces[i] == -1) for i in range(len(traces))}
        indices = range(len(traces))
    
    max_index_request = max(indices, key = max_requests.get)
    max_index_response = max(indices, key = max_responses.get)
    return len(indices) * (max_requests[max_index_request] + max_responses[max_index_response])


def compute_super_matrix(*tams):
    """
    Calculate the super matrix of a list of given tams (Traffic Aggregation Matrices), as defined in pallete


    Parameters
    ----------
        tams (np.ndarray): traces which are at the tam level (shape [2,n])
        

    Returns
    -------
        Super Matrix of given traces
    """

    
    #super_matrix = np.maximum.reduce(tams)

    super_matrix = np.max(tams, axis= 0) # I used to the above line, but they do this in the source code. don't think it makes a difference
    return super_matrix

def find_last_slot_in_tam(tam):
    # given a tam, finds the last time slot that has a packet (non zero entry)
    non_zero_columns = (tam != 0).any(axis=0)
    last_non_zero_col_index = np.max(np.where(non_zero_columns)[0])
    return last_non_zero_col_index

import numpy as np

def count_spikes(tam, threshold=5.0):
    """
    Count spikes in the second row of a 2xn array and return their indices.
    A spike occurs when the value at index i represents a significant increase
    compared to the previous value, determined by the threshold parameter.
    
    Parameters:
    tam (numpy.ndarray): 2xn array where n is the number of columns
    threshold (float): Minimum difference required to consider a change as a spike
    
    Returns:
    tuple: (spike_count, spike_indices)
        - spike_count (int): Number of spikes found in the second row
        - spike_indices (list): Indices where spikes were found
    """
    if tam.shape[0] != 2:
        raise ValueError("Array must have exactly 2 rows")
    if tam.shape[1] < 2:  # Now we only need 2 points minimum
        return 0, []
        
    row = tam[1]  # Get second row
    spike_count = 0
    spike_indices = []
    
    # Check each point except the first
    for i in range(1, len(row)):
        # Calculate difference with previous element
        diff_prev = row[i] - row[i-1]
        
        # Check if difference exceeds the threshold
        if diff_prev > threshold:
            spike_count += 1
            spike_indices.append(i)
            
    return spike_count, spike_indices