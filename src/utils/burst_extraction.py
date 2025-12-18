# a set of functions that help in converting a trace file, to burst format.
import numpy as np
import os
from os.path import join
from .config_utils import *
import utils
import argparse
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
from .trace_operations import load_single_trace, truncate_trace
from .file_operations import extract_label

def get_burst(trace, fdir):

    """
    Return the burst format of a given trace

    Args:
        trace is an np array with form of [[t1,d1], [t2,d2],...]
        fdir is the path to that trace. it is just used for logging purposes
        logger for logging the details
    Returns:
        merged_burst_seqs: burst sequence, nparray in the form of [[t1,b1], [t2,b2],...], where ti is the starting time of bi. Also, I think bi has + - direction.
        original_trace_size: number packets in the trace before cutting
        modified_trace_size: number of packets in the trace after cutting
    """
   
    original_trace_size = len(trace)
    # truncating the trace to remove the outliers and first packets that are incomming (we don't want the trace to start with -1)
    trace = truncate_trace(trace= trace, fdir = fdir)
    modified_trace_size = len(trace)
    burst_seqs = trace

    # merge bursts from the same direction
    merged_burst_seqs = []
    cnt = 0
    sign = np.sign(burst_seqs[0, 1])
    time = burst_seqs[0, 0]
    for cur_time, cur_size in burst_seqs:
        if np.sign(cur_size) == sign:
            cnt += cur_size
        else:
            merged_burst_seqs.append([time, cnt])
            sign = np.sign(cur_size)
            cnt = cur_size
            time = cur_time
    merged_burst_seqs.append([time, cnt])
    merged_burst_seqs = np.array(merged_burst_seqs)
    assert sum(merged_burst_seqs[::2, 1]) == sum(trace[trace[:, 1] > 0][:, 1])
    assert sum(merged_burst_seqs[1::2, 1]) == sum(trace[trace[:, 1] < 0][:, 1])
    merged_burst_seqs = np.array(merged_burst_seqs)
    return merged_burst_seqs, original_trace_size, modified_trace_size

def extract(trace, fdir):
    """
    Returns the burst array and timing array of a given trace

    Args:
        trace is an np array with form of [[t1,d1], [t2,d2],...]
        fdir is the path to that trace. it is just used for logging purposes
    Returns:
        bursts: nparray containing sizs of incomming, outcomming bursts: [b1, b2, b3, ...]
            all numbers are positive
            ** first element (index 0) is the burst length
            odd indices (1,3,...) are outgoing bursts, incomming bursts are even
            the size of the array has been truncated to trace_length in configs
            if normalize_bursts (config) is true, the burts sizes will be normalized with CELL_SIZE (config). this happens if the traces are not +1 -1, and are the actual sizes.
        times: nparray containing start time of original bursts [t1, t2, t3, ...]. this is not truncated
        original_burst_size: original length of burst before transforming to trace_length
        original_trace_size: number packets in the trace before cutting
        modified_trace_size: number of packets in the trace after cutting
    """
    
    burst_seq, original_trace_size, modified_trace_size = get_burst(trace, fdir)
    times = burst_seq[:, 0]
    bursts = abs(burst_seq[:, 1])
    if normalize_bursts:
        bursts /= CELL_SIZE
    bursts = list(bursts)
    bursts.insert(0, len(bursts))
    original_burst_size = len(bursts)
    bursts = bursts[:trace_length] + [0] * (trace_length - len(bursts))
    assert len(bursts) == trace_length
    return bursts, times, original_burst_size, original_trace_size, modified_trace_size

def extractfeature(fdir):
    """
    loads the trace as an np array with form of [[t1,d1], [t2,d2],...] and calls extract on that trace to get the burst information, and also adds a label of the cell
    
    Args:
        fdir is the path to that trace
    Returns:
        bursts: nparray containing sizs of incomming, outcomming bursts: [b1, b2, b3, ...]
            all numbers are positive
            ** first element (index 0) is the burst length
            odd indices (1,3,...) are outgoing bursts, incomming bursts are even
            the size of the array has been truncated to trace_length in configs
            if normalize_bursts (config) is true, the burts sizes will be normalized with CELL_SIZE (config). this happens if the traces are not +1 -1, and are the actual sizes.
        times: nparray containing start time of original bursts [t1, t2, t3, ...]. this is not truncated
        labels: true label of this burst
        original_burst_size: original length of burst before transforming to trace_length
        original_trace_size: number packets in the trace before cutting
        modified_trace_size: number of packets in the trace after cutting
    """
    
    
    trace, label = load_single_trace(fdir)
    bursts, times, original_burst_size, original_trace_size, modified_trace_size = extract(trace, fdir)
    return bursts, times, label, original_burst_size, original_trace_size, modified_trace_size

def parallel_burst_extraction(flist, n_jobs=70):
    """
    gets te list of trace paths, and obtoins their burst information using multiprocessing
    
    Args:
        flist: list containing paths to traces
    Returns:
    res:
        a list of elements, each consisting of the burst info of a given trace. the burst info for each trace are:
        bursts: nparray containing sizs of incomming, outcomming bursts: [b1, b2, b3, ...]
            all numbers are positive
            ** first element (index 0) is the burst length
            odd indices (1,3,...) are outgoing bursts, incomming bursts are even
            the size of the array has been truncated to trace_length in configs
            if normalize_bursts (config) is true, the burts sizes will be normalized with CELL_SIZE (config). this happens if the traces are not +1 -1, and are the actual sizes.
        times: nparray containing start time of original bursts [t1, t2, t3, ...]. this is not truncated
        labels: true label of this burst
        original_burst_size: original length of burst before transforming to trace_length
        original_trace_size: number packets in the trace before cutting
        modified_trace_size: number of packets in the trace after cutting
    """
    with mp.Pool(n_jobs) as p:
        res = p.map(extractfeature, flist)
        p.close()
        p.join() # I think it was not necessary to add this line
    return res


