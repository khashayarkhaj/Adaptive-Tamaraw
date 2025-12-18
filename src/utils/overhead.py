# functions for computing different kind of overheads
import numpy as np
import utils.config_utils as cm
from utils.trace_operations import find_last_slot_in_tam
from tqdm import tqdm
def total_data_overhead(traces_before, traces_after, trace_type = 'cell', return_percentage = True, verbose = False):
    """
    Calculate the total data (bandwith) overhead of our traces by comparing them before and after a defence is applied


    Parameters
    ----------
        trace_type: the type that our traces have. they can be cell, burst, or tam
        traces_before: the traces before they are defended. if type is cell or burst, this should only be the directions and not the timing.
        traces_after: the traces after they are defended. if type is cell or burst, this should only be the directions and not the timing.
        return_percentage: whether to return the ratio or percentage
        

    Returns
    -------
        data overhead incurred to our dataset, as mentioned in effective attacks and provable defenses, and also surakav:
        "total number of dummy packets divided by the total number of real packets over the whole dataset. "
    """
    total_bandwidth_undefended = 0
    total_bandwidth_defeneded = 0
    for idx in tqdm(range(len(traces_before)), desc= 'Computing the data overhead', disable= not verbose):
        trace_before = traces_before[idx]
        trace_after = traces_after[idx]

        if trace_type in ['cell', 'burst']:
            # whether we are in cell mode or burst mode, the abs values of the cells/bursts should be added up
            total_bandwidth_undefended += sum([abs(d) for d in trace_before])
            total_bandwidth_defeneded += sum([abs(d) for d in trace_after])
        elif trace_type == 'tam': # the tams will be 2 * n ndarrays
            total_bandwidth_undefended += np.abs(trace_before).sum()
            total_bandwidth_defeneded += np.abs(trace_after).sum()
    
    

    data_overhead = (total_bandwidth_defeneded - total_bandwidth_undefended)/ total_bandwidth_undefended
    

    if return_percentage:
        return data_overhead * 100
    else:
        return data_overhead

        

def total_time_overhead(traces_before, traces_after, trace_type = 'cell', return_percentage = True, verbose = False):
    """
    Calculate the total time overhead of our traces by comparing them before and after a defence is applied


    Parameters
    ----------
        trace_type: the type that our traces have. they can be cell, burst, or tam
        traces_before: the traces before they are defended. if type is cell or burst, this should only be the times and not the directions.
        traces_after: the traces after they are defended. if type is cell or burst, this should only be the times and not the directions.
        return_percentage: whether to return the ratio or percentage
        

    Returns
    -------
        time overhead incurred to our dataset, as mentioned in effective attacks and provable defenses, and also surakav:
        "total extra time divided by the total loading time in the undefended case over the whole dataset."
    """
    
    total_time_undefended = 0
    total_time_defeneded = 0
    for idx in tqdm(range(len(traces_before)), desc = 'Computing the time overhead', disable= not verbose):
        trace_before = traces_before[idx]
        trace_after = traces_after[idx]

        if trace_type in ['cell', 'burst']:
            # note that in this case trace_before and trace_after are the timings of each cell/burst
            total_time_undefended += trace_before[-1]
            total_time_defeneded += trace_after[-1]
        elif trace_type == 'tam': # the tams will be 2 * n ndarrays - for now, we will compare the last non zero timeslot 
            last_slot_index_undefended = find_last_slot_in_tam(trace_before)
            ending_time_before = (last_slot_index_undefended + 1) * cm.Time_Slot

            last_slot_index_defended = find_last_slot_in_tam(trace_after)
            ending_time_after = (last_slot_index_defended + 1) * cm.Time_Slot

            total_time_undefended += ending_time_before
            total_time_defeneded += ending_time_after


    
    time_overhead = (total_time_defeneded - total_time_undefended)/ total_time_undefended

    if return_percentage:
        return time_overhead * 100
    else:
        return time_overhead


def compute_overheads_tams(tam_undefended, tam_defended,  time_undefended = None, time_defended = None):
    # computing the overheads of two tams
    total_bandwidth_undefended = np.abs(tam_undefended).sum()
    total_bandwidth_defeneded = np.abs(tam_defended).sum()

    total_bandwidth_undefended_outgoing = np.abs(tam_undefended[0]).sum()
    total_bandwidth_defeneded_outgoing = np.abs(tam_defended[0]).sum()

    total_bandwidth_undefended_incomming = np.abs(tam_undefended[1]).sum()
    total_bandwidth_defeneded_incomming = np.abs(tam_defended[1]).sum()

    data_overhead = (total_bandwidth_defeneded - total_bandwidth_undefended)/ total_bandwidth_undefended

    data_overhead_outgoing = (total_bandwidth_defeneded_outgoing - total_bandwidth_undefended_outgoing)/ total_bandwidth_undefended

    data_overhead_incoming = (total_bandwidth_defeneded_incomming - total_bandwidth_undefended_incomming)/ total_bandwidth_undefended


    if time_undefended is None: # we need to approximate the end time in the tam
        last_slot_index_undefended = find_last_slot_in_tam(tam_undefended)
        ending_time_before = (last_slot_index_undefended + 1) * cm.Time_Slot

        last_slot_index_defended = find_last_slot_in_tam(tam_defended)
        ending_time_after = (last_slot_index_defended + 1) * cm.Time_Slot

        total_time_undefended = ending_time_before
        total_time_defeneded = ending_time_after
    else:
        total_time_undefended = time_undefended
        total_time_defeneded = time_defended
    time_overhead = (total_time_defeneded - total_time_undefended)/ total_time_undefended

    return data_overhead, data_overhead_outgoing, data_overhead_incoming, time_overhead



