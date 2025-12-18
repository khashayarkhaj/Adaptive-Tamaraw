"""
Implementation of multiple distance metrics, used to measure the difference between traces on a cell/burst level.
"""



import numpy as np
from .trace_operations import  compute_super_sequence, cell_inputs, scs
from tqdm import tqdm
import random
import utils.config_utils as cm

def cell_distance(cell1, cell2, use_abs = True):
    """
    Calculate the difference between two cells (values)

    Args:
        cel1, cell2 (np.ndarray): two cells (values), typically +1 and -1
        use_abs: whether to measure the absolute distance
    Returns:
        difference
    """
    difference = cell1 - cell2
    
    if use_abs:
        return abs(difference)
    return difference






@cell_inputs(flatten = True)
def dtw_distance(t1, t2):
    """
    Calculate the dynamic time warping distance matrix of two traces t1 and t2, mentioned in Glove.

    Args:
        t1, t2 (np.ndarray): traces which are at the cell level.

    Returns:
        alignemnt distance between the two traces
        dwt matrix (maybe later)
    """

   
 
    n = t1.shape[-1]
    m = t2.shape[-1]

    dwt_matrix = np.zeros([n + 1,m  +1])

    # initialization

    dwt_matrix[0,1 :] = np.inf
    dwt_matrix[1 :, 0] = np.inf
    #conceptually, we are measuring the cost of converting t2 to t1
    for i in range(1, n+1):
        for j in range(1, m+1):
            difference = cell_distance(t1[i - 1], t2[j - 1])
            previous_cost = min(dwt_matrix[i-1, j-1], dwt_matrix[i-1, j], dwt_matrix[i, j-1]) # i - 1 , j - 1 = match , i-1, j = insertion, i, j-1 = deletion

            dwt_matrix[i,j] = difference + previous_cost

            #TODO also save alignment strategy
    
    alignemnt_distance = dwt_matrix[-1,-1]

    return alignemnt_distance


# x = [7, -1, -2, 5, -10]
# y = [1, -8, 2, -4, 4, -2, 3]

# ret1, ret2 = dtw_distance(x, y, burst_mode = True)
# print(ret1)

@cell_inputs(flatten= True)
def molding_distance(t1, t2):
    """
    Calculate the cost of molding two traces, mentioned in Walkie-Talkie.

    this function first computes the molded trace of t1 and t2.
    then, it returns the cost of this procedure.
    #TODO define cost.

    Args:
        t1, t2 (np.ndarray): traces which are at the cell level

    Returns:
        molded trace (maybe later)
        cost
    """
    
    super_sequence= compute_super_sequence(t1, t2)

    # this part is a little bit tricky, because the cost can be different with respect to the shorter sequence or longer sequence.
    # based on effective attacks and provable defenses, we will use the formula 2 * |scs| - |p| - |q|

    return 2 * len(super_sequence) - len(t1) - len(t2)


@cell_inputs(flatten= True)
def dam_levenshtein_distance(t1, t2):
    """
    Calculate the Damerau-Levenstein distance between two traces, mentioned in Glove.

    #TODO define cost.

    Args:
        t1, t2 (np.ndarray): traces which are at the cell level.
        

    Returns:
        Damerau-Levenstein distance between two traces
    """
    n = t1.shape[-1]
    m = t2.shape[-1]

    lvs_matrix = np.zeros([n + 1,m  +1])

    lvs_matrix[0,1 :] = np.asarray([j for j in range(1, m + 1)])
    lvs_matrix[1 :, 0] = np.asarray([i for i in range(1, n + 1)])

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if t1[i - 1] == t2[j - 1] else 1
            lvs_matrix[i][j] = min(
                lvs_matrix[i - 1][j] + 1,  # deletion
                lvs_matrix[i][j - 1] + 1,  # insertion
                lvs_matrix[i - 1][j - 1] + cost,  # substitution
            )
            if i > 1 and j > 1 and t1[i - 1] == t2[j - 2] and t1[i - 2] == t2[j - 1]:
                lvs_matrix[i][j] = min(lvs_matrix[i][j], lvs_matrix[i - 2][j - 2] + cost)  # transposition

    return lvs_matrix[n][m]

@cell_inputs(flatten = True)
def scs_distance(t1, t2):
    """
    Calculate the similarity between two traces, based on their shortest common super sequence (effective attacks and provable defenses).


    Args:
        t1, t2 (np.ndarray): traces which are at the cell level
        

    Returns:
        scs distance between to traces
    """

    scs_trace = scs(t1, t2)

    # based on effective attacks and provable defenses, we will use the formula 2 * |scs| - |p| - |q|

    return 2 * len(scs_trace) - len(t1) - len(t2)


def tam_euclidian_distance(tam1, tam2) :
    """
    Calculate the euclidean distance between to tams (Traffic Aggregation Matrices)


    Parameters
    ----------
        tam1, tam2 (np.ndarray): traces which are at the tam level (shape [2,n])
        

    Returns
    -------
        eucliedean distance between two traces
    """


    # tam1_flattened = tam1.flatten()
    # tam2_flattened = tam2.flatten()

    # distance = np.sqrt(np.sum((tam1_flattened - tam2_flattened)**2))

    # I was doing the above lines first, but then changed it to the source code. don't think it makes any difference

    distance = np.linalg.norm(tam1 - tam2)

    #they have also used this somewhere else! : np.sqrt(np.sum(np.square(x - y)))
    return distance

def compute_distance_matrix_sequential():
    pass

def compute_distance_between_two_websites(distance_metric, repr1 = None, repr2 = None, sample_nums = None,
                                          dataset = None, website1 = None, website2 = None, random = None):
    """
    given two websites (class indexes) we want to measure (approximate) the distance between the traces of these two websites.
    if the representatives of these two websites are given, we will only compute the distance between these two traces.
    otherwise, we will take num_samples from the two websites, and compute their pairwise distance and return the average (simillar to pallete)


    Args:
        distance_metric: the distance_metric we will use
        repr1, repr2: the representatives of the two classes
        sample_nums: number of random samples we take from each class
        dataset: our dataset that comprises all traces
        website1, website2: the class indices of our desired websites
        random: if we want to choose random traces from each website, or choose based on the configs.
        

    Returns:
        distance between two classes
    """
    if repr1 is not None:
        return distance_metric(repr1, repr2)
    
    if random:
        traces1,_,_ = dataset.sample_randomly(sample_nums = sample_nums, class_num = website1 )
        traces2,_,_ = dataset.sample_randomly(sample_nums = sample_nums, class_num = website2 )
    
    else:
        traces1, _ = dataset.get_traces_of_class(class_number = website1)
        traces2, _ = dataset.get_traces_of_class(class_number = website2)
        # we expect the dataset to have the portion of the datat we want per class. so I commented the two lines below
        
        # traces1 = traces1[cm.MON_INST_START_IND: cm.MON_INST_START_IND + sample_nums]# for instance, we want the last 200 traces of this website. we set startind to 800 for tik tok
        # traces2 = traces2[cm.MON_INST_START_IND: cm.MON_INST_START_IND + sample_nums]

    
    distances = np.array([distance_metric(traces1[i], traces2[i]) for i in range(len(traces1))]) # pairwise distances
    
    
    return distances.mean()

   