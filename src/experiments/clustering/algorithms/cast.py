# An implementation of the CAST (Cluster Affinity Search Technique) algorithm for clustering time series based on similarity in time

import numpy as np
from utils.distance_metrics import tam_euclidian_distance
from utils.trace_operations import compute_super_matrix
from tqdm import tqdm
import utils.config_utils as cm
from utils.trace_dataset import TraceDataset
import os
from utils.file_operations import load_file, save_file
from ..clustering_utils import cluster_size_distribution
import copy
# from dtaidistance import similarity

def z_normalize(tams):
    """ Normalize each row of a (2, n) ndarray for a list of time series data. """
    # TODO maybe normalize based on the same time slots in different tams, rather than different timeslots in the same tam?
    normalized_tams = []
    for ts in tams:
        # ts.shape should be (2, n)
        normalized_ts = np.zeros_like(ts, dtype=float)
        for i in range(ts.shape[0]):  # Assuming ts has two rows (2, n)
            mean = np.mean(ts[i, :])
            std = np.std(ts[i, :])
            normalized_ts[i, :] = (ts[i, :] - mean) / std if std != 0 else ts[i, :]
        normalized_tams.append(normalized_ts)
    return normalized_tams

def compute_distance_matrix(tams):
    """ Compute the distance matrix for the given tams. """
    num_series = len(tams)
    distance_matrix = np.zeros((num_series, num_series))
    for i in tqdm(range(num_series), desc= 'Computing distance matrix for CAST'):
        for j in range(num_series):
            if i == j:
                continue
            elif i > j:
                distance_matrix[i,j] = distance_matrix[j, i]
            else:
                distance_matrix[i][j] = tam_euclidian_distance(tams[i], tams[j])
                
             
    return distance_matrix

def distance_to_similarity(distance_matrix, method='exponential', alpha=None, power_of_2 =False, rescaling_parameter = 7):
    """
    Convert a distance matrix to a similarity matrix using either simple inverse or exponential decay.

    Parameters:
    - distance_matrix (np.array): A square matrix of distances.
    - method (str): 'inverse' for simple inverse, 'exponential' for exponential decay.
    - alpha (float, optional): The decay rate for the exponential method. If None, alpha is set to 1/std of distances.
    Returns:
    - np.array: A matrix of similarities.
    """
    if method not in ['inverse', 'exponential', 'maximum', 'gaussian', 'self_scaling']:
        raise ValueError("Method must be 'inverse', 'maximum', 'gaussian', or 'exponential'.")

    
    # if power_of_2:
    #         distance_matrix = distance_matrix ** 2
    
    if alpha is None:
        # Calculate alpha as std of all distances (excluding diagonal where distances are zero)
        std = np.std(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])
        alpha = std if std != 0 else 1  # Prevent division by zero

        # mean = np.mean(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])
        # alpha = mean if mean != 0 else 1 # Prevent division by zero
        

    if method == 'inverse':
        # Avoid division by zero by adding a small number to distances
        similarity_matrix = 1 / (1 + distance_matrix)
    elif method == 'maximum':
        max_value = np.max(distance_matrix)  # Find the maximum value in the array
        normalized_matrix = distance_matrix / max_value  # Divide each element by the maximum value
        similarity_matrix = 1 - normalized_matrix
    elif method == 'exponential':
        similarity_matrix = np.exp(- distance_matrix/alpha)
    elif method == 'gaussian':
        alpha = alpha ** 2
        distance_matrix == distance_matrix ** 2
        similarity_matrix = np.exp(- distance_matrix/alpha)
    elif method == 'self_scaling':
        similarity_matrix = self_scale_similarity_matrix(distance_matrix= distance_matrix, K = rescaling_parameter)
    #print(similarity_matrix[:5,:5])
    return similarity_matrix

def self_scale_similarity_matrix(distance_matrix, K = 7):
    #self scaling the sigma in based on the distance matrix. paper: Self-Tuning Spectral Clustering
    # Number of data points
    n = distance_matrix.shape[0]
    
    # Step 1: Determine local scales
    # Sort each row and take the K-th nearest distance (K+1 because the smallest distance is to itself and is zero)
    sorted_distances = np.sort(distance_matrix, axis=1)
    local_scales = sorted_distances[:, K+1]
    
    # Ensure no scale is zero to avoid division by zero in the next step
    local_scales[local_scales == 0] = 1e-10
    
    # Step 2: Calculate the similarity matrix
    # We use an outer product to get σ_i * σ_j for each pair (i, j)
    scale_matrix = np.outer(local_scales, local_scales)
    
    # Compute the similarity matrix using the Gaussian kernel
    similarity_matrix = np.exp(-distance_matrix**2 / scale_matrix)
    
    return similarity_matrix

    
    

def update_threshold(base_threshold, C_open, rate =0.005):
    return min(base_threshold + rate * len(C_open), 1)

def dynamic_threshold(similarity_matrix, threshold = 0.5):
    # used for e-cast
    n = similarity_matrix.shape[0]
    total_similarity = 0
    count = 0
    
    # Only iterate over the upper triangle, excluding the diagonal
    for i in range(n):
        for j in range(i + 1, n):  # j starts from i+1 to exclude the diagonal and ensure each pair is counted once
            if similarity_matrix[i, j] >= threshold:
                total_similarity += (similarity_matrix[i, j] - threshold)
                count += 1

    if count == 0:
        return threshold  # Default threshold if no high similarities
    return (total_similarity / count) + threshold

def self_compute_affinity_threshold(similarity_matrix):
    # Extract the upper triangular part of the matrix, excluding the diagonal
    upper_triangular = np.triu(similarity_matrix, k=1)  # k=1 starts above the diagonal

    # Sum the elements of the upper triangular matrix
    sum_similarity = np.sum(upper_triangular)

    # Count the number of elements in the upper triangular part (which should be included)
    count = (len(similarity_matrix) * (len(similarity_matrix) - 1)) / 2

    # Compute the threshold T
    threshold_t = sum_similarity / count

    


    return threshold_t


import numpy as np

def calculate_expansion_ratios(similarity_matrix, clusters):
    """
    Calculate the expansion ratio for each cluster, excluding self-similarity in the degree calculation.
    This function does not modify the original similarity matrix provided. This funcion might be used for post processing, as mentioned
    in "A post-processing methodology for robust spectral embedded clustering of power networks"
    
    Args:
    similarity_matrix (np.ndarray): A square matrix where element (i, j) represents
                                    the weight of the edge between node i and node j, excluding self-links.
    clusters (list of lists): Each sublist contains indices of nodes that form a cluster.
    
    Returns:
    list: Expansion ratios for each cluster.
    """
    expansion_ratios = []
    n = similarity_matrix.shape[0]

    # Create a copy of the similarity matrix and set diagonal to zero
    matrix_copy = similarity_matrix.copy()
    np.fill_diagonal(matrix_copy, 0)  # Set diagonal to zero in the copy

    # Calculate weighted degree for each node (sum of the row in the similarity matrix copy, without diagonal values)
    weighted_degrees = np.sum(matrix_copy, axis=1)
    
    # Iterate over each cluster to calculate their expansion ratio
    for cluster in clusters:
        # Calculate volume of the cluster
        volume = np.sum(weighted_degrees[cluster])

        # Calculate cut of the cluster
        mask = np.ones(n, dtype=bool)  # Create a mask for nodes outside the cluster
        mask[cluster] = False
        cut = np.sum(matrix_copy[cluster, :][:, mask])
        
        # Calculate expansion ratio
        if volume > 0:
            expansion_ratio = cut / volume
        else:
            expansion_ratio = float('inf')  # Handle division by zero if volume is zero
        
        expansion_ratios.append(expansion_ratio)
    
    return expansion_ratios

def perform_post_processing(similarity_matrix, clusters, max_num_of_clusters):
    """
    perform post processing of the clusters to have more balanced and less clusters. inspired from:
    "A post-processing methodology for robust spectral embedded clustering of power networks"
    
    Args:
    similarity_matrix (np.ndarray): A square matrix where element (i, j) represents
                                    the weight of the edge between node i and node j, excluding self-links.
    clusters (list of lists): Each sublist contains indices of nodes that form a cluster.
    
    max_num_of_clusters: maximum number of allowed clusters
    Returns:
    list: Expansion ratios for each cluster.
    """
    if len(clusters) <= max_num_of_clusters:
        return clusters
    required_iterations = len(clusters) - max_num_of_clusters
    
    for iteration in tqdm(range(required_iterations), desc='Performing post processing on the clusters'):

        
        #finding cluster with minimum number of elements
        min_length = float('inf')  # Initialize with infinity to ensure any real list length is smaller
        min_index = -1  # Initialize with -1 to indicate no index found yet

        for index, cluster in enumerate(clusters):
            if len(cluster) < min_length:
                min_length = len(cluster)
                min_index = index
        
        smallest_cluster = clusters.pop(min_index) #removing the smallest cluster for adding it to other clusters
        lowest_max_expansion_ratio = float('inf')
        min_index = -1
        for idx,cluster in enumerate(clusters): # adding the smallest cluster to other clusters and checking which one gives the lowest maximum expansion rate
            clusters_copy = copy.deepcopy(clusters)
            
            clusters_copy[idx] += smallest_cluster
            
            
            expansion_ratios = calculate_expansion_ratios(similarity_matrix= similarity_matrix,
                                                          clusters= clusters_copy)
            
            max_expansion_ratio = max(expansion_ratios)

            if max_expansion_ratio < lowest_max_expansion_ratio:
                lowest_max_expansion_ratio = max_expansion_ratio
                min_index = idx
        
        #print(f'cluster with {len(smallest_cluster)} elements will be added to cluster with {len(clusters[min_index])} elements')
        clusters[min_index] += smallest_cluster

    return clusters


def cast_clustering(affinity_threshold = None, distance_matrix = None, tams=None, 
                    normalize=False, similarity_method = 'exponential', alpha = None, 
                    self_compute_affinity = False,
                    maximum_number_of_clusters = 8,
                    cleaning_step_max_iter = 0,
                    e_cast = False,
                    e_cast_parameter = 0.5,
                    rescaling_parameter = 7,
                    post_processing = True,
                    verbose = False):
    """
    Performs CAST clustering on given time series (or distance_matrix)

    Parameters:
    - affinity_threshold: threshold for adding a new element to the cluster
    - distance_matrix (np.array): A square matrix of distances. if None, the functions expects tams to compute it itself.
    - tams: tams we want to cluster
    - normalize: wether to z-normalize each tam.
    - similarity_method (str): method for converting distance matrix to similarity matrix 'reciprocal' for simple inverse, 'exponential' for exponential decay, 'gaussian' for gaussian.
    - alpha (float, optional): The decay rate for the exponential method. If None, alpha is set to 1/std of distances.
    - self_compute_affinity: if True, we will use the method found in a thesis to compute the threshold.
    - maximum_number_of_clusters: if post_processing is true, will force the number of clusters to be no more than this.
    - threshold_increase_rate: amount which the threshold gradually increases
    - e_cast: if True, we will perform e-cast instead of normal cast
    - e_cast_parameter: the parameter for defining the threshold in e_cast
    - cleaning_step_max_iter: the maximum number of iterations for our cleaning phase
    - min_cluster_size: we don't our clusters to be smaller than this
    - rescaling_parameter: used when we want to use self scaling for similarity matrix
    - post_processing: if True, will perform post processing.
    Returns:
    - A clustering of our data.
    """

    if tams is not None:
        if normalize:
            tams = z_normalize(tams)
        if distance_matrix is None:
            distance_matrix = compute_distance_matrix(tams)
            
    similarity_matrix = distance_to_similarity(distance_matrix= distance_matrix, 
                                               method= similarity_method, alpha= alpha, rescaling_parameter= rescaling_parameter)

    #similarity_matrix = similarity.distance_to_similarity(distance_matrix, method= similarity_method)
    
    
    n = distance_matrix.shape[0]
    C = []  # The collection of closed clusters
    C_open = []  # The currently open cluster
    U = list(range(n))  # Indices of elements not yet assigned to any cluster
    a = np.zeros(n)  # Affinity scores
    if e_cast: # the threshold will be computed dynamically
        affinity_threshold = dynamic_threshold(similarity_matrix= similarity_matrix, threshold= e_cast_parameter)
    
    if self_compute_affinity:
        affinity_threshold = self_compute_affinity_threshold(similarity_matrix)
        print(f'Affinity Threshold is {affinity_threshold}')
        
    while U or C_open:
        if U:
            if len(C_open) == 0: # initially, use the element with the biggest similarity as initialization
                unclustered = np.sort(list(U))
                current_similarity = similarity_matrix[unclustered, :][:, unclustered]

                # Compute maximum similarity for each element to any other remaining element
                max_similarities = np.max(current_similarity, axis=1)
                
                # Select the element with the highest maximum similarity
                if len(max_similarities) > 0:
                    max_sim_index = np.argmax(max_similarities)
                    u = unclustered[max_sim_index]
            # Find the element with maximal affinity in U     
            else:
                u = max(U, key=lambda x: a[x])
            #print(f'number of instances remaining : {len(U)}')
            
                
            if a[u] >= affinity_threshold * len(C_open):
                C_open.append(u)
                U.remove(u)
                # Update affinity scores
                for x in U + C_open:
                    a[x] += similarity_matrix[u][x]

                
            else:
                # If no high affinity elements outside C_open
                # TODO this is a modification. we find all elements that are outliers and remove them.
                # TODO should we do this sequentially?
                
                number_of_removed = 0
                while True:
                    v = min(C_open, key=lambda x: a[x]) if C_open else None
                    # TODO we will use a[element] - 1 because we don't want to count the affinity of the element with itself
                    if v is not None and (a[v] - 1) < affinity_threshold * (len(C_open) - 1):
                        C_open.remove(v)
                        U.append(v)
                        for x in U + C_open:
                            a[x] -= similarity_matrix[v][x]
                        number_of_removed += 1
                    else:
                        break
                if verbose:
                    if number_of_removed > 0:
                        print(f'{number_of_removed} removed from cluster {len(C)}')
                    
                        
                if number_of_removed == 0:
                    # Close the current cluster and reset
                    if C_open:
                        if verbose:
                            print(f'The cluster is being closed with {len(C_open)} number of elements')
                        C.append(C_open)
                        C_open = []
                        a = np.zeros(n)  # Reset affinity scores
                        
                        if e_cast:
                        # Recalculate the threshold with remaining unclustered points
                            unclustered = np.sort(list(U))
                            current_similarity = similarity_matrix[unclustered][:, unclustered]
                            affinity_threshold = dynamic_threshold(current_similarity, threshold= e_cast_parameter)
                            if verbose:
                                print(f'affinity threshold changed to {affinity_threshold}')
                        
        else:
            if C_open:  # If all elements are processed but C_open is not empty
                if verbose:
                    print(f'The final cluster is being closed with {len(C_open)} number of elements')
                C.append(C_open)
                break
    
    # Cleaning step: Reassign points for better fit
    def calculate_affinity(point, cluster, similarity_matrix, not_include_self = True):
        # not_include_self means we want to exclude the point if it is already in the cluster
        if not_include_self:
            cluster = [element for element in cluster if element != point]
        
        if not cluster:
            return 0        
        return np.mean([similarity_matrix[point, other] for other in cluster])
    

    for i in tqdm(range(cleaning_step_max_iter), desc= 'performing cleaning on the clusters'):
        
        cheked_already = [False for j in range(n)] #TODO we will inspect each element. I think we don't want to reinspect it if it is moved to another cluster, or do we?
        
        stable = True
        for idx, cluster in enumerate(C):
            if not cluster:
                continue
            for point in cluster:
                # if cheked_already[point]:
                #     continue TODO
                cheked_already[point] = True
                best_cluster = idx
                best_affinity = calculate_affinity(point, cluster, similarity_matrix)
                

                # Check for a better cluster
                for j, other_cluster in enumerate(C):
                    if j != idx:
                        affinity = calculate_affinity(point, other_cluster, similarity_matrix)
                        if affinity > best_affinity:
                            best_affinity = affinity
                            best_cluster = j
                # Reassign if a better cluster is found
                if best_cluster != idx:
                    cluster.remove(point)
                    C[best_cluster].append(point)
                    stable = False
            # removing empty clusters:
            C = [cluster for cluster in C if len(cluster) != 0]
        if stable:
            break
    
    if post_processing:
        C = perform_post_processing(similarity_matrix= similarity_matrix,
                                    clusters= C,
                                    max_num_of_clusters= maximum_number_of_clusters)
    
    return C

if __name__ == '__main__':
    cm.initialize_common_params('Tik_Tok')
    website_index = 13
    trace_mode = 'tam'
    
    dataset = TraceDataset(extract_traces= False, 
                           trace_mode= trace_mode, 
                           )
    load_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'distance_matrices', 'euc')
    filename = f'{website_index}.npy'
    
    distance_matrix = load_file(dir_path= load_dir, file_name= filename)
    tams, _ = dataset.get_traces_of_class(class_number = website_index)
    clusters = cast_clustering(distance_matrix= distance_matrix, 
                                similarity_method= 'self_scaling',
                                cleaning_step_max_iter= 20,
                                self_compute_affinity= True,
                                rescaling_parameter= 7,
                                maximum_number_of_clusters= 6,
                                post_processing= True)
    cluster_size_distribution(clusters = clusters, clustering_algorithm= 'CAST')