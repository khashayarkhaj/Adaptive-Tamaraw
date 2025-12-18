# a clustering algorithm specifically designed for tamaraw

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.distance_metrics import tam_euclidian_distance
from utils.trace_operations import compute_super_matrix, glove_cost_function
from sklearn.metrics import silhouette_score
import utils.config_utils as cm
from ..clustering_utils import group_clusters, find_elbow_point, tamaraw_overhead_vector
from ..clustering_metrics import is_diverse
from .palette_post_processing import refine_clusters_with_swaps_heavy, refine_clusters_with_directed_swaps_medium
import gc
from collections import Counter, defaultdict
import copy
def compute_diversity_factor(current_cluster_labels, new_element_label, diversity_threshold, diversity_penalty):
    """
    Computes a multiplicative diversity penalty factor based on the current cluster's labels,
    the new element's label, a diversity threshold (T), and a diversity penalty factor (eta).
    
    The penalty factor is designed to discourage clusters from having a high fraction of
    a single label. If adding the new element causes the projected fraction of its label to exceed
    the diversity threshold, the penalty factor increases linearly with the excess.
    
    Parameters:
    - current_cluster_labels (list): List of labels for the elements currently in the cluster.
    - new_element_label: Label of the new candidate element to be added.
    - diversity_threshold (float): Maximum allowed fraction for any single label (e.g., 0.3).
    - diversity_penalty (float): Penalty multiplier (eta) to scale the excess fraction.
    
    Returns:
    - float: A multiplicative penalty factor. A factor of 1.0 indicates no penalty; values >1.0
             indicate an increased penalty due to a lack of diversity.
    """
    
    # Count how many times the new element's label appears in the current cluster.
    label_count = current_cluster_labels.count(new_element_label)
    
    # Calculate the projected fraction for new_element_label if the candidate is added.
    # The +1 in both numerator and denominator accounts for the candidate element.
    projected_fraction = (label_count + 1) / (len(current_cluster_labels) + 1)
    
    # If the projected fraction is within the acceptable diversity threshold, no penalty is applied.
    if projected_fraction <= diversity_threshold:
        return 1.0, projected_fraction
    else:
        # If the projected fraction exceeds the threshold, compute the penalty factor.
        # The penalty increases linearly with the amount that the fraction exceeds the threshold.
        return 1.0 + diversity_penalty * (projected_fraction - diversity_threshold), projected_fraction
    

def find_pareto_pairs(values):
    """
    Find Pareto-optimal pairs from a list of alternating values [b1,t1, b2,t2, b3,t3, ...]
    Returns list of tuples containing Pareto-optimal pairs.
    
    Args:
        values (list): List of alternating values [b1,t1, b2,t2, ...]
    
    Returns:
        list: List of tuples [(b,t), ...] containing Pareto-optimal pairs
    """
    # Convert list to pairs
    pairs = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
    pareto_pairs = []
    
    for pair1 in pairs:
        is_pareto = True
        for pair2 in pairs:
            # Check if pair2 dominates pair1
            if pair1 != pair2:
                if (pair2[0] >= pair1[0] and pair2[1] >= pair1[1]) and \
                   (pair2[0] > pair1[0] or pair2[1] > pair1[1]):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_pairs.append(pair1)
    
    return pareto_pairs


#Function to Find Pareto-optimal Pairs

def find_top_k_pareto_pairs_sum(values, k = 10):
    """
    Find top k Pareto-optimal pairs from a list of alternating values [b1,t1, b2,t2, b3,t3, ...]
    and return the sum of bi + ti for these k pairs.
    
    Args:
        values (list): List of alternating values [b1,t1, b2,t2, ...]
        k (int): Number of top pairs to consider
    
    Returns:
        float: Sum of bi + ti for the top k Pareto-optimal pairs
    """
    # Convert list to pairs
    pairs = [(values[i], values[i+1]) for i in range(0, len(values) - 1, 2)]
    pareto_pairs = []
    
    # Find all Pareto-optimal pairs
    for pair1 in pairs:
        is_pareto = True
        for pair2 in pairs:
            # Check if pair2 dominates pair1
            if pair1 != pair2:
                if (pair2[0] >= pair1[0] and pair2[1] >= pair1[1]) and \
                   (pair2[0] > pair1[0] or pair2[1] > pair1[1]):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_pairs.append(pair1)
    
    # Sort Pareto pairs by their sum in descending order
    pareto_pairs.sort(key=lambda x: x[0] + x[1], reverse=True)
    
    # Take top k pairs (or all if k is larger than available pairs)
    k = min(k, len(pareto_pairs))
    top_k_pairs = pareto_pairs[:min(len(pareto_pairs),k)]
    
    # Calculate total sum - properly unpacking the tuples
    total_sum = sum(
    sum(inner_tuple)         # sum of each inner tuple
    for outer_tuple in top_k_pairs  # each element in `data`
    for inner_tuple in outer_tuple
)
    denominator = min(len(top_k_pairs), k)
    denominator = max(1, denominator)
    return total_sum/ denominator


def calculate_utility(time_overhead, bandwidth_overhead):
        """
        Calculate utility based on time overhead and bandwidth overhead.
        
        Parameters:
        time_overhead (float): Time overhead value
        bandwidth_overhead (float): Bandwidth overhead value
        
        Returns:
        float: Calculated utility
        """
        return (time_overhead + bandwidth_overhead) / 2
        #return math.sqrt(time_overhead ** 2 + bandwidth_overhead ** 2)


def find_best_overhead_pair(overhead_vector, target_latency=None, target_overhead=None, verbose=True, oh_threshold=0.1):
    """
    Find the best bandwidth-time overhead pair from a vector of alternating bandwidth and time overheads.
    
    Parameters:
    -----------
    overhead_vector : list
        List of alternating bandwidth and time overheads
        [bw_oh1, time_oh1, bw_oh2, time_oh2, ...]
    target_latency : float, optional
        Target time overhead
    target_overhead : float, optional
        Target bandwidth overhead
    verbose : bool, optional
        Whether to print detailed information about the selection process
    oh_threshold : float, optional
        Threshold for considering values close to target (default: 0.1 or 10%)
    
    Returns:
    --------
    tuple
        (best_bandwidth_overhead, best_time_overhead)
    """
    if len(overhead_vector) % 2 != 0:
        raise ValueError("Overhead vector must have even length (pairs of bandwidth and time)")
    
    if target_latency is None and target_overhead is None:
        raise ValueError("At least one target (latency or overhead) must be specified")
    
    # Convert vector to pairs for easier processing
    pairs = [(overhead_vector[i], overhead_vector[i+1]) 
             for i in range(0, len(overhead_vector), 2)]
    
    best_pair = None
    
    # Step 1: Find pairs strictly better than targets
    better_pairs = [
        pair for pair in pairs
        if (target_overhead is None or pair[0] <= target_overhead) and
           (target_latency is None or pair[1] <= target_latency)
    ]
    
    if better_pairs:
        best_pair = min(better_pairs, 
                       key=lambda p: calculate_utility(p[1], p[0]))
        if verbose:
            print("Found pair within target bounds:")
            print(f"Bandwidth overhead: {best_pair[0]}, Time overhead: {best_pair[1]}")
        return best_pair
    
    # Step 2: Find pairs within threshold
    def within_threshold(value, target):
        if target is None:
            return True
        threshold_lower = target * (1 - oh_threshold)
        threshold_upper = target * (1 + oh_threshold)
        return threshold_lower <= value <= threshold_upper
    
    threshold_pairs = [
        pair for pair in pairs
        if (within_threshold(pair[0], target_overhead) and
            within_threshold(pair[1], target_latency))
    ]
    
    if threshold_pairs:
        best_pair = min(threshold_pairs,
                       key=lambda p: calculate_utility(p[1], p[0]))
        if verbose:
            print("Found pair within threshold bounds:")
            print(f"Bandwidth overhead: {best_pair[0]}, Time overhead: {best_pair[1]}")
        return best_pair
    
    # Step 3: Find closest pair by individual targets TODO maybe change step 3 (return the best pair that has at least one smaller element and has the best utility?)
    closest_pairs = []
    
    if target_overhead is not None:
        closest_bw = min(pairs, key=lambda p: p[0] - target_overhead)
        closest_pairs.append(closest_bw)
    
    if target_latency is not None:
        closest_time = min(pairs, key=lambda p: p[1] - target_latency)
        closest_pairs.append(closest_time)
    
    if closest_pairs:
        best_pair = min(closest_pairs,
                       key=lambda p: calculate_utility(p[1], p[0]))
        if verbose:
            print("Found closest pair by individual target:")
            print(f"Bandwidth overhead: {best_pair[0]}, Time overhead: {best_pair[1]}")
        return best_pair
    
    raise ValueError("No suitable pair found")
def compute_tamaraw_distance_pareto(cluster1_idx, cluster2_idx, trace_oh_dict, num_combinations = 121, verbose_index = False,
                                    target_oh_pairs = None):
    # given two tier 1 clusters (each list of indices in the original trace dataset), compute their distance by comparing their overhead 
    # we will focus on the pareto optimizers given. we will try to find an improvement on them for each pair.
    # the amount that we improve will be the distance
    # note that cluster 2 should be the original cluster we want to improve
    # target_oh_pairs are the global overhead pairs we want to imporve
    original_overhead_vector = tamaraw_overhead_vector(trace_oh_dict= trace_oh_dict, trace_indices= cluster2_idx,
                                               num_combinations= num_combinations)
    
    
    
    big_cluster = []
    big_cluster += cluster1_idx
    big_cluster += cluster2_idx
    overhead_vector_improved = tamaraw_overhead_vector(trace_oh_dict= trace_oh_dict, trace_indices= big_cluster,
                                               num_combinations= num_combinations)
    
    distance = 0
    for i in range(0,len(target_oh_pairs) - 1, 2):
        desired_bandwidth = target_oh_pairs[i]
        desired_time = target_oh_pairs[i+1]

        best_bw_before, best_time_before = find_best_overhead_pair(overhead_vector= original_overhead_vector,
                                                     target_latency= desired_time,
                                                     target_overhead= desired_bandwidth,
                                                     verbose= False)
        
        best_bw_after, best_time_after = find_best_overhead_pair(overhead_vector= overhead_vector_improved,
                                                     target_latency= desired_time,
                                                     target_overhead= desired_bandwidth,
                                                     verbose= False)
        distance += best_bw_after - best_bw_before
        distance += best_time_after - best_time_before
    
        # print(f'target was bw {desired_bandwidth}')
        # print(f'target was time {desired_time}')
        # print(f'best time was {best_bw}')
        # print(f'best time was {best_time}')
        # breakpoint()
    #     if verbose_index:
    #         breakpoint()
    # breakpoint()
    del big_cluster

    return distance

def compute_tamaraw_distance(cluster1_idx, cluster2_idx, trace_oh_dict, num_combinations = 121, specific_pairs = None):
    # given two tier 1 clusters (each list of indices in the original trace dataset), compute their distance by comparing their overhead for all parameters
    
    overhead_vector1 = tamaraw_overhead_vector(trace_oh_dict= trace_oh_dict, trace_indices= cluster1_idx,
                                               num_combinations= num_combinations, desired_pairs= specific_pairs)
    
    overhead_vector2 = tamaraw_overhead_vector(trace_oh_dict= trace_oh_dict, trace_indices= cluster2_idx,
                                               num_combinations= num_combinations, desired_pairs= specific_pairs)
    l2_distance = np.linalg.norm(np.array(overhead_vector1) - np.array(overhead_vector2))

    del overhead_vector1
    del overhead_vector2

    return l2_distance

def compute_tamaraw_distance_length(cluster1_idx, cluster2_idx, trace_L_dictionary, number_of_L_configs, l_value = None):
    # we want the traces that have the same length after applying tamaraw to be in the same cluster
    length_vector1 = []
    length_vector2 = []
    # l_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    if l_value is None:
        l_values = [500, 600, 700, 800, 900, 1000]
    else:
        l_values = [l_value]
    # for l in l_values:
    #     for config_idx in range(number_of_L_configs):
    #         sizes = [trace_L_dictionary[idx][(config_idx,l)] for idx in cluster1_idx]
    #         length_vector1+= sizes
    
    # for l in l_values:
    #     for config_idx in range(number_of_L_configs):
    #         sizes = [trace_L_dictionary[idx][(config_idx,l)] for idx in cluster2_idx]
    #         length_vector2+= sizes
    # l2_distance = np.linalg.norm(np.array(length_vector2) - np.array(length_vector1))


    # del length_vector1
    # del length_vector2

    for l in l_values:
        for config_idx in range(number_of_L_configs):
            sizes1 = [trace_L_dictionary[idx][(config_idx,l)] for idx in cluster1_idx]
            sizes2 = [trace_L_dictionary[idx][(config_idx,l)] for idx in cluster2_idx]

    # return l2_distance


def calculate_cluster_accuracy(sizes, websites, debug_mode = False) :
    """
    Calculate attacker's accuracy within a cluster based on the size information.
    
    Args:
        sizes: List of trace sizes after defense
        websites: List of website indices corresponding to each trace
        
    Returns:
        Accuracy of the attacker's predictions
    """
    # Group traces by size
    size_to_traces = defaultdict(list)
    for i, size in enumerate(sizes):
        size_to_traces[size].append(i)
    
    correct_predictions = 0
    total_predictions = 0
    
    # For each unique size
    for size, trace_indices in size_to_traces.items():
        # Get the websites of traces with this size
        trace_websites = [websites[i] for i in trace_indices]
        
        # Determine most common website (attacker's prediction)
        website_counts = Counter(trace_websites)
        if website_counts:
            most_common_website = website_counts.most_common(1)[0][0]
            
            # Count correct predictions
            correct_predictions_this_size = sum(1 for website in trace_websites if website == most_common_website)
            correct_predictions += correct_predictions_this_size
            total_predictions += len(trace_websites)
        if debug_mode:
            breakpoint()
    
    # Calculate accuracy
    return correct_predictions / total_predictions if total_predictions > 0 else 0

def compute_cluster_accuracy(cluster1_idx, cluster2_idx, trace_L_dictionary, original_websites, number_of_L_configs, debug_mode = False, l_value = None):
    # l_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    if l_value is None:
        l_values = [500, 600, 700, 800, 900, 1000]
    else:
        l_values = [l_value]
    new_cluster = copy.deepcopy(cluster1_idx)
    new_cluster += cluster2_idx
    
    websites_in_this_cluster = [original_websites[i] for i in cluster1_idx]
    websites_in_new_element = [original_websites[i] for i in cluster2_idx]
    new_websites = websites_in_this_cluster
    new_websites += websites_in_new_element
    all_accuracies = []
    
    for l in l_values:
        
        for config_idx in range(number_of_L_configs):
            sizes = [trace_L_dictionary[idx][(config_idx,l)] for idx in new_cluster]
            max_accuracy = calculate_cluster_accuracy(sizes = sizes, websites = new_websites)
            all_accuracies.append(max_accuracy)
    if debug_mode:
        breakpoint()
            
    del new_cluster
    del new_websites
    del sizes
    return np.mean(np.array(all_accuracies))
def compute_max_accuracy_factor(cluster1_idx, cluster2_idx, trace_L_dictionary, original_websites, number_of_L_configs, 
                                penalty, max_accuracy_threshold, debug_mode = False, l_value = None):
    
    #l_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    if l_value is None:
        l_values = [500, 600, 700, 800, 900, 1000]
    else:
        l_values = [l_value]
    new_cluster = copy.deepcopy(cluster1_idx)
    new_cluster += cluster2_idx
    
    new_websites = [original_websites[i] for i in cluster1_idx]
    new_websites += [original_websites[i] for i in cluster2_idx]
    all_accuracies = []
    for l in l_values:
        for config_idx in range(number_of_L_configs):
            sizes = [trace_L_dictionary[idx][(config_idx,l)] for idx in new_cluster]
            max_accuracy = calculate_cluster_accuracy(sizes = sizes, websites = new_websites, debug_mode= debug_mode)
            all_accuracies.append(max_accuracy)
            if debug_mode:
                breakpoint()
    
    avg_max_accuracy = np.mean(np.array(all_accuracies))

    if  avg_max_accuracy > max_accuracy_threshold:
        factor = 1 + ((avg_max_accuracy - max_accuracy_threshold) * penalty)
    else:
        factor = 1
    
    return factor, avg_max_accuracy
            


def tamaraw_palette(
    tier1_clusters,
    k,
    tier1_labels,
    trace_L_dictionary,
    original_websites,
    number_of_L_configs,
    l_value,
    first_random_cluster=None,
    diversity_threshold=1.0,
    diversity_penalty=1,
    verbose=False,
):
    """
    Create anonymity sets using a greedy clustering algorithm that groups tier-1 clusters
    to minimize classification accuracy while maintaining diversity.
    
    This implements a two-tier clustering approach where tier-1 clusters are grouped into
    larger anonymity sets. The algorithm:
    1. Initializes with a random cluster
    2. Iteratively finds the farthest remaining cluster to start new anonymity sets
    3. Greedily adds k-1 nearest clusters to each anonymity set
    4. Assigns remaining clusters to their nearest anonymity set
    
    Parameters
    ----------
    tier1_clusters : list of lists
        Each inner list contains trace indices belonging to a tier-1 cluster.
        
    k : int
        Minimum number of tier-1 clusters in each anonymity set.
        
    tier1_labels : list
        Website index for each tier-1 cluster (used for diversity calculation).
        
    trace_L_dictionary : dict
        Maps trace_idx -> (config_idx, L) -> padded_size for tracking accuracy.
        
    original_websites : list
        True website label for each trace index.
        
    number_of_L_configs : int
        Number of L configuration options to consider for accuracy calculation.
        
    l_value : int or None
        Specific L value to use for padding configuration.
        
    first_random_cluster : int, optional
        Index of tier-1 cluster to use for initialization. If None, chooses randomly.
        
    diversity_threshold : float, optional (default=1.0)
        Maximum threshold to define a set as diverse.
        
    diversity_penalty : float, optional (default=1)
        Penalty factor applied when diversity constraints are violated.
        
    verbose : bool, optional (default=False)
        If True, prints detailed progress information.
        
    
    Returns
    -------
    S_A : list of lists
        Anonymity sets, where each inner list contains tier-1 cluster indices.
        
    overall_clusters : list of lists
        Final two-tier clusters, where each inner list contains trace indices.
    """
    # Convert numpy arrays to lists if necessary
    tier1_clusters = [
        arr.tolist() if hasattr(arr, 'tolist') else arr 
        for arr in tier1_clusters
    ]
    
    C = len(tier1_clusters)
    num_anonymity_sets = C // k
    
    print(f'Performing palette for {C} tier-1 clusters with k={k}')
    print(f'Will create {num_anonymity_sets} anonymity sets')
    
    # Initialize data structures
    S_A = [[] for _ in range(num_anonymity_sets)]  # Anonymity sets (tier-1 cluster indices)
    overall_clusters = [[] for _ in range(num_anonymity_sets)]  # Trace indices in each set
    overall_max_accuracies = [-1 for _ in range(num_anonymity_sets)]  # Max accuracy per set
    remaining_instances = set(range(C))  # Unassigned tier-1 cluster indices

    def find_min_distance_index(
        remaining_instances,
        tier1_clusters,
        current_cluster,
        tier2_indices_in_cluster=None,
        labels=None,
        consider_diversity=False
    ):
        """
        Find the tier-1 cluster with minimum distance to the current anonymity set.
        
        Distance is measured by classification accuracy - lower accuracy means better
        anonymization. Optionally applies diversity penalties.
        
        Parameters
        ----------
        remaining_instances : set
            Indices of tier-1 clusters not yet assigned to any anonymity set.
            
        tier1_clusters : list of lists
            All tier-1 clusters with their trace indices.
            
        current_cluster : list
            Trace indices in the current anonymity set being built.
            
        tier2_indices_in_cluster : list, optional
            Tier-1 cluster indices already in the current anonymity set (for diversity).
            
        labels : list, optional
            Website labels for each tier-1 cluster (for diversity calculation).
            
        consider_diversity : bool, optional
            Whether to apply diversity penalties.
        
        Returns
        -------
        min_index : int
            Index of the tier-1 cluster with minimum distance.
            
        min_distance : float
            The minimum distance/accuracy value found.
        """
        min_distance = float('inf')
        min_index = None
        best_original_dist = None
        best_diversity_factor = 0
        chosen_purity = None
        
        for i in tqdm(remaining_instances, desc='Finding minimum distance', disable=not verbose):
            # Calculate classification accuracy between candidate cluster and current set
            dist = compute_cluster_accuracy(
                cluster1_idx=tier1_clusters[i],
                cluster2_idx=current_cluster,
                trace_L_dictionary=trace_L_dictionary,
                original_websites=original_websites,
                number_of_L_configs=number_of_L_configs,
                l_value=l_value
            )
            
            original_dist = dist
            diversity_factor = 1
            purity = -1
            
            # Apply diversity penalty if enabled
            if consider_diversity and current_cluster and len(current_cluster) > 0:
                current_cluster_labels = [labels[idx] for idx in tier2_indices_in_cluster]
                new_element_label = labels[i]
                
                diversity_factor, purity = compute_diversity_factor(
                    current_cluster_labels=current_cluster_labels,
                    new_element_label=new_element_label,
                    diversity_threshold=diversity_threshold,
                    diversity_penalty=diversity_penalty
                )
                
                # Currently diversity factor is set to 1 (disabled)
                diversity_factor = 1
                dist = dist * diversity_factor
            
            # Track the cluster with minimum distance
            if dist < min_distance:
                min_distance = dist
                min_index = i
                best_original_dist = original_dist
                best_diversity_factor = diversity_factor
                chosen_purity = purity
        
        if verbose:
            print(f'Min index: {min_index}, Distance: {min_distance:.2f}, Purity: {chosen_purity:.2f}')
            print(f'Original dist: {best_original_dist:.2f}, Diversity factor: {best_diversity_factor:.2f}')
        
        gc.collect()
        return min_index, min_distance
    
    def find_farthest_trace_index(remaining_instances, tier1_clusters, overall_clusters):
        """
        Find the tier-1 cluster with maximum distance from the last anonymity set.
        
        Used to initialize new anonymity sets by choosing clusters that are far from
        existing sets, promoting diversity across anonymity sets.
        
        Parameters
        ----------
        remaining_instances : set
            Indices of tier-1 clusters not yet assigned.
            
        tier1_clusters : list of lists
            All tier-1 clusters with their trace indices.
            
        overall_clusters : list of lists
            Already formed anonymity sets (by trace indices).
        
        Returns
        -------
        max_index : int
            Index of the tier-1 cluster with maximum distance.
        """
        max_average_distance = float('-inf')
        max_index = None
        
        for i in tqdm(remaining_instances, desc='Finding farthest cluster', disable=not verbose):
            # Calculate distance from last formed anonymity set
            avg_distance = compute_cluster_accuracy(
                cluster1_idx=tier1_clusters[i],
                cluster2_idx=overall_clusters[-1],
                trace_L_dictionary=trace_L_dictionary,
                original_websites=original_websites,
                number_of_L_configs=number_of_L_configs,
                l_value=l_value
            )
            
            if avg_distance > max_average_distance:
                max_average_distance = avg_distance
                max_index = i
        
        return max_index

    # ==================== INITIALIZATION ====================
    # Choose initial tier-1 cluster randomly or use provided one
    if first_random_cluster is None:
        random_initialize_instance = np.random.randint(0, C)
    else:
        random_initialize_instance = first_random_cluster
    
    S_A[0].append(random_initialize_instance)
    overall_clusters[0] += tier1_clusters[random_initialize_instance]
    remaining_instances.remove(random_initialize_instance)
    
    if verbose:
        print(f'Cluster {random_initialize_instance} chosen as random initialization')
    
    # ==================== BUILD ANONYMITY SETS ====================
    for i in tqdm(range(num_anonymity_sets), desc=f'Building {num_anonymity_sets} anonymity sets'):
        # For sets after the first, initialize with the farthest cluster
        if i != 0:
            c_bar = find_farthest_trace_index(
                remaining_instances=remaining_instances,
                tier1_clusters=tier1_clusters,
                overall_clusters=overall_clusters[:i]
            )
            
            if verbose:
                print(f'Cluster {c_bar} chosen as initial element for anonymity set {i}')
            
            S_A[i].append(c_bar)
            overall_clusters[i] += tier1_clusters[c_bar]
            remaining_instances.remove(c_bar)
        
        # Greedily add k-1 more clusters to this anonymity set
        while len(S_A[i]) < k:
            c_bar, max_accuracy = find_min_distance_index(
                remaining_instances=remaining_instances,
                tier1_clusters=tier1_clusters,
                current_cluster=overall_clusters[i],
                tier2_indices_in_cluster=S_A[i],
                labels=tier1_labels,
                consider_diversity=True  # Enable diversity during set building
            )
            
            if verbose:
                print(f'Cluster {c_bar} added to anonymity set {i}')
            
            S_A[i].append(c_bar)
            overall_clusters[i] += tier1_clusters[c_bar]
            remaining_instances.remove(c_bar)
        
        overall_max_accuracies[i] = max_accuracy
        gc.collect()
    
    if verbose:
        print("=" * 50)
        print("Assigning remaining clusters to nearest anonymity sets")
    
    # ==================== ASSIGN REMAINING CLUSTERS ====================
    # Disable diversity constraints for residual assignment
    for instance in tqdm(remaining_instances, desc='Assigning residual clusters'):
        i_bar, max_accuracy = find_min_distance_index(
            remaining_instances=set(range(num_anonymity_sets)),
            tier1_clusters=overall_clusters,
            current_cluster=tier1_clusters[instance],
            consider_diversity=False
        )
        
        S_A[i_bar].append(instance)
        overall_clusters[i_bar] += tier1_clusters[instance]
        overall_max_accuracies[i_bar] = max_accuracy
    
    return S_A, overall_clusters