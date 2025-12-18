#an Implementation of the three phase l-diversity algorithm
import numpy as np
from utils.distance_metrics import tam_euclidian_distance
from utils.trace_operations import compute_super_matrix
import random
from tqdm import tqdm
#helper functions
def sample_indices_by_label_distribution(labels, num_samples):
    #Select num_samples distinct records based on their frequency in sensitive attribute values (labels)
    #labels is a list of labels of each trace e.g., [0,0,2,1,5,...]

    # Determine unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Calculate probabilities for each label
    probabilities = counts / counts.sum()
    
    # Map each label to its indices
    label_to_indices = {label: [] for label in unique_labels}
    for index, label in enumerate(labels):
        label_to_indices[label].append(index)
    
    # Flatten the list of indices, adjust probabilities accordingly
    flat_indices = []
    adjusted_probabilities = []
    for label, indices in label_to_indices.items():
        flat_indices.extend(indices)
        adjusted_probabilities.extend([probabilities[np.where(unique_labels == label)[0][0]]] * len(indices))
    
    # Adjust probabilities to sum to 1
    adjusted_probabilities = np.array(adjusted_probabilities)
    adjusted_probabilities /= adjusted_probabilities.sum()
    
    # Sample indices based on adjusted probabilities
    sampled_indices = np.random.choice(flat_indices, num_samples, replace=False, p=adjusted_probabilities)
    
    return sampled_indices



def sort_indices_by_distance(tams, centroids, r):
    # returns the sorted indices of centroids based on their proximity to tams[r]
    # Compute distances from tams[r] to each centroid
    distances = [tam_euclidian_distance(tams[r], centroid) for centroid in centroids]

    # Pair each distance with its corresponding index
    indexed_distances = list(enumerate(distances))

    # Sort pairs by distance
    indexed_distances.sort(key=lambda x: x[1])

    # Extract the sorted indices
    sorted_indices = [idx for idx, _ in indexed_distances]

    return sorted_indices

def clustering_phase(tams, k, l, labels, centroid_type = 'super_matrix', verbose = False):
    """
   Performs the clustering step of the three step (first) algorithm, by (trying to) partitioning the data into clusters that have at least k elements.

    Parameters
    ----------
    tams : list of tams
        
    k : value k for k-anonymity
    
    l : value l for l-diversity

    labels : the labels (website indexes) of each tam. They will serve as our sensitive attributes

    centroid_type : indicates how the centroids should be computed. currently we only support super_matrix
    
    verbose : wether to show internal results

    Returns
    -------
    A partition P = {P1, P2 …PK}
    """
    C = len(tams)
    num_anonimity_sets = C // k

    
    initial_tams = sample_indices_by_label_distribution(labels= labels, num_samples= num_anonimity_sets) # initializing the anonimitys sets
    partitions = [[initial_tams[i]] for i in range(num_anonimity_sets)] # our desired clusters
    centroids = [] # list of centroids of each partition

    # for now, we will use the supermatrixes as centroids # TODO check average tam or tam with lowest distance to others as well
    T = set(range(C)) # indices of instances that have not been assigned to any cluster yet - set is a good option for efficient removal
    for i in initial_tams:
        centroids.append(tams[i])
        T.remove(i)
    
    pbar = tqdm(desc= 'Performing Clustering Phase', disable= not verbose)
    while T:
        r = list(T)[0] # Let r be the first record in T
        # Order {Pi } according to their distances from r;
        ordered_partition_indices = sort_indices_by_distance(tams= tams, centroids= centroids, r = r)

        i = 0
        flag = 0
        for i in ordered_partition_indices:
            if flag == 1:
                break
            labels_of_this_partition = set([labels[idx] for idx in partitions[i]])
            if len(partitions[i]) < k or ((labels[r] not in labels_of_this_partition) and len(labels_of_this_partition) < l):
                partitions[i].append(r) # add r to Pi
                if centroid_type == 'super_matrix':
                    centroids[i] = compute_super_matrix(tams[r], centroids[i])# Update centroid of Pi - TODO maybe a better way
                flag = 1
                T.remove(r)
            else:
                i += 1
        
        if flag == 0: # tams[r] has not been assigned to any cluster TODO - maybe also consider size
            nearest_cluster_idx = ordered_partition_indices[0]
            partitions[nearest_cluster_idx].append(r) 
            if centroid_type == 'super_matrix':
                centroids[nearest_cluster_idx] = compute_super_matrix(tams[r], centroids[nearest_cluster_idx])# Update centroid of Pi - TODO maybe a better way
            T.remove(r)
        if verbose:
            pbar.update(1)
    pbar.close()
    return partitions

def adjustment_phase(partitions, k, tams, centroid_type = 'super_matrix', verbose = False):
    """
    Performs the adjustment step of the three step (second) algorithm, which assures that every cluster contains at least k records.

    Parameters
    ----------
    partitions : P = {P1, P2 …PK}, partitions obtained from the clustering phase. each partition contains the indices of the tams
        
    k : value k for k-anonymity
    
    tams : list of tams
    
    centroid_type : indicates how the centroids should be computed. currently we only support super_matrix

    verbose : wether to show internal results

    Returns
    -------
    an adjusted partitioning P = {P1 , P2 ...Pk }


    """

    """
    I tried to implement the algorithm in "An Algorithm to Achieve k-Anonymity and l-Diversity Anonymisation in Social
    Networks" but the pseudo code is a mess. Thus, initially, I will implement the code in "An Improved l-Diversity Anonymisation
    Algorithm".

    """

    #TODO in the original algorithm, it is not obvious if the number of partitions remains constant or not.
    # it seems like the clusters with size less than k/2 are added to the bigger ones in the beginning.
    #two helper functions we will use
    def find_closest_cluster(r, tams, clusters, cluster_mask = None):
        #given a trace r and a set of clusters, find the closest cluster to that trace
        # cluster_mask, if given, contains the cluster indices we actually want in our computation

        min_distance = float('inf')  # Start with infinitely large distance
        min_index = None  # Placeholder for the index with minimum distance

        for idx, cluster in enumerate(clusters):
            if cluster_mask and idx not in cluster_mask:
                continue
            tams_of_this_cluster = [tams[p] for p in cluster]
            centroid = None
            if centroid_type == 'super_matrix':
                centroid = compute_super_matrix(*tams_of_this_cluster) 

            dist = tam_euclidian_distance(r, centroid)

            if dist < min_distance:
                min_distance = dist
                min_index = idx
        
        return min_index

    def find_farthest_element(cluster, tams):
        # given a cluster (list of indices) find the farthest element from the centroid

        max_distance = float('-inf')  # Start with a very small number
        max_element = None  # Placeholder for the index with maximum average distance
        tams_of_this_cluster = [tams[p] for p in cluster]
        centroid = None
        if centroid_type == 'super_matrix':
            centroid = compute_super_matrix(*tams_of_this_cluster) 

        for element in cluster:
            dist = tam_euclidian_distance(tams[element], centroid)

            if dist > max_distance:
                max_distance = dist
                max_element = element
        
        return max_element
            

    S = set() # cluster indices that have less than k/2 elements
    small_clusters_mapping = {} # mapping from each element in S to its cluster idx
    U = set() # cluster indices that have size between k/2 and k
    V = set() # cluster indices that have size greater k
    
    for idx, partition in enumerate(partitions):
        if len(partition) <= k // 2:
            S.update(partition)
            for element in partition:
                small_clusters_mapping[element] = idx
        elif   k//2 < len(partitions) < k:
            U.add(idx)
        else:
            V.add(idx)
    
    pbar = tqdm(desc= 'Performing Adjustment Phase - step 1', disable= not verbose)
    while len(S) != 0:
        random_indice = random.choice(list(S))
        # removing the indice from its previous cluster
        initial_cluster_idx = small_clusters_mapping[random_indice]
        partitions[initial_cluster_idx].remove(random_indice)
        if len(U) != 0: # we still have clusters with size between k/2 and k
            # find closest cluster to random_indice and add it to this cluster
            closest_cluster_idx = find_closest_cluster(r = tams[random_indice],
                                                       tams= tams,
                                                       clusters = partitions,
                                                       cluster_mask= U)
            partitions[closest_cluster_idx].append(random_indice)
        else: # add r to the closest cluster in V
            # this part is vague. should we remove the element from its original cluster and then iterate all clusters? TODO
            closest_cluster_idx = find_closest_cluster(r = tams[random_indice],
                                                       tams= tams,
                                                       clusters = partitions,
                                                       cluster_mask= V) # later we can remove V and see the effect
            partitions[closest_cluster_idx].append(random_indice)
        
        S.remove(random_indice)
        if verbose:
            pbar.update(1)
    pbar.close()
    # removing the empty clusters
    partitions = [partition for partition in partitions if len(partition) != 0]

    
    R = set() # records belonging to partitions with more than k elements
    C = {} # mapping from index to original cluster index

    small_clusters = set() # clusters that have less than k elements
    for idx, partition in enumerate(partitions):
        if len(partition) > k:
            while len(partition) > k :
                farthest_element = find_farthest_element(cluster= partition,
                                                         tams= tams)
                R.add(farthest_element)
                partition.remove(farthest_element)
                C[farthest_element] = idx
        
        elif len(partition) < k:
            small_clusters.add(idx)
    
    
    
    pbar = tqdm(desc= 'Performing Adjustment Phase - step 2', disable= not verbose)
    while len(R) != 0:
        random_indice = random.choice(list(R))

        if len(small_clusters) != 0: # we still have clusters with size less than k
            # find closest cluster to random_indice and add it to this cluster
            closest_cluster_idx = find_closest_cluster(r = tams[random_indice],
                                                       tams= tams,
                                                       clusters = partitions,
                                                       cluster_mask= small_clusters)
            partitions[closest_cluster_idx].append(random_indice)
        else: # add r to its original cluster in P
            
           
            partitions[C[random_indice]].append(random_indice)
        R.remove(random_indice)
        if verbose:
            pbar.update(1)
    pbar.close()
    return partitions

                
    

def l_diversity_phase(partitions, tams, l, labels, centroid_type = 'super_matrix', verbose = False):
    """
    Performs the L - Diversity Phase of the three step (third) algorithm, which assures that every cluster contains at least l diverse records.

    Parameters
    ----------
    partitions : P = {P1, P2 …PM}, partitions obtained from the adjusment phase. Each partition contains the indices of the tams
        
    tams : list of tams

    l : value l for k-diversity

    labels : the labels (website indexes) of each tam. They will serve as our sensitive attributes
    
    centroid_type : indicates how the centroids should be computed. currently we only support super_matrix

    verbose : wether to show internal results

    Returns
    -------
    an adjusted partitioning P = {P1 , P2 ...Pk } satisfying l-diversity

    """

    # helper functions
    def find_closest_tam_to_centroid(indices, tams, centroid):
        # find index of the closest tam to a given centroid based on given indices
        min_distance = float('inf')  # Start with a very large number
        closest_tam_index = None

        # Iterate over each index in tams_in_cj to find the closest tam to centroid_i
        for index in indices:
            current_tam = tams[index]
            distance = tam_euclidian_distance(current_tam, centroid)
            if distance < min_distance:
                min_distance = distance
                closest_tam_index = index
        
        return closest_tam_index, min_distance

    def sort_diversity_matrix(P):
        # sorts the diversity matrix defined (refer to next lines after the function)
        # Count non-zero elements in each column (excluding the last row in the count) as the diversity
        non_zero_counts = np.count_nonzero(P[:-1], axis=0)
        P[-1, :] = non_zero_counts

        # Get the indices that would sort the columns by non-zero count
        sorted_indices = np.argsort(non_zero_counts)

        # Sort the array columns according to the sorted indices
        P = P[:, sorted_indices]

        return P, sorted_indices
    
    def find_closest_cluster(partitions, target_idx, tams, centroid_type = 'super_matrix'):
        # given the list of partitions and a target partition idx, find another partition which is the closest to this partition
        target_centroid = None

        if centroid_type == 'super_matrix':
            tams_of_this_cluster = [tams[p] for p in partitions[target_idx]]
            target_centroid = compute_super_matrix(*tams_of_this_cluster)
        
        min_distance = float('inf')  # Start with a very large number
        closest_cluster_index = None
        for idx, partition in enumerate(partitions):
            if idx == target_idx:
                continue
            
            candidate_centroid = None
            if centroid_type == 'super_matrix':
                tams_of_this_cluster = [tams[p] for p in partitions[idx]]
                candidate_centroid = compute_super_matrix(*tams_of_this_cluster)
            
            distance = tam_euclidian_distance(candidate_centroid, target_centroid)

            if distance < min_distance:
                min_distance = distance
                closest_cluster_index = idx
        
        return closest_cluster_index

        
        
    # Convert the list to a set to find unique labels
    unique_labels = set(labels)

    # Count the unique labels
    number_of_unique_labels = len(unique_labels)

    P = np.zeros([number_of_unique_labels + 1, len(partitions)]) # The last row contains the diversity values for each cluster (number of different websites in the cluster)
    tams_to_partition = {} # a mapping from each tam idx to its partition idx
    for idx, partition in enumerate(partitions):
        for element in partition:
            P[labels[element], idx] += 1
            tams_to_partition[element] = idx
    
    P, sorted_indices = sort_diversity_matrix(P)

    
    #sort the corresponding partitions as well
    partitions = [partitions[i] for i in sorted_indices]

    # Use searchsorted to find the first index where the value would exceed l
    index = np.searchsorted(P[-1, :], l)

    # Adjust the index to get the last one where the value is less than l
    q = index - 1
    
    # we want to construct diverse_websites. diverse_websites is a list of partitions. for each partition, we keep a dictionary of label : [tams] which labels are the ones with frequency greater than 1.
    # diverse websites stores the websites in each cluster that have frequency > 1
    diverse_websites = []
    for i in range(len(partitions)):
        partition = partitions[i]
        diverse_websites.append({})
        diverse_websites[-1]['partition index'] = i #we also store the index of the partition so that the implementation would be easier later on
        for tam_idx in partition:
            label = labels[tam_idx]
            if P[label, i] > 1:
                if label in diverse_websites[-1]:
                    diverse_websites[-1][label].append(tam_idx)
                else:
                    diverse_websites[-1][label] = [tam_idx]


    for i in tqdm(range(q + 1), desc= 'Perfroming the L-diversity step', disable= not verbose): # find elements in diverse clusters and add them to clusters with less diversity
        for j in range(q+1, len(partitions)):
            if P[-1, i] >= l: # Ci has become diverse
                break
            # Construct F, the sensitive attribute values which are in Cj but not in Ci and have frequency greater than 1
            condition = (P[:-1, i] == 0) & (P[:-1, j] > 1)
            F = np.where(condition)[0]
            mi = int(min(l - P[-1, i], len(F)))
            if mi == 0: # Cj has nothing to offer to Ci
                continue
            # swapping mi elements between Cj and Ci
            # note that since the elements in F have frequency > 1, the diversity of Cj won't be decreased
            # Also, since we are swapping elements, the cluster sizes won't change
            # TODO the only thing I'm not sure about is the information loss. how can we swap elements that incur minimum extra loss?

            # Construct Q, the sensitive attribute values which are in Ci that have frequency greater than 1
            condition = (P[:-1, i] > 1) 
            Q = np.where(condition)[0]

            
            # find the centroid of the Ci
            tams_of_ci= [tams[p] for p in partitions[i]]
            centroid_i = None
            if centroid_type == 'super_matrix':
                centroid_i = compute_super_matrix(*tams_of_ci)

            # we will find mi tams in Cj|F which are closest to Ci.
            candidates_in_cj = [] # candidates in Cj for swapping 
            distances_of_candidates_cj = []
            for website_index in F:
                tams_in_cj_index = diverse_websites[j][website_index] # list of tams in Cj that belong to the certain index in F
                closest_tam_index, distance = find_closest_tam_to_centroid(indices= tams_in_cj_index,
                                                                 tams = tams,
                                                                 centroid= centroid_i)
                candidates_in_cj.append(closest_tam_index)
                distances_of_candidates_cj.append(distance)
            
            # find the top mi candidates

            distances_of_candidates_cj = np.array(distances_of_candidates_cj)
            sorted_indices = np.argsort(distances_of_candidates_cj)

            
            selected_candidates_cj = [candidates_in_cj[idx] for idx in sorted_indices[:mi]]

            # we will also find mi tams in Ci|Q which are closest to Cj (excluding the candidates). TODO this is customized by me, check if it's correct

            tams_of_cj= [tams[p] for p in partitions[i] if p not in selected_candidates_cj]
            centroid_j = None
            if centroid_type == 'super_matrix':
                centroid_j = compute_super_matrix(*tams_of_cj)

            
            candidates_in_ci = [] # candidates in Cj for swapping 
            distances_of_candidates_ci = []
            for website_index in Q: 
                tams_in_ci_index = diverse_websites[i][website_index] # list of tams in Ci that belong to the certain index in F
                closest_tam_index, distance = find_closest_tam_to_centroid(indices= tams_in_ci_index,
                                                                 tams = tams,
                                                                 centroid= centroid_j)
                candidates_in_ci.append(closest_tam_index)
                distances_of_candidates_ci.append(distance)
            
            # find the top mi candidates

            distances_of_candidates_ci = np.array(distances_of_candidates_ci)
            sorted_indices = np.argsort(distances_of_candidates_ci)

            selected_candidates_ci = [candidates_in_ci[idx] for idx in sorted_indices[:mi]]

            # swap the candidates from the two groups
            for candidate_j in selected_candidates_cj:
                partitions[i].append(candidate_j)
                label_of_candidate = labels[candidate_j]
                P[label_of_candidate, i] += 1
                P[label_of_candidate, j] -= 1
                if label_of_candidate in diverse_websites[i]:
                    diverse_websites[i][label_of_candidate].append(candidate_j)
                else:
                    diverse_websites[i][label_of_candidate] = [candidate_j]
                
                diverse_websites[j][label_of_candidate].remove(candidate_j)
                if len(diverse_websites[j][label_of_candidate]) <= 1: # this cluster doesn't have > 1 elements of this website anymore
                    del diverse_websites[j][label_of_candidate]
            for candidate_i in selected_candidates_ci:
                partitions[j].append(candidate_j)
                label_of_candidate = labels[candidate_i]
                P[label_of_candidate, j] += 1
                P[label_of_candidate, i] -= 1
                if label_of_candidate in diverse_websites[j]:
                    diverse_websites[j][label_of_candidate].append(candidate_i)
                else:
                    diverse_websites[j][label_of_candidate] = [candidate_i]
                
                diverse_websites[i][label_of_candidate].remove(candidate_i)
                if len(diverse_websites[i][label_of_candidate]) <= 1: # this cluster doesn't have > 1 elements of this website anymore
                    del diverse_websites[i][label_of_candidate]
            #updating P as well
            P[-1, i] = np.count_nonzero(P[:-1, i])
            P[-1, j] = np.count_nonzero(P[:-1, j])
    
    # checking if we still have clusters with diversity < l
    P, sorted_indices = sort_diversity_matrix(P)

    #sort the corresponding partitions as well
    partitions = [partitions[i] for i in sorted_indices]

    # Use searchsorted to find the first index where the value would exceed l
    index = np.searchsorted(P[-1, :], l)

    # Adjust the index to get the last one where the value is less than l
    q = index - 1
    # Again, lines 10 to 13 of the algorithm are vague. what does it mean by perform steps 5 to 8? why just for the first element?
    if q >= 1: # equal to |S| > 1
        # defining S = {Ci: diversity of Ci < L }
        S = partitions[:q]
    
    elif q == 0: # only the first element has diversity < l
        """
        Else merge the element in S with the nearest of
        the clusters with diversity l or more obtained
        above.
        """
        closest_cluster = find_closest_cluster(partitions= partitions, target_idx= 0,
                                               tams= tams,
                                               centroid_type= centroid_type)
        partitions[closest_cluster] += partitions[0]

        del partitions[0]

    return partitions

def three_phase_oka(tams, k, l, labels, centroid_type = 'super_matrix', verbose = False):
    """
   Performs the three step OKA (one pass kmeans), which tries to take care of k-anonymity and l-diversity.

    Parameters
    ----------
    tams : list of tams
        
    k : value k for k-anonymity
    
    l : value l for l-diversity

    labels : the labels (website indexes) of each tam. They will serve as our sensitive attributes

    centroid_type : indicates how the centroids should be computed. currently we only support super_matrix
    
    verbose : whether to show internal results

    Returns
    -------
    A partition P = {P1, P2 …PK}
    """
    print(f'Initial partitioning being performed with k = {k} and l = {l}...')
    initial_partitions = clustering_phase(tams= tams,
                                          k = k,
                                          l = l,
                                          labels = labels,
                                          centroid_type= centroid_type,
                                          verbose= verbose)
    print('Initial partitioning done')

    print(f'Adjusted phase being performed k = {k} and l = {l}...')
    adjusted_partitions = adjustment_phase(partitions= initial_partitions,
                                           k = k,
                                           tams = tams,
                                           centroid_type= centroid_type,
                                           verbose= verbose)
    print('Adjustment phase done')


    print(f'l diversity phase being performed k = {k} and l = {l}...')
    l_diversity_partitions = l_diversity_phase(partitions= adjusted_partitions, 
                                               tams = tams,
                                               l = l,
                                               labels= labels,
                                               centroid_type= centroid_type,
                                               verbose= True)
    print('l diversity phase being done')
    return initial_partitions, adjusted_partitions, l_diversity_partitions
                
             
        
       

