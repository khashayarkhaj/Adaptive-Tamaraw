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
def perform_kmeans(tam_traces, n_clusters, max_iter=300, tol=1e-4, random_state=42):
    """
        a function that performs k_means on the provided traces.
        Args:
            tam_traces: a list of traces, each in tam format (since this is the only format supporting euclidean distance)
            the other arguments are self explanatory
    """
    # Flatten each matrix into a vector
    flattened_matrices = [matrix.flatten() for matrix in tam_traces]
    
    # Convert list of flattened matrices to a numpy array
    X = np.array(flattened_matrices)
    
    # Standardize features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X) # TODO check if this helps later
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state)
    kmeans.fit(X)
    
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Get cluster labels
    labels = kmeans.labels_
    
    # Get inertia (cost) after each iteration
    history = kmeans.inertia_
    
    # Get number of iterations until convergence
    n_iter = kmeans.n_iter_
    
    return cluster_centers, labels, history, n_iter
    




def k_medoids_custom(k, distance_matrix=None, max_iter=300, logger = None, random_state= 0, verbose = True):

    assert distance_matrix is not None, "distance_matrix should contain pairwise distances between traces, but is None instead"
    m, _ = distance_matrix.shape
    np.random.seed(random_state)
    
    # Step 2: Initialize medoids randomly
    medoids = np.random.choice(m, k, replace=False)
    clusters = np.zeros(m)
    cost_history = []
    iteration_num = 0
    # Step 3: Iterate until convergence or max_iter is reached
    for _ in tqdm(range(max_iter), desc= f'performing costumized k-medoids with k = {k}', disable= not verbose):
        # Step 4: Assign each point to the nearest medoid
        for i in range(m):
            distances = distance_matrix[i, medoids]
            clusters[i] = np.argmin(distances)

        # Step 5: Update medoids
        new_medoids = np.zeros(k, dtype=int)
        for medoid_idx in range(k):
            cluster_points_indices = np.where(clusters == medoid_idx)[0]
            if len(cluster_points_indices) > 0:
                cluster_distances = distance_matrix[np.ix_(cluster_points_indices, cluster_points_indices)]
                medoid_distances = np.sum(cluster_distances, axis=1)
                new_medoids[medoid_idx] = cluster_points_indices[np.argmin(medoid_distances)]
            else:
                new_medoids[medoid_idx] = medoids[medoid_idx]

        if np.all(medoids == new_medoids):
            break
        medoids = new_medoids


        # Step 6: Calculate the current cost
        current_cost = np.sum([distance_matrix[i, int(medoids[int(clusters[i])])] for i in range(m)])
        cost_history.append(current_cost)
        iteration_num += 2

    # Step 7: Return final medoids, cluster assignments, and cost history
    return medoids, clusters.astype(int), cost_history, iteration_num




def pallete(tams, k, first_random_tam = None, consider_diversity = False, 
            tam_labels = None, diversity_threshold = 1.0 ,diversity_penalty = 1, verbose = False, post_processing = 'None',
            post_processing_max_iter = 200):
    """
    Compute the anonimity sets with at least k number of elements, given a list of tams.


    Parameters
    ----------
        tams: list of traces. Can be all traces or just the supermatrix of each class
        
        k: Minimum number of elements in each anonimity set
        
        first_random_tam: if given, we will use it as the initialization. otherwise, we will choose a random tam
        
        consider_diversity: Also take into account that each cluster should have elements from different websites
        
        tam_labels: Website index of each tam. Expected to be given if consider_diversity is true

        diversity_threshold: the maximum threshold we define a set to be diverse

        diversity_penalty = Penalty added when trying to find the nearest trace to a cluster

        post_processing: wether to perform post processing (can be 'heavy', 'medium', None).
        

    Returns
    -------
        S_A: list of anonimity sets
        
        S_M: list of super matrixes
    """
    
    C = len(tams)
    num_anonimity_sets = C // k
    print(f'performing palette for {C} tams and k = {k}')

    S_A = [[] for i in range(num_anonimity_sets)] # list of anonimity sets
    S_M = [] # list of super matrixes
    remaining_instances = set(range(C)) # indices of instances that have not been assigned to any cluster yet - set is a good option for efficient removal

    def find_min_distance_index(remaining_instances, tams, m, current_cluster = None, tam_labels = None):
        #find trace that has the lowest distance to M (from the traces that have not been assigned any clusters yet)
        # current cluster is a list of indices in the current cluster to check if it is diverse
        # tan_labels is the labels of all tams

        min_distance = float('inf')  # Start with infinitely large distance
        min_index = None  # Placeholder for the index with minimum distance

        # Iterate over each index in the remaining instances
        for i in remaining_instances:
            # Calculate the distance from tams[i] to m
            
            dist = tam_euclidian_distance(tams[i], m) # TODO also check the approach of M' - M (as stated in greedy k members)

            if consider_diversity and current_cluster:
                diversity_flag, majority_label = is_diverse(cluster_labels= [tam_labels[idx] for idx in current_cluster],
                                  threshold= diversity_threshold)
                if not diversity_flag:
                    #dist = dist + diversity_penalty TODO check if this is better

                    if majority_label == tam_labels[i]:
                        dist = dist * diversity_penalty
            # Check if the calculated distance is less than the current minimum distance
            if dist < min_distance:
                min_distance = dist
                min_index = i
        if verbose:
            print(f'min index is {min_index} with min distance {min_distance}')
        return min_index
    
    def find_farthest_trace_index(remaining_instances, tams, S_M):
        #find instance that has the most average distance from all the super matrixes
        # in the paper it states that we should find the index with the most average distance
        # in the code but it looks for the index with the greatest distance from any of the centers
        
        
        max_average_distance = float('-inf')  # Start with a very small number
        max_index = None  # Placeholder for the index with maximum average distance

        # Iterate over each index in the remaining instances
        for i in remaining_instances:
            # Calculate the average distance from tams[i] to each element in S_M
            total_distance = sum(tam_euclidian_distance(tams[i], m) for m in S_M)
            avg_distance = total_distance / len(S_M)

            # Check if this average distance is greater than the current maximum
            if avg_distance > max_average_distance:
                max_average_distance = avg_distance
                max_index = i


        # max_distance = float('-inf')  # Start with a very small number
        # max_index = None  # Placeholder for the index with maximum average distance

        # # Iterate over each index in the remaining instances
        # for i in remaining_instances:
        #     # Calculate the average distance from tams[i] to each element in S_M
        #     for m in S_M:
        #         distance = tam_euclidian_distance(tams[i], m)
            

        #         # Check if this distance is greater than the current maximum
        #         if distance > max_distance:
        #             max_distance = distance
        #             max_index = i
        return max_index

    #initialization
    if not first_random_tam:
        random_initialize_instance = np.random.randint(0,C) #todo add fixed random state
    else:
        random_initialize_instance = first_random_tam
    S_A[0].append(random_initialize_instance)
    S_M.append(tams[random_initialize_instance])
    remaining_instances.remove(random_initialize_instance)
    
    if verbose:
        print(f'{random_initialize_instance} chosen as random initial website')
    for i in tqdm(range(num_anonimity_sets), desc= f'performing pallete for k = {k} and {num_anonimity_sets} clusters'):
        if i!= 0:
            c_bar = find_farthest_trace_index(remaining_instances, tams, S_M)
            if verbose:
                print(f'{c_bar} chosen as initial website')
            S_A[i].append(c_bar)
            S_M.append(tams[c_bar])
            remaining_instances.remove(c_bar)

        while len(S_A[i]) < k:
            
            c_bar = find_min_distance_index(remaining_instances,tams, S_M[i], current_cluster= S_A[i], tam_labels= tam_labels)
            # print(f'{c_bar} chosen as next point')
            S_A[i].append(c_bar)
            S_M[i] = compute_super_matrix(S_M[i], tams[c_bar]) 
            remaining_instances.remove(c_bar)
    if verbose:
        print("#########################")
    #Assigning remaining instances to nearest set
    for instance in tqdm(remaining_instances, desc= 'Assining the residual elements to their best clusters'):
        i_bar = find_min_distance_index(remaining_instances= set(range(0,num_anonimity_sets)), tams= S_M, m = tams[instance], current_cluster= None)
        S_A[i_bar].append(instance)
        S_M[i_bar] =  compute_super_matrix(S_M[i_bar], tams[instance])
    
    # print('final clusters are:')
    # for group in S_A:
    #     print(group)
    if post_processing == 'medium':
        S_A_refined, S_M_refined = refine_clusters_with_directed_swaps_medium(tams= tams, S_A= S_A, S_M= S_M, max_iterations= post_processing_max_iter)
        return S_A_refined, S_M_refined
    elif post_processing == 'heavy':
        S_A_refined, S_M_refined = refine_clusters_with_swaps_heavy(tams= tams, S_A= S_A, S_M= S_M, max_iterations= post_processing_max_iter)
        return S_A_refined, S_M_refined
    return S_A, S_M


def perform_kmedoids_on_website(dataset, class_num, ks, logger = None, metric = 'elbow'):
    """
    Given the traces of a website, perform kmedoids on them with different ks and perform best clusters


    Parameters
    ----------
        dataset: dataset containing the tam traces of all websites TODO maybe also cover other trace modes
        class_num: website number to be clusterd
        ks: list of ks to be used as for kmeans
        metric: how to choose the best cluster. currently it can be elbow, silhoette, glove, and data overhead?
        

    Returns
    -------
        the best clustering
    """

    if logger is None:
        logger = cm.global_logger
    tams, _ = dataset.get_traces_of_class(class_number= class_num)
    distance_matrix = np.zeros([len(tams), len(tams)])
    for i in tqdm(range(len(tams)), desc = f'Computing pair-wise distance of traces of website {class_num}'):
        for j in tqdm(range(len(tams)), desc = f'computing distance for instance {i + 1} / {len(tams)}', disable= True):
            if j > i:
                distance_matrix[i,j] = tam_euclidian_distance(tams[i], tams[j])
            elif j < i:
                distance_matrix[i,j] = distance_matrix[j, i]
    
    sse = []
    silhouette_scores = []
    glove_scores = []
    data = tams.reshape([tams.shape[0], -1]) # flattened tams

    #for glove cost function
    
    max_requests = {i : np.sum(tams[i][0]) for i in range(len(tams))}
    max_responses = {i : np.sum(data[i][1]) for i in range(len(tams))}
    # Perform k-means clustering for each k
    final_clusterings = []
    for k in tqdm(ks, desc= f'Kmeans with different ks for website {class_num}'):
        # kmeans = KMeans(n_clusters=k, random_state=42, n_init= 10)
        # kmeans.fit(data)
        # sse.append(kmeans.inertia_)
        # silhouette_avg = silhouette_score(data, kmeans.labels_)

        centers, cluster_labels, cost_history, iteration_num = k_medoids_custom(k = k, 
                                                                                        distance_matrix= distance_matrix,
                                                                                        logger = logger,
                                                                                        random_state= 42,
                                                                                        verbose= False)
        final_intra_distance = cost_history[-1]



        # elbow score
        sse.append(final_intra_distance)


        # glove score
        clusters = group_clusters(cluster_labels= cluster_labels, num_clusters= k) # dictionary of cluster index to cluster
        glove_cost = 0
        for i in range(k):
            glove_cost += glove_cost_function(max_requests = max_requests, max_responses = max_responses,
                                            indices = clusters[i])
        glove_scores.append(glove_cost)

        # silhoette score

        score = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
        silhouette_scores.append(score)


        # TODO add overhead section



        # saving the clustering to return the best at the end
        final_clusterings.append(clusters)

    if metric == 'elbow':
        best_k, best_index, _ = find_elbow_point(k_values= ks, sse= sse)

        return list(final_clusterings[best_index].values()), best_k, sse[best_index]

    elif metric == 'silhoette':
        best_index = silhouette_scores.index(max(silhouette_scores))
        return list(final_clusterings[best_index].values()), ks[best_index], silhouette_scores[best_index]

    elif metric == 'glove':
        best_index = glove_scores.index(max(glove_scores))
        return list(final_clusterings[best_index].values()), ks[best_index], glove_scores[best_index]

    elif metric == 'overhead':
        pass


