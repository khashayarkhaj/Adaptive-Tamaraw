# A set of metrics to evaluate the results of a clustering proccess
import numpy as np
from collections import Counter
from collections.abc import Mapping 

# helper function
def is_diverse(cluster_labels, threshold=1.0):
    """
    Given a certain cluster, we want to check if the elements in that cluster are diverse
    and identify the label that has the majority if it exists.

    Parameters
    ----------
        cluster_labels: list of labels (indices) of the traces in that cluster
        
        threshold: if the percentage of majority is >= threshold, then we declare the cluster as not diverse
    
    Returns
    -------
        tuple (boolean, int or None): 
            - boolean: whether the cluster is diverse
            - int or None: the label that has the majority or None if no single label meets the threshold
    """
    
    # Count the occurrences of each label in the cluster
    label_counts = Counter(cluster_labels)
    # Find the label with the maximum count
    max_label, max_count = label_counts.most_common(1)[0]
    # Calculate the fraction of the most common label
    max_label_fraction = max_count / len(cluster_labels)
    
    # Check if the cluster is diverse based on the threshold
    is_cluster_diverse = max_label_fraction < threshold
    # Return the majority label 
    majority_label = max_label 
    
    return is_cluster_diverse, majority_label

    

def ed_metric(clusters, labels, diversity_threshold = 1.0):
    """
    Returns the Equal Diversity Cost (ED) of a clustering result.


    Parameters
    ----------
        clusters: list of clusters, each having the indices that belong to a cluster like [[1,3,5], [0,2]]
        
        labels: list of labels of each element like [1,1,1,0, 5, 4, ...]

        diversity_threshold: the maximum fraction of majority in a cluster we consider it diverse
    Returns
    -------
        the Equal Diversity Cost (ED)
    """
    #this concept is used from the paper EFFICIENT K-ANONYMITY USING CLUSTERING TECHNIQUE
    ED = 0
    for cluster in clusters:
        cluster_labels = [labels[i] for i in cluster]
        weight = 1
        if is_diverse(cluster_labels= cluster_labels, threshold= diversity_threshold):
            weight = 0
        
        ED +=  weight * len(cluster)
    
    return ED
    



def compute_purity_of_a_cluster(cluster_labels):
    #computes the purity of a cluster, defined as max number of same labels divided be the length of the cluster
    label_count = Counter(cluster_labels)
    
    
    # Most frequent label in the cluster
    most_frequent = label_count.most_common(1)[0]
    # Purity of the current cluster
    
    

    cluster_purity = most_frequent[1] / len(cluster_labels)

    return cluster_purity, most_frequent, label_count

def purity_metrics(cluster_indices, labels):
    """
    Calculates purity metrics including the overall purity of the clustering, the purity of each cluster,
    and the maximum percentage of each class that is found in a single cluster.

    Parameters
    ----------
    cluster_indices : list of lists
        Each sublist contains the indices of elements that belong to one cluster.
    labels : list
        The labels corresponding to each element in the dataset. The indices in cluster_indices refer to this list.

    Returns
    -------
    tuple
        Contains three elements:
        - overall_purity (float): The overall purity of the clustering, calculated as the sum of the most frequent label in each cluster
          divided by the total number of points.
        - cluster_purities (list): A list containing the purity of each cluster, defined as the maximum number of elements of the same class
          divided by the cluster length.
        - max_percentages (dict): A dictionary where each key is a label and the value is the maximum percentage of that label found in any single cluster.
    """
    total_points = sum(len(cluster) for cluster in cluster_indices)
    overall_purity = 0
    cluster_purities = []
    label_max_percentage = {}
    all_labels_count = Counter(labels) # gives a dictionary of form { label: number_of_samples}


    # Process each cluster
    for cluster in cluster_indices:
        # Extract labels for the current cluster
        cluster_labels = [labels[i] for i in cluster]
        cluster_purity, most_frequent, label_count = compute_purity_of_a_cluster(cluster_labels)
        cluster_purities.append(cluster_purity)
        # Update overall purity calculation
        overall_purity += most_frequent[1]
        # Update label distribution for maximum percentage calculation
        for label, count in label_count.items():
            percentage = count / all_labels_count[label]
            if label not in label_max_percentage or percentage > label_max_percentage[label]:
                label_max_percentage[label] = percentage

    # Calculate overall purity
    overall_purity /= total_points



    return overall_purity, cluster_purities, label_max_percentage


def compute_website_protections(cluster_indices, intra_cluster_mapping, intra_cluster_percentages, intra_cluster_sizes):
    """
    Calculates how much each website is protected after two tier clustering.
    by protection, we mean the weighted average of protection a website gets for each cluster that it is in.

    Parameters
    ----------
    cluster_indices : list of lists
        Each sublist contains the indices of elements that belong to one cluster.
    intra_cluster_mapping : dict
        A mapping from the indices in the cluster to the actual website. this will be in form of 5 : [2,4] were 2 is the website number and 4 is the 4th cluster in website 2
    intra_cluster_percentages : list of lists
        Each sublist contains the percentages of elements that belong to that sub cluster.
    intra_cluster_sizes: list of lists
        Each sublist contains the number of elements that belong to that sub cluster.
    Returns
    -------
    list:
        The protection each website gets
    """

    
    if isinstance(intra_cluster_percentages, Mapping):
    # It’s a dict-like object → keep the same keys
        websites_protections = {key: 0 for key in intra_cluster_percentages}
    else:
        # It’s a sequence (list, tuple, numpy array, …) → keep the same length
        websites_protections = [0] * len(intra_cluster_percentages)

    """
        this part will probably be hard to understand!
        imagine we have cluster which has 5 elements from 3 websites. the labeling is like this:
        [0,1,1,0,2]
        imagine each of this elements, have a percentage like this from their original website:
        [0.3, 0.2, 0.5, 0.3, 0.7]
        this means that the first element has 30 percent of website zero, the second has 20 percent of website 1, and ...
        imagine each of this elements correspond to this much traces:
        [200, 400, 1000, 200, 700].
        thus, the relative percentage of each element in the cluster will be:
        [0.08, 0.16, 0.4, 0.08, 0.28].
        now if we want to compute the protection website 0 gets from this cluster, we do this:
        website 0 has 0.3 + 0.3 = 0.6 of its elements in this cluster
        the percentage of other websites present in this cluster is 1 - (0.08 + 0.08) = 0.84
        thus, the protection that website 0 gets is 0.6 * 0.84. 
        to get the overall protection website 0 gets, we need to sum this with protection that the rest of the 40 percent get.

    """
    for cluster in cluster_indices:
        percentages_in_this_cluster = {} # a mapping from actual website index to the percentage of that website present in this cluster
        size_distribition_of_this_cluster = [] # the fraction of the number of traces corresponding to each element divided by the total number of traces in this cluster
        labels_in_this_cluster = [] # the label of each element in this cluster
        for indice in cluster:
            website_of_this_element = intra_cluster_mapping[indice][0]
            intra_index_of_this_element = intra_cluster_mapping[indice][1] # the index of this element (e.g. supermatrix in the intra clusters of the website)
            percentage_of_this_element = intra_cluster_percentages[website_of_this_element][intra_index_of_this_element]
            percentages_in_this_cluster[website_of_this_element] = percentages_in_this_cluster.get(website_of_this_element, 0) + percentage_of_this_element

            size_distribition_of_this_cluster.append(intra_cluster_sizes[website_of_this_element][intra_index_of_this_element])
            labels_in_this_cluster.append(website_of_this_element)
        number_of_all_traces = sum(size_distribition_of_this_cluster)
        size_distribition_of_this_cluster = [element/number_of_all_traces for element in size_distribition_of_this_cluster]

        for website in percentages_in_this_cluster: # computing the protection for each website in this cluster
            # the percentage that this website allocates this cluster
            website_presence = sum([element for idx, element in enumerate(size_distribition_of_this_cluster) if labels_in_this_cluster[idx] == website])
            website_protection = 1 - website_presence
            websites_protections[website] += website_protection * percentages_in_this_cluster[website]
    
    return websites_protections
            
