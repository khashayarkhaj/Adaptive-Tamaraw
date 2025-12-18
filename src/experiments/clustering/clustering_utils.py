# a set of functions that can be used for different clustering algorithms and even visualizations
import numpy as np
import matplotlib.pyplot as plt
import utils.config_utils as cm
from utils.file_operations import save_file, load_file
from utils.overhead import total_data_overhead, total_time_overhead
from utils.trace_dataset import TraceDataset
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
from collections import Counter
#The next three functions are used for drawing the boxplots
def calculate_intra_cluster_distances_from_matrix(distance_matrix, clusters, k):
    intra_distances = []
    for cluster_idx in range(k):
        cluster_points = np.where(clusters == cluster_idx)[0]
        if len(cluster_points) > 1:
            dists = distance_matrix[np.ix_(cluster_points, cluster_points)]
            #The np.ix_ function takes multiple sequences (typically 1D arrays or lists) and returns a tuple of arrays. 
            # Each array in the tuple can be used to index a specific dimension of a multidimensional array.


            intra_distances.append(np.mean(dists[np.triu_indices_from(dists, k=1)]))
            #np.triu_indices_from is a function that returns the indices of the upper triangular part of a matrix.
            # The argument k=1 specifies that we want the upper triangular part excluding the diagonal. 
            # If k=0, it would include the diagonal, and if k is positive, it excludes the first k diagonals above the main diagonal.
        else:
            intra_distances.append(0)
    return intra_distances

def calculate_inter_cluster_distances_from_matrix(distance_matrix, clusters, k):
    inter_distances = [[] for _ in range(k)]
    
    for i in range(k):
        for j in range(k):
            if i != j:
                # Find points in clusters i and j
                cluster_i_points = np.where(clusters == i)[0]
                cluster_j_points = np.where(clusters == j)[0]
                
                # Extract distances between points in cluster i and cluster j
                dists = distance_matrix[np.ix_(cluster_i_points, cluster_j_points)]
                
                # Calculate the mean distance
                mean_dist = np.mean(dists)
                
                # Append the mean distance to the corresponding cluster's list
                inter_distances[i].append(mean_dist)
    
    
    return inter_distances

def plot_clusters_with_distance_matrix(distance_matrix, clusters, k, dir_path = None, text_title = None, sampling = False, file_name = None):
    intra_distances = calculate_intra_cluster_distances_from_matrix(distance_matrix, clusters, k)
    inter_distances = calculate_inter_cluster_distances_from_matrix(distance_matrix, clusters, k)
    
    
    plt.figure(figsize=(12, 6))
    
    # # Create boxplot for inter-cluster distances
    # plt.boxplot(inter_distances, positions=range(1, k+1), widths=0.6, showfliers=False)
    
    # # Overlay intra-cluster distances
    # plt.scatter(range(1, k+1), intra_distances, color='blue', label='Intra-cluster Distance', zorder=3)
    
    # plt.xlabel('Cluster Index')
    # plt.ylabel('Distance')
    # if text_title:
    #     plt.title(text_title)
    # else:
    #     plt.title('Similarity Measure Among Clusters')
    # plt.legend()
    # first I was using the code above, but then I'm using the code below based on the source code of palette


    plt.xticks(fontsize = 20)

    plt.yticks(fontsize = 20)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)


    plt.boxplot(inter_distances,
                showfliers=False,
                labels=[str(i) for i in range(1, k + 1)],
                medianprops={"color": "red", "linewidth": 1},  # 设置中位数的属性，如线的类型、粗细等
                boxprops={"color": "red", "linewidth": 1},  # 设置箱体的属性，如边框色，填充色等
                whiskerprops={"color": "red", "linewidth": 1},  # 设置须的属性，如颜色、粗细、线的类型等
                capprops={"color": "red", "linewidth": 1})
    #plt.title('Boxplot Example')
    plt.ylabel('Distance',fontsize = 20)
    plt.xlabel('Anonymity Set Index',fontsize = 20)
    for i in range(len(intra_distances)):
        plt.plot(i+1, intra_distances[i], marker = "*",color = "blue",markersize = 9)
    custom_legend = [
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=15, label='Intra-cluster Distance'),
]
    plt.grid(axis="y", linestyle='--')
    plt.legend(handles=custom_legend,fontsize = 20,loc = "upper right")
    plt.gca().spines['top'].set_linewidth(2)  # 设置顶部边框粗细为2
    plt.gca().spines['right'].set_linewidth(2)  # 设置右侧边框粗细为2
    plt.gca().spines['bottom'].set_linewidth(2)  # 设置底部边框粗细为2
    plt.gca().spines['left'].set_linewidth(2)  # 设置左侧边框粗细为2

    if text_title:
        plt.title(text_title)
    else:
        plt.title('Similarity Measure Among Clusters')
    if dir_path:
        if not file_name:
            file_name = 'boxplots-sampling.png' # the distances have been computed through smapling
            if not sampling:
                file_name = 'boxplots-representatives.png'
        save_file(dir_path= dir_path, file_name= file_name)
    else:
        plt.show()
def group_clusters(cluster_labels, num_clusters):
    #given a group of cluster labels (ndarray), divide them into their respective clusters and return a list of ndarrays
    # clusters = []
    # for k in range(num_clusters):
    #     cluster_indices = np.where(clusters == k)
    #     cluster_elements = clusters[cluster_elements]
    #     clusters.append(cluster_elements)
    
    # return clusters

    # gpt proposed a better implementation in which we use dictionary
    clusters = {label: [] for label in range(num_clusters)}

    for i in range(len(cluster_labels)):
        clusters[cluster_labels[i]].append(i)
    
    return clusters

def visualize_clusters(cluster_labels, num_clusters, save_dir, visualization_type = 'heatmap'):
    # different ways of visualizing the class-based(for now TODO) clustering results
    clusters = group_clusters(cluster_labels, num_clusters)
    clusters = [clusters[i] for i in range(num_clusters)] # converting clusters from dict to list
    if visualization_type == 'heatmap':
        cluster_membership = pd.DataFrame({'Instance': range(len(clusters)), 'Cluster': clusters})

        # Create a count of instances per cluster
        cluster_counts = cluster_membership.groupby('Cluster')['Instance'].count().reset_index()

        # Pivot the data for the heatmap
        cluster_pivot = cluster_membership.pivot_table(index='Instance', columns='Cluster', aggfunc='size', fill_value=0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cluster_pivot, cmap='viridis', cbar_kws={'label': 'Count'})
        plt.title('Cluster Membership Heatmap')
        plt.xlabel('Cluster')
        plt.ylabel('Instance')
        save_file(save_dir, 'heatmap.png')

def visualize_overall_clusters(cluster_labels, num_clusters, save_dir):
    # different ways of visualizing the overall-based clustering results
    
    true_labels = [] # which website each instance belongs to
    for website_number in range(cm.MON_SITE_NUM):
        true_labels += [website_number for i in range(cm.MON_INST_NUM)]
    
    df = pd.DataFrame({'Cluster': cluster_labels, 'Class': true_labels})

    # Creating a contingency table
    contingency_table = pd.crosstab(df['Cluster'], df['Class'])

    # Visualizing the contingency table using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title('Heatmap of Contingency Table')
    plt.ylabel('Clusters')
    plt.xlabel('Classes')
    save_file(save_dir, 'contingency.png')


    



    percentage_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    percentage_table.plot(kind='bar', stacked=True, colormap='viridis')
    plt.ylabel('Percentage')
    plt.title('Percentage of Classes per Cluster')
    plt.xticks(rotation=0)  # Keep the cluster names horizontal for better readability
    save_file(save_dir, 'percentage.png')



    total_instances = contingency_table.sum(axis=1)
    fig, axes = plt.subplots(5, 2, figsize=(20, 25))  # Adjust subplot grid as needed
    axes = axes.flatten()

    for i, cluster in enumerate(percentage_table.index):
        top_classes = percentage_table.loc[cluster].sort_values(ascending=False).head(10)
        ax = top_classes.plot(kind='bar', ax=axes[i], color='teal', ylim=(0, 100))
        axes[i].set_title(f'Cluster {cluster} (Total instances: {total_instances[cluster]})')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Percentage')
        axes[i].set_ylim(0, 100)


    # Annotate custom percentages on the bars
    for idx, p in enumerate(ax.patches):
        class_name = top_classes.index[idx]  # Get the class name from the top_classes index
        custom_percentage = custom_percentage_table.loc[cluster, class_name]  # Retrieve the custom percentage
        ax.annotate(f"{custom_percentage:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')


    plt.tight_layout()
    save_file(save_dir, 'top-websites.png')


def create_interval_hsitogram(distance_matrix, start, end, top_k = 5, super_matrix = False):
# display a histogram of how many times each index has appeared in the interval [start, end) of other indices when ranking the similarities between elements
# Assuming 'distance_matrix' is your [95, 95] ndarray
    np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances by setting them to infinity

    # Initialize a dictionary to hold the frequency of each class
    frequency_dict = {i: 0 for i in range(95)}

    number_of_elements = distance_matrix.shape[0]
    # Extract the indices for the interval and count frequencies
    for i in range(number_of_elements):
        sorted_indices = np.argsort(distance_matrix[i])[start:end]
        for index in sorted_indices:
            frequency_dict[index] += 1

    # Data for plotting
    classes = list(frequency_dict.keys())
    frequencies = list(frequency_dict.values())
    class_indices = np.arange(0, number_of_elements)


    
        
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.bar(classes, frequencies, color='teal')

    # Find the indices of the top k values
    top_k_indices = np.argsort(frequencies)[-top_k:]

    

    # Annotate the top k values
    for idx in top_k_indices:
        plt.text(class_indices[idx], frequencies[idx] + 1, str(class_indices[idx]), 
                ha='center', va='bottom', color='red')
        
        
    plt.xlabel('Class Index')
    plt.ylabel('Frequency of Appearance')
    if super_matrix:
        plt.title(f'Frequency of Each Class Appearing as a {start}-{end}th Nearest Neighbor using SuperMatrix')
    else:
        plt.title(f'Frequency of Each Class Appearing as a {start}-{end}th Nearest Neighbor using samples')
    plt.xticks(np.arange(0, number_of_elements, 5))  # Set x-ticks to show every 5th class for clarity
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    folder_name = 'samples'
    if super_matrix:
        folder_name = 'super-matrix'
    save_dir = os.path.join(cm.BASE_DIR, 'results', 'dataset-statistics', cm.data_set_folder, 'class-wise-distance', folder_name)
    file_name = f'{start}_{end}.png'
    save_file(dir_path= save_dir, file_name= file_name)


def compute_clustering_overheads(clusters, dataset, cluster_representatives, granularity = 'class',
                                  trace_mode = 'tam', plot = True, save_dir = None, percentage = False,
                                 intra_clusters = None, super_matrix_mapping = None, verbose = False):
    """
    Calculate the total data (bandwith) and time overhead that is incured by our clustering.



    Parameters
    ----------
        clusters : a list that contains the cluster number each instance is assigned to e.g. [2,0,5,1,...]
        dataset : dataset containing the traces used for the clustering
        cluster_representatives: the representative that is assigned to each class
        granularity: the level that our clustering has been performed. if it is class, the representatives of all the instances of the class will be the same.
        trace_mode: the mode of each trace. can be tam, cell, burst - currently if we have tam, we can find the time overhead - otherwise we can only compute bandwidth overhead
        plot: if true, it also plots the results and saves them based on the configs in cm
        save_dir: if available, it saves the plot
        percentage: whether to show the results as percents
        intra_clusters: for two tier clustering. the clusters in each website
        super_matrix_mapping: for two tier clustering. a mapping that is index: [website_num, cluster_num]
        verbose : wether to show the progress
        
    Returns
    -------
        data overhead incurred to our dataset, based on the representatives (e.g. supermatrix - super sequence) of each website
    """
    overall_data_overhead = 0
    overall_time_overhead = 0

    per_website_data_overhead = []
    per_website_time_overhead = []

    if trace_mode == 'tam':
        tams_before = np.zeros([0,2, dataset.tam_length])
        tams_after = np.zeros([0,2, dataset.tam_length])
        if granularity == 'class': # TODO implement other cases as well
            
            for element_idx in range(len(clusters)):
                tams_of_website_before, _ = dataset.get_traces_of_class(class_number = element_idx)
                tams_of_website_after = np.array([cluster_representatives[clusters[element_idx]] for i in range(len(tams_of_website_before))])
                # after defense, the representative will be the new trace of all the traces of this website

                ### computing the data/time overhead for this website
                data_overhead_of_this_website = total_data_overhead(traces_before= tams_of_website_before,
                                                                    traces_after= tams_of_website_after,
                                                                    trace_type = 'tam',
                                                                    return_percentage= percentage,
                                                                    verbose= verbose)
                
                time_overhead_of_this_website = total_time_overhead(traces_before= tams_of_website_before,
                                                                    traces_after= tams_of_website_after,
                                                                    trace_type = 'tam',
                                                                    return_percentage= percentage,
                                                                    verbose= verbose)

                per_website_data_overhead.append(data_overhead_of_this_website)
                per_website_time_overhead.append(time_overhead_of_this_website)


                ### storing the before/after tams to compute the overall overhead at the end
                tams_before = np.append(tams_before, tams_of_website_before, axis =0)
                tams_after = np.append(tams_after, tams_of_website_after, axis = 0)
                 

            ### computing the total overhead incurred on the dataset
            overall_data_overhead = total_data_overhead(traces_before= tams_before,
                                                        traces_after= tams_after,
                                                        trace_type= 'tam',
                                                        return_percentage= percentage,
                                                        verbose= verbose)  
            
            overall_time_overhead = total_time_overhead(traces_before= tams_before,
                                                        traces_after= tams_after,
                                                        trace_type= 'tam',
                                                        return_percentage= percentage,
                                                        verbose= verbose)  
        elif granularity == 'two-tier':
             # Initialize an empty set to store unique website indexes
            tams_before = np.zeros([0,2, dataset.tam_length])
            tams_after = np.zeros([0,2, dataset.tam_length])
            unique_website_indexes = set()

            # Iterate over the dictionary values
            for value in super_matrix_mapping.values():
                website_index = value[0]  # Extract the website index
                unique_website_indexes.add(website_index)  # Add to the set

            # Convert the set to a list if needed
            unique_website_indexes_list = sorted(list(unique_website_indexes))
            for website_index in tqdm(unique_website_indexes_list, desc= 'computing the overhead of two tier clusters', disable= not verbose):
                tams_of_this_website, _ = dataset.get_traces_of_class(class_number = website_index)
                tams_of_website_before = []
                tams_of_website_after = []
                for key, value in super_matrix_mapping.items():
                    if value[0] == website_index:
                        cluster_index = value[1]
                        indices_in_the_cluster = intra_clusters[website_index][cluster_index]
                        tams_of_this_cluster = tams_of_this_website[indices_in_the_cluster]
                        tams_of_website_before += tams_of_this_cluster.tolist()
                        tams_of_website_after += [cluster_representatives[clusters[key]] for i in range(len(indices_in_the_cluster))]
                ### computing the data/time overhead for this website
                tams_of_website_before = np.array(tams_of_website_before)
                tams_of_website_after = np.array(tams_of_website_after)

                data_overhead_of_this_website = total_data_overhead(traces_before= tams_of_website_before,
                                                                    traces_after= tams_of_website_after,
                                                                    trace_type = 'tam',
                                                                    return_percentage= percentage,
                                                                    verbose= verbose)
                
                time_overhead_of_this_website = total_time_overhead(traces_before= tams_of_website_before,
                                                                    traces_after= tams_of_website_after,
                                                                    trace_type = 'tam',
                                                                    return_percentage= percentage,
                                                                    verbose= verbose)

                per_website_data_overhead.append(data_overhead_of_this_website)
                per_website_time_overhead.append(time_overhead_of_this_website)


                ### storing the before/after tams to compute the overall overhead at the end
                tams_before = np.append(tams_before, tams_of_website_before, axis =0)
                tams_after = np.append(tams_after, tams_of_website_after, axis = 0)
                 

            ### computing the total overhead incurred on the dataset
            overall_data_overhead = total_data_overhead(traces_before= tams_before,
                                                        traces_after= tams_after,
                                                        trace_type= 'tam',
                                                        return_percentage= percentage,
                                                        verbose= verbose)  
            
            overall_time_overhead = total_time_overhead(traces_before= tams_before,
                                                        traces_after= tams_after,
                                                        trace_type= 'tam',
                                                        return_percentage= percentage,
                                                        verbose= verbose)  
    

    if plot:
        # Data for plotting
        websites = np.arange(len(per_website_data_overhead))  # Assuming consecutive integers for website IDs
        # print(per_website_data_overhead)
        # Plot for Data Overhead
        plt.figure(figsize=(10, 5))
        plt.bar(websites, per_website_data_overhead, color='blue', label='Data Overhead per Website')
        plt.text(0.5, 0.95, f'Overall Data Overhead: {overall_data_overhead:.2f}', transform=plt.gca().transAxes,
         horizontalalignment='center', verticalalignment='top', fontsize=12, color='red')

        plt.xlabel('Website ID')
        if percentage:
            plt.ylabel('Data Overhead (%)')
        else:
            plt.ylabel('Data Overhead (%)')
        plt.title('Data Overhead per Website')
        plt.legend()
        if save_dir:
            save_file(dir_path= save_dir, file_name= 'data_overhead.png')
        else:
            plt.show()

        # Plot for Time Overhead
        plt.figure(figsize=(10, 5))
        plt.bar(websites, per_website_time_overhead, color='green', label='Time Overhead per Website')
        plt.text(0.5, 0.95, f'Overall Time Overhead: {overall_time_overhead:.2f}', transform=plt.gca().transAxes,
         horizontalalignment='center', verticalalignment='top', fontsize=12, color='red')

        plt.xlabel('Website ID')
        if percentage:
            plt.ylabel('Time Overhead (%)')
        else:
            plt.ylabel('Time Overhead')
        plt.title('Time Overhead per Website')
        plt.legend()
        if save_dir:
            save_file(dir_path= save_dir, file_name= 'time_overhead.png')
        else:
            plt.show()
    
    return overall_data_overhead, overall_time_overhead, per_website_data_overhead, per_website_time_overhead


def find_elbow_point(k_values, sse):
    # Ensure k_values and sse have the same length
    assert len(k_values) == len(sse), "k_values and sse must be the same length"
    
    # Coordinates of all points
    points = np.array(list(zip(k_values, sse)))
    
    # Line between the first and last points
    line = np.array([points[0], points[-1]])
    
    # Calculate distance of each point from the line
    distances = []
    for i in range(len(points)):
        p = points[i]
        a, b = line[0], line[1]
        # Vector AB
        ab = b - a
        # Vector AP
        ap = p - a
        # Cross product of AB and AP to find the area of parallelogram
        cross_product = np.linalg.norm(np.cross(ab, ap))
        # Distance is area divided by base length (|AB|)
        distance = cross_product / np.linalg.norm(ab)
        distances.append(distance)
    
    # Find the index of the maximum distance
    elbow_index = np.argmax(distances)
    
    # The best k is the k_value at the elbow point
    best_k = k_values[elbow_index]
    
    return best_k, elbow_index, distances


def show_percentage_of_clusters(clusters_of_each_website,visualization_type = 'plot', save_path = None):
    # visualizing the clustering percentage of each website
    # Initialize an empty list to store the results
    percentage_clusters_of_each_website = []

    # Calculate the percentages
    for website_idx in clusters_of_each_website:
        website_clusters = clusters_of_each_website[website_idx]
        total_sequences = sum(len(cluster) for cluster in website_clusters)
        percentages = [(len(cluster) / total_sequences) * 100 for cluster in website_clusters]
        percentages = sorted(percentages, reverse = True)
        percentage_clusters_of_each_website.append(percentages)

    if visualization_type == 'plot':
        # Visualization
        num_websites = len(percentage_clusters_of_each_website)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set up bar width and positions
        bar_width = 0.35
        index = np.arange(max(len(website) for website in percentage_clusters_of_each_website))

        # Generate bars for each website
        for i, percentages in enumerate(percentage_clusters_of_each_website):
            bar_positions = index + i * bar_width
            ax.bar(bar_positions[:len(percentages)], percentages, bar_width, label=f'Website {i+1}')

        # Adding labels and title
        ax.set_xlabel('Clusters')
        ax.set_ylabel('Percentage')
        ax.set_title('Cluster Percentages for Each Website')
        ax.set_xticks(index + bar_width * (num_websites - 1) / 2)
        ax.set_xticklabels([f'Cluster {i+1}' for i in range(max(len(website) for website in percentage_clusters_of_each_website))])
        ax.legend()

        plt.tight_layout()

        if save_path:
            save_file(dir_path= save_path, file_name = 'clusters.png')
        else:
            plt.show()
    
    elif visualization_type == 'text':
        # Initialize an empty string to store the text content
        content = ""

        # Build the content
        for idx, percentages in enumerate(percentage_clusters_of_each_website):
            line = f'Website number {idx}, {len(percentages)} clusters : '
            line += f' '.join(f'{percentage:.2f}%' for percentage in percentages)
            content += line + ' \n'
            content += '\n'
            content += '#######################' + '\n'
        save_file(dir_path= save_path, file_name= 'clusters.txt', content= content)



def cluster_size_distribution(clusters, clustering_algorithm = '', dir_path = None, website_index = None):
    """
    Calculates the distribution of cluster sizes from a list of clusters.

    Args:
    - clusters (list of lists): Each sublist represents a cluster and contains indices.
    - clustering_algorithm: the algorithm used for clustering
    - dir_path: If given, the directory path we want to save our plot in.
    - website_index: if given, the specific website which has been clustered

    Returns:
    dict: A dictionary where keys are cluster sizes and values are counts of those sizes.
    """
    # Calculate the size of each cluster
    cluster_sizes = [len(cluster) for cluster in clusters]
    
    # Count the frequency of each cluster size
    size_distribution = Counter(cluster_sizes)
    
    # Optionally plot the distribution
    total_clusters = len(clusters)  # Total number of clusters
    plt.figure(figsize=(8, 5))
    plt.bar(size_distribution.keys(), size_distribution.values(), color='blue')
    plt.xlabel('Cluster Size')
    plt.ylabel('Count')
    if website_index:
        plt.title(f'Distribution of Cluster Sizes of website {website_index} in {clustering_algorithm}')
    else:
        plt.title(f'Distribution of Cluster Sizes {clustering_algorithm}')
    plt.grid(True)
    # Annotate the total number of clusters on the plot
    plt.annotate(f'Total clusters: {total_clusters}',
                    xy=(0.75, 0.95), xycoords='axes fraction',
                    fontsize=12, color='red')

    if dir_path:
        filename = f'cluster_dist_{website_index}_{clustering_algorithm}.png' 
        save_file(dir_path= dir_path, file_name= filename)
    else:
        plt.show()
    
    return size_distribution


import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain

def combined_cluster_size_distribution(cluster_lists, clustering_algorithm = '', dir_path = None, website_index = None):
    """
    Calculates the combined distribution of cluster sizes from multiple lists of clusters.

    Args:
    - cluster_lists (list of list of lists): Each inner list represents a cluster and contains indices.
    - clustering_algorithm: the algorithm used for clustering
    - dir_path: If given, the directory path we want to save our plot in.
    - website_index: if given, the specific website which has been clustered

    Returns:
    dict: A dictionary where keys are cluster sizes and values are counts of those sizes across all lists.
    """
    # Flatten the list of lists of clusters into a single list of clusters
    all_clusters = list(chain.from_iterable(cluster_lists))
    
    # Calculate the size of each cluster
    cluster_sizes = [len(cluster) for cluster in all_clusters]
    
    # Count the frequency of each cluster size
    size_distribution = Counter(cluster_sizes)
    
    # Optionally plot the distribution
    
    plt.figure(figsize=(10, 5))
    plt.bar(size_distribution.keys(), size_distribution.values(), color='blue')
    plt.xlabel('Cluster Size')
    plt.ylabel('Count')
    if website_index:
        plt.title(f'Overall Distribution of Cluster Sizes of website {website_index} in {clustering_algorithm}')
    else:
        plt.title(f'Overall Distribution of Cluster Sizes {clustering_algorithm}')
    plt.grid(True)

    if dir_path:
        filename = f'cluster_dist_overall_{clustering_algorithm}.png' 
        save_file(dir_path= dir_path, file_name= filename)
    else:
        plt.show()
    
    return size_distribution

def number_of_clusters_distribution(cluster_lists, clustering_algorithm = '', dir_path = None):
    """
    Calculates the distribution of the number of clusters across multiple lists of clusters.

    Args:
    - cluster_lists (list of list of lists): Each sublist contains clusters, which are themselves lists of indices.
    - clustering_algorithm: the algorithm used for clustering
    - dir_path: If given, the directory path we want to save our plot in.

    Returns:
    dict: A dictionary where keys are the numbers of clusters and values are counts of lists with those cluster numbers.
    """
    # Calculate the number of clusters in each list
    num_clusters = [len(clusters) for clusters in cluster_lists]
    
    # Count the frequency of each number of clusters
    cluster_counts_distribution = Counter(num_clusters)
    
    # Optionally plot the distribution
    
    plt.figure(figsize=(10, 5))
    plt.bar(cluster_counts_distribution.keys(), cluster_counts_distribution.values(), color='blue')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Count of Lists')
    plt.title(f'Distribution of Number of Clusters {clustering_algorithm}')
    plt.grid(True)
    if dir_path:
        filename = f'cluster_nums_{clustering_algorithm}.png' 
        save_file(dir_path= dir_path, file_name= filename)
    else:
        plt.show()
    
    return cluster_counts_distribution

def identify_pareto(scores):
    # Initialize a boolean array to identify Pareto points
    is_pareto = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_pareto[i]:
            # Keep only points that are not dominated by the current point
            is_pareto[is_pareto] = np.any(scores[is_pareto] > c, axis=1)  # Must be better in at least one dimension to be considered non-dominated
    return is_pareto


def load_two_tier_clusters(
    k,  # Number of second-tier clusters
    algorithm_tier1='palette',  # First-tier clustering algorithm (e.g., 'palette', 'cast')
    algorithm_tier2='cast',  # Second-tier clustering algorithm (e.g., 'cast', 'kmeans')
    preload_clusters=True,  # Whether to preload first-tier clusters from disk
    max_clusters=5,  # Maximum number of first-tier clusters per website
    extract_ds=False,  # Whether to extract dataset traces
    post_processing='',  # Post-processing method applied to clusters (e.g., 'merging', 'splitting')
    trace_mode='tam',  # Trace representation mode (e.g., 'tam', 'raw')
    trim_traces=False,  # Whether to trim/pad traces to fixed length
    replace_negative_ones=False,  # Whether to replace -1 values in traces
    trim_length=None,  # Fixed length for trace trimming (None uses default from config)
    return_original_traces=False,  # Whether to return untransformed original traces
    second_tier_file_name=None,  # Override file name for second-tier clusters
    second_tier_cluster_path=None,  # Override directory path for second-tier clusters
    l_tamaraw=None,  # Tamaraw L parameter for defense simulation
):
    """
    Load and organize two-tier clustering results for website fingerprinting analysis.
    
    This function loads hierarchical clusters where:
    - Tier 1: Clusters within each website (intra-website clustering)
    - Tier 2: Clusters across websites (inter-website clustering)
    
    Args:
        k: Number of second-tier (cross-website) clusters
        algorithm_tier1: Algorithm used for first-tier clustering
        algorithm_tier2: Algorithm used for second-tier clustering
        preload_clusters: Load pre-computed first-tier clusters from disk
        max_clusters: Maximum first-tier clusters per website
        extract_ds: Extract traces from dataset
        post_processing: Post-processing method name (empty string for none)
        trace_mode: Trace representation format
        trim_traces: Normalize all traces to same length
        replace_negative_ones: Replace -1 padding values
        trim_length: Target length for trace normalization
        return_original_traces: Include raw trace data in output
        second_tier_file_name: Custom filename for tier-2 clusters
        second_tier_cluster_path: Custom path for tier-2 clusters
        l_tamaraw: Tamaraw defense parameter L
        
    
    Returns:
        dict: Dictionary containing:
            - tier1_clusters_of_each_website: List of first-tier clusters per website
            - tier2_clusters: Second-tier cluster assignments
            - super_matrix_mapping: Maps super-matrix indices to (website, tier1_cluster)
            - ordered_traces: All traces ordered by original dataset indices
            - ordered_labels: Second-tier cluster label for each trace
            - ordered_websites: True website label for each trace
            - overall_mapping: Maps (website_idx, trace_idx) to (tier1, tier2) clusters
            - ordered_original_traces: Original untransformed traces (if requested)
            - reverse_mapping: Maps (website, tier1_cluster) to tier2_cluster
            - ordered_tier1_labels: First-tier cluster label for each trace
            - cluster_to_website_mapping: Set of websites in each tier-2 cluster
    """
    
    # ========================================
    # STEP 1: Load First-Tier Clusters
    # ========================================
    # First-tier clusters group similar traces within each website
    tier1_clusters_of_each_website = []
    
    if algorithm_tier1 == 'cast':
        if preload_clusters:
            # Load pre-computed CAST clusters for each monitored website
            for website_num in range(cm.MON_SITE_START_IND, cm.MON_SITE_END_IND):
                load_path = os.path.join(
                    cm.BASE_DIR, 'data', cm.data_set_folder, 'clustering', 
                    'cast', f'max_clusters_{max_clusters}', str(website_num)
                )
                # Each cluster file contains lists of trace indices belonging to that cluster
                clusters_of_this_website = load_file(
                    dir_path=load_path, 
                    file_name='clusters.pkl'
                )
                tier1_clusters_of_each_website.append(clusters_of_this_website)
    
    # ========================================
    # STEP 2: Load Second-Tier Clusters
    # ========================================
    # Second-tier clusters group first-tier clusters across different websites
    
    # Build the directory path for second-tier clusters
    load_dir = os.path.join(
        cm.BASE_DIR, 'data', cm.data_set_folder, 'clustering', 'two-tier', 
        f'{algorithm_tier1}-{algorithm_tier2}',
        *([]),
        f'{algorithm_tier1}-{max_clusters}', 
        f'{algorithm_tier2}-{k}'
    )
    
    # Determine the search term for cluster files
    search_term = 'second_tier_clusters'
    if l_tamaraw is not None:
        # Tamaraw defense simulation uses special directory structure
        load_dir = os.path.join(load_dir, f'l_{l_tamaraw}')
        search_term = f'second_tier_clusters_L_{l_tamaraw}'
    
    if post_processing != '':
        # Override path if post-processing was applied
        load_dir = os.path.join(
            cm.BASE_DIR, 'data', cm.data_set_folder, 'clustering', 'two-tier',
            f'{algorithm_tier1}-{algorithm_tier2}',
            f'{algorithm_tier1}-{max_clusters}', 
            f'{algorithm_tier2}-{k}',
            f'with_{post_processing}_post_processing'
        )
    
    # Load second-tier cluster file
    if second_tier_file_name is None and second_tier_cluster_path is None:
        # Auto-detect cluster file name
        second_tier_cluster_file_name = None
        for filename in os.listdir(load_dir):
            print(load_dir)
            if search_term in filename:
                second_tier_cluster_file_name = filename
        
        tier2_clusters = load_file(
            dir_path=load_dir, 
            file_name=second_tier_cluster_file_name
        )
    else:
        # Use provided path and filename
        print('Loading tier-2 clusters from provided path')
        tier2_clusters = load_file(
            dir_path=second_tier_cluster_path, 
            file_name=second_tier_file_name
        )
    
    # Load super-matrix mapping
    # Maps super-matrix index (first-tier cluster ID) to [website_num, cluster_num]
    # Example: super_matrix_mapping[5] = [2, 4] means super-matrix 5 is 
    # the 4th cluster of website 2
    mapping_file_name = 'super_matrix_mappings.pickle'
    super_matrix_mapping = load_file(
        dir_path=load_dir, 
        file_name=mapping_file_name
    )
    
    # ========================================
    # STEP 3: Load and Process All Traces
    # ========================================
    # Load the complete trace dataset
    overall_dataset = TraceDataset(
        extract_traces=extract_ds,
        trace_mode=trace_mode,
        keep_original_trace=return_original_traces
    )
    
    # Apply trace preprocessing if requested
    if replace_negative_ones:
        overall_dataset.replace_negative_ones()
    
    if trim_traces:
        trace_length = trim_length if trim_length is not None else cm.trace_length
        overall_dataset.trim_or_pad_traces(trace_length=trace_length)
    
    # ========================================
    # STEP 4: Organize Traces by Cluster Labels
    # ========================================
    # Initialize arrays to preserve original dataset ordering
    # This ensures trace indices remain consistent with the original dataset
    num_traces = len(overall_dataset.directions)
    ordered_traces = [None] * num_traces  # Trace data
    ordered_labels = [None] * num_traces  # Second-tier cluster labels
    ordered_tier1_labels = [None] * num_traces  # First-tier cluster labels
    ordered_websites = [None] * num_traces  # True website labels
    ordered_original_traces = [None] * num_traces  # Untransformed traces
    
    # Mappings to track cluster relationships
    overall_mapping = {}  # (website_idx, trace_idx) -> (tier1_cluster, tier2_cluster)
    super_matrix_labeling = {}  # super_matrix_idx -> tier2_cluster
    cluster_to_website_mapping = {}  # tier2_cluster -> set of website indices
    
    # ========================================
    # STEP 5: Assign Labels to Each Trace
    # ========================================
    # Iterate through second-tier clusters and assign labels
    for tier2_idx, tier2_cluster in enumerate(tier2_clusters):
        cluster_to_website_mapping[tier2_idx] = set()
        
        # Process each super-matrix (first-tier cluster) in this second-tier cluster
        for super_matrix_idx in tier2_cluster:
            super_matrix_labeling[super_matrix_idx] = tier2_idx
            
            # Get website and first-tier cluster for this super-matrix
            website_index, tier1_cluster_index = super_matrix_mapping[super_matrix_idx]
            
            # Get traces belonging to this first-tier cluster
            traces_in_this_subcluster = tier1_clusters_of_each_website[website_index][tier1_cluster_index]
            traces_of_this_website, _ = overall_dataset.get_traces_of_class(class_number=website_index)
            
            cluster_to_website_mapping[tier2_idx].add(website_index)
            
            # Label each trace in this first-tier cluster
            for trace_index in traces_in_this_subcluster:
                # Get the true index in the original dataset
                real_trace_idx = overall_dataset.get_real_indice_of_a_trace(
                    website_idx=website_index,
                    instance_idx=trace_index
                )
                
                # Store trace and labels at original position
                ordered_traces[real_trace_idx] = traces_of_this_website[trace_index]
                ordered_labels[real_trace_idx] = tier2_idx
                ordered_websites[real_trace_idx] = website_index
                ordered_tier1_labels[real_trace_idx] = tier1_cluster_index
                
                # Record mapping
                overall_mapping[(website_index, trace_index)] = (tier1_cluster_index, tier2_idx)
                
                # Store original trace if requested
                if return_original_traces:
                    ordered_original_traces[real_trace_idx] = overall_dataset.original_traces[real_trace_idx]
    
    # ========================================
    # STEP 6: Convert to Numpy Arrays
    # ========================================
    try:
        ordered_traces = np.array(ordered_traces)
    except ValueError as e:
        print("Traces have different shapes, keeping as list")
    
    # ========================================
    # STEP 7: Create Reverse Mapping
    # ========================================
    # Build reverse lookup: (website, tier1_cluster) -> tier2_cluster
    reverse_mapping = {}
    for second_tier_element, (website, cluster) in super_matrix_mapping.items():
        if website not in reverse_mapping:
            reverse_mapping[website] = {}
        reverse_mapping[website][cluster] = super_matrix_labeling[second_tier_element]
    
    # ========================================
    # STEP 8: Return Results
    # ========================================
    results = {
        'tier1_clusters_of_each_website': tier1_clusters_of_each_website,
        'tier2_clusters': tier2_clusters,
        'super_matrix_mapping': super_matrix_mapping,
        'ordered_traces': ordered_traces,
        'ordered_labels': np.array(ordered_labels),
        'ordered_websites': np.array(ordered_websites),
        'overall_mapping': overall_mapping,
        'ordered_original_traces': ordered_original_traces,
        'reverse_mapping': reverse_mapping,
        'ordered_tier1_labels': ordered_tier1_labels,
        'cluster_to_website_mapping': cluster_to_website_mapping
    }
    
    return results


def tamaraw_overhead_vector(trace_oh_dict, trace_indices, num_combinations = 196, desired_combination = None, desired_pairs = None):
    """
    Compute bandwidth and time overheads for each combination.
    
    Args:
        trace_oh_dict: Dictionary containing trace data
        trace_indices: List of trace indices to process
        num_combinations: Total number of parameter combinations
        desired_combination: some times we only want to check this for one combination
        desired_pairs: list containing the combination we want
    Returns:
        list: Vector of alternating bandwidth and time overheads for each combination
    """
    # Initialize result vector
    result = []
    
    # Process each combination
    for combo_idx in range(num_combinations):
        if desired_combination is not None and combo_idx != desired_combination:
            continue
        if desired_pairs is not None and combo_idx not in desired_pairs:
            continue
        # Initialize sums for undefended metrics
        total_undefended_bw = 0
        total_undefended_time = 0
        
        # Initialize sums for defended metrics
        total_defended_bw = 0
        total_defended_time = 0
        
        # Process each trace
        
        for trace_idx in trace_indices:
            # Add undefended metrics
            total_undefended_bw += trace_oh_dict[trace_idx]['bw']
            total_undefended_time += trace_oh_dict[trace_idx]['time']
            
            # Check if this combination exists for this trace
            if combo_idx in trace_oh_dict[trace_idx]:
                # Add defended metrics
                total_defended_bw += trace_oh_dict[trace_idx][combo_idx][0]  # len(defended)
                total_defended_time += trace_oh_dict[trace_idx][combo_idx][1]  # time(defended)
        
        # Calculate overheads
        bw_overhead = (total_defended_bw - total_undefended_bw) / total_undefended_bw
        time_overhead = (total_defended_time - total_undefended_time) / total_undefended_time
        
        # Add to result vector
        result.extend([bw_overhead, time_overhead])
    
    return result