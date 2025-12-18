# performing clusters on traces within a website class to see if we can get better clusters
import argparse
import utils.config_utils as cm
from utils.distance_metrics import  tam_euclidian_distance
import pickle
import os 
from utils.trace_dataset import TraceDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.file_operations import save_file, load_file
from utils.trace_operations import  compute_super_matrix
from multiprocessing import Pool
from .algorithms.clustering_algorithms import  pallete
from .algorithms.tamaraw_clustering import tamaraw_palette
from .clustering_utils import plot_clusters_with_distance_matrix, compute_clustering_overheads, show_percentage_of_clusters, identify_pareto
from utils.parser_utils import str2bool
from .clustering_metrics import ed_metric, purity_metrics, compute_website_protections
from .algorithms.l_diversity import three_phase_oka
import pickle
import copy
from collections import Counter, defaultdict
from collections.abc import Mapping

def plot_second_tier_overall_results(save_path, band_width_overheads, average_protections,
                                     purities, 
                                     eds,
                                     l_diversity_penalty_range = None,
                                     l_range = None,
                                        second_tier_algorithm = 'palette' ):
    #plotting the overall results obtained from two levels of clustering
    #if algorithm is palette, pass l_diversity_penalty_range penalty range to it
    # otherwise, pass l_range to it
    band_width_overheads = np.array(band_width_overheads)
    average_protections = np.array(average_protections)
    
    
    plt.figure(figsize=(8, 4))
    if second_tier_algorithm in ['palette', 'palette-tamaraw']:
        plt.plot(l_diversity_penalty_range, band_width_overheads, label='Bandwidth Overheads', marker='o')
        plt.title('Bandwidth Overheads over L-diversity Penalty Range with Palette')
        plt.xlabel('L-diversity Penalty')
    else:
        plt.plot(l_range, band_width_overheads, label='Bandwidth Overheads', marker='o')
        plt.title(f'Bandwidth Overheads over L Range with {second_tier_algorithm}')
        plt.xlabel('L')
    
    plt.ylabel('Bandwidth Overheads')
    plt.legend()
    plt.grid(True)
    if args.save:
        save_file(dir_path= save_path, file_name= 'Bandwidth_OH.png')
    else:
        plt.show()

    # Plot for Purities
    plt.figure(figsize=(8, 4))
    if second_tier_algorithm in ['palette', 'palette-tamaraw']:
        plt.plot(l_diversity_penalty_range, purities, label='Purities', marker='x')
        plt.title('Purities over L-diversity Penalty Range with Palette')
        plt.xlabel('L-diversity Penalty')
    else:
        plt.plot(l_range, purities, label='Purities', marker='x')
        plt.title(f'Purities over L Range with {second_tier_algorithm}')
        plt.xlabel('L')
    
    plt.ylabel('Purities')
    plt.legend()
    plt.grid(True)
    if args.save:
        save_file(dir_path= save_path, file_name= 'Purities.png')
    else:
        plt.show()

    # Plot for EDS
    plt.figure(figsize=(8, 4))
    if second_tier_algorithm in ['palette', 'palette-tamaraw']:
        plt.plot(l_diversity_penalty_range, eds, label='EDS', marker='^')
        plt.title('ED metric over L-diversity Penalty Range with Palette')
        plt.xlabel('L-diversity Penalty')
    else:
        plt.plot(l_range, eds, label='EDS', marker='^')
        plt.title(f'ED metric over L Range with {second_tier_algorithm}')
        plt.xlabel('L')
    
    plt.ylabel('ED')
    plt.legend()
    plt.grid(True)
    if args.save:
        save_file(dir_path= save_path, file_name= 'ED.png')
    else:
        plt.show()
    
    # Plot Average protections
    plt.figure(figsize=(8, 4))
    if second_tier_algorithm in ['palette', 'palette-tamaraw']:
        plt.plot(l_diversity_penalty_range, average_protections, label='Average Protections', marker='^')
        plt.title('Average Protections over L-diversity Penalty Range with Palette')
        plt.xlabel('L-diversity Penalty')
    else:
        plt.plot(l_range, average_protections, label='Average Protections', marker='^')
        plt.title(f'Average Protections over L Range with {second_tier_algorithm}')
        plt.xlabel('L')
    
    plt.ylabel('Average protections')
    plt.legend()
    plt.grid(True)
    if args.save:
        save_file(dir_path= save_path, file_name= 'Average_Protections.png')
    else:
        plt.show()

    # showing the tradeoff between bandwidth and protection
    # Combine into a single array, negate bandwidth to align both objectives towards minimization
    scores = np.column_stack((band_width_overheads, 1 - average_protections))  # Negate protection to convert it into a minimization problem

    pareto = identify_pareto(scores)
    pareto_points = scores[pareto]
    # Identify Pareto points
    pareto_indices = identify_pareto(scores)
    pareto_points = scores[pareto_indices]

    # Plotting all points
    plt.figure(figsize=(10, 5))
    plt.scatter(band_width_overheads, average_protections, color='blue', label='All Points')

    # Highlighting the Pareto frontier
    pareto_bandwidth = band_width_overheads[pareto_indices]
    pareto_protection = 1 - (1 - average_protections[pareto_indices])  # Reverting the negation for protection
    plt.scatter(pareto_bandwidth, pareto_protection, color='red', label='Pareto Frontier')
    # Plotting
    
    
    plt.title('Pareto Frontier')
    plt.xlabel('Bandwidth Overhead')
    plt.ylabel('Protection')
    plt.legend()
    plt.grid(True)
    if args.save:
        save_file(dir_path= save_path, file_name= 'Pareto.png')
    else:
        plt.show()

    protection_weight = args.protection_weight
    bandwidth_weight = 1 - args.protection_weight

    #plotting the weighted sum of bandwidth and protection
    
    # Normalize bandwidth_overheads
    bandwidth_min, bandwidth_max = band_width_overheads.min(), band_width_overheads.max()
    normalized_bandwidth_overheads = (band_width_overheads - bandwidth_min) / (bandwidth_max - bandwidth_min)
    normalized_bandwidth_overheads = 1 - ((band_width_overheads - bandwidth_min) / (bandwidth_max - bandwidth_min)) # we want lower overhead to be considered as better

    weighted_sum = bandwidth_weight * normalized_bandwidth_overheads + protection_weight * average_protections

    best_index_weighted_sum = np.argmax(weighted_sum)  # Index of the best weighted sum


    # Calculate F1 score
    f1_scores = 2 * (average_protections * normalized_bandwidth_overheads) / (average_protections + normalized_bandwidth_overheads)
    best_index_f1_score = np.argmax(f1_scores)


    plt.figure(figsize=(8, 4))
    if second_tier_algorithm in ['palette', 'palette-tamaraw']:
        plt.plot(l_diversity_penalty_range, weighted_sum, marker='o', linestyle='-', color='blue')
        plt.xlabel('L-diversity Penalty')
        plt.ylabel('Weighted Sum Score with Palette')
        plt.xticks(l_diversity_penalty_range)
        plt.scatter(l_diversity_penalty_range[best_index_weighted_sum], 
                    weighted_sum[best_index_weighted_sum], color='red', s=100, label=f'Best: {l_diversity_penalty_range[best_index_weighted_sum]}')
    else:
        plt.plot(l_range, weighted_sum, marker='o', linestyle='-', color='blue')
        plt.xlabel('L')
        plt.ylabel(f'Weighted Sum Score with {second_tier_algorithm}')
        plt.xticks(l_range)
        plt.scatter(l_range[best_index_weighted_sum], 
                    weighted_sum[best_index_weighted_sum], color='red', s=100, label=f'Best: {l_range[best_index_weighted_sum]}')
    
    if args.save:
        save_file(dir_path= save_path, file_name= 'Weighted_Sum.png')
    else:
        plt.show()

    # Plot for F1 Scores
    plt.figure(figsize=(8, 4))
    if second_tier_algorithm in ['palette', 'palette-tamaraw']:
        plt.plot(l_diversity_penalty_range, f1_scores, marker='o', linestyle='-', color='green')
        plt.scatter(l_diversity_penalty_range[best_index_f1_score], f1_scores[best_index_f1_score], color='red', s=100, label=f'Best: {l_diversity_penalty_range[best_index_f1_score]}')
        plt.title('F1 Scores Between Bandwidth and Protection with Palette')
        plt.xlabel('L-diversity Penalty')
        plt.ylabel('F1 Score')
        plt.xticks(l_diversity_penalty_range)
    else:
        plt.plot(l_range, f1_scores, marker='o', linestyle='-', color='green')
        plt.scatter(l_range[best_index_f1_score], f1_scores[best_index_f1_score], color='red', s=100, label=f'Best: {l_range[best_index_f1_score]}')
        plt.title(f'F1 Scores Between Bandwidth and Protection with {second_tier_algorithm}')
        plt.xlabel('L')
        plt.ylabel('F1 Score')
        plt.xticks(l_range)
    
    
    if args.save:
        save_file(dir_path= save_path, file_name= 'F1_Scores.png')
    else:
        plt.show()
    
    return best_index_f1_score
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
def compute_cluster_accuracy(cluster, trace_L_dictionary, original_websites, number_of_L_configs, debug_mode = False, l_value = None):
    # l_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    if l_value is None:
        l_values = [500, 600, 700, 800, 900, 1000]
    else:
        l_values = [l_value]
    new_cluster = copy.deepcopy(cluster)
    
    
    new_websites = [original_websites[i] for i in cluster]
    
   
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



if __name__ == '__main__':


    # arguments
    parser = argparse.ArgumentParser(description='Two tier clustering of all the clusters')
    parser.add_argument('-alg1', '--algorithm_tier1',
                        choices=['k_medoids', 'cast', 'cast_tamaraw'],
                        help='type of clusering algorithm we want to perform in the first tier',
                        default= 'cast') # TODO implement others
    
    parser.add_argument('-alg2', '--algorithm_tier2',
                        choices=['palette', 'oka', 'palette_tamaraw'],
                        help='type of clusering algorithm we want to perform in the second tier',
                        default= 'palette') # TODO implement others

    parser.add_argument('-oka_steps',  type = int ,choices= [1,2,3], default= 3 , help = 'number of phases we want oka to have') 

    parser.add_argument('-k', type = int , default= 0 , help = 'Minimum number of elements in each cluster for palette')
    
    
    parser.add_argument('-e', '--extract_ds', type=str2bool, nargs='?', const=True, default=False,
                        help='should we extract the dataset or is it already stored'
                        )
    
    parser.add_argument('-conf', '--config', help='which config file to use', default = 'default')


    parser.add_argument('-time_slot', type = float,
                         help='If we are using tam, what amount of time slot (in ms) do we want?', default = None)
    
    
    parser.add_argument('-min_ds_interval', type = float,
                         help='the lower bound of the dataset partition we want to cluster', default = 0)
    
    parser.add_argument('-max_ds_interval', type = float,
                         help='the upper bound of the dataset partition we want to cluster', default = 1)

    parser.add_argument('-time_threshold', type = float,
                         help='Do we want the traces to be truncated to a specific time ?', default = None)
    
    
    parser.add_argument('-max_clusters', help='maximum number of clusters for each website when performing cast', default = 5, type =int)

    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to save the computed elements')
    
    parser.add_argument('-div_threshold', type = float,
                         help='the diversity threshold we use in two tier', default = 1)
    
    
    
    
    
    parser.add_argument('-preload_clusters', type=str2bool, nargs='?', const=True, default=False,
                         help='Whether to load the pre-computed clusters', )
    
    parser.add_argument('-save_clusters', type=str2bool, nargs='?', const=True, default=False,
                         help='Whether to save the computed clusters', )
    
    parser.add_argument('-palette_l_diversity', type = float,
                         help='the ldiversity coefficient for palette', default = -1)
    # -1 means we want a range of of diversities between 1 and 2
    
    parser.add_argument('-l_diversity',  type = int , default= -1 , help = 'l for oka')
    # -1 means we want a range of of diversities between 2 and 10

    parser.add_argument('-l_tamaraw',  type = int , default= None , help = 'fixed L for tamaraw')
    parser.add_argument('-palette_pp', 
                        choices=['heavy', 'medium', 'None'],
                        help='how should the palette post processing be done',
                        default= 'None') 

    parser.add_argument('-protection_weight', type = float,
                         help='the weight we give to protection when we want to choose the best parameter', default = 0.5)
    
    parser.add_argument('-n_cores', type = int,
                         help='number of cores to use ', default = 1) #  1 means don't do parallelism
    
    
    
    logger = cm.init_logger(name = 'Two Tier Clustering')


    import matplotlib
    matplotlib.use('Agg') # apparently these two lines are necessary for saving plots

    


    args = parser.parse_args()
    
    cm.initialize_common_params(args.config)
    
    algorithm_tier1 = args.algorithm_tier1
    algorithm_tier2 = args.algorithm_tier2
    

    #initializing the global params between different files, based on the chosen config file
    
    

    



    trace_mode = 'tam'
    if args.time_slot:
        cm.Time_Slot = args.time_slot/1000 # converting from ms to s
        cm.Max_tam_matrix_len = None # we don't want the max_matrix_len anymore
         
    

    dataset_interval = [args.min_ds_interval, args.max_ds_interval]
    dataset = TraceDataset(extract_traces= args.extract_ds, 
                           trace_mode= trace_mode, 
                           interval= dataset_interval, 
                           time_threshold= args.time_threshold)
    

    clusters_of_each_website = {}
    cluster_lengths_of_each_website = {} # for each website, this will be a list like [200, 300, 500], indicating how many traces of this website are encapsulated in that subcluster
    cluster_percentages_of_each_website = {} # for each website, this will be a list like [0.2, 0.3, 0.5], indicating how much of the website is encapsulated in that subcluster
    
    
    
    
    if algorithm_tier1 in ['cast', 'cast_tamaraw']:
        if args.preload_clusters:
            
            
            for website_num in tqdm(range (cm.MON_SITE_START_IND, cm.MON_SITE_END_IND), desc= f'loading fierst tier {algorithm_tier1} clusters'):
                
                # if website_num not in clusters_of_each_website.keys():
                #     clusters_of_each_website[website_num] = []
                #     cluster_percentages_of_each_website[website_num] = []
                #     cluster_lengths_of_each_website[website_num] = []
                
                load_path = os.path.join(cm.BASE_DIR,  'data', cm.data_set_folder, 'clustering', algorithm_tier1, f'max_clusters_{args.max_clusters}' , str(website_num))
                clusters_of_this_website = load_file(dir_path= load_path, file_name= 'clusters.pkl')
                clusters_of_each_website[website_num] = clusters_of_this_website
                cluster_sizes = [len(subcluster) for subcluster in clusters_of_this_website]
                total_number_of_elements = sum(cluster_sizes)
                cluster_percentages_of_each_website[website_num] = [len(subcluster)/ total_number_of_elements for subcluster in clusters_of_this_website]
                cluster_lengths_of_each_website[website_num] = cluster_sizes
            
        else:   
            pass
            #TODO maybe add case where we don't have the clusters beforehand

    
    super_matrixes_of_all_clusters = [] # the super matrixes we want to cluster
    super_matrix_mapping = {} 
    # a mapping from the super matrix index to the actual cluster index in clusters_of_each_website
    # this will be in form of 5 : [2,4] were 2 is the website number and 4 is the 4th cluster in website 2
    super_matrix_labels = [] # the website each supermatrix belongs to
    
    #max_clusters added
    save_path = os.path.join(cm.BASE_DIR, 'results', 'clustering', 'two-tier', f'{cm.data_set_folder}', f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}',
                             *([]),f'k = {args.k}')
     
    show_percentage_of_clusters(clusters_of_each_website= clusters_of_each_website,
                                save_path= save_path, visualization_type= 'text')
    
    
    
    total_number_of_clusters = 0
    overall_tier1_clusters = [] # list of lists. each list has trace indices in it (for tamaraw palette)
    
    for website_num in range (cm.MON_SITE_START_IND, cm.MON_SITE_END_IND):
        
        tams, _ , indices= dataset.get_traces_of_class(class_number= website_num, return_indices= True) # TODO if the initial clustering is not tam-based, change the traces to tam first
        # print(clusters_of_each_website[website_num])
        for idx, intra_cluster in enumerate(clusters_of_each_website[website_num]):
            
            tams_of_this_cluster = tams[intra_cluster]
            overall_tier1_clusters.append(indices[intra_cluster])
            
            # print(tams_of_this_cluster.shape)
            super_matrix_of_this_cluster = compute_super_matrix(*tams_of_this_cluster)

            super_matrix_mapping[len(super_matrixes_of_all_clusters)] = [website_num, idx]

            super_matrixes_of_all_clusters.append(super_matrix_of_this_cluster)

            super_matrix_labels.append(website_num)
            total_number_of_clusters += 1
    
    save_dirrr = os.path.join(cm.BASE_DIR,  'data', cm.data_set_folder, 'clustering', 'two-tier', f'{algorithm_tier1}-{algorithm_tier2}',
                                f'{algorithm_tier1}-{args.max_clusters}', f'{algorithm_tier2}-{args.k}' )
        
    #saving the mapping
    save_file(dir_path= save_dirrr, file_name= 'super_matrix_mappings.pickle', content= super_matrix_mapping, 
                protocol = pickle.HIGHEST_PROTOCOL)
    logger.info(f'Total number of subclusters: {total_number_of_clusters}')

    
    band_width_overheads = []
    purities = []
    eds = []
    average_protections = []
    obtained_clusters = [] # the obtained clusters based on different hyperparameters.
    # if save_clusters is Trrue, the best clustering will be stored
    if algorithm_tier2 == 'palette':
        ### Performing pallete on the obtained super matrixes
        if args.palette_l_diversity:
            l_diversity_penalty_range = [1 + 0.5 * i for i in range (0, 1)]
            range_name = f'range1-2'
            # global path changed
            #max_clusters added
            global_path = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                   f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', range_name, f'k = {args.k}')
            if args.div_threshold != 1:
                global_path = os.path.join(global_path, f'div_{args.div_threshold}')
            if args.palette_pp != 'None':
                global_path = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                   f'{algorithm_tier1}_{algorithm_tier2}', f'with_{args.palette_pp}_post_processing', range_name, f'k = {args.k}')
            # l_diversity_penalty_range = [1] + [ 2 ** i for i in range (1, 8)]
            #range_name = f'range1-128'
            for idx, penalty in enumerate(tqdm(l_diversity_penalty_range, desc= 'perform palette with different penalties')):
                cluster_groups, super_matrixes_of_each_anonimity_set = pallete(tams = super_matrixes_of_all_clusters, k = args.k,
                                                                        diversity_penalty= penalty, consider_diversity= True, 
                                                                        tam_labels= super_matrix_labels, 
                                                                        post_processing= args.palette_pp)
                obtained_clusters.append(cluster_groups)
                cluster_labels = [None] * sum(len(cluster) for cluster in cluster_groups)  # Create a list to store labels

                for label, cluster in enumerate(cluster_groups):
                    for index in cluster:
                        cluster_labels[index] = label
                cluster_labels = np.array(cluster_labels) # this is like [10, 2, 0 , ..., 1]

                #computing the distance matrix between the super_matrixes
                distance_matrix_super_matrixes = np.zeros([len(super_matrixes_of_all_clusters), len(super_matrixes_of_all_clusters)])

                for i in tqdm(range(len(super_matrixes_of_all_clusters)), desc = f'Computing pair-wise distance of all super matrixes'):
                    for j in tqdm(range(len(super_matrixes_of_all_clusters)), desc = f'computing distance for instance {i + 1} / {len(tams)}', disable= True):
                        if j > i:
                            distance_matrix_super_matrixes[i,j] = tam_euclidian_distance(super_matrixes_of_all_clusters[i], 
                                                                                        super_matrixes_of_all_clusters[j])
                        elif j < i:
                            distance_matrix_super_matrixes[i,j] = distance_matrix_super_matrixes[j, i]


                clusters = cluster_labels # this is like [10, 2, 0 , ..., 1]
                number_of_clusters = len(super_matrixes_of_all_clusters) // args.k

                
                plot_title = f'Similarity Measure Among Clusters and super matrixes using {args.algorithm_tier1}-{algorithm_tier2} and diversity penalty = {penalty:.2f} and {args.palette_pp} post processing'
                save_path = os.path.join(global_path, f'k = {args.k}',f'results_{idx}')
                plot_clusters_with_distance_matrix(distance_matrix= distance_matrix_super_matrixes, clusters= clusters, k = number_of_clusters, 
                                        dir_path = save_path, text_title= plot_title, file_name= f'box_plots_{idx}.png')
                
                logger.info('Computing Overheads...')
                band_width_overhead,_,_,_ = compute_clustering_overheads(clusters = clusters, cluster_representatives= super_matrixes_of_each_anonimity_set,
                                    save_dir= save_path, dataset= dataset, 
                                    intra_clusters= clusters_of_each_website,
                                    super_matrix_mapping= super_matrix_mapping,
                                    granularity= 'two-tier',
                                    verbose = True)
                logger.info('Overheads Computed')
                
                overall_purity, cluster_purities, max_percentages = purity_metrics(cluster_indices= cluster_groups,
                                                                                   labels = super_matrix_labels)
                
                ed = ed_metric(clusters= cluster_groups,
                               labels= super_matrix_labels)
                website_protections = compute_website_protections(cluster_indices= cluster_groups,
                                                                  intra_cluster_mapping= super_matrix_mapping,
                                                                  intra_cluster_percentages= cluster_percentages_of_each_website,
                                                                  intra_cluster_sizes= cluster_lengths_of_each_website)
                band_width_overheads.append(band_width_overhead)
                purities.append(overall_purity)
                eds.append(ed)
                average_protections.append(np.mean(website_protections))


                #plot cluster purites and max_percentages
                save_path = os.path.join(global_path, 'purities')
                plt.figure(figsize=(10, 5))
                plt.plot(cluster_purities, label='Cluster Purities', marker='o')
                plt.title(f'Purity of Each Cluster with Palette and {penalty:.2f} penalty and {args.palette_pp} post processing')
                plt.xlabel('Cluster Number')
                plt.ylabel('Purity')
                plt.xticks(range(len(cluster_purities)), range(1, len(cluster_purities) + 1))  # Adjust x-ticks to match cluster numbers
                plt.legend()
                plt.grid(True)
                if args.save:
                    save_file(dir_path= save_path, file_name= f'purities_{penalty:.2f}.png')
                else:
                    plt.show()

                # Plot for Max Percentages
                save_path = os.path.join(global_path, 'percentages')
                plt.figure(figsize=(10, 5))
                labels, percentages = zip(*max_percentages.items())  # Unpacking the keys and values for plotting
                plt.bar(labels, percentages, color='skyblue')
                plt.title(f'Maximum Percentage of Each Website in Any Cluster with Palette and {penalty:.2f} penalty and {args.palette_pp} post processing')
                plt.xlabel('Wbsite')
                plt.ylabel('Maximum Percentage')
                plt.xticks(rotation=45)  # Rotate labels to prevent overlap
                plt.grid(axis='y')  # Only horizontal grid lines
                if args.save:
                    save_file(dir_path= save_path, file_name= f'class_max_distributions_{penalty:.2f}.png')
                else:
                    plt.show()
                
                # Plot for Protections
                save_path = os.path.join(global_path, 'protections')
                plt.figure(figsize=(10, 5))
                
                plt.bar(range (cm.MON_SITE_START_IND, cm.MON_SITE_END_IND), website_protections, color='skyblue')
                plt.title(f'Protection of Each Website with Palette and {penalty} penalty and {args.palette_pp} post processing')
                plt.xlabel('Websites')
                plt.ylabel('Protection')
                plt.xticks(rotation=45)  # Rotate labels to prevent overlap
                plt.grid(axis='y')  # Only horizontal grid lines
                if args.save:
                    save_file(dir_path= save_path, file_name= f'protections_{penalty:.2f}.png')
                else:
                    plt.show()
                
            
            

            #plotting the overall results
            print(f'l diversity penalty range was {l_diversity_penalty_range}')
            print(f'band_width_overheads were {band_width_overheads}')
            print(f'purities were {purities}')
            best_cluster_index = plot_second_tier_overall_results(save_path= global_path, band_width_overheads= band_width_overheads,
                                             average_protections= average_protections,
                                             purities= purities,
                                             eds = eds,
                                             l_diversity_penalty_range= l_diversity_penalty_range,
                                             second_tier_algorithm='palette')
    


    if algorithm_tier2 == 'palette_tamaraw':
        
        load_param_dir  = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'tamaraw_params')
        if args.l_tamaraw is not None:
            load_param_dir = os.path.join(load_param_dir, f'L_{args.l_tamaraw}')
        trace_oh_dict = load_file(dir_path= load_param_dir,  file_name= f'trace_params.json',  string_to_num = True )
        
        
        ### Performing pallete on the obtained super matrixes
        if args.palette_l_diversity is not None:
            # l_diversity_penalty_range = [1 + 0.1 * i for i in range (0, 1)]
            
            # l_diversity_penalty_range = [0, 1,2, 4, 8,16,32,64,128]
            l_diversity_penalty_range = [0] # bad coding!

            accuracy_penalty_range = [16]
            range_name = f'range1-128'
            
            #max_clusters added
            
            global_path = os.path.join(
                cm.BASE_DIR, 
                'results', 
                'clustering', 
                'two-tier', 
                f'{cm.data_set_folder}',
                f'{algorithm_tier1}_{algorithm_tier2}', 
                f'{algorithm_tier1}-{args.max_clusters}',
                *([]),
                *(([f'l_{args.l_tamaraw}'] if args.l_tamaraw is not None else [])),
                range_name, 
                f'k = {args.k}'
            )
            
            
            load_path_sizes = os.path.join(cm.BASE_DIR,  'results',  'tamaraw_params', f'{cm.data_set_folder}')

            # need to change this
            if args.l_tamaraw is None:
                trace_L_dictionary = load_file(dir_path= load_path_sizes, file_name= 'tamaraw_sizes_per_L_top_10.json' ,use_pickle = True)
            else:
                 trace_L_dictionary = load_file(dir_path= load_param_dir, file_name= 'tamaraw_sizes_per_L.json' ,use_pickle = True)
            for idx, penalty in enumerate(tqdm(l_diversity_penalty_range, desc= 'perform tamaraw palette with different accuracy penalties')):

                
                cluster_groups, overall_clusters = tamaraw_palette(tier1_clusters= overall_tier1_clusters,
                                                                   k  = args.k,
                                                                   diversity_penalty= penalty,
                                                                   tier1_labels= super_matrix_labels,
                                                                   verbose= True,
                                                                   diversity_threshold= args.div_threshold,
                                                                   trace_L_dictionary= trace_L_dictionary,
                                                                   number_of_L_configs= 196,
                                                                   original_websites= dataset.labels,
                                                                   l_value= args.l_tamaraw)
                # cluster groups is a list of lists. each list is an anomity sets. its elements are indices of the intraclusters
                # overall_clusters is again a list of lists. but this time in each list, we have the actual trace indices
                
                
                super_matrixes_of_each_anonimity_set = []
                for two_tier_cluster in overall_clusters:
                    tams_in_this_tier2_cluster = [dataset.directions[i] for i in two_tier_cluster]
                    super_matrixes_of_each_anonimity_set.append(compute_super_matrix(*tams_in_this_tier2_cluster))
                # computing the supermatrix of each anonimity set

                obtained_clusters.append(cluster_groups)
                cluster_labels = [None] * sum(len(cluster) for cluster in cluster_groups)  # Create a list to store labels

                for label, cluster in enumerate(cluster_groups):
                    for index in cluster:
                        cluster_labels[index] = label
                cluster_labels = np.array(cluster_labels) # this is like [10, 2, 0 , ..., 1]

                #computing the distance matrix between the super_matrixes
                distance_matrix_super_matrixes = np.zeros([len(super_matrixes_of_all_clusters), len(super_matrixes_of_all_clusters)])

                for i in tqdm(range(len(super_matrixes_of_all_clusters)), desc = f'Computing pair-wise distance of all super matrixes'):
                    for j in tqdm(range(len(super_matrixes_of_all_clusters)), desc = f'computing distance for instance {i + 1} / {len(tams)}', disable= True):
                        if j > i:
                            distance_matrix_super_matrixes[i,j] = tam_euclidian_distance(super_matrixes_of_all_clusters[i], 
                                                                                        super_matrixes_of_all_clusters[j])
                        elif j < i:
                            distance_matrix_super_matrixes[i,j] = distance_matrix_super_matrixes[j, i]


                clusters = cluster_labels # this is like [10, 2, 0 , ..., 1]
                number_of_clusters = len(super_matrixes_of_all_clusters) // args.k

                
                plot_title = f'Similarity Measure Among Clusters and super matrixes using {args.algorithm_tier1}-{algorithm_tier2} and diversity penalty = {penalty:.2f} and {args.palette_pp} post processing'
                save_path = os.path.join(global_path, f'k = {args.k}',f'results_{idx}')
                plot_clusters_with_distance_matrix(distance_matrix= distance_matrix_super_matrixes, clusters= clusters, k = number_of_clusters, 
                                        dir_path = save_path, text_title= plot_title, file_name= f'box_plots_{idx}.png')
                
                logger.info('Computing Overheads...')
                band_width_overhead,_,_,_ = compute_clustering_overheads(clusters = clusters, cluster_representatives= super_matrixes_of_each_anonimity_set,
                                    save_dir= save_path, dataset= dataset, 
                                    intra_clusters= clusters_of_each_website,
                                    super_matrix_mapping= super_matrix_mapping,
                                    granularity= 'two-tier',
                                    verbose = True)
                logger.info('Overheads Computed')
                
                overall_purity, cluster_purities, max_percentages = purity_metrics(cluster_indices= cluster_groups,
                                                                                   labels = super_matrix_labels)
                
                ed = ed_metric(clusters= cluster_groups,
                               labels= super_matrix_labels)
                website_protections = compute_website_protections(cluster_indices= cluster_groups,
                                                                  intra_cluster_mapping= super_matrix_mapping,
                                                                  intra_cluster_percentages= cluster_percentages_of_each_website,
                                                                  intra_cluster_sizes= cluster_lengths_of_each_website)
                if isinstance(website_protections, Mapping): 
                    website_protections = [website_protections[web_idx] for web_idx in sorted(website_protections.keys())]
                
                band_width_overheads.append(band_width_overhead)
                purities.append(overall_purity)
                eds.append(ed)
                average_protections.append(np.mean(website_protections))


                #plot cluster purites and max_percentages
                save_path = os.path.join(global_path, 'purities')
                plt.figure(figsize=(10, 5))
                plt.plot(cluster_purities, label='Cluster Purities', marker='o')
                plt.title(f'Purity of Each Cluster with Palette and {penalty:.2f} penalty and {args.palette_pp} post processing')
                plt.xlabel('Cluster Number')
                plt.ylabel('Purity')
                plt.xticks(range(len(cluster_purities)), range(1, len(cluster_purities) + 1))  # Adjust x-ticks to match cluster numbers
                plt.legend()
                plt.grid(True)
                if args.save:
                    save_file(dir_path= save_path, file_name= f'purities_{penalty:.2f}.png')
                else:
                    plt.show()

                # Plot for Max Percentages
                save_path = os.path.join(global_path, 'percentages')
                plt.figure(figsize=(10, 5))
                labels, percentages = zip(*max_percentages.items())  # Unpacking the keys and values for plotting
                plt.bar(labels, percentages, color='skyblue')
                plt.title(f'Maximum Percentage of Each Website in Any Cluster with Palette and {penalty:.2f} penalty and {args.palette_pp} post processing')
                plt.xlabel('Wbsite')
                plt.ylabel('Maximum Percentage')
                plt.xticks(rotation=45)  # Rotate labels to prevent overlap
                plt.grid(axis='y')  # Only horizontal grid lines
                if args.save:
                    save_file(dir_path= save_path, file_name= f'class_max_distributions_{penalty:.2f}.png')
                else:
                    plt.show()
                
                # Plot for Protections
                save_path = os.path.join(global_path, 'protections')
                plt.figure(figsize=(10, 5))
                
                plt.bar(range (cm.MON_SITE_START_IND, cm.MON_SITE_END_IND), website_protections, color='skyblue')
                plt.title(f'Protection of Each Website with Palette and {penalty} penalty and {args.palette_pp} post processing')
                plt.xlabel('Websites')
                plt.ylabel('Protection')
                plt.xticks(rotation=45)  # Rotate labels to prevent overlap
                plt.grid(axis='y')  # Only horizontal grid lines
                if args.save:
                    save_file(dir_path= save_path, file_name= f'protections_{penalty:.2f}.png')
                else:
                    plt.show()
                
                if args.l_tamaraw is not None:
                    overall_accuracies = []
                    for cluster in overall_clusters:
                        overall_accuracies.append(compute_cluster_accuracy(cluster= cluster,
                                                                           trace_L_dictionary= trace_L_dictionary,
                                                                           original_websites= dataset.labels,
                                                                           number_of_L_configs= 196,
                                                                           l_value=  args.l_tamaraw))
                        
                        mean_accuracy = np.mean(overall_accuracies)

                        # Create indices for x-axis
                        cluster_indices = range(len(overall_accuracies))

                        # Create the plot
                        plt.figure(figsize=(10, 6))
                        plt.plot(cluster_indices, overall_accuracies, 'bo-', linewidth=2, markersize=8)
                        plt.axhline(y=mean_accuracy, color='r', linestyle='--', label=f'Mean: {mean_accuracy:.4f}')

                        # Add labels and title
                        plt.xlabel('Cluster Index')
                        plt.ylabel('Accuracy')
                        plt.title('Accuracy by Cluster')
                        plt.grid(True, alpha=0.3)
                        plt.legend()

                        # Set x-axis ticks to integers
                        plt.xticks(cluster_indices)

                        # Add text annotation for mean value
                        plt.text(len(overall_accuracies)/2, mean_accuracy + 0.01, 
                                f'Mean Accuracy: {mean_accuracy:.4f}', 
                                horizontalalignment='center', 
                                color='darkred')

                        plt.tight_layout()
                        save_path = os.path.join(global_path, 'max_accuracies')
                        if args.save:
                            save_file(dir_path= save_path, file_name= f'accuracies_{penalty:.2f}.png')
                        else:
                            plt.show()

            
            

            
            best_cluster_index = 0
    

                
                



    elif algorithm_tier2 == 'oka':
        
        l_range = [l for l in range(2,min(11, args.k))]
        for l in l_range:
            initial_partitions, adjusted_partitions, l_diversity_partitions = three_phase_oka(tams= super_matrixes_of_all_clusters, 
                                             labels= super_matrix_labels, k = args.k, l = l, verbose= True)
            
            if args.oka_steps == 1:
                cluster_groups = initial_partitions
            elif args.oka_steps == 2:
                cluster_groups = adjusted_partitions
            elif args.oka_steps == 3:
                cluster_groups = l_diversity_partitions


            cluster_labels = [None] * sum(len(cluster) for cluster in cluster_groups)  # Create a list to store labels

            for label, cluster in enumerate(cluster_groups):
                for index in cluster:
                    cluster_labels[index] = label
            cluster_labels = np.array(cluster_labels) # this is like [10, 2, 0 , ..., 1]

            #computing the distance matrix between the super_matrixes
            distance_matrix_super_matrixes = np.zeros([len(super_matrixes_of_all_clusters), len(super_matrixes_of_all_clusters)])

            for i in tqdm(range(len(super_matrixes_of_all_clusters)), desc = f'Computing pair-wise distance of all super matrixes'):
                for j in tqdm(range(len(super_matrixes_of_all_clusters)), desc = f'computing distance for instance {i + 1} / {len(tams)}', disable= True):
                    if j > i:
                        distance_matrix_super_matrixes[i,j] = tam_euclidian_distance(super_matrixes_of_all_clusters[i], 
                                                                                    super_matrixes_of_all_clusters[j])
                    elif j < i:
                        distance_matrix_super_matrixes[i,j] = distance_matrix_super_matrixes[j, i]


            clusters = cluster_labels # this is like [10, 2, 0 , ..., 1]
            number_of_clusters = len(super_matrixes_of_all_clusters) // args.k

            plot_title = f'Similarity Measure Among Clusters and super matrixes using {args.algorithm_tier1}-{algorithm_tier2} and l = {l}'
            save_path = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                   f'{algorithm_tier1}_{algorithm_tier2}', f'steps = {args.oka_steps}', f'k = {args.k}',f'results_{l}')
            plot_clusters_with_distance_matrix(distance_matrix= distance_matrix_super_matrixes, clusters= clusters, k = number_of_clusters, 
                                    dir_path = save_path, text_title= plot_title, file_name= f'box_plots_{l}.png')
            ####TODO compute supermatrix of each anonimity set
            super_matrixes_of_each_anonimity_set = []
            for cluster in cluster_groups:
                tams_in_this_cluster = [super_matrixes_of_all_clusters[i] for i in cluster]
                super_matrixes_of_each_anonimity_set.append(compute_super_matrix(*tams_in_this_cluster))
            logger.info('Computing Overheads...')
            band_width_overhead, _, _, _= compute_clustering_overheads(clusters = clusters, cluster_representatives= super_matrixes_of_each_anonimity_set,
                                save_dir= save_path, dataset= dataset, 
                                intra_clusters= clusters_of_each_website,
                                super_matrix_mapping= super_matrix_mapping,
                                granularity= 'two-tier',
                                verbose= True)
            logger.info('Overheads Computed')
            
            overall_purity, cluster_purities, max_percentages = purity_metrics(cluster_indices= cluster_groups,
                                                                                labels = super_matrix_labels)
            
            ed = ed_metric(clusters= cluster_groups,
                            labels= super_matrix_labels)
            website_protections = compute_website_protections(cluster_indices= cluster_groups,
                                                                  intra_cluster_mapping= super_matrix_mapping,
                                                                  intra_cluster_percentages= cluster_percentages_of_each_website,
                                                                  intra_cluster_sizes= cluster_lengths_of_each_website)
            
            
            band_width_overheads.append(band_width_overhead)
            purities.append(overall_purity)
            eds.append(ed)
            average_protections.append(np.mean(website_protections))



            #plot cluster purites and max_percentages
            save_path = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                f'{algorithm_tier1}_{algorithm_tier2}',  f'steps = {args.oka_steps}', f'k = {args.k}', 'purities')
            plt.figure(figsize=(10, 5))
            plt.plot(cluster_purities, label='Cluster Purities', marker='o')
            plt.title(f'Purity of Each Cluster with Oka {args.oka_steps} and l = {l}')
            plt.xlabel('Cluster Number')
            plt.ylabel('Purity')
            plt.xticks(range(len(cluster_purities)), range(1, len(cluster_purities) + 1))  # Adjust x-ticks to match cluster numbers
            plt.legend()
            plt.grid(True)
            if args.save:
                save_file(dir_path= save_path, file_name= f'purities_{l}.png')
            else:
                plt.show()

            # Plot for Max Percentages
            save_path = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                f'{algorithm_tier1}_{algorithm_tier2}',  f'steps = {args.oka_steps}', f'k = {args.k}', 'percentages')
            plt.figure(figsize=(10, 5))
            labels, percentages = zip(*max_percentages.items())  # Unpacking the keys and values for plotting
            plt.bar(labels, percentages, color='skyblue')
            plt.title(f'Maximum Percentage of Each Label in Any Cluster with Oka {args.oka_steps} and l = {l}')
            plt.xlabel('Labels')
            plt.ylabel('Maximum Percentage')
            plt.xticks(rotation=45)  # Rotate labels to prevent overlap
            plt.grid(axis='y')  # Only horizontal grid lines
            if args.save:
                save_file(dir_path= save_path, file_name= f'class_max_distributions_{l}.png')
            else:
                plt.show()






                
        save_path = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                   f'{algorithm_tier1}_{algorithm_tier2}',  f'steps = {args.oka_steps}', f'k = {args.k}')
            

        best_cluster_index = plot_second_tier_overall_results(save_path= save_path, band_width_overheads= band_width_overheads,
                                            average_protections= average_protections,
                                            purities= purities,
                                            eds = eds,
                                            l_range = l_range,
                                            second_tier_algorithm=f'OKA {args.oka_steps}')


        

    if args.save_clusters: # saving the obtained clusters and also the supermatrix mapping
        # saving cluster groups                
        save_dir = os.path.join(cm.BASE_DIR,  'data', cm.data_set_folder, 'clustering', 'two-tier', f'{algorithm_tier1}-{algorithm_tier2}',
                                *([]),
                                f'{algorithm_tier1}-{args.max_clusters}', f'{algorithm_tier2}-{args.k}')
        if args.l_tamaraw is not None:
            save_dir = os.path.join(save_dir, f'l_{args.l_tamaraw}')
        
        if args.palette_pp != 'None':
            save_dir = os.path.join(cm.BASE_DIR,  'data', cm.data_set_folder, 'clustering', 'two-tier', f'{algorithm_tier1}-{algorithm_tier2}',
                                f'{algorithm_tier1}-{args.max_clusters}', f'{algorithm_tier2}-{args.k}', f'with_{args.palette_pp}_post_processing' )
        
        chosen_clusters = obtained_clusters[best_cluster_index]
        if args.div_threshold != 1:
            save_dir = os.path.join(save_dir, f'div_{args.div_threshold}')
        
        if args.algorithm_tier2 in ['palette', 'palette_tamaraw', 'palette_tamaraw_pareto']:
            best_penalty = l_diversity_penalty_range[best_cluster_index]
            best_penalty = 16.00 # TODO change this
            if args.l_tamaraw is not None:
                file_name = f'second_tier_clusters_L_{args.l_tamaraw}.pkl'
            else:
                file_name = f'second_tier_clusters_max_accuracy.pkl'
            #file_name = f'second_tier_clusters_1.pkl'
        else:
            best_l = l_range[best_cluster_index]
            file_name = f'second_tier_clusters_L_{best_l}_div_{args.div_threshold}.pkl'
        save_file(dir_path= save_dir, file_name= file_name, content= chosen_clusters)

        #saving the mapping
        save_file(dir_path= save_dir, file_name= 'super_matrix_mappings.pickle', content= super_matrix_mapping, 
                    protocol = pickle.HIGHEST_PROTOCOL)
        




# python3 -m experiments.clustering.two_tier_clustering -alg1 cast -alg2 palette  -conf Tik_Tok -k 5 -preload_clusters True  -save True -save_clusters True

# python3 -m experiments.clustering.two_tier_clustering -alg1 cast -alg2 palette_tamaraw  -conf Tik_Tok -k 5 -preload_clusters True  -save True -save_clusters True


# python3 -m experiments.clustering.two_tier_clustering -alg1 cast -alg2 palette_tamaraw  -conf Tik_Tok -k 5 -preload_clusters True  -save True -save_clusters True -l_tamaraw 100

# python3 -m experiments.clustering.two_tier_clustering -alg1 cast -alg2 palette_tamaraw  -conf Tik_Tok -k 5 -preload_clusters True  -save True -save_clusters True -l_tamaraw 100 -remove_websites 10


            

    


