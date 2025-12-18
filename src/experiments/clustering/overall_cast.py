# perform cast on all websites
#perform cast on a random website and see the results.

import argparse
from utils.parser_utils import str2bool
from utils.trace_dataset import TraceDataset
from utils.file_operations import save_file, load_file
from .algorithms.cast import cast_clustering
import utils.config_utils as cm
import os
from utils.trace_operations import compute_super_matrix
from utils.overhead import total_data_overhead
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':


    # arguments
    parser = argparse.ArgumentParser(description='perform cast on a random website')

    parser.add_argument('-conf', '--config', help='which config file to use', default = 'default')

    parser.add_argument('-preload', type=str2bool, nargs='?', const=True, default=False,
                         help='Whether to load the pre-computed distance_matrix' )
    
    parser.add_argument('-website', help='which website to cluster', default = -1, type =int)

    parser.add_argument('-max_clusters', help='maximum number of clusters for each website', default = -1, type =int)

    parser.add_argument('-e', '--extract_ds', type=str2bool, nargs='?', const=True, default=False,
                        help='should we extract the dataset or is it already stored'
                        )
    
    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to save the computed elements')
    
    parser.add_argument('-save_clusters', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to save the computed clusters')
    
    parser.add_argument('-use_tamaraw', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to do cast based on tamaraw')
    
    parser.add_argument('-s_threshold', type=int,  default=10,
                        help='cluster with elements less than this will be considered small'
                        )
    
    parser.add_argument('-min_ds_interval', type = float,
                         help='the lower bound of the dataset partition we want to cluster', default = 0)
    
    parser.add_argument('-max_ds_interval', type = float,
                         help='the upper bound of the dataset partition we want to cluster', default = 1)
    

    logger = cm.init_logger(name = 'Clustering the traces')

    


    args = parser.parse_args()
    
    cm.initialize_common_params(args.config)
    
    website = args.website
    trace_mode = 'tam'
    dataset_interval = [args.min_ds_interval, args.max_ds_interval]
    dataset = TraceDataset(extract_traces= args.extract_ds, 
                           trace_mode= trace_mode, 
                           interval= dataset_interval, 
                           )
    if args.max_clusters != -1:
        max_number_of_clusters_range = [args.max_clusters]
    else:
        max_number_of_clusters_range =[i for i in range(5,11)]
    
    
    
    if website== -1: # performing  cast on all websites
        website_list = [w for w in range(cm.MON_SITE_START_IND, cm.MON_SITE_END_IND)]
    else:
        website_list = [website]

    
    maximum_bandwith_overheads = [] # the maimum overhead we have if we don't cluster
    for website_index in website_list:
        logger.info(f'Performing Cast on website {website_index}')
        if args.use_tamaraw:
            logger.info('doing it based on tamaraw')
        bandwidth_over_heads = []
        smallest_cluster_sizes = []
        biggest_cluster_sizes = []
        

        if args.preload:
            load_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'distance_matrices', 'euc')
            if args.use_tamaraw:
                load_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'distance_matrices', 'tamaraw_L')
            filename = f'{website_index}.npy'
            
            distance_matrix = load_file(dir_path= load_dir, file_name= filename)
            
        else:
            #compute distance list manually
            pass
        tams, _ = dataset.get_traces_of_class(class_number = website_index)

        traces_after = [None for i in range(len(tams))] 
        
        overall_super_matrix = compute_super_matrix(*tams)
        for element in range(len(tams)):
            traces_after[element] = overall_super_matrix
        maximum_bandwidth_overhead = total_data_overhead(traces_before= tams,
                                                        traces_after= traces_after,
                                                        trace_type= 'tam',
                                                        return_percentage= False)
        saved_clusters = None # the clusters we might save in the future

        for max_num_of_clusters in tqdm(max_number_of_clusters_range, desc= f'performing cast with different max number of clusters on website {website_index}'):
            
            if max_num_of_clusters > 1:
                clusters = cast_clustering(distance_matrix= distance_matrix,
                                        similarity_method= 'self_scaling',
                                        cleaning_step_max_iter= 20,
                                        post_processing= True,
                                        self_compute_affinity= True,
                                        rescaling_parameter= 7,
                                        maximum_number_of_clusters= max_num_of_clusters)
            else:
                # we consider a case that we don't have intra patterns and just do website level clustering. we will store all traces in a single cluster for compatibility with later steps
                clusters = [[i for i in range(distance_matrix.shape[0])]]
            

            # computing the overal bandwidth overhead
            representatives = [] # the representatives of each cluster

            traces_after = [None for i in range(len(tams))] 
            smallest_cluster_size = float("inf")
            biggest_cluster_size = float("-inf")
            
            for cluster in clusters:
                if len(cluster) < smallest_cluster_size:
                    smallest_cluster_size = len(cluster)
                if len(cluster) > biggest_cluster_size:
                    biggest_cluster_size = len(cluster)
                tams_of_this_cluster = [tams[i] for i in cluster]
                representative = compute_super_matrix(*tams_of_this_cluster)
                for element in cluster:
                    traces_after[element] = representative
            
            bandwidth_over_head = total_data_overhead(traces_before= tams,
                                                        traces_after= traces_after,
                                                        trace_type= 'tam',
                                                        return_percentage= False)
            
            
            bandwidth_over_heads.append(bandwidth_over_head)
            smallest_cluster_sizes.append(smallest_cluster_size)
            biggest_cluster_sizes.append(biggest_cluster_size)

            if args.max_clusters != -1: # we only have one cluster
                saved_clusters = clusters

        #cluster_size_distribution(clusters= clusters, clustering_algorithm= 'cast', website_index= website_index)
    
        if args.max_clusters == -1: # we have multiple cluster sizes, and we want to plot the differences.
            
            save_dir = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'cast', f'{cm.data_set_folder}', 'different ks',
                                    str(website_index), )
            if args.use_tamaraw:
                 save_dir = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'cast_tamaraw', f'{cm.data_set_folder}', 'different ks',
                                    str(website_index), )
            # Create a plot
            
            plt.figure(figsize=(8, 5))  # Adjust the figure size as needed
            
            plt.plot(max_number_of_clusters_range, bandwidth_over_heads, label='Bandwidth overhead', color='red')
            plt.xlabel('Maximum number of clusters')
            plt.ylabel('Value')
            plt.title(f'Comparison of Bandwidth Overhead, for website {website_index}')
            plt.legend()
            plt.grid(True)
            plt.axhline(y= maximum_bandwidth_overhead, color='purple', linestyle=':', linewidth=1, label='Max bandwidth overhead')
            if args.save:
                save_file(dir_path= save_dir, file_name= 'bandwidth.png')
            else:
                plt.show()
                
            
            
            plt.figure(figsize=(8, 5))  # Adjust the figure size as needed

            plt.plot(max_number_of_clusters_range, smallest_cluster_sizes, label='Smallest Cluster Size', color='blue', linestyle='--')
            plt.plot(max_number_of_clusters_range, biggest_cluster_sizes, label='Biggest Cluster Size', color='green', linestyle='-')

            plt.xlabel('Maximum Number of Clusters')
            plt.ylabel('Cluster Size')
            plt.title(f'Comparison of Cluster Sizes for website {website_index}')
            plt.legend()
            plt.grid(True)
            if args.save:
                save_file(dir_path= save_dir, file_name= 'cluster_sizes.png')
            else:
                plt.show()
                
        
        else:
            # we only have one cluster size, and maybe we want to save the clusters.
            if args.save_clusters:
                save_dir = os.path.join(cm.BASE_DIR,  'data', cm.data_set_folder, 'clustering', 'cast', f'max_clusters_{args.max_clusters}', str(website_index) )
                if args.use_tamaraw:
                    save_dir = os.path.join(cm.BASE_DIR,  'data', cm.data_set_folder, 'clustering', 'cast_tamaraw', f'max_clusters_{args.max_clusters}', str(website_index) )
                
                save_file(dir_path= save_dir, file_name= 'clusters.pkl', content= saved_clusters)



    




# python3 -m experiments.clustering.overall_cast -conf default  -website -1 -preload True -save_clusters True -max_clusters 5 -save True


    