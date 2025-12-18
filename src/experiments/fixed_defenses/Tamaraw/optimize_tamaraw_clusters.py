# finding the best parameters of tamaraw for each cluster in two_tier clustering
import utils.config_utils as cm

import argparse
import numpy as np

from utils.trace_dataset import TraceDataset
from utils.parser_utils import str2bool
from tqdm import tqdm
import os
from ...clustering.clustering_utils import load_two_tier_clusters
from utils.file_operations import load_file

from fixed_defenses.tamaraw import grid_search_tamaraw_params

if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser(description='Visualizing Tams')
    
    parser.add_argument('-e',  '--extract_ds', type=str2bool, nargs='?', const=True, default=False, 
                         help='should we extract the dataset or is it already stored')
        
    parser.add_argument('-conf', '--config', help='which config file to use', default = 'Tik_Tok')

    parser.add_argument('-alg1', '--algorithm_tier1',
                        choices=['k_medoids', 'cast'],
                        help='type of clusering algorithm we have to performed in the first tier',
                        default= 'cast') # TODO implement others
    
    parser.add_argument('-alg2', '--algorithm_tier2',
                        choices=['palette', 'oka', 'palette_tamaraw', 'palette_tamaraw_pareto', 'palette_tamaraw_top_ten'],
                        help='type of clusering algorithm we have performed in the second tier',
                        default= 'palette') # TODO implement others
    
    parser.add_argument('-k', type = int , default= 0 , help = 'Minimum number of elements in each second tier cluster')
    parser.add_argument('-max_clusters', help='maximum number of clusters for each website when performing first tier cluster', default = 5, type =int)
    

    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to save the computed elements')
    
    parser.add_argument('-preload', type = str2bool, nargs='?', const=True, default=False,
                         help='Whether to load the pre-computed elements')
    
    parser.add_argument('-preload_ohs', type = str2bool, nargs='?', const=True, default=False,
                         help='Whether to load the pre-computed defended traces statistics')
    
    
    
    parser.add_argument('-n_cores', type = int,
                         help='number of cores to use for tamaraw', default = None) # None means use all, 1 means don't do parallelism
    
    parser.add_argument('-random_extend', type = str2bool, nargs='?', const=True, default=False,
                         help='Whether to randomly extend the tamaraw trace to a factor of L')
    
    parser.add_argument('-preload_clusters', type=str2bool, nargs='?', const=True, default=True,
                         help='Whether to load the pre-computed first tier clusters', )
    
   
    
    parser.add_argument('-start_cluster', type = int , default= 0 , help = 'for perfromance reasons, I might want to start from a different cluster rather than 0')

    parser.add_argument('-end_cluster', type = int , default= None , help = 'for perfromance reasons, I might want to end in a different cluster rather than the last one')
    
    parser.add_argument('-div_threshold', type = float,
                         help='the diversity threshold we use in two tier', default = None)
    
    parser.add_argument('-diversity_penalty', type = float,
                         help='the diversity penalty we use in two tier', default = None)
    
   
    
    parser.add_argument('-l_tamaraw',  type = int , default= None , help = 'fixed L for tamaraw')



    
    
    logger = cm.init_logger(name = 'Performing Tamaraw')
    args = parser.parse_args()

    cm.initialize_common_params(args.config)

    

    algorithm_tier1 = args.algorithm_tier1
    algorithm_tier2 = args.algorithm_tier2
    

    
    


    # do what tao does
    rho_in_initial = 0.012
    rho_out_initial = 0.04
    # n ranges from -5 to +5 (inclusive), giving us 11 values
    n_values = np.arange(-7, 7)  # [-10, -9, ..., 9, 10]

    # Calculate the exponential factor for each n
    factors = np.power(7, n_values / 7)  # factor = 10^(n/10)

    # Generate the rho_in and rho_out values
    ro_in_range = rho_in_initial * factors
    ro_out_range = rho_out_initial * factors


    second_tier_file_name = None
    second_tier_cluster_path = None
    if args.div_threshold is not None and args.diversity_penalty is not None:
         # if these two are given, we build our own path
         
         second_tier_file_name = f'second_tier_clusters_peanlty_{args.diversity_penalty:.2f}_div_{args.div_threshold}.pkl'
         second_tier_cluster_path = os.path.join(cm.BASE_DIR,  'data', cm.data_set_folder, 'clustering', 'two-tier', f'{algorithm_tier1}-{algorithm_tier2}',
                                f'{algorithm_tier1}-{args.max_clusters}', f'{algorithm_tier2}-{args.k}', f'div_{args.div_threshold}' )
    
    
    loading_results = load_two_tier_clusters(k = args.k,
                                            algorithm_tier1= algorithm_tier1,
                                            algorithm_tier2= algorithm_tier2,
                                            preload_clusters = args.preload_clusters,
                                            max_clusters= args.max_clusters,
                                            extract_ds= args.extract_ds,
                                            trace_mode= 'cell',
                                            trim_traces= False,
                                            replace_negative_ones= False,
                                            return_original_traces= True,
                                            second_tier_file_name= second_tier_file_name,
                                            second_tier_cluster_path= second_tier_cluster_path,
                                            l_tamaraw= args.l_tamaraw,
                                            )
    
    
    tier1_clusters_of_each_website = loading_results['tier1_clusters_of_each_website']
    tier2_clusters = loading_results['tier2_clusters']
    super_matrix_mapping = loading_results['super_matrix_mapping']
    ordered_traces = loading_results['ordered_traces']
    ordered_labels = loading_results['ordered_labels']
    ordered_websites = loading_results['ordered_websites']
    overall_mapping = loading_results['overall_mapping']
    ordered_original_traces = loading_results['ordered_original_traces']





        #### performing grid search on each of the clusters
        # First, get unique classes using set()
    unique_classes = set([label for label in ordered_labels if label is not None])
    num_classes = len(unique_classes)

    last_cluster = num_classes
    if args.end_cluster is not None:
        last_cluster = args.end_cluster + 1
    print(f"Number of unique classes: {num_classes}")
    logger.info(f'Optimization will start from cluster {args.start_cluster}')
    logger.info(f'Optimization will end in cluster number {last_cluster - 1}')
    # Iterate through each class and find corresponding instances
    trace_oh_dict = None
    if args.preload_ohs:
            param_load_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'tamaraw_params')
            if args.l_tamaraw is not None:
                 param_load_dir = os.path.join(param_load_dir, f'L_{args.l_tamaraw}')
            trace_oh_dict = load_file(dir_path= param_load_dir,  file_name= f'trace_params.json',  string_to_num = True )
    for class_label in range(args.start_cluster, last_cluster):
        # Get indices where ordered_labels matches current class
        class_indices = [i for i, label in enumerate(ordered_labels) if label == class_label]
        
        # Get corresponding traces for this class
        class_traces = [ordered_original_traces[i] for i in class_indices]
        
        print(f"\nClass {class_label} has {len(class_traces)} instances")
        # Work with class_traces as needed

        
        # changed the save path here to include max_cluster
        initial_save_path = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                    f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', f'k = {args.k}')
        
        
        if args.l_tamaraw is not None:
             save_dir = os.path.join(initial_save_path, f'fixed_L_{args.l_tamaraw}','tamaraw_optimization', f'cluster_{class_label}')
        elif args.div_threshold is not None and args.diversity_penalty is not None:
             save_dir = os.path.join(initial_save_path, f'div_{args.div_threshold}_l_penalty_{args.diversity_penalty}','tamaraw_optimization', f'cluster_{class_label}')
        
        
             
        else:
            save_dir = os.path.join(initial_save_path, 'tamaraw_optimization', f'cluster_{class_label}')

        
        cluster_indices = None

        if args.preload_ohs:
            
            cluster_indices = class_indices

        
        grid_search_tamaraw_params(undefended_traces= class_traces,
                                ro_in_range= ro_in_range,
                                ro_out_range= ro_out_range,
                                L = args.l_tamaraw, n_cores= args.n_cores, random_extend= args.random_extend,
                                save_dir= save_dir,
                                trace_oh_dict= trace_oh_dict,
                                cluster_indices= cluster_indices)


    # python3 -m experiments.fixed_defenses.Tamaraw.optimize_tamaraw_clusters -n_cores 1 -k 5 -alg2  palette_tamaraw -preload_ohs True -l_tamaraw 500 -conf Tik_Tok


   
