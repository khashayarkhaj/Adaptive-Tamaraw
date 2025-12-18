# create a switching table that tells us what parameters to use when we are in cluster x and our target oh is (a,b)
import utils.config_utils as cm
from time import strftime
import argparse
import numpy as np

from utils.trace_dataset import TraceDataset
from utils.parser_utils import str2bool
from tqdm import tqdm
import os
from ...clustering.clustering_utils import load_two_tier_clusters
from utils.file_operations import load_file, save_file


from fixed_defenses.tamaraw import perform_tamaraw_on_trace, obtain_pareto_points
from multiprocessing import Pool
from functools import partial

import pandas as pd





def process_single_trace_two_level(trace, trace_percentage, args, ro_in_global, ro_out_global, ro_in_cluster, ro_out_cluster):
    """Process a single trace with the given parameters using two-level approach."""

    defended_trace = perform_tamaraw_on_trace(trace= trace,
                                              pad_length=args.L,
                                                randomized_extension=args.random_extend,
                                                cell_mode='time_dir',
                                                ro_in= ro_in_global,
                                                ro_out= ro_out_global,
                                                second_ro_in= ro_in_cluster,
                                                second_ro_out= ro_out_cluster,
                                                switch_percentage= trace_percentage)
    
    # Calculate metrics
    bandwidth_undefended = sum([abs(d[1]) for d in trace])
    bandwidth_defended = sum([abs(d[1]) for d in defended_trace])
    time_undefended = trace[-1][0]
    time_defended = defended_trace[-1][0]
    
    return {
        'bandwidth_undefended': bandwidth_undefended,
        'bandwidth_defended': bandwidth_defended,
        'time_undefended': time_undefended,
        'time_defended': time_defended,
    }


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

def find_closest_config(all_configs, target_latency=None, target_overhead=None, verbose=True, oh_threshold = 0.1):
    """
    Find the closest Pareto-optimal configuration based on target latency and overhead.
    
    Parameters:
    -----------
    all_configs : pandas.DataFrame
        DataFrame containing Pareto-optimal configurations with columns:
        ro_in, ro_out, bandwidth_overhead, time_overhead
    target_latency : float, optional
        Target time overhead
    target_overhead : float, optional
        Target bandwidth overhead
    verbose : bool, optional
        Whether to print detailed information about the selection process
    
    Returns:
    --------
    pandas.Series
        The closest configuration row
    """

    best_config = None
    # Validate inputs
    if target_latency is None and target_overhead is None:
        raise ValueError("At least one target (latency or overhead) must be specified")
    
    
    
    # Step 1: Find rows strictly better than targets (if targets are specified)
    better_rows = all_configs.copy()
    
    if target_latency is not None:
        better_rows = better_rows[better_rows['time_overhead'] <= target_latency]
    
    if target_overhead is not None:
        better_rows = better_rows[better_rows['bandwidth_overhead'] <= target_overhead]
    
    # If better rows exist, return the best one based on utility
    if not better_rows.empty:
        best_row = better_rows.loc[better_rows.apply(
            lambda row: calculate_utility(row['time_overhead'], row['bandwidth_overhead']), 
            axis=1
        ).idxmin()]
        
        if verbose:
            print("Found configuration within target bounds:")
            #print(best_row)
        
        best_config = best_row
    
    
    # Step 2: If no strictly better rows, find rows within threshold
    def find_closest_by_target(configs, target, column, threshold):
        """
        Find closest configuration for a specific target and column.
        """
        if target is None:
            return configs
        
        # Define thresholds (you can adjust these as needed)
        threshold_lower = target * (1 - threshold)
        threshold_upper = target * (1 + threshold)
        
        threshold_rows = configs[
            (configs[column] >= threshold_lower) & 
            (configs[column] <= threshold_upper)
        ]
        
        return threshold_rows if not threshold_rows.empty else configs
    

    if best_config is None:
        # Apply thresholds for both time and bandwidth
        threshold_configs = all_configs.copy()
        
        if target_latency is not None:
            threshold_configs = find_closest_by_target(
                threshold_configs, target_latency, 'time_overhead', threshold= oh_threshold
            )
        
        if target_overhead is not None:
            threshold_configs = find_closest_by_target(
                threshold_configs, target_overhead, 'bandwidth_overhead', threshold= oh_threshold
            )
        
        # If threshold rows exist, return the best one
        if not threshold_configs.empty:
            best_threshold_row = threshold_configs.loc[threshold_configs.apply(
                lambda row: calculate_utility(row['time_overhead'], row['bandwidth_overhead']), 
                axis=1
            ).idxmin()]
            
            if verbose:
                print("Found configuration within threshold bounds:")
                #print(best_threshold_row)
            
            best_config = best_threshold_row
    

    if best_config is None:
        # Step 3: If no rows in thresholds, find closest by individual targets
        closest_configs = []
        
        if target_latency is not None:
            closest_latency = all_configs.loc[
                (all_configs['time_overhead'] - target_latency).abs().idxmin()
            ]
            closest_configs.append(closest_latency)
        
        if target_overhead is not None:
            closest_overhead = all_configs.loc[
                (all_configs['bandwidth_overhead'] - target_overhead).abs().idxmin()
            ]
            closest_configs.append(closest_overhead)
        
        # If multiple closest configs, choose the best by utility
        if closest_configs:
            best_closest = min(closest_configs, key=lambda row: 
                calculate_utility(row['time_overhead'], row['bandwidth_overhead'])
            )
            
            if verbose:
                print("Found closest configuration by individual target:")
                #print(best_closest)
            
            best_config =  best_closest
    
    # If no configuration found at all
    if best_config is None:
        raise ValueError("No suitable configuration found")


    ro_in = best_config['ro_in']
    ro_out = best_config['ro_out']
    time_overhead= best_config['time_overhead']
    bw_overhead = best_config['bandwidth_overhead']
    if verbose:
        print(f'found ro_in is {ro_in}')
        print(f'found ro_out is {ro_out}')
        print(f'found time overhead is {time_overhead}')
        print(f'found bw overhead is {bw_overhead}')

    
    return ro_in, ro_out, time_overhead, bw_overhead
    
def get_top_k_configs(pareto_configs, k=5, verbose=True):
    """
    Get the top k configurations based on their utility.
    
    Parameters:
    -----------
    pareto_configs : pandas.DataFrame
        DataFrame containing Pareto-optimal configurations with columns:
        ro_in, ro_out, bandwidth_overhead, time_overhead
    k : int, optional
        Number of top configurations to return (default is 5)
    verbose : bool, optional
        Whether to print detailed information about the top configurations
    
    Returns:
    --------
    pandas.DataFrame
        Top k configurations sorted by utility (lowest utility first)
    """
    # Calculate utility for each row
    pareto_configs['utility'] = pareto_configs.apply(
        lambda row: calculate_utility(row['time_overhead'], row['bandwidth_overhead']), 
        axis=1
    )
    
    # Sort by utility and get top k
    top_k_configs = pareto_configs.sort_values('utility').head(k)
    
    if verbose:
        print(f"Top {k} Configurations:")
        print(top_k_configs)
    
    # Optional: drop the utility column if you don't want it in the final output
    top_k_configs = top_k_configs.drop(columns=['utility'])
    
    return top_k_configs

# # Example usage:
# # For latency-based search:
# ro_in, ro_out, time_oh, bw_oh = find_closest_config(all_configs, target_latency=0.04)

# # For overhead-based search:
# ro_in, ro_out, time_oh, bw_oh = find_closest_config(all_configs, target_latency=None, target_overhead=5.5)


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
                        choices=['palette', 'oka', 'palette_tamaraw_pareto', 'palette_tamaraw', 'palette_tamaraw_top_ten'],
                        help='type of clusering algorithm we have performed in the second tier',
                        default= 'palette') # TODO implement others
    
    parser.add_argument('-k', type = int , default= 0 , help = 'Minimum number of elements in each second tier cluster')
    parser.add_argument('-max_clusters', help='maximum number of clusters for each website when performing first tier cluster', default = 5, type =int)
    

    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to save the computed elements')
    
    parser.add_argument('-preload', type = str2bool, nargs='?', const=True, default=False,
                         help='Whether to load the pre-computed elements')
    
    parser.add_argument('-vizualize_tams', type=str2bool, nargs='?', const=True, default=True,
                    help='Whether to visualize five random tams after tamaraw')
    
    
    parser.add_argument('-L', type = int,
                         help='the L param for tamaraw', default = 50)
    
    parser.add_argument('-n_cores', type = int,
                         help='number of cores to use for tamaraw', default = None) # None means use all, 1 means don't do parallelism
    
    parser.add_argument('-top_configs', type = int,
                         help='how many of the top global parameter combinations we want to minimize', default = 10)
    parser.add_argument('-random_extend', type = str2bool, nargs='?', const=True, default=False,
                         help='Whether to randomly extend the tamaraw trace to a factor of L')
    
    parser.add_argument('-preload_clusters', type=str2bool, nargs='?', const=True, default=True,
                         help='Whether to load the pre-computed first tier clusters', )
    
    
    

    parser.add_argument('-start_config', type = int , default= 0 , help = 'for perfromance reasons, I might want to start from a different top_config rather than 0')

    parser.add_argument('-end_config', type = int , default= None , help = 'for perfromance reasons, I might want to start from a different top_config rather than 0')
    
    parser.add_argument('-div_threshold', type = float,
                         help='the diversity threshold we use in two tier', default = None)
    
    parser.add_argument('-diversity_penalty', type = float,
                         help='the diversity penalty we use in two tier', default = None)
    
    parser.add_argument('-l_tamaraw',  type = int , default= None , help = 'fixed L for tamaraw')

    
    
    
    logger = cm.init_logger(name = 'Creating Switching Table')
    args = parser.parse_args()

    cm.initialize_common_params(args.config)

    

    algorithm_tier1 = args.algorithm_tier1
    algorithm_tier2 = args.algorithm_tier2

    global_load_dir = os.path.join(cm.BASE_DIR, 'results', 'fixed_defenses', 'tamaraw', cm.data_set_folder, 'optimization')
    if args.l_tamaraw is not None:
        global_load_dir = os.path.join(cm.BASE_DIR, 'results', 'fixed_defenses', 'tamaraw', cm.data_set_folder, 'optimization', f'Fixed_L_{args.l_tamaraw}')
    # find the rho_in and rho out that gives us the closest latency to the given latency
    all_configs = load_file(dir_path= global_load_dir, file_name= 'all_params.csv')
    # we will find the best pareto point (otherwise we might choose a bad point)
    pareto_configs = obtain_pareto_points(results_df= all_configs)
    top_global_configs = get_top_k_configs(pareto_configs= pareto_configs, k= args.top_configs)
    
    
    switching_dictionary = {} # second tier cluster idx -> dict: target bw -> dict: target time -> (rho_in, rho_out)
    
    

    
    



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


    percentages_to_analyze = [round(i * 0.1, 1) for i in range(11)]
    unique_classes = set([label for label in ordered_labels if label is not None])
    num_classes = len(unique_classes)
    #we will load the best parameters for global tamaraw
    
    
    print(f"Number of unique classes: {num_classes}")
    start_config = args.start_config
    end_config = args.end_config
    if end_config is None:
        end_config = args.top_configs - 1
    
    config_counter = -1
    
    for config_index, top_global_config in top_global_configs.iterrows():   

        config_counter += 1  

        if config_counter < start_config :
            continue
        if config_counter > end_config :
            break

        
        
        over_heads = {}
        logger.info(f'global config {config_counter}')
        print(top_global_config)
        target_time_oh = top_global_config['time_overhead']
        target_bw_oh = top_global_config['bandwidth_overhead']
        ro_in_global = top_global_config['ro_in']
        ro_out_global = top_global_config['ro_out']
    
        

        for class_label in tqdm(unique_classes):
            
            # Get indices where ordered_labels matches current class
            class_indices = [i for i, label in enumerate(ordered_labels) if label == class_label]
            
            # Get corresponding traces for this class
            class_traces = [ordered_original_traces[i] for i in class_indices]
            
            #print(f"\nClass {class_label} has {len(class_traces)} instances")

            
            
            initial_save_path = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                    f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', f'k = {args.k}')
        
           
            if args.l_tamaraw is not None:
                cluster_load_dir = os.path.join(initial_save_path, f'fixed_L_{args.l_tamaraw}','tamaraw_optimization', f'cluster_{class_label}')
            elif args.div_threshold is not None and args.diversity_penalty is not None:
                cluster_load_dir = os.path.join(initial_save_path, f'div_{args.div_threshold}_l_penalty_{args.diversity_penalty}','tamaraw_optimization', f'cluster_{class_label}')
            
                
            else:
                cluster_load_dir = os.path.join(initial_save_path, 'tamaraw_optimization', f'cluster_{class_label}')
            
            
            all_cluster_configs = load_file(dir_path= cluster_load_dir, file_name= 'all_params.csv')
            pareto_configs_this_cluster = obtain_pareto_points(results_df= all_cluster_configs)

            

            ro_in_cluster, ro_out_cluster, _, _  = find_closest_config(all_configs= pareto_configs_this_cluster, target_latency= target_time_oh,
                                                                    target_overhead= target_bw_oh)

            if int(class_label) not in switching_dictionary.keys():
                switching_dictionary[int(class_label)] = {}
            if target_bw_oh not in switching_dictionary[class_label].keys():
                switching_dictionary[int(class_label)][target_bw_oh] = {}
            switching_dictionary[int(class_label)][target_bw_oh][target_time_oh] = (ro_in_cluster, ro_out_cluster)
    


    
    initial_save_path = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                    f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', f'k = {args.k}')
        
    
    if args.l_tamaraw is not None:
        save_dir = os.path.join(initial_save_path, f'fixed_L_{args.l_tamaraw}' ,'tamaraw_switching')
    elif args.div_threshold is not None and args.diversity_penalty is not None:
        save_dir = os.path.join(initial_save_path,f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}' ,'tamaraw_switching')
    else:
        save_dir = os.path.join(initial_save_path, 'tamaraw_switching')
    save_file(dir_path= save_dir, content = switching_dictionary, file_name= 'switching_dictionary.json')


# python3 -m experiments.fixed_defenses.Tamaraw.create_switching_table  -k 5 -alg2 palette_tamaraw -div_threshold 0.3 -diversity_penalty 16.00
# python3 -m experiments.fixed_defenses.Tamaraw.create_switching_table  -k 5 -alg2 palette_tamaraw -l_tamaraw 500 -top_configs 50
