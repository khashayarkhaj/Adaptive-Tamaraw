# check adaptive tamaraw vs sota attacks

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
from utils.dataset_utils import to_dt
from models.Tik_Tok.train_utils import tt_input_processor, tt_training_loop

from fixed_defenses.tamaraw import perform_tamaraw_on_trace, obtain_pareto_points
from multiprocessing import Pool
from functools import partial
from utils.dataset_utils import  to_tam
import pandas as pd
from training.train_utils import train_wf_model, stratified_split
from training.config_manager import ConfigManager
from models.RF.rf_train import  rf_training_loop, RF_input_processor
import gc
from models.kfp.kfp_train import train_kfp
import random







def process_single_trace_two_level(trace, trace_percentage, L, ro_in_global, ro_out_global, ro_in_cluster, ro_out_cluster, return_defended = False):
    """Process a single trace with the given parameters using two-level approach."""

    defended_trace = perform_tamaraw_on_trace(trace= trace,
                                              pad_length= L,
                                                randomized_extension= False,
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
    if return_defended is False:
        defended_trace = None
    
    return {
        'bandwidth_undefended': bandwidth_undefended,
        'bandwidth_defended': bandwidth_defended,
        'time_undefended': time_undefended,
        'time_defended': time_defended,
        'defended_trace': defended_trace
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
def convert_numeric_keys(d, second_type = 'float'):
    # New dictionary to store converted keys
    converted = {}
    
    for key, value in d.items():
        # Convert outer key to int
        new_key = int(key) if key.isdigit() else float(key)
        
        # If value is also a dictionary, convert its keys to float
        if isinstance(value, dict):
            if second_type == 'float':
                new_value = {float(k): v for k, v in value.items()}
            elif second_type =='int':
                new_value = {int(k): v for k, v in value.items()}
            else:
                new_value = value
        else:
            new_value = value
            
        converted[new_key] = new_value
    
    return converted
def convert_nested_numeric_keys(d):
    converted = {}
    
    for key, value in d.items():
        # Convert first level key (class_label) to int
        new_key = int(key) if key.isdigit() else key
        
        if isinstance(value, dict):
            # Convert second level dictionary (target_bw_oh)
            second_level = {}
            for k2, v2 in value.items():
                # Convert second level key to float
                new_k2 = float(k2)
                
                if isinstance(v2, dict):
                    # Convert third level dictionary (target_time_oh)
                    third_level = {float(k3): v3 for k3, v3 in v2.items()}
                    second_level[new_k2] = third_level
                else:
                    second_level[new_k2] = v2
                    
            converted[new_key] = second_level
        else:
            converted[new_key] = value
            
    return converted

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
    

    parser.add_argument( '-attack_type',
                        choices=['RF', 'Tik_Tok', 'Laserbeak', 'kfp'],
                        help='type of attack',
                        default= 'RF') # TODO implement others

    parser.add_argument('-k', type = int , default= 0 , help = 'Minimum number of elements in each second tier cluster')
    parser.add_argument('-max_clusters', help='maximum number of clusters for each website when performing first tier cluster', default = 5, type =int)
    

    
    
    
    
    
    parser.add_argument('-train_config', help='which config file to use for training the model', default = 'RF_Tik_Tok_Tamaraw')
    
    parser.add_argument('-n_cores', type = int,
                         help='number of cores to use for tamaraw', default = None) # None means use all, 1 means don't do parallelism
    
    parser.add_argument('-top_configs', type = int,
                         help='how many of the top global parameter combinations we want to minimize', default = 50)
    
    
    parser.add_argument('-preload_clusters', type=str2bool, nargs='?', const=True, default=True,
                         help='Whether to load the pre-computed first tier clusters', )
    
    

    parser.add_argument('-start_config', type = int , default= 0 , help = 'for perfromance reasons, I might want to start from a different top_config rather than 0')

    parser.add_argument('-end_config', type = int , default= None , help = 'for perfromance reasons, I might want to start from a different top_config rather than 0')
    
    
    parser.add_argument('-l_tamaraw',  type = int , default= None , help = 'fixed L for tamaraw')
    
    parser.add_argument('-holmes_min_percentage', type = float,
                         help='minimum percentage of trace percentage we want holmes to have before prediciting', default = 0.2)
    logger = cm.init_logger(name = 'Performing Tamaraw')
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
    # [59, 60, 71, 58, 48, 70, 49, 72, 47, 46]
    '''
    59        59.0  0.012000  0.028991            1.172731       0.470899
    60        60.0  0.012000  0.040000            1.089567       0.561351
    71        71.0  0.016557  0.040000            0.929415       0.808849
    58        58.0  0.012000  0.021012            1.333530       0.407944
    48        48.0  0.008697  0.028991            1.442229       0.314202
    70        70.0  0.016557  0.028991            1.049691       0.710793
    49        49.0  0.008697  0.040000            1.396220       0.402220
    72        72.0  0.016557  0.055189            0.879026       0.946516
    47        47.0  0.008697  0.021012            1.578668       0.259557
    46        46.0  0.008697  0.015229            1.804958       0.222727
    '''
   
    # dir where the parameters for tamaraw performed on all traces is saved
    
    
    

    
    



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
                                            l_tamaraw= args.l_tamaraw)
    
    
    tier1_clusters_of_each_website = loading_results['tier1_clusters_of_each_website']
    tier2_clusters = loading_results['tier2_clusters']
    super_matrix_mapping = loading_results['super_matrix_mapping']
    ordered_traces = loading_results['ordered_traces']
    ordered_labels = loading_results['ordered_labels']
    ordered_websites = loading_results['ordered_websites']
    overall_mapping = loading_results['overall_mapping']
    ordered_original_traces = loading_results['ordered_original_traces']

    original_dataset_cell = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'cell')
    percentages_to_analyze = [round(i * 0.1, 1) for i in range(11)]
    percentages_to_analyze.insert(0, percentages_to_analyze.pop())
    unique_classes = set(ordered_labels)
    num_classes = len(unique_classes)
    #we will load the best parameters for global tamaraw
    
    
    ### loading the safetimes for each cluster
    perc_accs = [0.9] #precision thresholds for performing ecdire - 0 means we want the original params
    
    safe_times_per_acc = {} # perc_acc --> dictionary of safe times
    for perc_acc in perc_accs:
        if perc_acc != 0:
            if args.l_tamaraw is not None:
                load_path_safe_times = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                   f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}',f'k = {args.k}',  f'fixed_L_{args.l_tamaraw}', 'holmes_kfp_ECDIRE', f'perc_acc_{perc_acc}')
            elif args.div_threshold is not None and args.diversity_penalty is not None:
                load_path_safe_times = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                   f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}',f'k = {args.k}',  f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}', 'holmes_kfp_ECDIRE', f'perc_acc_{perc_acc}')
            else:
                load_path_safe_times = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                    f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', f'k = {args.k}', 'holmes_kfp_ECDIRE', f'perc_acc_{perc_acc}')

            minimum_holmes_percentage = int(args.holmes_min_percentage * 100)

            loaded = load_file(dir_path= load_path_safe_times, file_name= f'safe_times_holmes_min_{minimum_holmes_percentage}.json')
            safe_times_per_acc[perc_acc] = convert_numeric_keys(loaded, second_type=  None)



    













    print(f"Number of unique classes: {num_classes}")
    training_config_dir = os.path.join(cm.BASE_DIR, 'configs', 'training', args.train_config + '.yaml')
    hyperparam_manager = ConfigManager(config_path= training_config_dir)
    start_config = args.start_config
    end_config = args.end_config
    if end_config is None:
        end_config = args.top_configs - 1
    
    
    random_configs = random.sample(range(33), 4)
    config_counter = -1
    for config_index, top_global_config in top_global_configs.iterrows():   
        
        config_counter += 1 

        if config_counter not in random_configs:
            continue
         

        

        

        
        
        over_heads = {}
        logger.info(f'global config {config_counter}')
        print(top_global_config)
        target_time_oh = top_global_config['time_overhead']
        target_bw_oh = top_global_config['bandwidth_overhead']
        ro_in_global = top_global_config['ro_in']
        ro_out_global = top_global_config['ro_out']
        accuracies_per_percentage = {} # threshold -> accuracy
        for perc_acc in perc_accs:
            logger.info(f'Performing two-tiered tamaraw for {perc_acc} perc acc')
            over_heads[perc_acc] = {}
            total_bandwidth_undefended = 0
            total_bandwidth_defeneded = 0
            total_time_undefended = 0
            total_time_defeneded = 0
            defended_traces = []
            defended_labels = []

            # rf
            defended_tams = []

            # tik tok
            defended_dts = []

            # kfp 
            total_defended_directions = []
            total_defended_times = [] 


            for class_label in unique_classes:
                
                logger.info(f'global config {config_counter} vs cluster {class_label} and switching at {perc_acc} perc acc')
                over_heads[perc_acc][class_label] = {}
                total_bandwidth_undefended_for_this_class = 0
                total_bandwidth_defeneded_for_this_class = 0
                total_time_undefended_for_this_class = 0
                total_time_defeneded_for_this_class = 0
                # Get indices where ordered_labels matches current class
                class_indices = [i for i, label in enumerate(ordered_labels) if label == class_label]
                
                # Get corresponding traces for this class
                class_traces = [ordered_original_traces[i] for i in class_indices]
                
                print(f"\nClass {class_label} has {len(class_traces)} instances")

                
                # we will load the best parameters for cluster-based tamaraw
                cluster_load_dir = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                        f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}',f'k = {args.k}',
                                          f'fixed_L_{args.l_tamaraw}',
                                          'tamaraw_optimization', f'cluster_{class_label}')
                
                all_cluster_configs = load_file(dir_path= cluster_load_dir, file_name= 'all_params.csv')
                pareto_configs_this_cluster = obtain_pareto_points(results_df= all_cluster_configs)

                
                
                ro_in_cluster, ro_out_cluster, _, _  = find_closest_config(all_configs= pareto_configs_this_cluster, target_latency= target_time_oh,
                                                                        target_overhead= target_bw_oh)
                
                
                switch_percentages_in_this_class = []
                switch_time_in_this_class = safe_times_per_acc[perc_acc][class_label]['threshold_timestep']
                
                for trace_indice in tqdm(class_indices, desc=f'finding the switch percentages of traces in cluster {class_label}'):
                    times_of_this_trace = original_dataset_cell.times[trace_indice]
                    final_time = times_of_this_trace[-1]
                    idx = np.argmax(times_of_this_trace > switch_time_in_this_class)           # first True → index
                    idx = len(times_of_this_trace) if (times_of_this_trace[idx] <= switch_time_in_this_class) else idx

                    final_time = times_of_this_trace[-1]

                    
                    # check this later on
                    chosen_percentage = min(idx/len(times_of_this_trace), 1)
                    switch_percentages_in_this_class.append(chosen_percentage)
                    # breakpoint()
                
                
                
                if args.n_cores == 1:
                    # Single-core processing
                    results = []
                    for trace_idx, trace in enumerate(tqdm(class_traces, desc=f'Sequential Two tier tamaraw for class {class_label} and {perc_acc} perc acc')):
                        result = process_single_trace_two_level(
                            trace=trace,
                            trace_percentage= switch_percentages_in_this_class[trace_idx],
                            L = args.l_tamaraw,
                            ro_in_global=ro_in_global,
                            ro_out_global=ro_out_global,
                            ro_in_cluster=ro_in_cluster,
                            ro_out_cluster=ro_out_cluster,
                            return_defended= True
                        )
                        results.append(result)
                else:
                    
                    trace_percentage_pairs = list(zip(class_traces, switch_percentages_in_this_class))
                    def process_trace_with_percentage(trace_percentage_pair):
                        trace, trace_percentage = trace_percentage_pair
                        return process_single_trace_two_level(
                            trace=trace,
                            trace_percentage=trace_percentage,
                            L = args.l_tamaraw,
                            ro_in_global=ro_in_global,
                            ro_out_global=ro_out_global,
                            ro_in_cluster=ro_in_cluster,
                            ro_out_cluster=ro_out_cluster,
                            return_defended= True
                        )

                    with Pool(processes=args.n_cores) as pool:
                        results = list(tqdm(
                            pool.imap(process_trace_with_percentage, trace_percentage_pairs),
                            total=len(trace_percentage_pairs),
                            desc=f' Multi core ({args.n_cores}) Two tier tamaraw for class {class_label} with variable switching'
                        ))   
                    
                    
                    
                # adding the overheads
                # Aggregate results
                defended_traces = [np.array(r['defended_trace']) for r in results]
                defended_labels += [ordered_websites[i] for i in class_indices]
                defended_times, defended_directions = zip(*[trace.T for trace in defended_traces])
                del defended_traces
                if args.attack_type == 'RF':
                    defended_tams += to_tam(directions= defended_directions, times = defended_times).tolist()
                elif args.attack_type in ['Tik_Tok', 'Laserbeak']:
                    defended_dts += to_dt(directions= defended_directions, times = defended_times, padding_length= 10000)
                elif args.attack_type == 'kfp':
                    total_defended_directions += defended_directions
                    total_defended_times += defended_times
                    defended_tams += to_tam(directions= defended_directions, times = defended_times).tolist() # bad coding! keeping this so that we can use startified split
                
                del defended_times
                del defended_directions
            
                gc.collect()
            # converting the defended traces to tam
            
            
            if args.attack_type in ['RF', 'kfp']:
                defended_tams = np.array(defended_tams)
                X_train_def, y_train_def, X_val_def, y_val_def, X_test_def, y_test_def, train_indices, val_indices, test_indices = stratified_split(defended_tams, defended_labels,
                                                                                                                            train_ratio= 0.64,
                                                                                                                            val_ratio= 0.16,
                                                                                                                        test_ratio= 0.2)
                

                if args.attack_type == 'kfp':
                    directions_train = [total_defended_directions[i] for i in train_indices]
                    times_train = [total_defended_times[i] for i in train_indices]


                    directions_test = [total_defended_directions[i] for i in test_indices]
                    times_test = [total_defended_times[i] for i in test_indices]

            elif args.attack_type == 'Tik_Tok':
                defended_dts = np.array(defended_dts)
                X_train_def, y_train_def, X_val_def, y_val_def, X_test_def, y_test_def, train_indices, val_indices, test_indices = stratified_split(defended_dts, defended_labels,
                                                                                                                            train_ratio= 0.64,
                                                                                                                            val_ratio= 0.16,
                                                                                                                        test_ratio= 0.2)


            if args.attack_type == 'Laserbeak':
                defended_dts = np.array(defended_dts)

                # convert defended dts to the laserbeak required pkl. train and test the pkl seperatly on laserbeak code
                # it is a dict of website num, to a list of traces. each element of that list is also a list of size 1
                dt_dict = {}
                for trace_idx in range(len(defended_labels)):
                    label = defended_labels[trace_idx]
                    if label not in dt_dict:
                        dt_dict[label] = []
                    dt_dict[label].append([defended_dts[trace_idx].tolist()])
                save_path_dt = os.path.join(cm.BASE_DIR,  'laserbeak',  'adaptive_tamaraw')
                save_file(dir_path = save_path_dt, file_name = f'adaptive_tamaraw-mon_{config_counter}.pkl', content = dt_dict)
                continue
                # save the dt files
            # we want the retraining to be as fast as possible. we will not evaluate the model during each epoch and do it only once at the end
            if args.attack_type == 'RF':
                training_results = train_wf_model(
                                                    logger= logger, 
                                training_loop= rf_training_loop  ,                                                                                                          
                                num_classes= cm.MON_SITE_NUM,
                                input_processor= RF_input_processor,
                                hyperparam_manager= hyperparam_manager,
                                save_model= False,
                                x_train = X_train_def,
                                y_train = y_train_def,
                                x_test= X_test_def,
                                y_test= y_test_def,
                                x_val= X_val_def,
                                y_val= y_val_def,
                                if_use_gpu= True,
                                report_train_accuracy= True,
                                wf_model_type= 'RF',
                                should_evaluate= True,
                                training_loop_tqdm= True)
            
            elif args.attack_type == 'Tik_Tok':
                training_results = train_wf_model(
                                                    logger= logger, 
                                training_loop= tt_training_loop  ,                                                                                                          
                                num_classes= cm.MON_SITE_NUM,
                                input_processor= tt_input_processor,
                                hyperparam_manager= hyperparam_manager,
                                save_model= False,
                                x_train = X_train_def,
                                y_train = y_train_def,
                                x_test= X_test_def,
                                y_test= y_test_def,
                                x_val= X_val_def,
                                y_val= y_val_def,
                                if_use_gpu= True,
                                report_train_accuracy= True,
                                wf_model_type= 'tt',
                                should_evaluate= True,
                                training_loop_tqdm= True)
            elif args.attack_type == 'kfp':
                test_acc, _ = train_kfp( directions_train= directions_train, directions_test= directions_test, 
                            times_train= times_train, times_test= times_test,
                            y_train= y_train_def, y_test= y_test_def, save_features= False,
                            num_trees= 500, verbose= True, return_predictions= False)
                
                training_results = {'test_accuracy' : test_acc}
                print('kfp test accuracy is :', test_acc)

            
            if args.attack_type != 'Laserbeak':
                accuracies_per_percentage[perc_acc] = training_results['test_accuracy']
                print('accuracies up to this point')
                print(accuracies_per_percentage)

                save_dir = os.path.join(cm.BASE_DIR,  'results',  'clustering', 'two-tier', f'{cm.data_set_folder}',
                                                    f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', f'k = {args.k}', 'two_tier_tamaraw_utility_holmes_kfp', f'global_cfg_{config_counter}')
                

                
                save_file(dir_path = save_dir, file_name = f'{args.attack_type}_accuracies.json', content = accuracies_per_percentage)







# python3 -m experiments.fixed_defenses.Tamaraw.tamaraw_two_tier_vs_attacks -n_cores 1 -k 7 -l_tamaraw 100 -start_config 0 -alg2 palette_tamaraw -train_config TT -attack_type Tik_Tok

# python3 -m experiments.fixed_defenses.Tamaraw.tamaraw_two_tier_vs_attacks -n_cores 1 -k 7 -l_tamaraw 100 -start_config 0 -alg2 palette_tamaraw  -attack_type kfp


#python3 -m experiments.fixed_defenses.Tamaraw.tamaraw_two_tier_vs_attacks -n_cores 1 -k 7 -l_tamaraw 100  -start_config 0 -alg2 palette_tamaraw  -attack_type Laserbeak
