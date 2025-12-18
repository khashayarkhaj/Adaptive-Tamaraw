# switching based on the predictions of holmes + kfp and the switching table and saving the overheads

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
from models.kfp.kfp_train import get_kfp_feature_set, get_kfp_prediction
import pandas as pd
from sklearn.metrics import accuracy_score
from training.train_utils import train_wf_model, stratified_split
import traceback
# Plotting Overheads with Matplotlib

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter




def plot_overheads(improvements_per_config, global_rhos_used, save_dir = None, file_name= 'overheads.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(improvements_per_config)))
    
    for idx, (overheads, rhos) in enumerate(zip(improvements_per_config, global_rhos_used)):
        trace_percentages = sorted(overheads.keys())
        bw_overheads = [overheads[tp]['bw_oh_total'] for tp in trace_percentages]
        time_overheads = [overheads[tp]['time_oh_total'] for tp in trace_percentages]
        
        label = f'ρin={rhos[0]:.2f}, ρout={rhos[1]:.2f}'
        
        # Plot on first axis without legend
        ax1.plot(trace_percentages, bw_overheads, 
                marker='o', color=colors[idx])
        # Plot on second axis with legend
        ax2.plot(trace_percentages, time_overheads, 
                marker='o', color=colors[idx], label=label)
    
    ax1.set_xlabel('Trace Percentage')
    ax1.set_ylabel('Bandwidth Overhead')
    ax1.set_title('Bandwidth Overhead vs Trace Percentage')
    ax1.grid(True)
    
    ax2.set_xlabel('Trace Percentage')
    ax2.set_ylabel('Time Overhead')
    ax2.set_title('Time Overhead vs Trace Percentage')
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent legend from being cut off
    plt.tight_layout()
    
    if save_dir is None:
        plt.show()
    else:
        save_file(dir_path= save_dir, file_name= file_name)



def process_single_trace_two_level(trace, trace_percentage, L, ro_in_global, ro_out_global, ro_in_cluster, ro_out_cluster):
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









def organize_clusters_by_timestep(safe_times, args_repetition=False):
    """
    Organize clusters by their threshold timesteps.
    
    Parameters:
    - safe_times: Dictionary of clusters with their time-related information
    - args_repetition: Boolean to determine if clusters should be repeated in subsequent timesteps
    
    Returns:
    - Dictionary mapping threshold timesteps to lists of cluster numbers
    """
    # Dictionary to store timesteps and their corresponding clusters
    timestep_to_clusters = {}
    
    # Iterate through the safe_times dictionary
    for cluster_num, cluster_info in safe_times.items():
        threshold_timestep = cluster_info['threshold_timestep']
        
        # If repetition is allowed, add the cluster to its own and subsequent timesteps
        if args_repetition:
            # Find all timesteps greater than or equal to the current timestep
            relevant_timesteps = [ts for ts in set(safe_times[c]['threshold_timestep'] 
                                                   for c in safe_times) 
                                  if ts >= threshold_timestep]
            
            # Add cluster to each of these timesteps
            big_count = 0
            for ts in relevant_timesteps:
                if ts == threshold_timestep:
                    pass
                if ts > threshold_timestep:
                    big_count += 1
                    # if big_count > 1:
                    #     continue
                if ts not in timestep_to_clusters:
                    timestep_to_clusters[ts] = []
                timestep_to_clusters[ts].append(cluster_num)
        
        # If no repetition, add cluster only to its exact timestep
        else:
            if threshold_timestep not in timestep_to_clusters:
                timestep_to_clusters[threshold_timestep] = []
            timestep_to_clusters[threshold_timestep].append(cluster_num)
    
    return timestep_to_clusters



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
    
    
    parser.add_argument('-n_cores', type = int,
                         help='number of cores to use for tamaraw', default = None) # None means use all, 1 means don't do parallelism
    
    parser.add_argument('-top_configs', type = int,
                         help='how many of the top global parameter combinations we want to minimize', default = 10)
    parser.add_argument('-random_extend', type = str2bool, nargs='?', const=True, default=False,
                         help='Whether to randomly extend the tamaraw trace to a factor of L')
    
    parser.add_argument('-preload_clusters', type=str2bool, nargs='?', const=True, default=True,
                         help='Whether to load the pre-computed first tier clusters', )
    parser.add_argument('-preload_results', type=str2bool, nargs='?', const=True, default=False,
                         help='Whether to lperform the code or just load and visualize', )
    
    

    parser.add_argument('-repetition', type=str2bool, nargs='?', const=True, default=False,
                         help='Whether a class can be chosen after its safe timestep again', )
    

    parser.add_argument('-start_config', type = int , default= 0 , help = 'for perfromance reasons, I might want to start from a different top_config rather than 0')

    parser.add_argument('-end_config', type = int , default= None , help = 'for perfromance reasons, I might want to start from a different top_config rather than 0')
    
    
    parser.add_argument('-div_threshold', type = float,
                         help='the diversity threshold we use in two tier', default = None)
    
    parser.add_argument('-diversity_penalty', type = float,
                         help='the diversity penalty we use in two tier', default = None)
    
    
    
    parser.add_argument('-l_tamaraw',  type = int , default= None , help = 'fixed L for tamaraw')

    
    parser.add_argument('-holmes_min_percentage', type = float,
                         help='minimum percentage of trace percentage we want holmes to have before prediciting', default = 0.2)
    
    
    logger = cm.init_logger(name = 'Performing Tamaraw')
    args = parser.parse_args()

    cm.initialize_common_params(args.config)

    

    algorithm_tier1 = args.algorithm_tier1
    algorithm_tier2 = args.algorithm_tier2

    # ==============================================================================
# TWO-TIER TAMARAW DEFENSE WITH HOLMES-KFP-ECDIRE SWITCHING
# ==============================================================================
# This script evaluates a two-tier Tamaraw defense mechanism that uses:
# - Global defense parameters for baseline protection
# - Cluster-specific parameters that switch based on ECDIRE predictions
# - Multiple precision thresholds to balance security and overhead
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: SETUP AND CONFIGURATION LOADING
# ------------------------------------------------------------------------------

# Determine the path for loading global Tamaraw optimization results
global_load_dir = os.path.join(cm.BASE_DIR, 'results', 'fixed_defenses', 'tamaraw', cm.data_set_folder, 'optimization')
if args.l_tamaraw is not None:
    # If a fixed L parameter is specified, load from the corresponding subdirectory
    global_load_dir = os.path.join(cm.BASE_DIR, 'results', 'fixed_defenses', 'tamaraw', cm.data_set_folder, 'optimization', f'Fixed_L_{args.l_tamaraw}')

# Load all configuration parameters and select the best Pareto-optimal points
# These represent optimal trade-offs between bandwidth overhead and time overhead
all_configs = load_file(dir_path=global_load_dir, file_name='all_params.csv')
pareto_configs = obtain_pareto_points(results_df=all_configs)
top_global_configs = get_top_k_configs(pareto_configs=pareto_configs, k=args.top_configs)

# ------------------------------------------------------------------------------
# SECTION 2: TEST DATA AND PREDICTIONS LOADING
# ------------------------------------------------------------------------------

# Load the indices of test traces to evaluate
load_path__test_indices = os.path.join(cm.BASE_DIR, 'data', 'holmes', cm.data_set_folder)
test_indices_file_name = 'test_indices.npy'
labels_to_websites_dict = {}
desired_trace_indices = load_file(dir_path=load_path__test_indices, file_name=test_indices_file_name).tolist()

# Construct the base path for loading cluster predictions
initial_save_path = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'clustering', 'two-tier', f'{algorithm_tier1}-{algorithm_tier2}',
                                f'{algorithm_tier1}-{args.max_clusters}', f'{algorithm_tier2}-{args.k}')

# Adjust the prediction loading path based on experiment parameters
if args.l_tamaraw is not None:
    load_path_predictions = os.path.join(initial_save_path, f'fixed_L_{args.l_tamaraw}', 'holmes_kfp_ECDIRE')
elif args.div_threshold is not None and args.diversity_penalty is not None:
    load_path_predictions = os.path.join(initial_save_path, f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}', 'holmes_kfp_ECDIRE')
else:
    load_path_predictions = os.path.join(initial_save_path, 'holmes_kfp_ECDIRE')

# Load the predictions for which cluster each trace belongs to at different time steps
minimum_holmes_percentage = int(args.holmes_min_percentage * 100)
test_predictions_filename = f'test_predictions_holmes_min_{minimum_holmes_percentage}.json'
predicted_second_tier_cluster = load_file(dir_path=load_path_predictions, file_name=test_predictions_filename)
predicted_second_tier_cluster = convert_numeric_keys(predicted_second_tier_cluster)

# ------------------------------------------------------------------------------
# SECTION 3: SAFE TIMES LOADING FOR DIFFERENT PRECISION THRESHOLDS
# ------------------------------------------------------------------------------

# Define precision thresholds for ECDIRE switching (0 = baseline with no switching)
perc_accs = [0, 0.5, 0.6, 0.7, 0.8, 0.9]


# Load safe times for each precision threshold
# Safe times indicate when it's safe to switch to cluster-specific parameters
safe_times_per_acc = {}
for perc_acc in perc_accs:
    if perc_acc != 0:
        # Construct path for safe times based on experiment configuration
        initial_save_path = os.path.join(cm.BASE_DIR, 'results', 'clustering', 'two-tier', f'{cm.data_set_folder}',
                                         f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', f'k = {args.k}')
        
        if args.l_tamaraw is not None:
            load_path_safe_times = os.path.join(initial_save_path, f'fixed_L_{args.l_tamaraw}', 'holmes_kfp_ECDIRE', f'perc_acc_{perc_acc}')
        elif args.div_threshold is not None and args.diversity_penalty is not None:
            load_path_safe_times = os.path.join(initial_save_path, f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}', 'holmes_kfp_ECDIRE', f'perc_acc_{perc_acc}')
        else:
            load_path_safe_times = os.path.join(initial_save_path, 'holmes_kfp_ECDIRE', f'perc_acc_{perc_acc}')
        
        # Load and convert safe times
        loaded = load_file(dir_path=load_path_safe_times, file_name=f'safe_times_holmes_min_{minimum_holmes_percentage}.json')
        safe_times_per_acc[perc_acc] = convert_numeric_keys(loaded, second_type=None)

# ------------------------------------------------------------------------------
# SECTION 4: DATASET AND CLUSTER LOADING
# ------------------------------------------------------------------------------

# Load the original undefended dataset for timing calculations
original_dataset_cell = TraceDataset(extract_traces=args.extract_ds, trace_mode='cell')

# Prepare parameters for loading cluster assignments
second_tier_file_name = None
second_tier_cluster_path = None
if args.div_threshold is not None and args.diversity_penalty is not None:
    # Build custom path if diversity parameters are specified
    second_tier_file_name = f'second_tier_clusters_peanlty_{args.diversity_penalty:.2f}_div_{args.div_threshold}.pkl'
    second_tier_cluster_path = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'clustering', 'two-tier', f'{algorithm_tier1}-{algorithm_tier2}',
                                            f'{algorithm_tier1}-{args.max_clusters}', f'{algorithm_tier2}-{args.k}', f'div_{args.div_threshold}')

# Load two-tier cluster assignments and related data
loading_results = load_two_tier_clusters(k=args.k,
                                         algorithm_tier1=algorithm_tier1,
                                         algorithm_tier2=algorithm_tier2,
                                         preload_clusters=args.preload_clusters,
                                         max_clusters=args.max_clusters,
                                         extract_ds=args.extract_ds,
                                         trace_mode='cell',
                                         trim_traces=False,
                                         replace_negative_ones=False,
                                         return_original_traces=True,
                                         second_tier_file_name=second_tier_file_name,
                                         second_tier_cluster_path=second_tier_cluster_path,
                                         l_tamaraw=args.l_tamaraw)

# Extract cluster assignments and mappings
tier1_clusters_of_each_website = loading_results['tier1_clusters_of_each_website']
tier2_clusters = loading_results['tier2_clusters']
super_matrix_mapping = loading_results['super_matrix_mapping']
ordered_traces = loading_results['ordered_traces']
ordered_labels = loading_results['ordered_labels']
ordered_websites = loading_results['ordered_websites']
overall_mapping = loading_results['overall_mapping']
ordered_original_traces = loading_results['ordered_original_traces']
website_ft_to_st = loading_results['reverse_mapping']
ordered_tier1_labels = loading_results['ordered_tier1_labels']

# Calculate number of unique classes
unique_classes = set([label for label in ordered_labels if label is not None])
num_classes = len(unique_classes)
print(f"Number of unique classes: {num_classes}")

# Set configuration range for evaluation
start_config = args.start_config
end_config = args.end_config



# ------------------------------------------------------------------------------
# SECTION 6: CONFIGURATION RANGE AND TRACKING SETUP
# ------------------------------------------------------------------------------

# Set default end configuration if not specified
if end_config is None:
    end_config = args.top_configs - 1

# Initialize tracking variables
config_counter = -1
improvements_per_config = []  # Store overhead improvements for each config
global_rhos_used = []  # Track which global rho values were used

# ------------------------------------------------------------------------------
# SECTION 7: SWITCHING DICTIONARY LOADING
# ------------------------------------------------------------------------------

# Load the switching dictionary that maps clusters to their optimal parameters
# given a target overhead (time and bandwidth)
initial_save_path = os.path.join(cm.BASE_DIR, 'results', 'clustering', 'two-tier', f'{cm.data_set_folder}',
                                f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', f'k = {args.k}')

if args.l_tamaraw is not None:
    switching_load_path = os.path.join(initial_save_path, f'fixed_L_{args.l_tamaraw}', 'tamaraw_switching')
elif args.div_threshold is not None and args.diversity_penalty is not None:
    switching_load_path = os.path.join(initial_save_path, f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}', 'tamaraw_switching')
else:
    switching_load_path = os.path.join(initial_save_path, 'tamaraw_switching')

switching_dictionary = load_file(dir_path=switching_load_path, file_name='switching_dictionary.json')
switching_dictionary = convert_nested_numeric_keys(switching_dictionary)

# ==============================================================================
# MAIN EVALUATION LOOP
# ==============================================================================

# Iterate through each top global configuration
for config_index, top_global_config in top_global_configs.iterrows():
    
    config_counter += 1
    
    # Skip configurations outside the specified range
    if config_counter < start_config:
        continue
    if config_counter > end_config:
        break
    
    # ------------------------------------------------------------------------------
    # SECTION 8: CURRENT CONFIGURATION SETUP
    # ------------------------------------------------------------------------------
    
    over_heads = {}
    logger.info(f'global config {config_counter}')
    print(top_global_config)
    
    # Extract target overheads and global rho parameters for this configuration
    target_time_oh = top_global_config['time_overhead']
    target_bw_oh = top_global_config['bandwidth_overhead']
    ro_in_global = top_global_config['ro_in']
    ro_out_global = top_global_config['ro_out']
    
    # --------------------------------------------------------------------------
    # SECTION 9: EVALUATION FOR EACH PRECISION THRESHOLD
    # --------------------------------------------------------------------------
    
    for perc_acc in perc_accs:
        logger.info(f'performing switching for perc_acc = {perc_acc}')
        
        # Load safe times and organize by timestep for ECDIRE switching
        if perc_acc != 0:
            safe_times = safe_times_per_acc[perc_acc]
            preditiction_time_steps = organize_clusters_by_timestep(safe_times=safe_times, args_repetition=args.repetition)
            # preditiction_time_steps: Maps each time_step to a list of clusters that can be safely predicted
        
        # ----------------------------------------------------------------------
        # SECTION 10: DETERMINE PREDICTED CLUSTER FOR EACH TRACE
        # ----------------------------------------------------------------------
        
        trace_predicted_cluster = {}  # Maps trace_idx to predicted cluster (-1 if no prediction)
        trace_predicted_percentage = {}  # Maps trace_idx to percentage of trace used for switching
        
        for original_idx, trace_idx in enumerate(tqdm(desired_trace_indices, desc='finding the predictioned cluster and switch percentage for the instances')):
            if perc_acc != 0:
                # Find the earliest prediction time where the predicted cluster is safe
                prediction_times = sorted(preditiction_time_steps.keys())
                predicted_cluster = -1
                chosen_percentage = None
                
                for prediction_time in prediction_times:
                    prediction = predicted_second_tier_cluster[trace_idx][prediction_time]
                    
                    # Check if prediction is in the list of safe clusters for this time
                    if prediction in preditiction_time_steps[prediction_time]:
                        predicted_cluster = prediction
                        chosen_time = prediction_time
                        
                        # Calculate what percentage of the trace this time represents
                        times_of_this_trace = original_dataset_cell.times[trace_idx]
                        final_time = times_of_this_trace[-1]
                        chosen_percentage = min(prediction_time / final_time, 1)
                        break
                
                trace_predicted_cluster[trace_idx] = predicted_cluster
                trace_predicted_percentage[trace_idx] = chosen_percentage
            else:
                # Baseline case: no switching, use original parameters for entire trace
                trace_predicted_cluster[trace_idx] = -1
                trace_predicted_percentage[trace_idx] = 1
        
        # ----------------------------------------------------------------------
        # SECTION 11: OPTIONAL PLOTTING OF SWITCHING DISTRIBUTIONS
        # ----------------------------------------------------------------------
        
        if perc_acc != 0:
            # Construct save directory for switching distribution plots
            initial_save_path = os.path.join(cm.BASE_DIR, 'results', 'clustering', 'two-tier', f'{cm.data_set_folder}',
                                            f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', f'k = {args.k}')
            
            if args.l_tamaraw is not None:
                save_dir_plot = os.path.join(initial_save_path, f'fixed_L_{args.l_tamaraw}', 'holmes_kfp_ECDIRE', 'switching_distributions', f'perc_acc_{perc_acc}')
            elif args.div_threshold is not None and args.diversity_penalty is not None:
                save_dir_plot = os.path.join(initial_save_path, f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}', 'holmes_kfp_ECDIRE', 'switching_distributions', f'perc_acc_{perc_acc}')
            else:
                save_dir_plot = os.path.join(initial_save_path, 'holmes_kfp_ECDIRE', 'switching_distributions', f'perc_acc_{perc_acc}')
            
            filename_percentage = None
        
        # ----------------------------------------------------------------------
        # SECTION 12: OVERHEAD CALCULATION FOR EACH CLUSTER
        # ----------------------------------------------------------------------
        
        logger.info(f'Performing two-tiered tamaraw perc_acc = {perc_acc} switching for global config {config_counter}')
        over_heads[perc_acc] = {}
        
        # Initialize overhead tracking variables
        total_bandwidth_undefended = 0
        total_bandwidth_defeneded = 0
        total_time_undefended = 0
        total_time_defeneded = 0
        number_of_traces_analyzed = 0
        
        # Process each cluster (including -1 for unpredicted traces)
        clusters_plus_baseline = [-1]
        clusters_plus_baseline += unique_classes
        
        for class_label in clusters_plus_baseline:
            logger.info(f'global config {config_counter} vs cluster {class_label} and switching with perc_acc = {perc_acc}')
            
            # Initialize overhead tracking for this cluster
            total_bandwidth_undefended_for_this_class = 0
            total_bandwidth_defeneded_for_this_class = 0
            total_time_undefended_for_this_class = 0
            total_time_defeneded_for_this_class = 0
            
            # ------------------------------------------------------------------
            # SECTION 13: COLLECT TRACES PREDICTED FOR THIS CLUSTER
            # ------------------------------------------------------------------
            
            traces_in_this_class = []
            switch_percentages_in_this_class = []
            
            for original_idx, trace_idx in enumerate(tqdm(desired_trace_indices, desc='finding the instances for this cluster')):
                predicted_second_cluster = trace_predicted_cluster[trace_idx]
                
                if predicted_second_cluster == class_label:
                    traces_in_this_class.append(ordered_original_traces[trace_idx])
                    switch_percentages_in_this_class.append(trace_predicted_percentage[trace_idx])
            
            # Skip if no traces were predicted for this cluster
            if len(traces_in_this_class) == 0:
                continue
            
            # ------------------------------------------------------------------
            # SECTION 14: DETERMINE CLUSTER-SPECIFIC PARAMETERS
            # ------------------------------------------------------------------
            
            if perc_acc == 0:
                # Baseline: no cluster-specific parameters
                ro_in_cluster = None
                ro_out_cluster = None
            else:
                if class_label != -1:
                    # Look up optimized parameters for this cluster
                    ro_in_cluster, ro_out_cluster = switching_dictionary[class_label][target_bw_oh][target_time_oh]
                else:
                    # ECDIRE couldn't predict a cluster, so don't switch
                    ro_in_cluster = None
                    ro_out_cluster = None
            
            print(f"\nClass {class_label} has {len(traces_in_this_class)} instances predicted to it")
            number_of_traces_analyzed += len(traces_in_this_class)
            
            # ------------------------------------------------------------------
            # SECTION 15: PROCESS TRACES (SINGLE-CORE OR MULTI-CORE)
            # ------------------------------------------------------------------
            
            if args.n_cores == 1:
                # Single-core processing with progress bar
                results = []
                for trace_idx, trace in enumerate(tqdm(traces_in_this_class, desc=f'Sequential Two tier tamaraw for class {class_label} and perc_acc {perc_acc} switching')):
                    result = process_single_trace_two_level(
                        trace=trace,
                        trace_percentage=switch_percentages_in_this_class[trace_idx],
                        L=args.l_tamaraw,
                        ro_in_global=ro_in_global,
                        ro_out_global=ro_out_global,
                        ro_in_cluster=ro_in_cluster,
                        ro_out_cluster=ro_out_cluster
                    )
                    results.append(result)
            else:
                # Multi-core processing using multiprocessing Pool
                trace_percentage_pairs = list(zip(traces_in_this_class, switch_percentages_in_this_class))
                
                # Define worker function that unpacks trace-percentage pairs
                def process_trace_with_percentage(trace_percentage_pair):
                    trace, trace_percentage = trace_percentage_pair
                    return process_single_trace_two_level(
                        trace=trace,
                        trace_percentage=trace_percentage,
                        L=args.l_tamaraw,
                        ro_in_global=ro_in_global,
                        ro_out_global=ro_out_global,
                        ro_in_cluster=ro_in_cluster,
                        ro_out_cluster=ro_out_cluster
                    )
                
                # Process traces in parallel
                with Pool(processes=args.n_cores) as pool:
                    results = list(tqdm(
                        pool.imap(process_trace_with_percentage, trace_percentage_pairs),
                        total=len(trace_percentage_pairs),
                        desc=f' Multi core ({args.n_cores}) Two tier tamaraw for class {class_label} with variable switching'
                    ))
            
            # ------------------------------------------------------------------
            # SECTION 16: AGGREGATE RESULTS FOR THIS CLUSTER
            # ------------------------------------------------------------------
            
            total_bandwidth_undefended_for_this_class = sum(r['bandwidth_undefended'] for r in results)
            total_bandwidth_defeneded_for_this_class = sum(r['bandwidth_defended'] for r in results)
            total_time_undefended_for_this_class = sum(r['time_undefended'] for r in results)
            total_time_defeneded_for_this_class = sum(r['time_defended'] for r in results)
            
            # Add to overall totals
            total_bandwidth_undefended += total_bandwidth_undefended_for_this_class
            total_bandwidth_defeneded += total_bandwidth_defeneded_for_this_class
            total_time_undefended += total_time_undefended_for_this_class
            total_time_defeneded += total_time_defeneded_for_this_class
        
        # ----------------------------------------------------------------------
        # SECTION 17: CALCULATE TOTAL OVERHEADS FOR THIS PRECISION THRESHOLD
        # ----------------------------------------------------------------------
        
        over_heads[perc_acc]['bw_oh_total'] = (total_bandwidth_defeneded - total_bandwidth_undefended) / total_bandwidth_undefended
        over_heads[perc_acc]['time_oh_total'] = (total_time_defeneded - total_time_undefended) / total_time_undefended
        
        print(f'{number_of_traces_analyzed} number of traces were analyzed')
        print(over_heads)
    
    # --------------------------------------------------------------------------
    # SECTION 18: SAVE RESULTS FOR THIS CONFIGURATION
    # --------------------------------------------------------------------------
    
    # Construct save directory based on experiment parameters
    initial_save_path = os.path.join(cm.BASE_DIR, 'results', 'clustering', 'two-tier', f'{cm.data_set_folder}',
                                    f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', f'k = {args.k}')
    
    if args.l_tamaraw is not None:
        save_dir = os.path.join(initial_save_path, f'fixed_L_{args.l_tamaraw}',
                               'holmes_kfp_ECDIRE', f'two_tier_tamaraw_utility_repitition_{args.repetition}', f'global_cfg_{config_counter}')
    elif args.div_threshold is not None and args.diversity_penalty is not None:
        save_dir = os.path.join(initial_save_path, f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}', 'holmes_kfp_ECDIRE', f'two_tier_tamaraw_utility_repitition_{args.repetition}', f'global_cfg_{config_counter}')
    else:
        save_dir = os.path.join(initial_save_path, 'holmes_kfp_ECDIRE', f'two_tier_tamaraw_utility_repitition_{args.repetition}', f'global_cfg_{config_counter}')
    
    # Add subdirectory for corrected results
    save_dir = os.path.join(save_dir, 'after_correct_fixing')
    
    # Log configuration details
    logger.info(f'k was {args.k}')
    logger.info(f'target time overhead was {target_time_oh}')
    logger.info(f'target bandwidth overhead was {target_bw_oh}')
    
    # Create final save path with overhead values
    save_dir = os.path.join(save_dir, f'bw_oh={target_bw_oh:.2f}_time_oh={target_time_oh:.2f}')
    
    # Save overhead results
    file_name_overhead = f'overheads_holmes_min_{minimum_holmes_percentage}.json'
    save_file(dir_path=save_dir, file_name=file_name_overhead, content=over_heads)
    
    # Track results for plotting
    global_rhos_used.append((ro_in_global, ro_out_global))
    improvements_per_config.append(over_heads)

# ==============================================================================
# FINAL RESULTS VISUALIZATION
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 19: PLOT OVERHEAD IMPROVEMENTS ACROSS ALL CONFIGURATIONS
# ------------------------------------------------------------------------------

# Construct final save directory for plots
initial_save_path = os.path.join(cm.BASE_DIR, 'results', 'clustering', 'two-tier', f'{cm.data_set_folder}',
                                f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}', f'k = {args.k}')

if args.l_tamaraw is not None:
    save_dir_overheads = os.path.join(initial_save_path, f'fixed_L_{args.l_tamaraw}', 'holmes_kfp_ECDIRE', f'two_tier_tamaraw_utility_repitition_{args.repetition}')
elif args.div_threshold is not None and args.diversity_penalty is not None:
    save_dir_overheads = os.path.join(initial_save_path, f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}', 'holmes_kfp_ECDIRE', f'two_tier_tamaraw_utility_repitition_{args.repetition}')
else:
    save_dir_overheads = os.path.join(initial_save_path, 'holmes_kfp_ECDIRE', f'two_tier_tamaraw_utility_repitition_{args.repetition}')

# Add subdirectory for corrected results
save_dir_overheads = os.path.join(save_dir_overheads, 'after_correct_fixing')

# Generate overhead comparison plot
filename_overhead_plot = f'overheads_holmes_min_{minimum_holmes_percentage}.png'

plot_overheads(improvements_per_config=improvements_per_config,
               global_rhos_used=global_rhos_used,
               save_dir=save_dir_overheads,
               file_name=filename_overhead_plot)


# python3 -m experiments.fixed_defenses.Tamaraw.tamaraw_two_tier_holmes_kfp_ecdire_switch -n_cores 30 -k 5 -alg2 palette_tamaraw -cc True -repetition False -l_tamaraw 100 -top_configs 33 -start_config 0 -end_config 10 





















            
        


        


