# trying to combine ecdire and kfp + holmes in tamaraw clustering
# this code loads the predictions of holmes for wesbsites for different percentages
# loads the predicitons of kfp models in each website for different time stamps
# for different time steps, predicts the second tier cluster of each trace
# stores the accuracies for each timestep and each cluster seperatly
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
matplotlib.use('Agg')
import gc







def make_json_serializable(data):
    """
    Convert a nested dictionary containing numpy values to JSON-serializable format.
    
    Args:
        data (dict): Nested dictionary where keys can be integers/floats and values
                    can be numpy numbers or regular Python numbers
    
    Returns:
        dict: A JSON-serializable version of the input dictionary
    """
    def convert_value(v):
        # Convert numpy numbers to Python native types
        if isinstance(v, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                         np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(v)
        if isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
            return float(v)
        if isinstance(v, dict):
            return {str(k): convert_value(v) for k, v in v.items()}
        return v

    # Convert the outer dictionary
    serializable_dict = {}
    for trace_idx, threshold_dict in data.items():
        # Convert the trace_idx to string (JSON requires string keys)
        serializable_dict[str(trace_idx)] = {}
        for threshold, cluster in threshold_dict.items():
            # Convert both threshold and cluster values
            serializable_dict[str(trace_idx)][str(float(threshold))] = convert_value(cluster)
    
    return serializable_dict



def calculate_accuracy_by_percentage(true_websites, early_website_prediction_mapping, desired_trace_indices, percentages):
    """
    Calculate classification accuracy for different percentage values.
    
    Parameters:
    -----------
    true_websites : dict
        Dictionary mapping trace indices to true labels
    early_website_prediction_mapping : dict
        Nested dictionary mapping trace index and percentage to predictions
    desired_trace_indices : list
        List of trace indices to evaluate
    percentages : list
        List of percentage values to evaluate (e.g., [0.2, 0.3, ..., 1.0])
    
    Returns:
    --------
    dict
        Dictionary mapping percentage values to their corresponding accuracy scores
    """
    accuracy_results = {}
    
    for percentage in percentages:
        accuracy_results[percentage] = calculate_accuracy_for_percentage(
            true_websites,
            early_website_prediction_mapping,
            desired_trace_indices,
            percentage
        )
            
    return accuracy_results
def fix_dictionary_keys(early_website_prediction_mapping):
    """
    Convert string keys in the early_website_prediction_mapping dictionary to their proper numeric types.
    Cluster keys are converted to integers, and timestep keys are converted to floats.
    
    Parameters:
    - early_website_prediction_mapping: Dictionary with string keys for both clusters and timesteps
    
    Returns:
    - fixed_dict: Dictionary with proper numeric keys
    """
    fixed_dict = {}
    
    # Process each cluster
    for trace_str, timestep_dict in early_website_prediction_mapping.items():
        # Convert cluster key to integer
        try:
            trace_key = int(trace_str)
        except ValueError:
            # If conversion to int fails, keep as string
            trace_key = trace_str
        
        # Create a new dictionary for this cluster with float timestep keys
        fixed_timestep_dict = {}
        for timestep_str, cluster in timestep_dict.items():
            # Convert timestep key to float
            try:
                timestep_key = float(timestep_str)
                fixed_timestep_dict[timestep_key] = cluster
            except ValueError:
                # Skip entries where timestep can't be converted to float
                print(f"Warning: Could not convert timestep '{timestep_str}' to float. Skipping.")
                continue
        
        # Add the fixed timestep dictionary to the fixed result
        fixed_dict[trace_key] = fixed_timestep_dict
    
    return fixed_dict
def calculate_accuracy_for_percentage(true_websites, early_website_prediction_mapping, desired_trace_indices, percentage):
    """
    Calculate classification accuracy for a single percentage value.
    
    Parameters:
    -----------
    true_websites : dict
        Dictionary mapping trace indices to true labels
    early_website_prediction_mapping : dict
        Nested dictionary mapping trace index and percentage to predictions
    desired_trace_indices : list
        List of trace indices to evaluate
    percentage : float
        The percentage value to evaluate (e.g., 0.2, 0.3, etc.)
    
    Returns:
    --------
    float
        Accuracy score for the given percentage
    """
    correct_predictions = 0
    total_predictions = 0
    
    for trace_idx in desired_trace_indices:
        if trace_idx in early_website_prediction_mapping and percentage in early_website_prediction_mapping[trace_idx]:
            prediction = early_website_prediction_mapping[trace_idx][percentage]
            true_label = true_websites[trace_idx]
            
            
            if prediction is not None and prediction == true_label:
                correct_predictions += 1
            total_predictions += 1
    # print(f'number of instances were {total_predictions}')
    if total_predictions > 0:
        
        return correct_predictions / total_predictions
    
    
    return 0.0



def find_closest_percentage(times_of_this_trace, time_step, percentages):
    # If the trace is empty, return None
    if times_of_this_trace is None:
        return None
    
    # Calculate what percentage of the trace this time_step represents
    # First, find the position of time_step in the trace
    total_duration = times_of_this_trace[-1] - times_of_this_trace[0]
    
    # Find the closest time in the trace that doesn't exceed time_step
    if time_step < times_of_this_trace[0]:
        # time_step is before the start of the trace
        return None
    
    if time_step >= times_of_this_trace[-1]:
        # time_step is at or after the end of the trace
        actual_percentage = 100.0
    else:
        # Find the position of time_step
        for i, t in enumerate(times_of_this_trace):
            if t > time_step:
                prev_time = times_of_this_trace[i-1] if i > 0 else times_of_this_trace[0]
                time_position = prev_time - times_of_this_trace[0]
                actual_percentage = (time_position / total_duration) * 100.0
                break
    
    # If the actual percentage is less than the minimum in percentages, return None
    if actual_percentage < min(percentages):
        return None
    
    # Find the closest percentage in the percentages list
    closest_percentage = min(percentages, key=lambda x: abs(x - actual_percentage))
    
    return closest_percentage
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
    
    
    
    
    
    
    parser.add_argument('-preload_clusters', type=str2bool, nargs='?', const=True, default=True,
                         help='Whether to load the pre-computed first tier clusters', )
    parser.add_argument('-preload_results', type=str2bool, nargs='?', const=True, default=False,
                         help='Whether to lperform the code or just load and visualize', )
    

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
    

    
   
    
    
    """
    Two-Tier Clustering Prediction Pipeline
    This script performs early website prediction using Holmes and k-FP classifiers
    on a two-tier clustering architecture.
    """


    # ============================================================================
    # CONFIGURATION AND PATH SETUP
    # ============================================================================

    # Determine second-tier cluster file paths based on diversity parameters
    second_tier_file_name = None
    second_tier_cluster_path = None

    if args.div_threshold is not None and args.diversity_penalty is not None:
        # Build custom path for diversity-based clustering
        second_tier_file_name = f'second_tier_clusters_penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}.pkl'
        second_tier_cluster_path = os.path.join(
            cm.BASE_DIR, 'data', cm.data_set_folder, 'clustering', 'two-tier',
            f'{algorithm_tier1}-{algorithm_tier2}',
            f'{algorithm_tier1}-{args.max_clusters}',
            f'{algorithm_tier2}-{args.k}',
            f'div_{args.div_threshold}'
        )

    # ============================================================================
    # LOAD TWO-TIER CLUSTERING DATA
    # ============================================================================

    loading_results = load_two_tier_clusters(
        k=args.k,
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
        l_tamaraw=args.l_tamaraw
    )

    # Extract clustering results
    tier1_clusters_of_each_website = loading_results['tier1_clusters_of_each_website']
    tier2_clusters = loading_results['tier2_clusters']
    super_matrix_mapping = loading_results['super_matrix_mapping']
    ordered_traces = loading_results['ordered_traces']
    ordered_labels = loading_results['ordered_labels']
    ordered_websites = loading_results['ordered_websites']
    overall_mapping = loading_results['overall_mapping']
    ordered_original_traces = loading_results['ordered_original_traces']
    website_ft_to_st = loading_results['reverse_mapping']  # Maps website -> first-tier -> second-tier
    ordered_tier1_labels = loading_results['ordered_tier1_labels']

    # ============================================================================
    # LOAD ORIGINAL DATASET AND TEST INDICES
    # ============================================================================

    # Load original dataset with cell-level traces
    original_dataset_cell = TraceDataset(extract_traces=args.extract_ds, trace_mode='cell')

    # Load test indices
    load_path_test_indices = os.path.join(cm.BASE_DIR, 'data', 'holmes', cm.data_set_folder)
    test_indices_file_name = 'test_indices.npy'
    desired_trace_indices = load_file(dir_path=load_path_test_indices, file_name=test_indices_file_name).tolist()

    # ============================================================================
    # SETUP HOLMES EARLY PREDICTION PARAMETERS
    # ============================================================================

    # Define time steps and percentages for early prediction
    minimum_holmes_percentage = int(args.holmes_min_percentage * 100)
    percentages = list(range(minimum_holmes_percentage, 101))  # e.g., [20, 21, ..., 100]
    time_steps = [0.8 * i for i in range(1, 101)]  # [0.8, 1.6, ..., 80.0]

    # ============================================================================
    # LOAD OR COMPUTE EARLY WEBSITE PREDICTIONS (HOLMES)
    # ============================================================================

    load_path_holmes = os.path.join(cm.BASE_DIR, 'data', 'holmes', cm.data_set_folder, 'holmes_predictions')
    save_path_early_websites = os.path.join(cm.BASE_DIR, 'data', 'holmes', cm.data_set_folder)
    save_file_name = f'early_website_prediction_holmes_min_{minimum_holmes_percentage}_ecdire.json'

    early_website_prediction_mapping = {}  # trace_idx -> time_step -> predicted_website

    try:
        # Try to load pre-computed predictions
        early_website_prediction_mapping = load_file(
            dir_path=save_path_early_websites,
            file_name=save_file_name,
        )
        early_website_prediction_mapping = fix_dictionary_keys(early_website_prediction_mapping)
        logger.info('Early website predictions mapping loaded from cache')
        
    except:
        # Compute early website predictions for each trace and time step
        logger.info('Computing early website predictions...')
        
        for idx, trace_index in enumerate(tqdm(desired_trace_indices, desc='Computing website predictions')):
            if trace_index not in early_website_prediction_mapping:
                early_website_prediction_mapping[trace_index] = {}
            
            times_of_this_trace = original_dataset_cell.times[trace_index]
            
            for time_step in time_steps:
                # Find the closest percentage threshold for this time step
                closest_percentage = find_closest_percentage(times_of_this_trace, time_step, percentages)
                
                if closest_percentage is not None:
                    # Load Holmes predictions for this percentage
                    predictions = load_file(
                        dir_path=load_path_holmes,
                        file_name=f'taf_test_p{closest_percentage}_Holmes_predictions.npy',
                        verbose=False
                    )
                    website_prediction = predictions[idx]
                    early_website_prediction_mapping[trace_index][time_step] = website_prediction
                else:
                    early_website_prediction_mapping[trace_index][time_step] = None
        
        # Save computed predictions
        early_website_prediction_mapping = make_json_serializable(early_website_prediction_mapping)
        save_file(
            content=early_website_prediction_mapping,
            dir_path=save_path_early_websites,
            file_name=save_file_name,
        )
        early_website_prediction_mapping = fix_dictionary_keys(early_website_prediction_mapping)

    # ============================================================================
    # PREPARE GROUND TRUTH DATA
    # ============================================================================

    unique_classes = set([label for label in ordered_labels if label is not None])
    num_classes = len(unique_classes)

    # Store ground truth for evaluation
    true_websites = {trace_idx: ordered_websites[trace_idx] for trace_idx in desired_trace_indices}
    true_second_tier_clusters = {trace_idx: ordered_labels[trace_idx] for trace_idx in desired_trace_indices}

    # ============================================================================
    # EVALUATE HOLMES WEBSITE PREDICTION ACCURACY
    # ============================================================================

    accuracy_results = calculate_accuracy_by_percentage(
        true_websites,
        early_website_prediction_mapping,
        desired_trace_indices,
        time_steps
    )

    print("\n=== Holmes Website Prediction Accuracy ===")
    for time_step, accuracy in sorted(accuracy_results.items()):
        print(f"Time step: {time_step:.2f}, Accuracy: {accuracy:.4f}")

    # ============================================================================
    # LOAD OR COMPUTE k-FP FEATURES
    # ============================================================================

    kfp_model_load_path = os.path.join(
        cm.BASE_DIR, 'data', cm.data_set_folder, 'kfp_predictions',
        f'{algorithm_tier1}-{args.max_clusters}', 'kfp_timesteps'
    )

    save_path_features = kfp_model_load_path
    features_file_name = 'kfp_features.json'

    try:
        # Try to load pre-computed features
        kfp_features_of_traces = load_file(
            dir_path=save_path_features,
            file_name=features_file_name,
        )
        kfp_features_of_traces = {float(k): v for k, v in kfp_features_of_traces.items()}
        print('k-FP features loaded from cache')
        
    except:
        # Compute k-FP features for each time step
        kfp_features_of_traces = {}  # time_step -> feature_set
        
        if args.max_clusters > 1:
            for timestep_idx, time_step in enumerate(tqdm(time_steps, desc='Extracting k-FP features')):
                is_last_time_step = (timestep_idx == len(time_steps) - 1)
                
                # Extract traces up to current time step
                if is_last_time_step:
                    # Use full traces for final time step
                    directions_test = [original_dataset_cell.directions[i] for i in desired_trace_indices]
                    times_test = [original_dataset_cell.times[i] for i in desired_trace_indices]
                else:
                    # Truncate traces at current time step
                    times = original_dataset_cell.times
                    directions = original_dataset_cell.directions
                    
                    directions_test = [
                        directions[i][:next((j for j, t in enumerate(times[i]) if t > time_step), len(times[i]))]
                        for i in desired_trace_indices
                    ]
                    times_test = [
                        times[i][:next((j for j, t in enumerate(times[i]) if t > time_step), len(times[i]))]
                        for i in desired_trace_indices
                    ]
                
                labels_test = [original_dataset_cell.labels[i] for i in desired_trace_indices]
                
                # Extract k-FP features
                kfp_features_of_traces[time_step] = get_kfp_feature_set(
                    directions=directions_test,
                    times=times_test,
                    y=labels_test,
                    verbose=False
                )
            
            # Save computed features
            save_file(
                dir_path=save_path_features,
                file_name=features_file_name,
                content=kfp_features_of_traces,
            )

    # ============================================================================
    # LOAD OR COMPUTE FIRST-TIER PREDICTIONS
    # ============================================================================

    predicted_first_tier_clusters = {}  # trace_idx -> time_step -> predicted_first_tier
    found_predicted_first_tiers = False

    save_path_first_tier_predictions = os.path.join(
        cm.BASE_DIR, 'data', cm.data_set_folder, 'kfp_predictions',
        f'{algorithm_tier1}-{args.max_clusters}'
    )
    prediction_file_name = 'predictions_first_tier.json'

    try:
        # Try to load pre-computed first-tier predictions
        predicted_first_tier_clusters = load_file(
            dir_path=save_path_first_tier_predictions,
            file_name=prediction_file_name,
        )
        predicted_first_tier_clusters = fix_dictionary_keys(predicted_first_tier_clusters)
        found_predicted_first_tiers = True
        print('First-tier predictions loaded from cache')
        
    except Exception as e:
        print('First-tier cluster predictions not found, will compute them')
        print(f"Error: {e}")
        traceback.print_exc()

    # ============================================================================
    # PERFORM SECOND-TIER CLUSTER PREDICTION
    # ============================================================================

    predicted_second_tier_cluster = {}  # trace_idx -> time_step -> predicted_cluster
    accuracy_per_timestep = {}  # time_step -> accuracy
    accuracy_per_cluster_per_timestep = {}  # cluster -> time_step -> accuracy

    if not args.preload_results:
        for times_step_idx, time_step in enumerate(time_steps):
            print(f'\n=== Processing time step: {time_step} ({times_step_idx + 1}/{len(time_steps)}) ===')
            
            # Load k-FP models for all websites if computing first-tier predictions
            if args.max_clusters != 1 and not found_predicted_first_tiers:
                kfp_models_websites = {}
                for website_idx in tqdm(range(cm.MON_SITE_NUM), desc='Loading k-FP models'):
                    kfp_model_website = load_file(
                        dir_path=kfp_model_load_path,
                        file_name=f'website_{website_idx}_thresh_{time_step:.2f}.joblib',
                        verbose=False
                    )
                    kfp_models_websites[website_idx] = kfp_model_website
            
            # Predict second-tier clusters for each trace
            for original_idx, trace_idx in enumerate(tqdm(desired_trace_indices, desc='Predicting clusters')):
                if trace_idx not in predicted_second_tier_cluster:
                    predicted_second_tier_cluster[trace_idx] = {}
                
                # Get Holmes website prediction
                website_prediction = early_website_prediction_mapping[trace_idx][time_step]
                
                if website_prediction is None:
                    # No valid Holmes prediction available
                    predicted_second_tier_cluster[trace_idx][time_step] = None
                else:
                    # Get or compute first-tier prediction
                    if args.max_clusters == 1:
                        # Website-level only (no first-tier clustering)
                        first_tier_prediction = 0
                    elif not found_predicted_first_tiers:
                        # Compute first-tier prediction using k-FP
                        kfp_model = kfp_models_websites[website_prediction]
                        first_tier_prediction = get_kfp_prediction(
                            features=[kfp_features_of_traces[time_step][original_idx]],
                            kfp_model=kfp_model
                        )[0]
                        
                        # Cache first-tier prediction
                        if trace_idx not in predicted_first_tier_clusters:
                            predicted_first_tier_clusters[trace_idx] = {}
                        predicted_first_tier_clusters[trace_idx][time_step] = first_tier_prediction
                    else:
                        # Use cached first-tier prediction
                        first_tier_prediction = predicted_first_tier_clusters[trace_idx][time_step]
                    
                    # Map to second-tier cluster
                    predicted_second_cluster = website_ft_to_st[website_prediction][first_tier_prediction]
                    predicted_second_tier_cluster[trace_idx][time_step] = predicted_second_cluster
            
            # Calculate overall accuracy for this time step
            accuracy = calculate_accuracy_for_percentage(
                true_second_tier_clusters,
                predicted_second_tier_cluster,
                desired_trace_indices,
                time_step
            )
            accuracy_per_timestep[time_step] = accuracy
            
            # Calculate per-cluster accuracy
            for second_tier_cluster in range(num_classes):
                indices_in_this_cluster = [
                    desired_trace_indices[i]
                    for i in range(len(desired_trace_indices))
                    if ordered_labels[desired_trace_indices[i]] == second_tier_cluster
                ]
                
                accuracy_of_cluster = calculate_accuracy_for_percentage(
                    true_second_tier_clusters,
                    predicted_second_tier_cluster,
                    indices_in_this_cluster,
                    time_step
                )
                
                if second_tier_cluster not in accuracy_per_cluster_per_timestep:
                    accuracy_per_cluster_per_timestep[second_tier_cluster] = {}
                accuracy_per_cluster_per_timestep[second_tier_cluster][time_step] = accuracy_of_cluster
            
            # Clean up memory
            if args.max_clusters != 1 and not found_predicted_first_tiers:
                del kfp_models_websites
                gc.collect()
        
        # ============================================================================
        # SAVE RESULTS
        # ============================================================================
        
        # Save first-tier predictions if newly computed
        if not found_predicted_first_tiers:
            predicted_first_tier_clusters = make_json_serializable(predicted_first_tier_clusters)
            save_file(
                dir_path=save_path_first_tier_predictions,
                file_name=prediction_file_name,
                content=predicted_first_tier_clusters,
            )
        
        # Determine save path based on configuration
        initial_save_path = os.path.join(
            cm.BASE_DIR, 'data', cm.data_set_folder, 'clustering', 'two-tier',
            f'{algorithm_tier1}-{algorithm_tier2}',
            f'{algorithm_tier1}-{args.max_clusters}',
            f'{algorithm_tier2}-{args.k}'
        )
        
        if args.l_tamaraw is not None:
            save_path_predictions = os.path.join(initial_save_path, f'fixed_L_{args.l_tamaraw}', 'holmes_kfp_ECDIRE')
        elif args.div_threshold is not None and args.diversity_penalty is not None:
            save_path_predictions = os.path.join(
                initial_save_path,
                f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}',
                'holmes_kfp_ECDIRE'
            )
        else:
            save_path_predictions = os.path.join(initial_save_path, 'holmes_kfp_ECDIRE')
        
        # Define file names
        prediction_accuracies_filename = f'prediction_accuracies_holmes_min_{minimum_holmes_percentage}.json'
        test_predictions_filename = f'test_predictions_holmes_min_{minimum_holmes_percentage}.json'
        prediction_accuracies_per_cluster_file_name = f'prediction_accuracies_per_cluster_holmes_min_{minimum_holmes_percentage}.json'
        
        # Save all results
        save_file(
            dir_path=save_path_predictions,
            file_name=prediction_accuracies_filename,
            content=accuracy_per_timestep,
        )
        
        try:
            predicted_second_tier_cluster = make_json_serializable(predicted_second_tier_cluster)
            save_file(
                dir_path=save_path_predictions,
                file_name=test_predictions_filename,
                content=predicted_second_tier_cluster,
            )
            
            accuracy_per_cluster_per_timestep = make_json_serializable(accuracy_per_cluster_per_timestep)
            save_file(
                dir_path=save_path_predictions,
                file_name=prediction_accuracies_per_cluster_file_name,
                content=accuracy_per_cluster_per_timestep,
            )
            
            print('\n=== Results saved successfully ===')
            
        except:
            print('Error saving results:')
            traceback.print_exc()

        
        
        
        


        
# python3 -m experiments.fixed_defenses.Tamaraw.tamaraw_two_tier_holmes_kfp_ECDIRE -k 5 -alg2 palette_tamaraw -l_tamaraw 500 

