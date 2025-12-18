#assuming a saved dictionary of accuracies per time step, try to perfrom ecdire and find the switching time stamps for each cluster

import utils.config_utils as cm

import argparse
import numpy as np

from utils.trace_dataset import TraceDataset
from utils.parser_utils import str2bool
from tqdm import tqdm
import os
from utils.file_operations import load_file, save_file



import pandas as pd

import traceback
# Plotting Overheads with Matplotlib

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# matplotlib.use('Agg')
from ..clustering.clustering_utils import load_two_tier_clusters
import seaborn as sns
from scipy import stats
import pandas as pd

def find_max_accuracy_and_threshold_timestep(accuracy_per_cluster_per_timestep, perc_acc, time_steps):
    """
    Find the maximum accuracy and the first timestep where accuracy exceeds a percentage of the maximum for each cluster.
    
    Parameters:
    - accuracy_per_cluster_per_timestep: Dictionary mapping cluster indices to a nested dictionary
                                         mapping timesteps to accuracy values.
    - perc_acc: Float between 0 and 1, representing the percentage of maximum accuracy to use as threshold.
    - time_steps: List of timestep values in ascending order.
    
    Returns:
    - results: Dictionary mapping each cluster to its max accuracy and threshold timestep.
    """
    # Initialize an empty dictionary to store results
    results = {}
    
    # Iterate through each cluster and its corresponding timestep-to-accuracy mapping
    for cluster_str, timestep_to_accuracy in accuracy_per_cluster_per_timestep.items():
        # Convert cluster key to integer if possible
        try:
            cluster = int(cluster_str)
        except ValueError:
            # If conversion fails, keep as string
            cluster = cluster_str
            
        # Create a dictionary with float timestep keys
        float_timestep_to_accuracy = {}
        for ts_str, acc in timestep_to_accuracy.items():
            try:
                ts = float(ts_str)
                float_timestep_to_accuracy[ts] = acc
            except ValueError:
                # Skip if timestep can't be converted to float
                continue
        
        # Skip this cluster if no valid timesteps
        if not float_timestep_to_accuracy:
            continue
            
        # Find maximum accuracy value for this cluster across all timesteps
        max_accuracy = max(float_timestep_to_accuracy.values())
        
        # Calculate the threshold accuracy (percentage of maximum)
        threshold = perc_acc * max_accuracy
        
        # Initialize variable to store the first timestep that exceeds the threshold
        first_threshold_timestep = None
        chosen_accuracy = 0
        # Iterate through timesteps in order
        for timestep in time_steps:
            # Check if this timestep exists in the data and its accuracy exceeds the threshold
            if timestep in float_timestep_to_accuracy and float_timestep_to_accuracy[timestep] >= threshold:
                # Save this timestep and exit the loop
                chosen_accuracy = float_timestep_to_accuracy[timestep]
                first_threshold_timestep = timestep
                break
        
        # Store both the max accuracy and the threshold timestep for this cluster
        results[cluster] = {
            'max_accuracy': max_accuracy,
            'threshold_timestep': first_threshold_timestep,
            'chosen accuracy': chosen_accuracy
        }
    
    return results

def visualize_cluster_accuracy_results(results, accuracy_per_cluster_per_timestep, time_steps, save_dir = None):
    """
    Create comprehensive visualizations for cluster accuracy analysis.
    
    Parameters:
    - results: Dictionary from find_max_accuracy_and_threshold_timestep function
               with cluster -> {'max_accuracy', 'threshold_timestep'}
    - accuracy_per_cluster_per_timestep: Original data dictionary with 
                                         cluster -> timestep -> accuracy
    - time_steps: List of all possible timesteps
    """
    # Extract data for plotting
    clusters = list(results.keys())
    max_accuracies = [results[c]['max_accuracy'] for c in clusters]
    threshold_timesteps = [results[c]['threshold_timestep'] for c in clusters]
    
    # Calculate threshold accuracies - handle string keys properly
    threshold_accuracies = []
    for c in clusters:
        if results[c]['threshold_timestep'] is not None:
            # Convert cluster key to string since original dict has string keys
            c_str = str(c)
            ts_str = str(results[c]['threshold_timestep'])
            
            # Check if the cluster and timestep exist in the original data
            if c_str in accuracy_per_cluster_per_timestep and ts_str in accuracy_per_cluster_per_timestep[c_str]:
                threshold_accuracies.append(accuracy_per_cluster_per_timestep[c_str][ts_str])
    
    # Filter out None values for timesteps (if any clusters didn't meet threshold)
    valid_indices = [i for i, t in enumerate(threshold_timesteps) if t is not None]
    valid_clusters = [clusters[i] for i in valid_indices]
    valid_timesteps = [threshold_timesteps[i] for i in valid_indices]
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 15))
    
    # 1. Maximum accuracies across all clusters
    plt.subplot(2, 2, 1)
    plt.bar(range(len(clusters)), max_accuracies, alpha=0.7)
    plt.xlabel('Cluster Index')
    plt.ylabel('Maximum Accuracy')
    plt.title('Maximum Accuracy per Cluster')
    plt.xticks(range(0, len(clusters), max(1, len(clusters)//10)))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Threshold timesteps across all clusters
    plt.subplot(2, 2, 2)
    plt.bar(range(len(valid_clusters)), valid_timesteps, alpha=0.7)
    plt.xlabel('Cluster Index')
    plt.ylabel('Threshold Timestep')
    plt.title('First Timestep Exceeding Threshold per Cluster')
    plt.xticks(range(0, len(valid_clusters), max(1, len(valid_clusters)//10)))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. CDF of threshold timesteps
    plt.subplot(2, 2, 3)
    # Calculate CDF
    sorted_timesteps = np.sort(valid_timesteps)
    cumulative_prob = np.arange(1, len(sorted_timesteps) + 1) / len(sorted_timesteps)
    
    plt.step(sorted_timesteps, cumulative_prob, where='post', lw=2)
    plt.xlabel('Threshold Timestep')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Threshold Timesteps')
    plt.grid(linestyle='--', alpha=0.7)
    
    # 4. PDF (histogram) of threshold timesteps
    plt.subplot(2, 2, 4)
    sns.histplot(valid_timesteps, kde=True, bins=min(20, len(set(valid_timesteps))))
    plt.xlabel('Threshold Timestep')
    plt.ylabel('Frequency')
    plt.title('Distribution of Threshold Timesteps')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_file(dir_path= save_dir, file_name= 'cluster_accuracy_overview.png')
    # plt.savefig('cluster_accuracy_overview.png', dpi=300)
    # plt.close()
    
    # Additional visualization: Scatter plot comparing max acc vs threshold acc
    if threshold_accuracies:  # Only create this plot if we have threshold accuracies
        plt.figure(figsize=(10, 8))
        plt.scatter(max_accuracies[:len(threshold_accuracies)], threshold_accuracies, alpha=0.7)
        plt.xlabel('Maximum Accuracy')
        plt.ylabel('Threshold Accuracy')
        plt.title('Maximum Accuracy vs. Threshold Accuracy')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Add a diagonal line representing y = x
        max_val = max(max(max_accuracies), max(threshold_accuracies))
        min_val = min(min(max_accuracies), min(threshold_accuracies))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        plt.tight_layout()
        save_file(dir_path= save_dir, file_name= 'max_vs_threshold_accuracy.png')
        # plt.savefig('max_vs_threshold_accuracy.png', dpi=300)
        # plt.close()
    
    # Heatmap of accuracy progression for a subset of clusters
    visualize_accuracy_progression(accuracy_per_cluster_per_timestep, time_steps, save_dir= save_dir)
    
    # Print statistics
    print_statistics(max_accuracies, valid_timesteps, threshold_accuracies)

def visualize_accuracy_progression(accuracy_per_cluster_per_timestep, time_steps, num_clusters=10, save_dir = None):
    """
    Create a heatmap showing accuracy progression over time for a subset of clusters.
    
    Parameters:
    - accuracy_per_cluster_per_timestep: Dictionary mapping cluster to timestep to accuracy
    - time_steps: List of all possible timesteps
    - num_clusters: Number of clusters to visualize (default: 10)
    """
    # Select a subset of clusters
    clusters = list(accuracy_per_cluster_per_timestep.keys())
    selected_clusters = clusters[:min(num_clusters, len(clusters))]
    
    # Create a DataFrame for the heatmap
    data = []
    for cluster_str in selected_clusters:
        cluster_data = []
        for ts in time_steps:
            ts_str = str(ts)
            if ts_str in accuracy_per_cluster_per_timestep[cluster_str]:
                cluster_data.append(accuracy_per_cluster_per_timestep[cluster_str][ts_str])
            else:
                cluster_data.append(np.nan)
        data.append(cluster_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data, index=selected_clusters, columns=time_steps)
    
    # Create heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(df, cmap='viridis', xticklabels=10, yticklabels=True)
    plt.xlabel('Timestep')
    plt.ylabel('Cluster')
    plt.title('Accuracy Progression Over Time for Selected Clusters')
    plt.tight_layout()
    save_file(dir_path= save_dir, file_name= 'accuracy_progression_heatmap.png')
    # plt.savefig('accuracy_progression_heatmap.png', dpi=300)
    # plt.close()

def print_statistics(max_accuracies, threshold_timesteps, threshold_accuracies):
    """
    Print descriptive statistics about the results.
    
    Parameters:
    - max_accuracies: List of maximum accuracies for each cluster
    - threshold_timesteps: List of threshold timesteps for each cluster
    - threshold_accuracies: List of accuracies at the threshold timesteps
    """
    # Statistics for maximum accuracies
    max_acc_mean = np.mean(max_accuracies)
    max_acc_std = np.std(max_accuracies)
    max_acc_min = np.min(max_accuracies)
    max_acc_max = np.max(max_accuracies)
    
    # Statistics for threshold timesteps
    ts_mean = np.mean(threshold_timesteps) if threshold_timesteps else np.nan
    ts_std = np.std(threshold_timesteps) if threshold_timesteps else np.nan
    ts_min = np.min(threshold_timesteps) if threshold_timesteps else np.nan
    ts_max = np.max(threshold_timesteps) if threshold_timesteps else np.nan
    
    # Statistics for threshold accuracies
    thresh_acc_mean = np.mean(threshold_accuracies) if threshold_accuracies else np.nan
    thresh_acc_std = np.std(threshold_accuracies) if threshold_accuracies else np.nan
    
    # Print statistics
    print("\n===== STATISTICS =====")
    print(f"Number of clusters: {len(max_accuracies)}")
    print(f"Number of clusters with valid threshold: {len(threshold_timesteps)}")
    
    print("\nMaximum Accuracies:")
    print(f"  Mean: {max_acc_mean:.4f}")
    print(f"  Std Dev: {max_acc_std:.4f}")
    print(f"  Min: {max_acc_min:.4f}")
    print(f"  Max: {max_acc_max:.4f}")
    
    print("\nThreshold Timesteps:")
    print(f"  Mean: {ts_mean:.2f}")
    print(f"  Std Dev: {ts_std:.2f}")
    print(f"  Min: {ts_min:.2f}")
    print(f"  Max: {ts_max:.2f}")
    
    print("\nAccuracies at Threshold:")
    print(f"  Mean: {thresh_acc_mean:.4f}")
    print(f"  Std Dev: {thresh_acc_std:.4f}")
    
    # Only calculate ratio if we have threshold accuracies
    if threshold_accuracies:
        print(f"  Mean ratio to max accuracy: {thresh_acc_mean/np.mean(max_accuracies[:len(threshold_accuracies)]):.4f}")
    
    # Create a DataFrame for more detailed statistics
    # Ensure all lists are the same length for DataFrame creation
    max_length = len(max_accuracies)
    thresh_ts_padded = threshold_timesteps + [np.nan] * (max_length - len(threshold_timesteps))
    thresh_acc_padded = threshold_accuracies + [np.nan] * (max_length - len(threshold_accuracies))
    
    stats_df = pd.DataFrame({
        'Max Accuracy': max_accuracies,
        'Threshold Timestep': thresh_ts_padded,
        'Threshold Accuracy': thresh_acc_padded
    })
    
    # Calculate additional statistics
    percentiles = [10, 25, 50, 75, 90]
    print("\nPercentiles:")
    for col in stats_df.columns:
        print(f"\n{col}:")
        for p in percentiles:
            val = stats_df[col].quantile(p/100)
            print(f"  {p}th percentile: {val:.4f}")

def fix_dictionary_keys(accuracy_per_cluster_per_timestep):
    """
    Convert string keys in the accuracy_per_cluster_per_timestep dictionary to their proper numeric types.
    Cluster keys are converted to integers, and timestep keys are converted to floats.
    
    Parameters:
    - accuracy_per_cluster_per_timestep: Dictionary with string keys for both clusters and timesteps
    
    Returns:
    - fixed_dict: Dictionary with proper numeric keys
    """
    fixed_dict = {}
    
    # Process each cluster
    for cluster_str, timestep_dict in accuracy_per_cluster_per_timestep.items():
        # Convert cluster key to integer
        try:
            cluster_key = int(cluster_str)
        except ValueError:
            # If conversion to int fails, keep as string
            cluster_key = cluster_str
        
        # Create a new dictionary for this cluster with float timestep keys
        fixed_timestep_dict = {}
        for timestep_str, accuracy in timestep_dict.items():
            # Convert timestep key to float
            try:
                timestep_key = float(timestep_str)
                fixed_timestep_dict[timestep_key] = accuracy
            except ValueError:
                # Skip entries where timestep can't be converted to float
                print(f"Warning: Could not convert timestep '{timestep_str}' to float. Skipping.")
                continue
        
        # Add the fixed timestep dictionary to the fixed result
        fixed_dict[cluster_key] = fixed_timestep_dict
    
    return fixed_dict

# Example usage:
# Assuming you've loaded your data into accuracy_per_cluster_per_timestep
# fixed_data = fix_dictionary_keys(accuracy_per_cluster_per_timestep)


from matplotlib.ticker import MaxNLocator

def compare_threshold_and_avg_times(results, avg_time_in_each_cluster, sort_by='cluster', save_dir = None):
    """
    Create a comparison plot between threshold timesteps and average time in each cluster.
    
    Parameters:
    - results: Dictionary from find_max_accuracy_and_threshold_timestep function
               with cluster -> {'max_accuracy', 'threshold_timestep'}
    - avg_time_in_each_cluster: Dictionary mapping cluster index to average time in that cluster
    - sort_by: How to sort the data. Options:
               'cluster': Sort by cluster index (default)
               'threshold': Sort by threshold timestep values
               'avg_time': Sort by average time values
    
    Returns:
    - None (saves plot to file)
    """
    # Extract data and make sure keys are same type (int)
    data = []
    for cluster in results:
        if cluster in avg_time_in_each_cluster and results[cluster]['threshold_timestep'] is not None:
            data.append({
                'cluster': cluster,
                'threshold_timestep': results[cluster]['threshold_timestep'],
                'avg_time': avg_time_in_each_cluster[cluster]
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort based on preference
    if sort_by == 'threshold':
        df = df.sort_values('threshold_timestep')
    elif sort_by == 'avg_time':
        df = df.sort_values('avg_time')
    else:  # Default: sort by cluster
        df = df.sort_values('cluster')
    
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot threshold timesteps
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Threshold Timestep', color='tab:blue')
    line1 = ax1.plot(range(len(df)), df['threshold_timestep'], 'o-', color='tab:blue', 
                     label='Threshold Timestep', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Create second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Time in Cluster', color='tab:red')
    line2 = ax2.plot(range(len(df)), df['avg_time'], 's-', color='tab:red', 
                     label='Avg Time in Cluster', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Add cluster indices as x-tick labels
    plt.xticks(range(len(df)), df['cluster'], rotation=90 if len(df) > 20 else 0)
    if len(df) > 30:
        # If too many clusters, only show a subset of x-ticks
        ax1.xaxis.set_major_locator(MaxNLocator(20))
    
    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    # Add grid for readability
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient to title
    correlation = df['threshold_timestep'].corr(df['avg_time'])
    plt.title(f'Comparison of Threshold Timesteps vs. Average Time in Clusters\n'
              f'Correlation: {correlation:.3f}')
    
    # Add annotations for interesting points
    # 1. Point with max difference
    df['diff'] = abs(df['threshold_timestep'] - df['avg_time'])
    max_diff_idx = df['diff'].idxmax()
    max_diff_cluster = df.loc[max_diff_idx, 'cluster']
    max_diff_position = list(df['cluster']).index(max_diff_cluster)
    
    plt.annotate(f'Max Difference: Cluster {max_diff_cluster}',
                xy=(max_diff_position, df.loc[max_diff_idx, 'threshold_timestep']),
                xytext=(max_diff_position, df.loc[max_diff_idx, 'threshold_timestep'] * 1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                ha='center', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_file(dir_path= save_dir ,file_name= 'threshold_vs_avg_time_comparison.png')
    # plt.savefig('threshold_vs_avg_time_comparison.png', dpi=300)
    
    # Create a scatter plot to further visualize the relationship
    plt.figure(figsize=(10, 8))
    plt.scatter(df['threshold_timestep'], df['avg_time'], alpha=0.7)
    plt.xlabel('Threshold Timestep')
    plt.ylabel('Average Time in Cluster')
    plt.title(f'Scatter Plot: Threshold Timestep vs. Average Time\nCorrelation: {correlation:.3f}')
    
    # Add trend line
    z = np.polyfit(df['threshold_timestep'], df['avg_time'], 1)
    p = np.poly1d(z)
    plt.plot(df['threshold_timestep'], p(df['threshold_timestep']), "r--", alpha=0.7)
    
    # Add diagonal line (y=x) for reference
    min_val = min(df['threshold_timestep'].min(), df['avg_time'].min())
    max_val = max(df['threshold_timestep'].max(), df['avg_time'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Annotate some interesting points
    # 1. Clusters with largest differences
    top_diff_indices = df.nlargest(3, 'diff').index
    for idx in top_diff_indices:
        cluster = df.loc[idx, 'cluster']
        plt.annotate(f'Cluster {cluster}',
                    xy=(df.loc[idx, 'threshold_timestep'], df.loc[idx, 'avg_time']),
                    xytext=(df.loc[idx, 'threshold_timestep'] * 1.1, df.loc[idx, 'avg_time'] * 1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                    fontsize=9)
    
    # Save scatter plot
    plt.tight_layout()
    save_file(dir_path= save_dir ,file_name= 'threshold_vs_avg_time_scatter.png')
   
    
    # Print some statistics
    print("\n===== THRESHOLD VS AVERAGE TIME STATISTICS =====")
    print(f"Number of clusters analyzed: {len(df)}")
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"Mean threshold timestep: {df['threshold_timestep'].mean():.2f}")
    print(f"Mean average time in cluster: {df['avg_time'].mean():.2f}")
    print(f"Mean absolute difference: {df['diff'].mean():.2f}")
    print(f"Maximum absolute difference: {df['diff'].max():.2f} (Cluster {max_diff_cluster})")
    
    # Calculate how many times threshold is higher than avg and vice versa
    threshold_higher = sum(df['threshold_timestep'] > df['avg_time'])
    avg_higher = sum(df['avg_time'] > df['threshold_timestep'])
    print(f"Number of clusters where threshold timestep > average time: {threshold_higher} ({threshold_higher/len(df)*100:.1f}%)")
    print(f"Number of clusters where average time > threshold timestep: {avg_higher} ({avg_higher/len(df)*100:.1f}%)")
    
    return df  # Return the DataFrame for further analysis if needed


    

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
                        default= 'palette_tamaraw') # TODO implement others
    
    parser.add_argument('-k', type = int , default= 5 , help = 'Minimum number of elements in each second tier cluster')
    parser.add_argument('-max_clusters', help='maximum number of clusters for each website when performing first tier cluster', default = 5, type =int)
    
    parser.add_argument('-preload_clusters', type=str2bool, nargs='?', const=True, default=True,
                         help='Whether to load the pre-computed first tier clusters', )
    
    parser.add_argument("-perc_acc", type=float, default=0.5, help="full accuracy will be multiplied by this")

    

    parser.add_argument('-div_threshold', type = float,
                         help='the diversity threshold we use in two tier', default = None)
    
    parser.add_argument('-diversity_penalty', type = float,
                         help='the diversity penalty we use in two tier', default = None)
    
    parser.add_argument('-closed_world_assumption', type=str2bool, nargs='?', const=True, default=False,
            help='this will assume we know the website beforehand')
    
    
    parser.add_argument('-l_tamaraw',  type = int , default= None , help = 'fixed L for tamaraw')

   
    parser.add_argument('-holmes_min_percentage', type = float,
                         help='minimum percentage of trace percentage we want holmes to have before prediciting', default = 0.2)

    
    
    
    args = parser.parse_args()
    algorithm_tier1 = args.algorithm_tier1
    algorithm_tier2 = args.algorithm_tier2
    
    
    cm.initialize_common_params(args.config)
    

   

    # --- Configuration and Initial Data Loading ---

    # Initialize the dataset handler for 'cell' mode traces.
    # 'TraceDataset' is likely a custom class for loading and managing trace data.
    original_dataset_cell = TraceDataset(extract_traces=args.extract_ds, trace_mode='cell')

    # Dynamic path construction for second-tier clustering results based on diversity arguments.
    second_tier_file_name = None
    second_tier_cluster_path = None

    if args.div_threshold is not None and args.diversity_penalty is not None:
        # If diversity arguments are provided, construct a specific file and directory path for the second-tier clusters.
        second_tier_file_name = f'second_tier_clusters_peanlty_{args.diversity_penalty:.2f}_div_{args.div_threshold}.pkl'
        second_tier_cluster_path = os.path.join(
            cm.BASE_DIR, 'data', cm.data_set_folder, 'clustering', 'two-tier',
            f'{algorithm_tier1}-{algorithm_tier2}',
            f'{algorithm_tier1}-{args.max_clusters}', f'{algorithm_tier2}-{args.k}',
            f'div_{args.div_threshold}'
        )

    # Load the results of the two-tier clustering.
    # 'load_two_tier_clusters' is a function that fetches or computes the cluster assignments and trace data.
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
        l_tamaraw=args.l_tamaraw,
    )

    # Unpack the loaded clustering results.
    tier1_clusters_of_each_website = loading_results['tier1_clusters_of_each_website']
    tier2_clusters = loading_results['tier2_clusters']
    super_matrix_mapping = loading_results['super_matrix_mapping']
    ordered_traces = loading_results['ordered_traces']
    ordered_labels = loading_results['ordered_labels'] # Tier 2 cluster labels for each trace
    ordered_websites = loading_results['ordered_websites']
    overall_mapping = loading_results['overall_mapping']
    ordered_original_traces = loading_results['ordered_original_traces']
    website_ft_to_st = loading_results['reverse_mapping']
    ordered_tier1_labels = loading_results['ordered_tier1_labels']

    # Determine the number of unique second-tier clusters (classes).
    unique_classes = set([label for label in ordered_labels if label is not None])
    num_classes = len(unique_classes)

    # --- Average Trace Time Calculation ---

    # Dictionary to store the average total time for traces in each cluster.
    avg_time_in_each_cluster = {}

    # Define the path to load the predefined test indices.
    load_path__test_indices = os.path.join(
        cm.BASE_DIR, 'data', 'holmes', cm.data_set_folder
    )

    # Load the indices of the traces designated for testing.
    desired_trace_indices = load_file(
        dir_path=load_path__test_indices, file_name='test_indices.npy'
    ).tolist()

    # Calculate the average total time for all test traces belonging to each cluster.
    for cluster_idx in range(num_classes):
        # Filter test indices that belong to the current cluster.
        indices_in_this_cluster = [
            i for i in desired_trace_indices if ordered_labels[i] == cluster_idx
        ]
        
        total_time = 0
        number_of_elements = 0
        
        # Sum up the total time (last element of the times array) for each trace in the cluster.
        for indice in indices_in_this_cluster:
            # original_dataset_cell.times[indice][-1] is likely the total duration of the trace.
            total_time += original_dataset_cell.times[indice][-1]
            number_of_elements += 1
        
        # Calculate and store the average time.
        if number_of_elements > 0:
            avg_time_in_each_cluster[cluster_idx] = total_time / number_of_elements
        else:
            avg_time_in_each_cluster[cluster_idx] = 0 # Handle case with no elements

    # --- Load Prediction Accuracy Results ---

    # Base path for saving or loading intermediate clustering results.
    initial_save_path = os.path.join(
        cm.BASE_DIR, 'data', cm.data_set_folder, 'clustering', 'two-tier',
        f'{algorithm_tier1}-{algorithm_tier2}',
        f'{algorithm_tier1}-{args.max_clusters}', f'{algorithm_tier2}-{args.k}'
    )

    # Construct the full path to the directory containing Holmes prediction results,
    # considering 'l_tamaraw', 'diversity', and default cases.
    if args.l_tamaraw is not None:
        save_path_dictionary = os.path.join(
            initial_save_path, f'fixed_L_{args.l_tamaraw}', 'holmes_kfp_ECDIRE'
        )
    elif args.div_threshold is not None and args.diversity_penalty is not None:
        save_path_dictionary = os.path.join(
            initial_save_path,
            f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}',
            'holmes_kfp_ECDIRE'
        )
    else:
        save_path_dictionary = os.path.join(initial_save_path, 'holmes_kfp_ECDIRE')

    # Adjust path for Closed-World Assumption if applicable.
    if args.closed_world_assumption:
        save_path_dictionary = os.path.join(save_path_dictionary, 'CW_assumption')

    # Determine the minimum Holmes percentage threshold for the filename.
    minimum_holmes_percentage = int(args.holmes_min_percentage * 100)

    # Load the pre-computed prediction accuracies per cluster for different timesteps.
    accuracy_per_cluster_per_timestep = load_file(
        dir_path=save_path_dictionary,
        file_name=f'prediction_accuracies_per_cluster_holmes_min_{minimum_holmes_percentage}.json',
        
    )

    # Fix keys (e.g., converting string keys to integers) if necessary.
    # This dictionary maps cluster index -> {timestep -> accuracy}.
    accuracy_per_cluster_per_timestep = fix_dictionary_keys(accuracy_per_cluster_per_timestep)

    # Define the set of timesteps (e.g., in seconds) used for accuracy calculation.
    # These timesteps represent points in the trace where prediction accuracy was measured.
    time_steps = [0.8 * i for i in range(1, 101)] # 0.8, 1.6, ..., 80.0

    # --- Analyze and Save Safe Times (Threshold Timesteps) ---

    perc_acc = args.perc_acc # Target percentage accuracy for determining the "safe time".

    # Find the maximum accuracy and the 'threshold timestep' (the earliest time a target accuracy is reached).
    # The result maps cluster_num -> {'max_accuracy': ..., 'threshold_timestep': ..., 'chosen accuracy': ...}
    results = find_max_accuracy_and_threshold_timestep(
        accuracy_per_cluster_per_timestep, perc_acc, time_steps
    )

    # Base path for saving final *analysis* results.
    initial_save_path = os.path.join(
        cm.BASE_DIR, 'results', 'clustering', 'two-tier', f'{cm.data_set_folder}',
        f'{algorithm_tier1}_{algorithm_tier2}', f'{algorithm_tier1}-{args.max_clusters}',
        f'k = {args.k}'
    )

    # Construct the final save path for results, mirroring the logic used for loading.
    if args.l_tamaraw is not None:
        save_path = os.path.join(
            initial_save_path, f'fixed_L_{args.l_tamaraw}', 'holmes_kfp_ECDIRE', f'perc_acc_{perc_acc}'
        )
    elif args.div_threshold is not None and args.diversity_penalty is not None:
        save_path = os.path.join(
            initial_save_path,
            f'penalty_{args.diversity_penalty:.2f}_div_{args.div_threshold}',
            'holmes_kfp_ECDIRE', f'perc_acc_{perc_acc}'
        )
    else:
        save_path = os.path.join(initial_save_path, 'holmes_kfp_ECDIRE', f'perc_acc_{perc_acc}')

    # Adjust path for Closed-World Assumption.
    if args.closed_world_assumption:
        save_path = os.path.join(save_path, 'CW_assumption', f'perc_acc_{perc_acc}')

    # Save the calculated 'safe times' (threshold timesteps) for each cluster.
    save_file(
        dir_path=save_path, content=results,
        file_name=f'safe_times_holmes_min_{minimum_holmes_percentage}.json'
    )

    # Visualize the accuracy curve and chosen threshold point for each cluster.
    visualize_cluster_accuracy_results(
        results, accuracy_per_cluster_per_timestep, time_steps, save_dir=save_path
    )

    # Compare the calculated threshold times with the average overall trace times per cluster.
    compare_threshold_and_avg_times(
        results, avg_time_in_each_cluster, sort_by='threshold', save_dir=save_path
    )

    # --- Visualization of Max vs. Chosen Accuracy ---


    # Extract data for plotting.
    clusters = list(results.keys())
    max_accuracies = [results[cluster]['max_accuracy'] for cluster in clusters]
    chosen_accuracies = [results[cluster]['chosen accuracy'] for cluster in clusters]

    # Create the plot figure.
    plt.figure(figsize=(10, 6))
    plt.plot(
        clusters, max_accuracies, 'o-', linewidth=2,
        label='Max Accuracy', color='blue'
    )
    plt.plot(
        clusters, chosen_accuracies, 's-', linewidth=2,
        label='Chosen Accuracy', color='orange'
    )

    # Add labels, title, and grid.
    plt.xlabel('Cluster Index')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Max Accuracy vs Chosen Accuracy by Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add value labels (annotations) to the plot points.
    for i, (max_acc, chosen_acc) in enumerate(zip(max_accuracies, chosen_accuracies)):
        plt.text(clusters[i], max_acc + 0.01, f'{max_acc:.2f}', ha='center')
        plt.text(clusters[i], chosen_acc - 0.02, f'{chosen_acc:.2f}', ha='center')

    plt.tight_layout()

    # Save the generated plot.
    save_file(
        dir_path=save_path,
        file_name=f'max_acc_vs_chosen_acc_holmes_min_{minimum_holmes_percentage}.png'
    )
    


# python3 -m experiments.early_detection.ecdire -l_tamaraw -k 5 -conf Tik_Tok -cc True -alg2 palette_tamaraw -perc_acc 0.5

