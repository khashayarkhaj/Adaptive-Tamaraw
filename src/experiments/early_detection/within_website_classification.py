# training kfp models for each website traces per each time step and saving the model


# Core imports for model training
from models.kfp.kfp_train import train_kfp

# Training utilities
from training.train_utils import stratified_split


# Clustering utilities
from ..clustering.clustering_utils import load_two_tier_clusters

# General utilities
import utils.config_utils as cm
from utils.trace_dataset import TraceDataset
from utils.file_operations import save_file
from utils.parser_utils import str2bool


# Standard libraries
import argparse
import os
import gc
from tqdm import tqdm

# Scientific computing
import numpy as np
import pandas as pd



# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns



def plot_accuracy_distribution(detection_accuracy_website, threshold=1, plot_it=True):
    """
    Visualizes the distribution of detection accuracies across websites.
    
    Args:
        detection_accuracy_website: Dictionary mapping website numbers to accuracy values
        threshold: Current time threshold being analyzed (for plot title)
        plot_it: Whether to display the plot
    
    Returns:
        fig: Matplotlib figure object with three subplots
        summary_stats: Pandas Series with statistical summary of accuracies
    """
    # Convert accuracy dictionary to DataFrame for easier manipulation
    df = pd.DataFrame.from_dict(
        detection_accuracy_website, 
        orient='index', 
        columns=['accuracy']
    )
    df.index.name = 'website_num'
    df.reset_index(inplace=True)
    
    # Create visualization with three different views
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Histogram: Shows frequency distribution of accuracy values
    sns.histplot(data=df, x='accuracy', bins=20, ax=ax1)
    ax1.set_title(f'Histogram of Detection Accuracies for th = {threshold}')
    ax1.set_xlabel('Accuracy')
    ax1.set_ylabel('Count')
    
    # Box plot: Shows quartiles, median, and outliers
    sns.boxplot(y='accuracy', data=df, ax=ax2)
    ax2.set_title('Box Plot of Detection Accuracies')
    ax2.set_ylabel('Accuracy')
    
    # Scatter plot: Shows accuracy per individual website
    sns.scatterplot(data=df, x='website_num', y='accuracy', ax=ax3)
    ax3.set_title('Accuracy by Website Number')
    ax3.set_xlabel('Website Number')
    ax3.set_ylabel('Accuracy')
    
    plt.tight_layout()
    
    if plot_it:
        plt.show()
    
    # Calculate summary statistics (mean, std, min, max, quartiles)
    summary_stats = df['accuracy'].describe()
    print("\nSummary Statistics:")
    print(summary_stats)
    
    return fig, summary_stats


if __name__ == '__main__':
    # ===========================
    # Command Line Argument Setup
    # ===========================
    parser = argparse.ArgumentParser(
        description='Train KFP on Websites'
    )
    
    # Clustering algorithm selection
    parser.add_argument(
        '-alg1', '--algorithm_tier1',
        choices=['k_medoids', 'cast'],
        default='cast',
        help='First tier clustering algorithm (intra-website clustering)'
    )
    
    parser.add_argument(
        '-alg2', '--algorithm_tier2',
        choices=['palette', 'oka', 'palette_tamaraw', 'palette_tamaraw_pareto', 'palette_tamaraw_top_ten'],
        default='palette',
        help='Second tier clustering algorithm (inter-website clustering)'
    )
    
    
    # Clustering parameters
    parser.add_argument(
        '-k', type=int, default=5,
        help='Minimum number of elements in each second tier cluster'
    )
    
    parser.add_argument(
        '-max_clusters', type=int, default=5,
        help='Maximum number of first-tier clusters per website'
    )
    
    # Configuration files
    parser.add_argument(
        '-config', default='Tik_Tok',
        help='Configuration file for dataset parameters'
    )
    
    
    # Data splitting ratios
    
    
    #optimization
    
    parser.add_argument(
        '-compress', type=int, default=None,
        help='Compression level for saved joblib files'
    )
    
    
    parser.add_argument(
        '-e', '--extract_ds', default=False,
        help='Extract dataset from raw files (False if already preprocessed)'
    )
    
    parser.add_argument(
        '-save', type=str2bool, nargs='?', const=True, default=False,
        help='Save trained models to disk'
    )
    
    parser.add_argument(
        '-preload_clusters', type=str2bool, nargs='?', const=True, default=False,
        help='Load pre-computed cluster assignments instead of re-clustering'
    )
    
    
    
    # Path and execution control
    parser.add_argument(
        '-start_timestep_idx', type=int, default=0,
        help='Starting index for time step iteration (for resuming interrupted runs)'
    )
    
    parser.add_argument(
        '-end_timestep_idx', type=int, default=None,
        help='Ending index for time step iteration (None processes all remaining steps)'
    )
    
    parser.add_argument(
        '-l_tamaraw', type=int, default=None,
        help='Fixed L parameter for Tamaraw defense simulation'
    )
    
    args = parser.parse_args()
    
    # ===========================
    # Initialize Configuration
    # ===========================
    algorithm_tier1 = args.algorithm_tier1
    algorithm_tier2 = args.algorithm_tier2
    
    
    # Load common parameters from config file
    cm.initialize_common_params(args.config)
    
    # Set up result directory structure
    
    
    
    
    logger = cm.init_logger(name='Training KFP on wbsites', )
    
    # ===========================
    # Configure Trace Processing
    # ===========================
    # Default trace processing mode (Time-Amplitude Matrix)
    trace_mode = 'tam'
    trim_traces = False
    replace_negative_ones = False
    
    # Deep learning models require cell-level traces
    
    
    # ===========================
    # Load Clustered Data (helps us get first tier cluster num of each trace)
    # ===========================
    

    loading_results = load_two_tier_clusters(
        k=args.k,
        algorithm_tier1=algorithm_tier1,
        algorithm_tier2=algorithm_tier2,
        preload_clusters=args.preload_clusters,
        max_clusters=args.max_clusters,
        extract_ds=args.extract_ds,
        trace_mode=trace_mode,
        trim_traces=trim_traces,
        replace_negative_ones=replace_negative_ones,
        l_tamaraw=args.l_tamaraw,
    )
    
    # Extract clustering results
    ordered_labels = loading_results['ordered_labels']
    ordered_websites = loading_results['ordered_websites']
    overall_mapping = loading_results['overall_mapping']  # Maps (website, trace_idx) -> (tier1_cluster, tier2_cluster)
    
    # Dataset statistics
    number_of_websites = len(set(ordered_websites))
    number_of_traces = len(ordered_labels)
    print(f'Processing {number_of_websites} websites with {number_of_traces} total traces')
    
    # ===========================
    # Load Original Datasets
    # ===========================
    # TAM (Time-Amplitude Matrix) dataset for standard feature extraction
    original_dataset = TraceDataset(extract_traces=args.extract_ds, trace_mode='tam')
    
    # Cell-level dataset for KFP (k-Fingerprinting) features
    original_dataset_cell = TraceDataset(extract_traces=args.extract_ds, trace_mode='cell')
    
    # ===========================
    # Early Detection Parameters
    # ===========================
    # Maximum page load time and TAM dimensions
    maximum_load_time = cm.Maximum_Load_Time
    tam_length = cm.Max_tam_matrix_len
    
    # Time steps for early detection analysis (0.8s increments up to 80s)
    time_steps = [0.8 * i for i in range(1, 101)]
    
    # Control which time steps to process (useful for parallel execution)
    start_time_step = args.start_timestep_idx
    end_time_step = args.end_timestep_idx if args.end_timestep_idx is not None else len(time_steps)
    
    # ===========================
    # Initialize Result Storage
    # ===========================
    # Track accuracy at each time step
    detection_accuracy = {}  # timestep -> accuracy
    detection_accuracy_website = {}  # website_num -> accuracy at current timestep
    mean_accuracies = []  # List of mean accuracies across all websites per timestep
    
    # Store predictions for each test trace at each time threshold
    # Structure: {original_trace_index: {time_threshold: prediction}}
    overall_test_predictions = {}
    
    # Cache KFP features to avoid recomputation
    kfp_features_of_traces = {}  # threshold -> feature set
    
    # ===========================
    # Main Training Loop
    # ===========================
    for timestep_idx, time_step in enumerate(time_steps):
        # Skip time steps outside the specified range
        if timestep_idx < start_time_step or timestep_idx > end_time_step:
            continue
        
        # Reset per-website accuracy tracking for this time step
        accuracies = {}
        
        # Train a separate KFP model for each website at this time threshold
        for website_num in tqdm(
            range(number_of_websites),
            desc=f'Training KFP models at t={time_step:.1f}s ({timestep_idx + 1}/{len(time_steps)})'
        ):
            # ===========================
            # Prepare Training Data
            # ===========================
            
            if args.max_clusters > 1:
                # Multi-cluster scenario: assign tier-1 cluster labels
                tams_of_this_website, _ = original_dataset.get_traces_of_class(class_number=website_num)
                
                # Initialize labels and assign based on cluster membership
                labels = [-1 for _ in range(len(tams_of_this_website))]
                for tam_idx, tam in enumerate(tams_of_this_website):
                    correct_tier1_cluster = overall_mapping[(website_num, tam_idx)][0]
                    labels[tam_idx] = correct_tier1_cluster
                
                # Get cell-level traces for KFP feature extraction
                directions, times, original_indices = original_dataset_cell.get_traces_of_class(
                    class_number=website_num,
                    return_indices=True
                )
                
                # Stratified split to preserve cluster distribution
                X_train_tam, y_train, X_val_tam, y_val, X_test_tam, y_test, \
                train_indices, val_indices, test_indices = stratified_split(
                    tams_of_this_website, labels,
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                    verbose=False
                )
                
                # ===========================
                # Truncate Traces at Time Threshold
                # ===========================
                is_last_time_step = timestep_idx == len(time_steps) - 1
                
                # Helper function to truncate trace at time threshold
                def truncate_at_time(directions_list, times_list, indices, threshold):
                    if is_last_time_step:
                        # Use complete traces for final time step
                        return (
                            [directions_list[i] for i in indices],
                            [times_list[i] for i in indices]
                        )
                    else:
                        # Truncate at first packet exceeding threshold
                        truncated_dirs = []
                        truncated_times = []
                        for i in indices:
                            cutoff_idx = next(
                                (j for j, t in enumerate(times_list[i]) if t > threshold),
                                len(times_list[i])
                            )
                            truncated_dirs.append(directions_list[i][:cutoff_idx])
                            truncated_times.append(times_list[i][:cutoff_idx])
                        return truncated_dirs, truncated_times
                
                # Apply truncation to train/val/test sets
                directions_train, times_train = truncate_at_time(
                    directions, times, train_indices, time_step
                )
                directions_val, times_val = truncate_at_time(
                    directions, times, val_indices, time_step
                )
                directions_test, times_test = truncate_at_time(
                    directions, times, test_indices, time_step
                )
                
                # ===========================
                # Train KFP Classifier
                # ===========================
                # Combine train and validation sets
                directions_train += directions_val
                times_train += times_val
                y_train = np.array(y_train.tolist() + y_val.tolist())
                
                # Train k-Fingerprinting model
                test_acc, kfp_model, test_predictions = train_kfp(
                    directions_train=directions_train,
                    directions_test=directions_test,
                    times_train=times_train,
                    times_test=times_test,
                    y_train=y_train,
                    y_test=y_test,
                    save_features=False,
                    num_trees=100,
                    verbose=False,
                    return_predictions=True
                )
                
                # Store accuracy for this website
                detection_accuracy_website[website_num] = test_acc
                
                # ===========================
                # Save Predictions
                # ===========================
                # Map predictions back to original dataset indices
                for i in range(len(test_indices)):
                    original_index = original_indices[test_indices[i]]
                    if original_index not in overall_test_predictions:
                        overall_test_predictions[original_index] = {}
                    overall_test_predictions[original_index][time_step] = test_predictions[i]
                
                # Clean up memory
                del directions_train, times_train, times_test, directions_test
                gc.collect()
                
                # ===========================
                # Save Trained Model
                # ===========================
                save_path_model = os.path.join(
                    cm.BASE_DIR, 'data', cm.data_set_folder, 'kfp_predictions',
                    f'{algorithm_tier1}-{args.max_clusters}',
                    'kfp_timesteps',
                    'compressed' if args.compress is not None else ''
                )
                save_file(
                    dir_path=save_path_model,
                    content=kfp_model,
                    file_name=f'website_{website_num}_thresh_{time_step:.2f}.joblib',
                    compress=args.compress
                )
            else:
                # Single cluster per website: perfect accuracy at website level
                detection_accuracy_website[website_num] = 1.0
        
        # ===========================
        # Analyze Results for Time Step
        # ===========================
        # Generate accuracy distribution plots and statistics
        _, summary = plot_accuracy_distribution(
            detection_accuracy_website,
            threshold=time_step,
            plot_it=False
        )
        
        # Track mean accuracy across all websites
        mean_accuracies.append(summary['mean'])
    
    # ===========================
    # Save Final Results
    # ===========================
    save_dir = os.path.join(
        cm.BASE_DIR, 'data', cm.data_set_folder, 'clustering', 'two-tier',
        f'{algorithm_tier1}-{algorithm_tier2}',
        f'{algorithm_tier1}-{args.max_clusters}',
        f'{algorithm_tier2}-{args.k}',
        'kfp_timesteps'
    )
    
    # Save mean accuracy progression across time steps
    save_file(
        dir_path=save_dir,
        file_name=f'mean_accuracy_{start_time_step}_{end_time_step}.npy',
        content=np.array(mean_accuracies)
    )

# Example usage:
# python3 -m experiments.early_detection.within_website_classification -alg1 cast -alg2 palette_tamaraw -config Tik_Tok -k 7 -preload_clusters True -save True -l_tamaraw 100