# Analyzing different datasets and plotting their statistics
import numpy as np
import matplotlib.pyplot as plt
import utils.config_utils as cm
from utils.trace_dataset import TraceDataset
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from utils.file_operations import save_file
from scipy.stats import gaussian_kde
from scipy import stats
from utils.parser_utils import str2bool
def analyze_traffic_traces(number_of_cells, save_dir = None):
    """
    Perform comprehensive analysis of traffic trace packet counts.
    
    Parameters:
    number_of_cells (list): List of packet counts for each trace
    
    Returns:
    dict: Dictionary containing various statistical analyses
    """
    # Convert to numpy array for easier calculations
    data = np.array(number_of_cells)
    
    # Basic statistics
    stats_dict = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std_dev': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'total_traces': len(data)
    }
    
    # Create figure with subplots for different visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Histogram
    plt.subplot(2, 2, 1)
    plt.hist(data, bins='auto', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Packet Counts')
    plt.xlabel('Number of Packets')
    plt.ylabel('Frequency')
    
    # 2. Probability Density Function (PDF)
    plt.subplot(2, 2, 2)
    # Kernel Density Estimation for smooth PDF
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 100)
    plt.plot(x_range, kde(x_range))
    plt.title('Probability Density Function')
    plt.xlabel('Number of Packets')
    plt.ylabel('Density')
    
    # 3. Cumulative Distribution Function (CDF)
    plt.subplot(2, 2, 3)
    # Calculate CDF
    x_sorted = np.sort(data)
    y_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    
    plt.plot(x_sorted, y_cdf)
    plt.title('Cumulative Distribution Function')
    plt.xlabel('Number of Packets')
    plt.ylabel('Cumulative Probability')
    
    # Highlight specific quantiles on CDF
    quantiles = [0.5, 0.7, 0.9]
    for q in quantiles:
        quantile_value = np.quantile(data, q)
        quantile_prob = np.sum(data <= quantile_value) / len(data)
        plt.plot(quantile_value, quantile_prob, 'ro')  # Red dot
        plt.annotate(f'p{int(q*100)}: {quantile_value:.2f}', 
                     (quantile_value, quantile_prob), 
                     xytext=(10, 10), 
                     textcoords='offset points')
    
    # 4. Box Plot
    plt.subplot(2, 2, 4)
    plt.boxplot(data)
    plt.title('Box Plot of Packet Counts')
    plt.ylabel('Number of Packets')
    
    # Adjust layout and save
    plt.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        save_file(dir_path= save_dir, file_name= 'traffic_trace_size_analysis.png')
    
    
    # Calculate and store quantile information
    stats_dict['quantiles'] = {
        f'p{int(q*100)}': np.quantile(data, q) for q in quantiles
    }
    
    return stats_dict
if __name__ == '__main__':


    # arguments
    parser = argparse.ArgumentParser(description='Analyzing Trace Datasets')
    parser.add_argument('-conf', '--config', help='which config file to use', default = 'default')

    parser.add_argument('-e', '--extract_ds', help='should we extract the dataset or is it already stored', 
                        default = False)
    
    parser.add_argument('-threshold_time', type= float, help='the time threshold used for computing the percentage of outliers', 
                        default = None)
    
    parser.add_argument('-threshold_length', type= float, help='the length threshold used for computing the percentage of outliers', 
                        default = None)
    
    

    logger = cm.init_logger(name = 'Dataset Analysis')
    args = parser.parse_args()

    cm.initialize_common_params(args.config)
    ds = TraceDataset(extract_traces=  args.extract_ds, trace_mode = 'cell')
    trace_loader = DataLoader(ds, batch_size=1, shuffle=False)

    ending_times = []
    number_of_cells = []

    for directions, times, labels in tqdm(trace_loader, desc= f'Gathering statistics from the {cm.data_set_folder} dataset'):
        ending_times.append(times[0][-1])
        number_of_cells.append(len(directions[0]))
        #apparently, dataloader returns each of the returned elements from getitem in a seperate tensor, with batch size shape
        


    dir_path =  os.path.join(cm.BASE_DIR,  'results',  'dataset-statistics', cm.data_set_folder, 'statistics','traces')
    # Plot historgram of ending times
    plt.figure(figsize=(10, 4))
    plt.hist(ending_times, bins=20, color='blue', alpha=0.7)
    plt.title(f'Histogram of Ending Times in the {cm.data_set_folder} Dataset')
    plt.xlabel('Loading latency (s)')
    plt.ylabel('Frequency')
    if args.threshold_time:
        # Draw vertical line for the threshold in PDF
        above_threshold = sum(np.array(ending_times) > args.threshold_time)
        # Display number of occurrences above the threshold
        plt.axvline(x=args.threshold_time, color='k', linestyle='--', label=f'Threshold = {args.threshold_time}')
        plt.legend()
        plt.text(args.threshold_time + 0.1, max(ending_times) * 0.9, f'Counts > {args.threshold_time}: {above_threshold}', color='black')

    save_file(dir_path= dir_path, file_name= 'timing_hist.png')


    #plot pdf of ending times
    kde = gaussian_kde(ending_times)
    x_range = np.linspace(min(ending_times), max(ending_times), 500)
    plt.figure(figsize=(10, 4))
    plt.plot(x_range, kde(x_range) * 100, color='r') # the 100 is for percentage
    plt.title('Probability Density Function (Percentage)')
    plt.xlabel('Loading latency (s)')
    plt.ylabel('Probability (%)')
    plt.grid(True)
    if args.threshold_time:
        # Draw vertical line for the threshold in PDF
        percentage_above_threshold = np.mean(np.array(ending_times) > args.threshold_time) * 100
        plt.axvline(x=args.threshold_time, color='k', linestyle='--', label=f'Threshold = {args.threshold_time}')
        plt.legend()
        plt.text(args.threshold_time + 0.1, plt.ylim()[1] * 0.9, f'{percentage_above_threshold:.2f}% > {args.threshold_time}', color='black')
    save_file(dir_path= dir_path, file_name= 'timing_pdf.png')

    #plot cdf of ending times
    plt.figure(figsize=(6, 4))
    plt.hist(ending_times, bins=20, cumulative=True, color='blue', alpha=0.7, density=True)
    plt.title(f'CDF of Ending Times in the {cm.data_set_folder} Dataset')
    plt.xlabel('Time')
    plt.ylabel('Probability (%)')
    plt.grid(True)

    
    save_file(dir_path= dir_path, file_name= 'timing_cdf.png')


    # Plot Histogram of trace sizes
    plt.figure(figsize=(10, 4))
    plt.hist(number_of_cells, bins= 20, align='left', color='green', alpha=0.7)
    plt.title(f'Histogram of Trace Sizes in the {cm.data_set_folder} Dataset')
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    
    plt.tight_layout()

    if args.threshold_length:
        # Draw vertical line for the threshold in PDF
        above_threshold = sum(np.array(number_of_cells) > args.threshold_length)
        plt.axvline(x=args.threshold_length, color='k', linestyle='--', label=f'Threshold = {args.threshold_length}')
        plt.legend()
        plt.text(args.threshold_time + 0.1, max(number_of_cells) * 0.9, f'Counts > {args.threshold_lenght}: {above_threshold}', color='black')
    #plot pdf of trace sizes
    kde = gaussian_kde(number_of_cells)
    x_range = np.linspace(min(number_of_cells), max(number_of_cells), 500)
    plt.figure(figsize=(10, 4))
    plt.plot(x_range, kde(x_range) * 100, color='r') # the 100 is for percentage
    plt.title('Probability Density Function (Percentage)')
    plt.ylabel('Probability (%)')
    plt.grid(True)

    if args.threshold_length:
        # Draw vertical line for the threshold in PDF
        percentage_above_threshold = np.mean(np.array(number_of_cells) > args.threshold_length) * 100
        plt.axvline(x=args.threshold_length, color='k', linestyle='--', label=f'Threshold = {args.threshold_length}')
        plt.legend()
        plt.text(args.threshold_length + 0.1, plt.ylim()[1] * 0.9, f'{percentage_above_threshold:.2f}% > {args.threshold_length}', color='black')
    save_file(dir_path= dir_path, file_name= 'trace_length_pdf.png')

   

    # Plot CDF of sequence sizes
    plt.figure(figsize=(6, 4))
    plt.hist(number_of_cells, bins=20, cumulative=True, align='left', color='green', alpha=0.7, density=True)
    plt.title(f'CDF of Trace Sizes in the {cm.data_set_folder} Dataset')
    plt.xlabel('Size')
    plt.ylabel('Probability')
    plt.grid(True)
    save_file(dir_path= dir_path, file_name= 'trace_length_cdf.png')

    analyze_traffic_traces(number_of_cells, dir_path)
# python3 -m experiments.datasets.statistics -conf Tik_Tok