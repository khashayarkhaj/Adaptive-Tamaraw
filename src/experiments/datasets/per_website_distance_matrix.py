# compute the distance matrix of each website and save it

import argparse
import utils.config_utils as cm
from utils.file_operations import load_file, save_file
from utils.distance_metrics import scs_distance, molding_distance, dtw_distance, dam_levenshtein_distance, tam_euclidian_distance
from utils.trace_dataset import TraceDataset
from utils.parser_utils import str2bool
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import os
import gc
def compute_distance(indices):
                    #apparently if your function is func(a1, a2), it will be hard to implement python multiprocessing, and the function should be like func(a) where a is a tupple.
                    if distance_type != 'tamaraw_L':
                        i, j = indices
                        return (i, j, custom_distance(data[i], data[j]))
                    else:
                        
                        return (i, j, custom_distance(real_indices[i], real_indices[j]))

def compute_distance_matrix_parallel(data, num_process):            
            # Compute distance matrix
            distance_matrix = np.zeros((len(data), len(data)))
            
            if num_process == 1: # no multi processing
                logger.info(f'Computing Distance matrix of {len(data)} traces...')
                logger.info(f'The distance metric function used is {custom_distance.__name__}')
                for i in tqdm(range(len(data)), desc = 'computing pair-wise distance'):
                    for j in tqdm(range(len(data)), desc = f'computing distance for instance {i + 1} / {len(data)}', disable= True):
                        if j > i:
                            if distance_type != 'tamaraw_L':
                                distance_matrix[i,j] = custom_distance(data[i], data[j])
                            else:
                                distance_matrix[i,j] = custom_distance(real_indices[i], real_indices[j])
                        elif j < i:
                            distance_matrix[i,j] = distance_matrix[j, i]
                
            #distance_matrix = np.array([[custom_distance(d1, d2) for d2 in data] for d1 in tqdm(data, desc = 'computing pair-wise distance')])

            else:
                logger.info(f'Computing Distance matrix of {len(data)} traces using {num_process} cores...')
                # Compute distance matrix
                index_pairs = [(i, j) for i in range(len(data)) for j in range(i, len(data))]
                with Pool(num_process) as pool:
                    # Use imap for lazy iteration over results
                    results = tqdm(pool.imap(compute_distance, index_pairs), total=len(index_pairs))
                    # Wrap the results with tqdm for a progress bar
                    for i, j, distance in results:
                        distance_matrix[i][j] = distance
                        distance_matrix[j][i] = distance  # Assuming the distance function is symmetric
            logger.info(f'Distance matrix computed')

            return distance_matrix
def compute_tamaraw_L_distance(indice1, indice2):
    length_vector1 = []
    length_vector2 = []
    l_values = [100, 500, 1000]

    
    for l in l_values:
        for config_idx in range(10):
            
            sizes = [tamaraw_overhead_dict[indice1][(config_idx,l)]]
            length_vector1+= sizes

            sizes = [tamaraw_overhead_dict[indice2][(config_idx,l)]]
            length_vector2+= sizes
    
    
            
    l2_distance = np.linalg.norm(np.array(length_vector2) - np.array(length_vector1))
    return l2_distance
if __name__ == '__main__':

    #python3 -m experiments.per_website_distance_matrix -d tamaraw_L

    # arguments
    parser = argparse.ArgumentParser(description='Computing the distance matrix for traces in each website')
    
    parser.add_argument('-d', '--distance',
                        choices=['mold', 'dtw', 'dam', 'scs', 'euc', 'tamaraw_L'], # tamaraw_L tries to put the elements with same lenght after tamaraw into an internal cluster
                        help='type of distance metric used to compare the intances',
                        default= 'euc') # if algorithm is k-means, this will only be euc
    
    parser.add_argument('-n', '--num_process', type = int, help='number of processes we want to use in our multi processing setup, for computing the distance matrix', 
                        default = 1)
    
    parser.add_argument('-e', '--extract_ds', type=str2bool, nargs='?', const=True, default=False,
                        help='should we extract the dataset or is it already stored'
                        )
    
    parser.add_argument('-conf', '--config', help='which config file to use', default = 'Tik_Tok')

    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=True,
                    help='Whether to save the computed elements')
    
   
    

    parser.add_argument('-min_ds_interval', type = float,
                         help='the lower bound of the dataset partition we want to cluster', default = 0)
    
    parser.add_argument('-max_ds_interval', type = float,
                         help='the upper bound of the dataset partition we want to cluster', default = 1)
    
    
    
    logger = cm.init_logger(name = 'Per class distance matrix')

    args = parser.parse_args()
    
    cm.initialize_common_params(args.config)
    distance_metric_name = args.distance
    
    #initializing the global params between different files, based on the chosen config file
    
    

    #choosing the distance function we want ot use
    custom_distance = molding_distance
    distance_type = args.distance
    if args.distance == 'dtw':
        custom_distance = dtw_distance
    elif args.distance == 'dam':
        custom_distance = dam_levenshtein_distance
    elif args.distance == 'scs':
        custom_distance = scs_distance
    elif args.distance == 'euc':
        custom_distance = tam_euclidian_distance
    elif args.distance == 'tamaraw_L':
        custom_distance = compute_tamaraw_L_distance
    
    trace_mode = 'cell'
    if distance_metric_name == 'euc':
         args.distance = 'euc'
         distance_metric_name = 'euc'
         custom_distance = tam_euclidian_distance # just to make sure that tams only use euclidean distance
         trace_mode = 'tam'
    
    if distance_metric_name == 'tamaraw_L':
        load_path_sizes = os.path.join(cm.BASE_DIR,  'results',  'tamaraw_params', f'{cm.data_set_folder}')
        tamaraw_overhead_dict = load_file(dir_path= load_path_sizes, file_name= 'tamaraw_sizes_per_L_top_10.json' ,use_pickle = True)
        
    
    num_process = args.num_process
    
    dataset_interval = [args.min_ds_interval, args.max_ds_interval]
    dataset = TraceDataset(extract_traces= args.extract_ds, 
                           trace_mode= trace_mode, 
                           interval= dataset_interval)
    

    for website in range(cm.MON_SITE_START_IND, cm.MON_SITE_END_IND):
          
          logger.info(f'Computing the distance matrix for website {website}')
          data, _ , real_indices= dataset.get_traces_of_class(class_number= website, return_indices= True)

          distance_matrix = compute_distance_matrix_parallel(data= data, num_process = num_process)
          
          save_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'distance_matrices', args.distance)
          if args.save: # if we want to save our distance matrix for future use
            file_name = f'{website}.npy'
            
            save_file(dir_path= save_dir, file_name= file_name, content= distance_matrix)
          del distance_matrix
          gc.collect()

          

    
#python3 -m experiments.per_website_distance_matrix -d tamaraw_L

#python3 -m experiments.datasets.per_website_distance_matrix -conf default