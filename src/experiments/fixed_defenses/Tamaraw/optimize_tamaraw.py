# finding the best parameters of tamaraw for a given dataset
import utils.config_utils as cm
from time import strftime
import argparse
import numpy as np

from utils.trace_dataset import TraceDataset
from utils.parser_utils import str2bool
from utils.file_operations import load_file
from tqdm import tqdm
import os

from fixed_defenses.tamaraw import grid_search_tamaraw_params

if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser(description='Visualizing Tams')

    
    parser.add_argument('-e',  '--extract_ds', type=str2bool, nargs='?', const=True, default=False, 
                         help='should we extract the dataset or is it already stored')
        
    parser.add_argument('-conf', '--config', help='which config file to use', default = 'Tik_Tok')

    
    
   
    
    
    
    
    parser.add_argument('-L', type = int,
                         help='the L param for tamaraw', default = 50)
    
    parser.add_argument('-n_cores', type = int,
                         help='number of cores to use for tamaraw', default = 1) # None means use all, 1 means don't do parallelism
    
    parser.add_argument('-random_extend', type = str2bool, nargs='?', const=True, default=False,
                         help='Whether to randomly extend the tamaraw trace to a factor of L')
    
    
    

    parser.add_argument('-preload_ohs', type = str2bool, nargs='?', const=True, default=False,
                         help='Whether to load the pre-computed defended traces statistics')
    logger = cm.init_logger(name = 'Performing Tamaraw')
    args = parser.parse_args()

    cm.initialize_common_params(args.config)

    

    dataset = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'cell', keep_original_trace= True)
    undefended_traces = dataset.original_traces # each original trace is in the form [[t1, d1], [t2, d2] , ...]
    
    save_dir = os.path.join(cm.BASE_DIR, 'results', 'fixed_defenses', 'tamaraw', cm.data_set_folder, 'optimization', f'Fixed_L_{args.L}')



   
    
    
    # do what tao does
    rho_in_initial = 0.012
    rho_out_initial = 0.04
    # n ranges from -10 to +10 (inclusive), giving us 11 values
    n_values = np.arange(-7, 7)  # [-10, -9, ..., 9, 10]

    # Calculate the exponential factor for each n
    factors = np.power(7, n_values / 7)  # factor = 10^(n/10)

    # Generate the rho_in and rho_out values
    ro_in_range = rho_in_initial * factors
    ro_out_range = rho_out_initial * factors


    trace_oh_dict = None
    if args.preload_ohs:
            param_load_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'tamaraw_params')
            if args.L is not None:
                 param_load_dir = os.path.join(param_load_dir, f'L_{args.L}')
            trace_oh_dict = load_file(dir_path= param_load_dir,  file_name= f'trace_params.json', string_to_num = True )

    grid_search_tamaraw_params(undefended_traces= undefended_traces,
                               ro_in_range= ro_in_range,
                               ro_out_range= ro_out_range,
                               L = args.L, n_cores= args.n_cores, random_extend= args.random_extend,
                               save_dir= save_dir,
                               trace_oh_dict= trace_oh_dict)
# python3 -m experiments.fixed_defenses.Tamaraw.optimize_tamaraw -n_cores 1 -L 500  -preload_ohs True