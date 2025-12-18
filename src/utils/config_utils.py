import yaml
from os.path import join, abspath, dirname, pardir
import numpy as np
import logging
import os
import random
import torch
# Function to load YAML configuration file


# Global parameters used through out the execution

print('Global config parameters are being initialized...')
BASE_DIR = ''
configs = None
#outputdir = join(BASE_DIR, 'dump')
outputdir = ''
resultdir = ''
dModelDir = ''
LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
logger = None # this logger should be set to a logger in the code, and all the other functions will use it as well. 
CUT_OFF_THRESHOLD = np.inf
CELL_SIZE = 1
DUMMY_CODE = 888
trace_length = 1000
normalize_bursts = False
MON_SITE_NUM = 100
MON_INST_NUM = 100
MON_SITE_START_IND = 0
MON_SITE_END_IND = 100
UNMONITORED_SITE_NUM = 10000
MON_INST_START_IND = 0
data_set_folder = 'ds19'
data_set_format = '.cell'
dataset_category = None #some datasets like DF18 have multiple datasets like NoDef
TIME_THRESHOLD_TRACE = None 

'''
threshold we will use to shorten the traces (empirical evaluation of how much information a short trace has about the entire trace).
'''
Maximum_Load_Time = None
Time_Slot = None # used for traffic aggeregation matrix
Max_tam_matrix_len = None # # used for traffic aggeregation matrix. if this is provided, then time_slot will be aotomatically adjusted base on maximum load time



def load_config(filename, section = 'default'):
    with open(filename, 'r') as file:
        all_configs = yaml.safe_load(file)
    return all_configs[section]

# print(load_config('configs/default.yaml'))


#from jiajuns code"
def init_logger(name = 'main', log_dir=None, level = logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    if log_dir is not None:
        # Extract the directory part

        directory_path = os.path.dirname(log_dir)

        if not os.path.exists(directory_path):
            try:
                os.makedirs(directory_path) #creating the dir if it doesn't already exist
            except:
                pass # sometimes would throw file exists error
        ch2 = logging.StreamHandler(open(log_dir, 'w'))
        ch2.setFormatter(formatter)
        logger.addHandler(ch2)
    return logger



def initialize_common_params(confdir, section = 'default'):
    # this function initializes the parameters that might be used by different files during execution.
    # Thus, it should be executed at the begining of our script.
    
    print(f'cm is being initialized with {confdir}')
    global BASE_DIR, configs, outputdir, resultdir, CUT_OFF_THRESHOLD, CELL_SIZE, DUMMY_CODE

    global trace_length, normalize_bursts, MON_SITE_NUM, MON_INST_NUM, MON_SITE_START_IND,  MON_SITE_END_IND, MON_INST_START_IND, UNMONITORED_SITE_NUM
    global data_set_folder, MON_SITE_START_IND, data_set_format, global_logger
    global dataset_category
    global Maximum_Load_Time, Time_Slot, Max_tam_matrix_len


    

    BASE_DIR = abspath(join(dirname(__file__), '../..')) # since we are in src/utils, and we want to go to our main folder
    confdir = join(BASE_DIR, 'configs', confdir + '.yaml')
    configs = load_config(confdir, section)
    outputdir = join(BASE_DIR, 'outputs')
    resultdir = join(BASE_DIR, 'results')
    # setting the seeds for the experiments
    set_seed(configs.get("random_seed", 43))
    

    
    CUT_OFF_THRESHOLD = configs.get("CUT_OFF_THRESHOLD", np.inf)
    CELL_SIZE = configs.get("CELL_SIZE", 1)
    DUMMY_CODE = configs.get("DUMMY_CODE", 888)
    trace_length = configs.get("trace_length", 1000)
    normalize_bursts= configs.get("normalize_bursts", False)
    MON_SITE_NUM = configs.get("MONITORED_SITE_NUM", 100)
    MON_INST_NUM = configs.get("MONITORED_INST_NUM", 100)
    MON_SITE_START_IND = configs.get("MONITORED_SITE_START_IND", 0)
    MON_SITE_END_IND = configs.get("MON_INST_END_IND", MON_SITE_NUM)
    UNMONITORED_SITE_NUM = configs.get("UNMONITORED_SITE_NUM", 10000)
    data_set_folder = configs.get("data_set_folder", 'ds19')
    MON_INST_START_IND = configs.get("MON_INST_START_IND", 0)
    
    data_set_format =  configs.get("data_set_format", 0)
    dataset_category = configs.get("data_set_category", None)
    Maximum_Load_Time = configs.get("Maximum_Load_Time", None)
    Max_tam_matrix_len = configs.get("Max_tam_matrix_len", None)
    

    if Max_tam_matrix_len:
        Time_Slot = round(Maximum_Load_Time / Max_tam_matrix_len, 3) # the time bins in tam
    else:
        Time_Slot = configs.get("Time_Slot", None) 
        
    
    
    
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)   
    print(f'seed of randomness is {seed}') 







global_logger = init_logger('global')




