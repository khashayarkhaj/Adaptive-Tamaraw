 # a number of functions that facilitate working with files
import utils.config_utils as cm
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ray import tune
import yaml
import torch
import json
import pandas as pd
import dill
import joblib
import sys
from tqdm import tqdm

def create_trace_flist():
    
    # return a list of paths of trace files (like .cell) used in our experiments
    
    flist = []
    for i in tqdm(range(cm.MON_SITE_START_IND, cm.MON_SITE_START_IND + cm.MON_SITE_NUM), desc = 'creating the flist'):
        
        for j in range(cm.MON_INST_START_IND, cm.MON_INST_START_IND + cm.MON_INST_NUM):
            desired_path = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'dataset', 
                                            str(i) + "-" + str(j) + cm.data_set_format)
        
        
            if os.path.exists(desired_path):
                flist.append(desired_path)
               
    return flist


def extract_label(fdir):
    #given the path of a trace file, this function extracts the label of that trace
    #fname = fdir.split('/')[-1].split(".")[0]

    fname, path = os.path.splitext(os.path.basename(fdir))
    
    if '-' in fname:
        label = int(fname.split('-')[0])
    else:
        label = int(cm.MON_SITE_NUM)
    
    return label
def convert_string_keys_to_numbers(d): # some times, when I save dictionaries, the numbers are saved as string
    """
    Recursively converts dictionary string keys to numbers if they represent numbers.
    Works on nested dictionaries and lists.
    
    Args:
        d: Dictionary or list to convert
        
    Returns:
        Converted dictionary or list with numeric keys where applicable
    """
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            # Convert the key to number if possible
            try:
                new_key = float(key) if '.' in key else int(key)
            except (ValueError, TypeError):
                new_key = key
                
            # Recursively convert any nested structures
            new_value = convert_string_keys_to_numbers(value)
            new_dict[new_key] = new_value
        return new_dict
    
    elif isinstance(d, list):
        return [convert_string_keys_to_numbers(item) for item in d]
    
    elif isinstance(d, (str, int, float)):
        # Try to convert strings to numbers if possible
        if isinstance(d, str):
            try:
                return float(d) if '.' in d else int(d)
            except (ValueError, TypeError):
                return d
        return d
    
    else:
        return d
    

def check_directory_exists(directory_path):
    #checks if a certain directory exists
    assert os.path.isdir(directory_path), f"Directory does not exist: {directory_path}"

def check_file_exists(file_path):
    #checks if a certain file exists
    assert os.path.isfile(file_path), f"File does not exist: {file_path}"
def json_serializable_convert(obj):
    if isinstance(obj, (np.int64, np.float64)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
def make_json_friendly(obj):
    """
    Recursively convert np.int64/np.float64 in both keys and values
    to standard Python types.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Convert the key if it's a NumPy type
            if isinstance(k, (np.integer, np.floating)):
                k = k.item()  # or int(k), float(k) depending on your usage
            # Recursively convert the value
            new_dict[k] = make_json_friendly(v)
        return new_dict

    elif isinstance(obj, list):
        return [make_json_friendly(item) for item in obj]

    elif isinstance(obj, tuple):
        return tuple(make_json_friendly(item) for item in obj)

    else:
        # If it's a scalar NumPy type, convert it
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

def save_file(dir_path, file_name, content = None, logger =None, **kwargs):
    """
        a function that saves our file when we want to store the results in a specific place.
        Args:
            dir_path: the directory where we want to save our file
            file_name: the name of our file
            content: the content we want to save in our file
            kwargs: 
                if you want to np.savez_compressed, you should provide compress = True, feautures, labels
                you can also pass the protocol for pickle (pickle_protocol)
        """
    file_encoding = file_name.split('.')[-1]

    
    if not logger:
        logger = cm.global_logger
    logger.info(f'saving {file_name} ...')
    if file_encoding in ['npz', 'pkl', 'png', 'npy', 'txt', 'pickle', 'pth', 'json', 'csv', 'joblib']:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path) #creating the dir if it does'nt already exist
            except:
                pass # sometimes the folder already exists but os.path.exists doesn't show correctly and when we do os.makedirs it throws FileExistsError
        
        save_dir = os.path.join(dir_path, file_name)
        
        if file_encoding == 'npz':
            logger.info(f"Saving file as NPZ format to {save_dir}")
            if 'compress' in kwargs:
                np.savez_compressed(save_dir, features = kwargs['features'], labels = kwargs['labels'])
            else:
                if isinstance(content, dict):
                    np.savez(save_dir, **content)
                else:
                    np.savez(save_dir, *content)

        elif file_encoding in ['pkl', 'pickle'] or ('save_with_pickle' in kwargs and kwargs['save_with_pickle'] is True):
            logger.info(f"Saving file as pickle format to {save_dir}")
            with open(save_dir, 'wb') as file:
                if 'pickle_protocol' not in kwargs:
                    pickle.dump(content, file)
                else:
                    pickle.dump(content, file, protocol= kwargs['pickle_protocol'])

        elif file_encoding == 'png':
            logger.info(f"Saving plot as PNG image to {save_dir}") 
            plt.savefig(save_dir) #TODO check if this works if we call plt.savefig out of nowhere
            plt.close()

        elif file_encoding == 'npy':
            logger.info(f"Saving NumPy array to {save_dir}")
            np.save(save_dir, content)

        elif file_encoding == 'txt':
            logger.info(f"Saving text file to {save_dir}")
            with open(save_dir, 'w') as file:
                file.write(content)

        elif file_encoding == 'pth':
            logger.info(f"Saving PyTorch model state to {save_dir}") # content is expected to be wf_model.state_dict(
            torch.save(content, save_dir)

        elif file_encoding == 'json':
            logger.info(f"Saving JSON file to {save_dir}")
            if 'use_pickle' in kwargs:
                with open(save_dir, 'wb') as f:
                    pickle.dump(content, f)
            else:
                with open(save_dir, 'w') as f:
                    json.dump(content, f, indent=4, default=json_serializable_convert)

        elif file_encoding == 'csv':
            logger.info(f"Saving DataFrame as CSV to {save_dir}")
            content.to_csv(save_dir)  #we assume that content is pd df and save it accordingly

        elif file_encoding == 'joblib': # sklearn models
            logger.info(f"Saving joblin model to {save_dir}")
            if 'compress' in kwargs and kwargs['compress'] is not None:
                joblib.dump(content, save_dir, compress=9)
            else:
                joblib.dump(content, save_dir)

        logger.info(f'{file_name} saved')

def load_file(dir_path, file_name, logger=None, verbose=True, **kwargs):
    """
        a function that loads and returns the content of a specific file.
        Args:
            dir_path: the directory where we want to load our file from
            file_name: the name of our file
            logger: logger object to use (defaults to global logger)
            verbose: whether to show log messages (default: True)
            kwargs:
                if you want a specific encoding for pickles, set encoding = 'latin' for instance
    """

    if not logger:
        logger = cm.global_logger

    file_encoding = file_name.split('.')[-1]
    
    
    if verbose:
        logger.info(f'Loading {file_name}...')
    
    if file_encoding in ['npz', 'pkl', 'npy', 'pickle', 'json', 'csv', 'dill', 'joblib']:  
        check_directory_exists(dir_path)
        load_dir = os.path.join(dir_path, file_name)
        check_file_exists(load_dir)
        content = None
        
        if verbose:
            logger.info(f'path is {load_dir}')
            
        if file_encoding in ['npy', 'npz']:
            content = np.load(load_dir)
        
        elif file_encoding in ['pkl' , 'pickle']:
            with open(load_dir, 'rb') as file:
                if 'encoding' in kwargs:
                    content = pickle.load(file, encoding= kwargs['encoding'])
                else:
                    content = pickle.load(file)
        elif file_encoding == 'json':
            if 'use_pickle' in kwargs:
                with open(load_dir, 'rb') as f:
                    content = pickle.load(f)
            else:
                with open(load_dir, 'r') as file:
                    content = json.load(file)
                if 'string_to_num' in kwargs: # converting the strings in the dict to ints
                    content = convert_string_keys_to_numbers(content)

        elif file_encoding == 'csv':
            content = pd.read_csv(load_dir)
        elif file_encoding == 'dill':
            with open(load_dir, 'rb') as f:
                content = dill.load(f)
        elif file_encoding == 'joblib': # sklearn model
            content = joblib.load(load_dir)

        if verbose:
            logger.info(f'{file_name} loaded')
            
        assert content is not None, f"File had None content: {load_dir}"
        return content
      
      
    
      
           
    
def predict_np_file_size(arr, unit='gb', verbose=False, logger=None):
    """
    Takes a NumPy array and estimates its size in memory.
    
    Args:
        arr: NumPy ndarray
        unit: str, one of ['kb', 'mb', 'gb']
        verbose: bool, whether to print/log the size information
        logger: logging object, optional
    
    Returns:
        float: Memory usage in specified unit
    """
    assert isinstance(arr, np.ndarray), "Input must be a NumPy array"
    assert unit.lower() in ['kb', 'mb', 'gb'], "The given unit is not defined"

    memory_usage = arr.size * arr.itemsize
    
    unit = unit.lower()
    if unit == 'kb':
        memory_usage = memory_usage / 1024
    elif unit == 'mb':
        memory_usage = memory_usage / (1024 ** 2)
    elif unit == 'gb':
        memory_usage = memory_usage / (1024 ** 3)  # This was incorrect in original

    if verbose:
        text = f'Required memory for saving this file will probably be around {memory_usage:.2f} {unit}'
        if logger:
            logger.info(text)
        else:
            print(text)
    
    return memory_usage  # This was returning unit instead of memory_usage

def predict_model_size(model, unit='gb'):
    """
    Calculate the size of a PyTorch model in various units.
    
    Args:
        model: PyTorch model
        unit: str, one of ['b', 'kb', 'mb', 'gb'] for bytes, kilobytes, megabytes, or gigabytes
    
    Returns:
        float: Size of model in specified unit
    """
    # Get model state dict
    param_size = 0
    buffer_size = 0
    
    # Calculate size of parameters
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Calculate size of buffers (e.g., running mean in BatchNorm)
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    # Convert to desired unit
    if unit.lower() == 'b':
        return total_size
    elif unit.lower() == 'kb':
        return total_size / 1024
    elif unit.lower() == 'mb':
        return total_size / (1024 ** 2)
    elif unit.lower() == 'gb':
        return total_size / (1024 ** 3)
    else:
        raise ValueError("Unit must be one of ['b', 'kb', 'mb', 'gb']")

from typing import Dict, Any

def predict_npz_file_size(data_dict: Dict[str, Any], verbose = True) -> Dict[str, int]:
    """
    Estimate the size of an NPZ file before saving it.
    
    Args:
        data_dict: Dictionary containing arrays to be saved
        
    Returns:
        Dictionary with size estimates in bytes
    """
    total_raw_size = 0
    array_sizes = {}
    
    for key, value in data_dict.items():
        # Convert to numpy array if it isn't already
        if not isinstance(value, np.ndarray):
            value = np.array(value)
            
        # Calculate size of this array
        array_size = value.nbytes
        array_sizes[key] = array_size
        total_raw_size += array_size
        
    # NPZ files use ZIP compression
    # Estimate compressed size (very rough estimate - assumes 50% compression)
    estimated_compressed = total_raw_size // 2
    
    # Add overhead for ZIP format and NumPy metadata
    zip_overhead = 1024  # Rough estimate for ZIP headers and metadata
    overhead_per_array = 256  # Rough estimate for NumPy array metadata
    total_overhead = zip_overhead + (overhead_per_array * len(data_dict))
    
    size_info = {
        'total_raw_bytes': total_raw_size,
        'estimated_compressed_bytes': estimated_compressed + total_overhead,
        'overhead_bytes': total_overhead,
        'array_sizes': array_sizes
    }

    if verbose:
        def bytes_to_human_readable(bytes_size: int) -> str:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_size < 1024:
                    return f"{bytes_size:.2f} {unit}"
                bytes_size /= 1024
            return f"{bytes_size:.2f} TB"
        
        print("\nSize Analysis:")
        print("--------------")
        print(f"Total raw size: {bytes_to_human_readable(size_info['total_raw_bytes'])}")
        print(f"Estimated compressed size: {bytes_to_human_readable(size_info['estimated_compressed_bytes'])}")
        print(f"Overhead: {bytes_to_human_readable(size_info['overhead_bytes'])}")
        
        print("\nSize per array:")
        for array_name, size in size_info['array_sizes'].items():
            print(f"{array_name}: {bytes_to_human_readable(size)}")
    
    return size_info




def get_sklearn_model_size(model, filepath=None):
    """
    Get the size of a model, either in memory or on disk
    
    Args:
        model: The model to analyze
        filepath: Optional path to saved model file
        
    Returns:
        dict: Dictionary containing size information in bytes and human-readable format
    """
    def bytes_to_human_readable(size_in_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_in_bytes < 1024.0:
                return f"{size_in_bytes:.2f} {unit}"
            size_in_bytes /= 1024.0
        return f"{size_in_bytes:.2f} TB"

    # Get in-memory size
    memory_size = sys.getsizeof(model)
    
    # Get on-disk size if filepath provided
    disk_size = None
    if filepath and os.path.exists(filepath):
        disk_size = os.path.getsize(filepath)
    
    size_info = {
        'memory_size_bytes': memory_size,
        'memory_size_human': bytes_to_human_readable(memory_size),
    }
    
    if disk_size is not None:
        size_info['disk_size_bytes'] = disk_size
        size_info['disk_size_human'] = bytes_to_human_readable(disk_size)
        print(f"On-disk size: {size_info['disk_size_human']}")
    
    print(f"In-memory size: {size_info['memory_size_human']}")
    
    return size_info


