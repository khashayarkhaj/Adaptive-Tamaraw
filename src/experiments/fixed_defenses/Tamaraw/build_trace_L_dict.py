# given the L json files in tamaraw_params, combine them and create a dictionary that can be used for two tier clustering
import os
import json
import re
from tqdm import tqdm
import pickle
def process_trace_params(folder_path, min_index = None, max_index = None):
    # Initialize dictionaries to store results
    trace_params = {}
    tamaraw_sizes_per_L = {}

    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json') and f.startswith('trace_params_')]

    # Sort files to ensure consistent processing
    json_files.sort()
    L = int(os.path.basename(folder_path).split('_')[-1]) if 'L_' in os.path.basename(folder_path) else None
    for json_file in tqdm(json_files):
        # Extract trace indices using regex
        match = re.search(r'trace_params_(\d+)_(\d+)\.json', json_file)
        if not match:
            print(f"Skipping file {json_file} - unable to parse trace indices")
            continue
        
        trace_idx_start = int(match.group(1))
        trace_idx_end = int(match.group(2))
        
        # Full path to the file
        file_path = os.path.join(folder_path, json_file)
        
        # Read the JSON file
        with open(file_path, 'r') as f:
            
            file_data = json.load(f)
        
        for trace_idx in file_data.keys():
            if (min_index is not None and int(trace_idx) < min_index) or (max_index is not None and int(trace_idx) > max_index):
                continue
            tamaraw_sizes_per_L[int(trace_idx)] = {}
            trace_params[trace_idx] = file_data[trace_idx]

            # Process each config index
            for config_idx, values in file_data[trace_idx].items():
                if config_idx in ['bw', 'time']:
                    continue
                
                config_idx = int(config_idx)
                
                size = values[0]
                tamaraw_sizes_per_L[int(trace_idx)][(config_idx, L)] = size
                # Extract L from filename (assuming it's part of the filepath)
                
                    

    # Save trace_params
    # breakpoint()
    with open(os.path.join(folder_path, 'trace_params.json'), 'w') as f:
        json.dump(trace_params, f, indent=2)
    
    # Save tamaraw_sizes_per_L
    with open(os.path.join(folder_path, 'tamaraw_sizes_per_L.json'), 'wb') as f:
        pickle.dump(tamaraw_sizes_per_L, f)
    
    print("Processing complete. Files 'trace_params.json' and 'tamaraw_sizes_per_L.json' created.")

# Example usage
# folder_path= ''
# process_trace_params(folder_path, min_index= 0, max_index= 100 * 1000 - 1) for awf
# process_trace_params(folder_path)
