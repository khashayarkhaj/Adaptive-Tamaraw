from . import kFP, kFP_RF_fextract
from tqdm import tqdm
import numpy as np
import utils.config_utils as cm
from time import strftime
import argparse
import numpy as np
import torch
from utils.trace_dataset import TraceDataset
from utils.parser_utils import str2bool
from utils.file_operations import save_file, load_file
from training.train_utils import stratified_split


# sams code for kfp. I changed it a little bit

def kfp_features(X, T, Y, data_type, data_dict, max_cells=5000, verbose=True, preserve_order=False, return_features = False):
    """
    Implementation of kFP with optional order preservation
    """
    data_dict[data_type + '_feature'] = []
    data_dict[data_type + '_label'] = []
    
    if verbose:
        print(f'Creating KFP {data_type} Features...')
    
    if preserve_order:
        # Process instances in original order
        pbar = tqdm(range(len(X)), unit="instance", total=len(X), disable=not verbose)
        for idx in pbar:
            zero_cells = np.where(X[idx] == 0)[0]
            
            if zero_cells.any():
                last_cell_index = zero_cells[0]
            else:
                last_cell_index = max_cells
                
            list_data = list(zip(T[idx][:last_cell_index], X[idx][:last_cell_index].astype(int)))
            
            if len(list_data) <= 11:
                features = (0,) * 175
            else:
                try:
                    features = kFP_RF_fextract.TOTAL_FEATURES(list_data)
                except:
                    print(f'trace {idx} with {len(list_data)} elements was not convertible')
                    # Insert placeholder to maintain order
                    data_dict[data_type + '_feature'].append([(0,) * 175])
                    data_dict[data_type + '_label'].append((int(Y[idx]), idx))
                    continue
                    
            data_dict[data_type + '_feature'].append([features])
            data_dict[data_type + '_label'].append((int(Y[idx]), idx))
    
    else:
        # Original site-by-site processing
        unique_sites = np.unique(Y)
        pbar1 = tqdm(np.nditer(unique_sites), unit="site", total=unique_sites.shape[0], disable=not verbose)
        for site in pbar1:
            site_index = np.where(Y == site)[0]
            
            pbar2 = tqdm(np.ndenumerate(site_index), unit="instance", leave=False, total=site_index.shape[0], disable=not verbose)
            for instance_no, instance_index in pbar2:
                zero_cells = np.where(X[instance_index] == 0)[0]
                
                if zero_cells.any():
                    last_cell_index = zero_cells[0]
                else:
                    last_cell_index = max_cells
                    
                list_data = list(zip(T[instance_index][:last_cell_index], X[instance_index][:last_cell_index].astype(int)))
                
                if len(list_data) <= 11:
                    features = (0,) * 175
                else:
                    try:
                        features = kFP_RF_fextract.TOTAL_FEATURES(list_data)
                    except:
                        print(f'trace {instance_no} with {len(list_data)} elements was not convertible')
                         ######### I added this for awf rebuttal
                        data_dict[data_type + '_feature'].append([(0,) * 175])
                        data_dict[data_type + '_label'].append((int(site), instance_no[0]))
                        continue
                        
                data_dict[data_type + '_feature'].append([features])
                data_dict[data_type + '_label'].append((int(site), instance_no[0]))
    
    assert len(data_dict[data_type + '_feature']) == len(data_dict[data_type + '_label'])

def train_kfp(directions_train, times_train, directions_test, times_test, y_train, y_test, num_trees = 100, verbose = True, 
              preload_features = False, feature_path = None, save_features = False,
              return_predictions = False):
    
    
    if not preload_features:
        # create a data_dict for train
        data_dict_mon_train = {}
        # Populate data_dict for train
        kfp_features(directions_train,
                    times_train,
                    y_train,
                    'alexa',
                    data_dict_mon_train,
                    verbose= verbose)
        
        
        # Free up some memory
        del directions_train
        del times_train

        

        # create a data_dict for train
        data_dict_mon_test = {}
        # Populate data_dict for train
        kfp_features(directions_test,
                    times_test,
                    y_test,
                    'alexa',
                    data_dict_mon_test,
                    verbose= verbose)
        
        
        # Free up some memory
        del directions_test
        del times_test
        
        if save_features:
            save_file(dir_path= feature_path, file_name= 'train_features.json', content= data_dict_mon_train, apply_cc= True)
            save_file(dir_path= feature_path, file_name= 'test_features.json', content= data_dict_mon_test, apply_cc= True)
    else:
        data_dict_mon_train = load_file(dir_path= feature_path, file_name= 'train_features.json')
        data_dict_mon_test = load_file(dir_path= feature_path, file_name= 'test_features.json')

        for trace_idx in range(len(data_dict_mon_train['alexa_feature'])):
            data_dict_mon_train['alexa_feature'][trace_idx] = [tuple(data_dict_mon_train['alexa_feature'][trace_idx][0])]
        for trace_idx in range(len(data_dict_mon_test['alexa_feature'])):
            data_dict_mon_test['alexa_feature'][trace_idx] = [tuple(data_dict_mon_test['alexa_feature'][trace_idx][0])]
    
    _, train_set = kFP.mon_train_test_references(data_dict_mon_train, 0)
    _, test_set = kFP.mon_train_test_references(data_dict_mon_test, 0)

    
    accuracy, model, kfp_predictions = kFP.RF_closedworld(training=train_set, test=test_set, num_trees=num_trees, verbose= verbose,
                                                          return_predictions= return_predictions)

    
    if not return_predictions:
        return accuracy, model
    else:
        return accuracy, model, kfp_predictions
    

def get_kfp_feature_set(directions, times, y, verbose = True):
    # given a set of directions times, obtain a set of features for feeding into the kfp model

    data_dict_mon = {}
        # Populate data_dict for train
    kfp_features(directions,
                times,
                y,
                'alexa',
                data_dict_mon,
                verbose= verbose,
                preserve_order= True)
    
    for trace_idx in range(len(data_dict_mon['alexa_feature'])):
            data_dict_mon['alexa_feature'][trace_idx] = [tuple(data_dict_mon['alexa_feature'][trace_idx][0])]
    _, feature_set = kFP.mon_train_test_references(data_dict_mon, 0, shuffle= False)
    
    return feature_set

def get_kfp_prediction(features, kfp_model):
    te_data, _ = list(zip(*features))
    

    kfp_predictions = kfp_model.predict(te_data)
    return kfp_predictions



if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser(description=' Analyzing the results of a classifier on RegulaTor')
    parser.add_argument('-e',  '--extract_ds', type=str2bool, nargs='?', const=True, default=False, 
                         help='should we extract the dataset or is it already stored')
        
    parser.add_argument('-conf', '--config', help='which config file to use', default = 'Tik_Tok')

    

    logger = cm.init_logger(name = 'Normal KFP training check')
    args = parser.parse_args()

    cm.initialize_common_params(args.config)

    

    dataset = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'cell', keep_original_trace= True)

    dataset_tam = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'tam')

    X_train_tam, y_train, X_val, y_val, X_test_tam, y_test, train_indices, val_indices, test_indices = stratified_split(dataset_tam.directions, dataset_tam.labels,
                                                                                                                train_ratio= 0.8,
                                                                                                                val_ratio= 0,
                                                                                                                test_ratio= 0.2)
    
    # obtaining the train and test data in cell mode
    directions_train = [dataset.directions[i] for i in train_indices]
    directions_test = [dataset.directions[i] for i in test_indices]
    times_train = [dataset.times[i] for i in train_indices]
    times_test = [dataset.times[i] for i in test_indices]
    
    train_kfp( directions_train= directions_train, directions_test= directions_test, 
              times_train= times_train, times_test= times_test,
              y_train= y_train, y_test= y_test)