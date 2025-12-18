import torch
from torch.utils.data import Dataset, DataLoader
from .trace_operations import load_all_traces
import os
import numpy as np
import utils.config_utils as cm
from .burst_extraction import parallel_burst_extraction
from .file_operations import extract_label, create_trace_flist, save_file, load_file, predict_npz_file_size
import pickle
import random
from os.path import join
import glob
from .dataset_utils import to_tam, to_dt, to_taf, load_DF18, convert_ndarray_to_list, efficient_store_traces, reconstruct_efficient
from tqdm import tqdm
import gc

class TraceDataset(Dataset):
    def __init__(self, trace_mode = 'cell', cache_data = True, extract_traces = False, 
                  save_extracted = True, num_workers = 70, logger = None, time_threshold = None,
                  split = 'train', check_validity = False, random_seed = 43, interval = None, 
                  keep_original_trace = False, stored_path = None, stored_file_name = None, 
                  stored_labels_path = None, stored_labels_file = None,
                  load_traces_fast = True, store_traces_numpy = False, trim_length = None):
        """
        Args:
            trace_mode: specifies if we want our data be in form of cells, bursts, or tam (can be 'cell', 'burst', 'dt', 'tam', 'taf')
            cache_data: if True, the data will be stored in a variable for future reference. Otherwise, __getitem__ will 
            read from file every time. 
            extract_traces: if True, then the dataset will read the trace files and extract the information.
                if trace_mode is cell, then the traces and times will be extracted in +1 -1 order in an npz file.
                if trace_mode is burst, then the traces and times will be proccessed to capture the burst format.
            num_workers: number of workers in case we want to do multiprocessing for the extraction part
            time_threshold: threshold we will use to shorten the traces (empirical evaluation of how much information a short trace has about the entire trace)
            split: it can be one of train, val, test, indicating the specific file we want to extract e.g. used in DF18.
            check_validity : whether to analyze the dataset after it has been prepared and report some stats.
            random_seed: random number that will be used whenever we want to perform random actions (used for reproducibility of experiments)
            interval: sometimes we might not want all the data. for instance, we might want only the first 80% traces of a class. interval is expected to be [start, end] where 0<start, end< 1
            keep_original_trace: keep the original trace in self.original_trace 
            stored_path: if we want to preload the data, we can pass the path and file name and it will load it based on the trace mode
            store_traces_numpy: if we are loading the traces like [(t1,d1), ...], we will store them in ndarrays for efficient loading next time.
            load_traces_fast: if true, we will load traces like [(t1,d1), ...] efficiently from their numpy arrays
        """

        #TODO add place for augmentation
        #TODO add split to file paths
        if not logger:
            logger = cm.global_logger
        
        self.logger = logger
        self.trace_mode = trace_mode
        self.directions = None
        self.times = None
        self.data = None
        self.original_traces = None
        self.cache_data = cache_data
        self.trace_mode = trace_mode
        self.random_seed = random_seed
        self.interval = interval
        self.number_of_instances_per_class = cm.MON_INST_NUM # this will change if we just use an interval of the data
        self.tam_length = None # if the traces are in tam or taf format, the length of the trace
        cm.TIME_THRESHOLD_TRACE = time_threshold
        self.time_threshold = time_threshold
        '''
        we have two options for handling the time_threshold:
            1 - loading the entire traces, and enforcing the threshold in get_item
            2 - cutting the traces when they are loaded, so the dataset won't store the entire traces.

            since the dataset will likely be used during training, the first option will probably add computational burden. 
            Thus I opted for the second option.
        '''
        if cache_data:
            if trace_mode in ['cell', 'tam', 'taf', 'dt']:
                if extract_traces:
                    logger.info(f'Extracting traces at {trace_mode} level')
                    if time_threshold:
                        logger.info(f'The traces are trimmed to a time threshold of {time_threshold}')
                    if cm.data_set_format in ['.cell', '']: #we expect a list of .cell files (ds19) or no extenstions (tik tok)
                        data_dirs = create_trace_flist() #list of our trace files based on our config settings
                        results= load_all_traces(data_dirs)
                        
                        traces, labels = zip(*results)
                        traces = list(traces)
                        labels = list(labels)
                        times, directions = zip(*[trace.T for trace in traces])
                        
                        if keep_original_trace:
                            logger.info('The original traces are also stored in self.original_traces')
                            self.original_traces = traces # each original trace is in the form [[t1, d1], [t2, d2] , ...]
                    elif cm.data_set_format == '.pkl': # like DF18. this part is initialy written for df18, maybe I modify it later TODO
                        directions, labels = load_DF18(split= split, logger = logger)
                    elif cm.data_set_format == '.h5': # for QCSD
                        directions, times, labels = load_QCSD(logger= logger)
                        traces = [np.array([t, d]).T.tolist() for t, d in zip(times, directions)]
                        
                    if trace_mode == 'cell':
                        self.directions = list(directions)
                        self.times = list(times)
                        self.labels = labels
                    elif trace_mode == 'tam':
                        logger.info(f'Time slot for tams is {cm.Time_Slot * 1000}ms')
                        self.directions = to_tam(directions = list(directions), times = list(times))
                        self.times = None
                        self.labels = labels
                    elif trace_mode == 'dt':
                        self.directions = to_dt(directions = list(directions), times = list(times))
                        self.times = None
                        self.labels = labels
                    elif trace_mode == 'taf':
                        self.directions = to_taf(directions = list(directions), times = list(times))
                        self.times = None
                        self.labels = labels
                    
                    logger.info('Extraction procedure completed')
                    
                    if save_extracted:
                        self.save_data(traces= traces)

                        
                else:
                    logger.info(f'Loading traces at {trace_mode} level')
                    logger.info(f'The traces have a time threshold of {time_threshold}')

                    if stored_labels_path is None:
                        label_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, cm.data_set_folder + '-celltraces')
                    else:
                        label_dir = stored_labels_path
                    
                    if stored_labels_file is None:
                        label_file = cm.data_set_folder +'_labels.pkl'
                    else:
                        label_file = stored_labels_file
                    
                    labels = load_file(dir_path= label_dir, file_name= label_file)
                    self.labels = labels
                    
                    if trace_mode in ['tam', 'taf']:

                        if stored_path is None:
                            save_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, f'{trace_mode}s')
                        else:
                            save_dir = stored_path

                        if stored_file_name is None:
                            if time_threshold:
                                file_name =  f'{cm.Maximum_Load_Time}_{cm.Time_Slot}_{time_threshold:.2f}.npy'
                            else:
                                file_name =  f'{cm.Maximum_Load_Time}_{cm.Time_Slot}.npy'
                        else:
                            file_name = stored_file_name
                        
                        
                        self.directions = load_file(save_dir, file_name)
                        self.times = None


                       
                    


                    elif trace_mode == 'cell':       
                        if cm.data_set_format in ['.cell', '']:   # datasets like DS19 or Tik-Tok

                            if stored_path is None:
                                dir_path = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, cm.data_set_folder + '-celltraces')
                            else:
                                dir_path = stored_path
                            
                            if stored_file_name is None:
                                if time_threshold:
                                    file_name = cm.data_set_folder +f'_cells_{time_threshold:.2f}.npz'
                                else:
                                    file_name = cm.data_set_folder +'_cells.npz'
                            else:
                                file_name = stored_file_name
                            
                            
                            if not load_traces_fast:
                                traces_dict = load_file(dir_path, file_name) # we expect it to be a dictionay with keys in the form of trace_{i}
                                logger.info('Loading the cell traces the original way')
                                sorted_keys = sorted(traces_dict.files, key = lambda x : int(x.split('_')[1])) # sorting the keys just to make sure we don't mess up the order when loading back
                                traces = [traces_dict[key] for key in tqdm(sorted_keys, desc= 'converting the dictionary to an array')] # change this line to the lines below for optimization. check if they work
                                
                                
                                # #Faster version:
                                # # 1. Extract numbers once instead of repeatedly during sort
                                # numbers_and_keys = [(int(key.split('_')[1]), key) for key in traces_dict.files]
                                # # 2. Sort based on the pre-calculated numbers
                                # numbers_and_keys.sort()  # sorts based on first element of tuple by default
                                # # 3. Extract just the sorted keys and create the final list
                                # traces = [traces_dict[key] for _, key in tqdm(numbers_and_keys, desc= 'converting the dictionary to an array')]



                                #traces we will be in the format of [x,2], where the first column is times and the second is directions
                                logger.info('Traces array is now constructed')
                                if keep_original_trace:
                                    logger.info('The original traces are also stored in self.original_traces')
                                    self.original_traces = traces # each original trace is in the form [[t1, d1], [t2, d2] , ...]
                                

                                times, directions = zip(*[trace.T for trace in traces])
                                self.directions = list(directions)
                                self.times = list(times)
                                # if trim_length is not None:
                                #     self.trim_or_pad_traces(trace_length= trim_length)
                                if store_traces_numpy:
                                    dir_path = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'efficient_cells')
                                   

                                    data_array, metadata = efficient_store_traces(traces, trim_length= trim_length)
                                    save_file(dir_path= dir_path, file_name= 'data_array.npy', content= data_array)
                                    save_file(dir_path= dir_path, file_name= 'meta_data.npy', content= metadata)
                                    
                            else:
                                logger.info('Loading the cell traces the fast way')
                                dir_path = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'efficient_cells')
                            

                                data_array = load_file(dir_path= dir_path, file_name= 'data_array.npy')
                                metadata = load_file(dir_path= dir_path, file_name= 'meta_data.npy')
                                times, directions = reconstruct_efficient(data_array, metadata)
                                if keep_original_trace:
                                    traces_reconstructed = [np.column_stack((t, d))  # shape: (N_i, 2)
                                                            for t, d in zip(times, directions)
                                                                                                ]

                                    logger.info('The original traces are also stored in self.original_traces')
                                    self.original_traces = traces_reconstructed # each original trace is in the form [[t1, d1], [t2, d2] , ...]
                                

                                
                                self.directions = list(directions)
                                self.times = list(times)
                            


                        elif cm.data_set_format == '.pkl': # This is initially written for DF18. maybe it will be adjusted in the future
                            
                            directions, labels = load_DF18(split= split, logger = logger)

                            
                            self.directions = list(directions)
                            self.times = None
                            
                    elif trace_mode == 'dt':
                        if stored_path is None:
                            save_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'dts')
                        else:
                            save_dir = stored_path

                        if stored_file_name is None:
                            if time_threshold:
                                file_name = f'_dts_{time_threshold:.2f}.pkl'
                            else:
                                file_name = 'dts.pkl'
                        else:
                            file_name = stored_file_name
                        
                        self.directions = load_file(save_dir, file_name)
                        self.times = None
                        

                        
                       

                    logger.info('Loading procedure completed')
                
                
                
            
            elif trace_mode == 'burst':
                if extract_traces:
                    logger.info('Extracting traces at burst level')
                    if time_threshold:
                        logger.info(f'The traces are trimmed to a time threshold of {time_threshold}')
                    flist = create_trace_flist()
                    logger.info('In total {} files.'.format(len(flist)))
                    raw_data_dict = parallel_burst_extraction(flist, n_jobs= num_workers)
                    bursts, times, labels, original_burst_sizes, original_trace_sizes, modified_trace_sizes = zip(*raw_data_dict) 
                    """ In summary, this line restructures the data so that all 
                    bursts are grouped together in one tuple, all times are grouped together in another tuple, 
                    and all labels are in a third tuple, ...  """

                    bursts = np.array(bursts)
                    labels = np.array(labels)
                    logger.info("feature sizes:{}, label size:{}".format(bursts.shape, labels.shape))
                    logger.info('Extraction procedure completed')
                    if save_extracted:
                        if not time_threshold:
                            burst_save_file = "raw_feature_{}-{}x{}-{}.npz".format(cm.MON_SITE_START_IND, 
                                                                                cm.MON_SITE_START_IND + cm.MON_SITE_NUM,
                                                                                    cm.MON_INST_START_IND, 
                                                                                    cm.MON_INST_NUM + cm.MON_INST_START_IND)
                        else:
                            burst_save_file = "raw_feature_{}-{}x{}-{}-{}.npz".format(cm.MON_SITE_START_IND, 
                                                                                cm.MON_SITE_START_IND + cm.MON_SITE_NUM,
                                                                                    cm.MON_INST_START_IND, 
                                                                                    cm.MON_INST_NUM + cm.MON_INST_START_IND,
                                                                                    time_threshold)
                        
                        output_folder = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, cm.data_set_folder + '-bursts')

                        save_file(dir_path= output_folder, file_name= burst_save_file, compress = True,
                                  features = bursts, labels = bursts)
                       
                        logger.info("output to {}".format(join(output_folder, "raw_feature_{}-{}x{}-{}.npz".
                                                            format(cm.MON_SITE_START_IND, cm.MON_SITE_START_IND + cm.MON_SITE_NUM,
                                                                    cm.MON_INST_START_IND, cm.MON_INST_NUM + cm.MON_INST_START_IND))))

                        
                        # save the time information. The even indexes are outgoing timestamps and the odd indexes are incoming ones.

                        if not time_threshold:
                            time_file = "time_feature_{}-{}x{}-{}.npz".format(cm.MON_SITE_START_IND, 
                                                                            cm.MON_SITE_START_IND + cm.MON_SITE_NUM, cm.MON_INST_START_IND,
                                            cm.MON_INST_NUM + cm.MON_INST_START_IND)
                        else:
                            time_file = "time_feature_{}-{}x{}-{}-{}.npz".format(cm.MON_SITE_START_IND, 
                                                                            cm.MON_SITE_START_IND + cm.MON_SITE_NUM, cm.MON_INST_START_IND,
                                            cm.MON_INST_NUM + cm.MON_INST_START_IND, time_threshold)

                        
                        save_file(dir_path= output_folder, file_name= time_file, content= times)
                    self.directions = bursts
                    self.times = times
                    self.labels = labels
                else: # bursts have already been extracted
                    logger.info('Loading traces at burst level')
                    logger.info(f'The traces have a time threshold of {time_threshold}')

                    if stored_path is None:
                        output_folder = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, cm.data_set_folder + '-bursts')
                    else:
                        output_folder = stored_path

                    if stored_file_name is None:
                        if not time_threshold:
                                burst_load_file = "raw_feature_{}-{}x{}-{}.npz".format(cm.MON_SITE_START_IND, 
                                                                                    cm.MON_SITE_START_IND + cm.MON_SITE_NUM,
                                                                                        cm.MON_INST_START_IND, 
                                                                                        cm.MON_INST_NUM + cm.MON_INST_START_IND)
                        else:
                                burst_load_file = "raw_feature_{}-{}x{}-{}-{}.npz".format(cm.MON_SITE_START_IND, 
                                                                                    cm.MON_SITE_START_IND + cm.MON_SITE_NUM,
                                                                                        cm.MON_INST_START_IND, 
                                                                                        cm.MON_INST_NUM + cm.MON_INST_START_IND,
                                                                                        time_threshold)
                    else:
                        burst_load_file = stored_file_name
                        
                    burst_npz_file = load_file(dir_path= output_folder, file_name= burst_load_file)
                    

                    # Access the saved arrays using their respective keys
                    self.directions = burst_npz_file['features']
                    self.labels = burst_npz_file['labels']

                    if not time_threshold:
                            times_load_file = "time_feature_{}-{}x{}-{}.npz".format(cm.MON_SITE_START_IND, 
                                                                            cm.MON_SITE_START_IND + cm.MON_SITE_NUM, cm.MON_INST_START_IND,
                                            cm.MON_INST_NUM + cm.MON_INST_START_IND)
                    else:
                            times_load_file = "time_feature_{}-{}x{}-{}-{}.npz".format(cm.MON_SITE_START_IND, 
                                                                            cm.MON_SITE_START_IND + cm.MON_SITE_NUM, cm.MON_INST_START_IND,
                                            cm.MON_INST_NUM + cm.MON_INST_START_IND, time_threshold)
                    time_npz_file = load_file(dir_path= output_folder, file_name = times_load_file)
                    

                    # Initialize an empty list to hold your arrays
                    self.times = []

                    # Assuming the naming convention is 'arr_0', 'arr_1', etc.
                    for i in range(len(time_npz_file.files)):
                        self.times.append(time_npz_file['arr_{}'.format(i)])
                    
                    logger.info('Loading procedure completed')
            
            if interval and not (interval[0] == 0 and interval[1] == 1):
                        

                        # for each class, we want only the mentioned interval to be used

                        #first we make sure all the data are lists so it will be easier to work with them
                        self.directions = convert_ndarray_to_list(self.directions)
                        self.labels = convert_ndarray_to_list(self.labels)
                        if self.times:
                            self.times = convert_ndarray_to_list(self.times)

                        
                        trimmed_directions = []
                        trimmed_times = []
                        trimmed_labels = [] 
                        for class_number in tqdm(range(cm.MON_SITE_NUM), desc = 'trimming the trace intervals from each website'):
                            start_idx = class_number * cm.MON_INST_NUM # start and end index of the desired class
                            end_idx = start_idx + cm.MON_INST_NUM
                            

                            #choosing the traces and labels that belong to this specific class
                            traces_of_this_class = self.directions[start_idx : end_idx]
                            labels_of_this_class = self.labels[start_idx: end_idx]
                            number_of_traces = len(traces_of_this_class)

                            #selecting the traces and labels in the given interval
                            start_of_interval = int(interval[0] * number_of_traces)
                            end_of_interval = int(interval[1] * number_of_traces)
                            self.number_of_instances_per_class = end_of_interval - start_of_interval # this is not clean but it works!
                            
                            trimmed_directions += traces_of_this_class[start_of_interval : end_of_interval]
                            trimmed_labels += labels_of_this_class[start_of_interval: end_of_interval]

                            #selecting the times in the given interval, if we have timing
                            if self.times:
                                times_of_this_class = self.times[start_idx : end_idx]
                                trimmed_times += times_of_this_class[start_of_interval: end_of_interval]

                        if trace_mode in ['burst', 'tam', 'taf']: # in this case, the data should be ndarrays
                            trimmed_directions = np.array(trimmed_directions)
                            trimmed_labels = np.array(trimmed_labels)
                            if self.times:
                                trimmed_times = np.array(trimmed_times)

                        #replacing the data with the trimmed data
                        self.directions = trimmed_directions
                        self.labels = trimmed_labels
                        if self.times:
                            self.times = trimmed_times
            if trace_mode in ['tam', 'taf']:
                self.tam_length = self.directions[0].shape[-1]        
            

                    
        else:
            pass #TODO what if we don't want to cache the data?
        
        #since this is a shared parameter, it would be better to restore it to default. TODO: think about a better design
        cm.TIME_THRESHOLD_TRACE = None
        if check_validity:
            self.check_ds_validity()
        ######## saving only the websites that we want

        ### find first occurrence of cm.MONITORED_SITE_START_IND
        first_index = None
        for i, label in enumerate(self.labels):
            if label == cm.MON_SITE_START_IND:
                first_index = i
                break

        ### find last occurrence of cm.MONITORED_SITE_NUM
        last_index = None
        for i in range(len(self.labels) - 1, -1, -1):
            if self.labels[i] == cm.MON_SITE_NUM - 1:
                last_index = i
                break

        # Only proceed if both indices were found
        if first_index is not None and last_index is not None:
            self.directions = self.directions[first_index: last_index + 1]
            self.labels = self.labels[first_index: last_index + 1]
            
            if self.times is not None:
                self.times = self.times[first_index: last_index + 1]
            if self.original_traces is not None:
                self.original_traces = self.original_traces[first_index: last_index + 1]
        else:
            # Handle case where the range is not found
            print("Warning: Could not find the specified label range")
        

    def __len__(self):
        # Return the total number of samples in the dataset
        if self.cache_data:
            return len(self.directions)
        else:
            if not self.interval:
                return cm.MON_SITE_NUM * cm.MON_INST_NUM
            else:
                interval_length = self.interval[1] - self.interval[0]
                return int(cm.MON_SITE_NUM * cm.MON_INST_NUM * interval_length)

    

    def __getitem__(self, idx):
        # Implement logic to get a single data point from the dataset

        if self.cache_data:
            if self.times:
                return self.directions[idx], self.times[idx], self.labels[idx]
            else:
                return self.directions[idx], None, self.labels[idx]
        else:
            pass #TODO handle case where we haven't stored the data
    def save_data(self, file_name = None, traces = None, save_dir = None, label_dir = None, label_file = None):

        self.logger.info(f'Saving the dataset in {self.trace_mode} format')
        if self.trace_mode == 'cell': # we need the original traces as well if traces is None
            if traces is None:
                traces = self.original_traces
            # saving the trace (directions and time)
            traces_dict = {f'trace_{i}': trace for i, trace in enumerate(traces)}
            if save_dir is None:
                save_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, cm.data_set_folder + '-celltraces')
            
            if file_name is None:
                if self.time_threshold is not None:
                    file_name = cm.data_set_folder +f'_cells_{self.time_threshold:.2f}.npz'
                else:
                    file_name = cm.data_set_folder +'_cells.npz'
            #save_file(dir_path= save_dir, file_name= file_name, content= traces)
            save_file(dir_path= save_dir, file_name= file_name, content= traces_dict)
        elif self.trace_mode == 'tam':
            # saving the tam
            if save_dir is None:
                save_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'tams')

            if file_name is None:
                if self.time_threshold is not None:
                    file_name =  f'{cm.Maximum_Load_Time}_{cm.Time_Slot}_{self.time_threshold:.2f}.npy'
                else:
                    file_name =  f'{cm.Maximum_Load_Time}_{cm.Time_Slot}.npy'
            save_file(dir_path= save_dir, file_name= file_name, content= self.directions)

        elif self.trace_mode == 'taf':
            # saving the tam
            if save_dir is None:
                save_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'tafs')
            if file_name is None:
                if self.time_threshold is not None:
                    file_name =  f'{cm.Maximum_Load_Time}_{cm.Time_Slot}_{self.time_threshold:.2f}.npy'
                else:
                    file_name =  f'{cm.Maximum_Load_Time}_{cm.Time_Slot}.npy'
            save_file(dir_path= save_dir, file_name= file_name, content= self.directions)

        elif self.trace_mode == 'dt':
            if save_dir is None:
                save_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'dts')
            
            if file_name is None:
                if self.time_threshold is not None:
                    file_name = f'_dts_{self.time_threshold:.2f}.pkl'
                else:
                    file_name = 'dts.pkl'
                save_file(dir_path= save_dir, file_name= file_name, content= self.directions)

    

        # saving the labels
        
        if label_dir is None:
            label_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, cm.data_set_folder + '-celltraces')
        if label_file is None:
            label_file = cm.data_set_folder +'_labels.pkl'
        save_file(dir_path= label_dir, file_name= label_file, content= self.labels)
        
    def get_traces_of_class(self, class_number, return_indices = False):
        #returns the traces of a specific class. based on the dataset, we might have time values as well.
        
        
        start_idx = class_number * self.number_of_instances_per_class
        end_idx = start_idx + self.number_of_instances_per_class
        returned_indices = np.array([i for i in range(start_idx, end_idx)]) # the real indexes of each trace TODO modify this for when trace numbers of each class are not fixed
        
        returned_times = None
        if self.times is not None:
            returned_times = self.times[start_idx : end_idx]
        
        if not return_indices:
            return self.directions[start_idx : end_idx], returned_times
        else:
            return self.directions[start_idx : end_idx], returned_times, returned_indices
        #TODO handle the case where we have non-moitored class as well
    
    def get_indices_of_a_class(self, class_number):
        # get the start and end index of the traces of a class
        # TODO modify this when variable length classes are added
        start_idx = class_number * self.number_of_instances_per_class
        end_idx = start_idx + self.number_of_instances_per_class

        return start_idx, end_idx
    
    def get_real_indice_of_a_trace(self, website_idx, instance_idx): 
        # for instance, we want to see what the real index of the 5th trace of website 10 is
        # TODO modify this when variable length classes are added
        start_idx = website_idx * self.number_of_instances_per_class
        return instance_idx + start_idx
    

    def sample_randomly(self, sample_nums = 1, class_num = None, return_indices = False, use_random_seed = True):
        # samples randomly a number of traces from the dataset and returns them. based on the dataset, we might have time values as well.
        #if class num is given, we will only return random samples from the given class
        #if return indices is true, we will also return the indices that were chosen randomly
        # if use_random_seed, the datasets random seed will be used for reproducibility. otherwise, we will use a random number each time.
        
        if use_random_seed:
            # Save the current random state. Since the random module is global, we will revert back to this state at the end of the function so that th res of the code won't be effected by our seed.
            random_state_before = random.getstate()
            random.seed(self.random_seed) # setting the seed so that we can reproduce the experiments

        sampled_times = None # if our data has time as well, we will store it in this variable

        if self.trace_mode in ['cell', 'tam', 'taf']: # the data is a list
            if not class_num: # we want to sample from all classes
                sampled_indices = random.sample(range(self.__len__()), sample_nums)
                sampled_traces = [self.directions[i] for i in sampled_indices]
                if self.times:
                    sampled_times = [self.times[i] for i in sampled_indices]
                
                sampled_labels = [self.labels[i] for i in sampled_indices]
            else: # we only want samples of a certain class
                directions, times = self.get_traces_of_class(class_num)
                sampled_indices = random.sample(range(len(directions)), sample_nums)
                sampled_traces = [directions[i] for i in sampled_indices]
                if self.times:
                    sampled_times = [times[i] for i in sampled_indices]
                
                sampled_labels = [class_num for i in sampled_indices]
        else: # the data is an ndarray
            if not class_num: # we want to sample from all classes
                sampled_indices = np.random.choice(self.__len__(), size=sample_nums, replace=False)
                sampled_traces = self.directions[sampled_indices]
                if self.times:
                    sampled_times = self.times[sampled_indices]
                sampled_labels = self.labels[sampled_indices]
            else:
                directions, times = self.get_traces_of_class(class_num)
                sampled_indices = np.random.choice(len(directions), size=sample_nums, replace=False)
                sampled_traces = directions[sampled_indices]
                if times:
                    sampled_times = times[sampled_indices]
                sampled_labels = [class_num for i in sampled_indices]

        if use_random_seed:
            # Restore the original random state after operation so that the rest of the code won't be effected
            random.setstate(random_state_before)
        if return_indices:
            return sampled_traces, sampled_times, sampled_labels, sampled_indices
        return sampled_traces, sampled_times, sampled_labels
    
    def check_ds_validity(self):
        # checks some specific attributes that the traces should have
        #TODO this function will get updated gradually

        # 1 - checking if some traces start with an incomming packet
        if self.trace_mode != 'tam':
            number_of_incomming_starters = 0
            all_traces = self.__len__()
            self.logger.info(f'Evaluating the {cm.data_set_folder} dataset...')
            for direction in tqdm(self.directions, desc= 'Checking if traces do not start with an incomming cell'):
                if direction[0] < 0:
                    number_of_incomming_starters += 1
            
            self.logger.info(f'Number of traces starting with incomming packets: {number_of_incomming_starters}')
            self.logger.info(f'Percentage of traces starting with incomming packets in the dataset: {number_of_incomming_starters/all_traces}')
       

    def trim_or_pad_traces(self, trace_length=5000):
        for idx in tqdm(range(len(self.directions)), f'trimming or padding traces to length {trace_length}'):  # Use range to get the index for iteration
            # Trimming excess length
            if len(self.directions[idx]) > trace_length:
                self.directions[idx] = self.directions[idx][:trace_length]
                if self.times is not None:
                    self.times[idx] = self.times[idx][:trace_length]

            # Padding shorter lengths
            elif len(self.directions[idx]) < trace_length:
                
                # Calculate the amount to pad
                pad_length = trace_length - len(self.directions[idx])
            
                # Padding directions with zeros
                if isinstance(self.directions[idx], list):
                    self.directions[idx].extend([0] * pad_length)  # Extend the list with zeros
                elif isinstance(self.directions[idx], np.ndarray):
                    self.directions[idx] = np.pad(self.directions[idx], (0, pad_length), 'constant')
                    
                
                # Padding times by repeating the last time
                if self.times is not None:
                    last_time = self.times[idx][-1] if len(self.times[idx]) > 0 else 0  # Get the last time or use 0 if empty
                    if isinstance(self.times[idx], list):
                        self.times[idx].extend([last_time] * pad_length)
                    elif isinstance(self.times[idx], np.ndarray):
                        self.times[idx] = np.pad(self.times[idx], (0, pad_length), 'constant', constant_values=last_time)
    
    def to_numpy(self):
        # Convert directions to numpy array
        self.directions = np.array(self.directions)

        # Convert times to numpy array if it is not None
        if self.times is not None:
            self.times = np.array(self.times)

        # Convert labels to numpy array
        self.labels = np.array(self.labels)
    
    def replace_negative_ones(self, new_value=2):
        """
        Replaces all instances of -1 in each sublist or ndarray of self.directions with `new_value`.
        This function is designed for use with a wf transformer, which utilizes an embedding layer
        that requires non-negative integers for inputs. Negative values like -1 could lead to issues
        with embeddings, hence the need for this replacement.
        """
        for i in range(len(self.directions)):
            if isinstance(self.directions[i], list):
                # Replace -1 in list
                self.directions[i] = [new_value if x == -1 else x for x in self.directions[i]]
            elif isinstance(self.directions[i], np.ndarray):
                # Replace -1 in ndarray
                self.directions[i][self.directions[i] == -1] = new_value


    

if __name__ == '__main__':

    
    cm.initialize_common_params('Tik_Tok')
    # extracting the traces the first time 
    ds = TraceDataset(extract_traces=  True, trace_mode= 'cell', store_traces_numpy= True) # extracts and saves the Tik_Tok traces efficiently
    del ds
    gc.collect()

    # next time for building the ds we won't need to extract again
    ds = TraceDataset(extract_traces=  False, trace_mode= 'cell') 
    del ds
    gc.collect()


    ### extracting in other formats
    ds = TraceDataset(extract_traces=  True, trace_mode= 'tam')
    del ds
    gc.collect()
   
    ds = TraceDataset(extract_traces=  True, trace_mode= 'dt')
    



    #### python3 -m utils.trace_dataset


    