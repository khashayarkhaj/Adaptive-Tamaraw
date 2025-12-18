#visualize the tams or super matrixes of different websites
import argparse
import utils.config_utils as cm
from utils.visualization_utils import visualize_tam
from utils.trace_dataset import TraceDataset
import os
from utils.file_operations import load_file, save_file
from utils.trace_operations import compute_super_matrix
from utils.parser_utils import str2bool
from tqdm import tqdm


if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser(description='Visualizing Tams')
    parser.add_argument('-e',  '--extract_ds', type=str2bool, nargs='?', const=True, default=False, 
                         help='should we extract the dataset or is it already stored')
        
    parser.add_argument('-conf', '--config', help='which config file to use', default = 'default')

    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to save the computed elements')
    
    parser.add_argument('-preload', type = str2bool, nargs='?', const=True, default=False,
                         help='Whether to load the pre-computed elements')
    
    parser.add_argument('-super', type=str2bool, nargs='?', const=True, default=False,
                         help='Whether to do visualize supermatrixes or normal traces')
    
    parser.add_argument('-min_ds_interval', type = float,
                         help='the lower bound of the dataset partition we want to cluster', default = 0)
    
    parser.add_argument('-max_ds_interval', type = float,
                         help='the upper bound of the dataset partition we want to cluster', default = 1)
    parser.add_argument('-class_num', type = int,
                         help='Which class to visualize', default = -1) # if -1, we will visualize all classes
    
    parser.add_argument('-class_num2', type = int,
                         help='if we want to compare two classes', default = -1) # if -1, we won't visualize it
    
    parser.add_argument('-sample_num', type = int,
                         help='how many samples we want to viusalize for each class', default = 1) 
    
    parser.add_argument('-super_end', type=str2bool, nargs='?', const=True, default=False,
                         help='optionally show the super matrix of the desired class at the end as well')
    
    
    
    logger = cm.init_logger(name = 'Vizualizing Tams')
    args = parser.parse_args()

    cm.initialize_common_params(args.config)
    cm.compute_canada = args.compute_canada
    class_num = args.class_num
    class_num2 = args.class_num2
    

    if args.super or args.super_end: # computing/loading the super matrixes
        super_matrices = []
        if not args.preload: # computing the super matrixes
                dataset_interval = [args.min_ds_interval, args.max_ds_interval]
                dataset = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'tam', interval= dataset_interval)
                for i in tqdm(range(cm.MON_SITE_NUM), f'Computing the super matrixes for {cm.MON_SITE_NUM} clasees'):
                        tams,_ = dataset.get_traces_of_class(class_number = i)
                        
                        super_matrix = compute_super_matrix(*tams)
                        super_matrices.append(super_matrix)
                
                if args.save:
                    save_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'super_matrices')
                    save_file(dir_path= save_dir, file_name= f'super_matrices_{cm.Time_Slot}.pkl', content= super_matrices)
        else: # loading the super matrixes
            load_dir = os.path.join(cm.BASE_DIR, 'data', cm.data_set_folder, 'super_matrices')
            super_matrices = load_file(dir_path= load_dir, file_name= f'super_matrices_{cm.Time_Slot}.pkl', logger=logger)
        
        
    if args.super:
        #visualizing each of the super matrices and storing them
        save_dir = None
        if args.save:
            save_dir = os.path.join(cm.BASE_DIR, 'results', 'dataset-statistics', cm.data_set_folder, 'visualization', 'tams', 'super_matrices')
        if args.class_num == -1: # visualize all super matrices
            logger.info(f'Visualizing the super matrixes of {len(super_matrices)} websites')
            for idx, super_matrix in enumerate(super_matrices):
                visualize_tam(tams = [super_matrix], website_index= idx, super_matrix= True, save_dir= save_dir)
        else:
            if class_num2 == -1:
                logger.info(f'Visualizing the super matrix of website number {class_num}')
                visualize_tam(tams = [super_matrices[class_num]], website_index= class_num, super_matrix= True, save_dir= save_dir)
            else:
                logger.info(f'Visualizing the super matrices of websites {class_num} vs {class_num2}')
                visualize_tam(tams = [super_matrices[class_num], super_matrices[class_num2]], 
                              website_index= [class_num, class_num2], super_matrix= True, save_dir= save_dir)

    else:
        dataset_interval = [args.min_ds_interval, args.max_ds_interval]
        dataset = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'tam', interval= dataset_interval)
        save_dir = None
        if args.save:
                save_dir = os.path.join(cm.BASE_DIR, 'results', 'dataset-statistics', cm.data_set_folder, 'visualization', 'tams', 'random_traces')
        
        if args.class_num == -1: # visualize a random trace for each website
              logger.info(f'Visualizing {args.sample_num} random tam(s) for each of {cm.MON_INST_START_IND - cm.MON_SITE_START_IND} the websites')
              for website_idx in range(cm.MON_SITE_START_IND, cm.MON_INST_START_IND):
                    for i in range(args.sample_num):
                        random_tam, _, _, indices_list = dataset.sample_randomly(class_num= website_idx,
                                                                                sample_nums= 1,
                                                                                return_indices= True,
                                                                                use_random_seed= False)
                    
                        visualize_tam(tams = [random_tam], website_index= website_idx,
                                    super_matrix= False,
                                    save_dir= save_dir,
                                    trace_num = indices_list[0])
        else:
            if class_num2 == -1:
                logger.info(f'Visualizing {args.sample_num} random tam(s) from websites number {class_num}')

                
                
                for i in range(args.sample_num):
                    random_tams, _, _, indices_list = dataset.sample_randomly(class_num= class_num,
                                                                            sample_nums= 1,
                                                                            return_indices= True, 
                                                                            use_random_seed= False)
                    
                    visualize_tam(tams = [random_tams[0]], website_index= class_num,
                                    super_matrix= False,
                                    save_dir= save_dir,
                                    trace_num = indices_list[0])
                if args.super_end:
                    visualize_tam(tams = [super_matrices[class_num]], website_index= class_num, super_matrix= True, save_dir= save_dir)

            else:
                logger.info(f'Visualizing {args.sample_num} random tam(s) from websites {class_num} vs {class_num2}')

                
                
                for i in range(args.sample_num):
                    random_tams1, _, _, indices_list1 = dataset.sample_randomly(class_num= class_num,
                                                                            sample_nums= 1,
                                                                            return_indices= True, 
                                                                            use_random_seed= False)
                    
                    random_tams2, _, _, indices_list2 = dataset.sample_randomly(class_num= class_num2,
                                                                            sample_nums= 1,
                                                                            return_indices= True, 
                                                                            use_random_seed= False)
                    
                    # directions, times = dataset.get_traces_of_class(class_num)
                    # random_tams1 = [directions[817]]
                    # random_tams2 = [directions[247]]
                    print(f'instances number {i}')
                    print(f'{indices_list1[0]} vs {indices_list2[0]}')
                    print("#######################")
                    visualize_tam(tams = [random_tams1[0], random_tams2[0]], website_index= [class_num, class_num2],
                                    super_matrix= False,
                                    save_dir= save_dir,
                                    trace_num = [indices_list1[0], indices_list2[0]])
                    
                    
                if args.super_end:
                    visualize_tam(tams = [super_matrices[class_num], super_matrices[class_num2]], website_index= [class_num, class_num2], 
                                  super_matrix= True, save_dir= save_dir)


    