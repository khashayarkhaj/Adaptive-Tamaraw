# train tt on Tik_Tok dataset to see how it works

import utils.config_utils as cm
from time import strftime
import argparse
import numpy as np
import torch
from utils.trace_dataset import TraceDataset
from utils.parser_utils import str2bool
from os.path import join
from training.config_manager import ConfigManager
from .train_utils import tt_input_processor, tt_training_loop
from training.train_utils import train_wf_model, stratified_split
from training.train_utils import save_accuracy_plot, plot_per_class_accuracy, plot_train_metrics, plot_model_stats, plot_website_accuracies
import os
if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser(description='Training an RF model')

    parser.add_argument('-config', help='which config file to use for the data', default = 'Tik_Tok')
    parser.add_argument('-train_config', help='which config file to use for training the model', default = 'TT')
    parser.add_argument('-use_gpu', type = bool, help='whether gpu should be used for training (if availabe)', default = True)
    parser.add_argument('-e', '--extract_ds', help='should we extract the dataset or is it already stored', 
                        default = False)
    parser.add_argument('-save', type = bool, help='whether we should save the model', 
                        default = False)
    parser.add_argument('--cuda_id', default=0, type=int, help='CUDA device ID')
    

    logger = cm.init_logger(name = 'Training an RF model')
    args = parser.parse_args()
    cm.initialize_common_params(args.config)
    
    
    training_config_dir = join(cm.BASE_DIR, 'configs', 'training', args.train_config + '.yaml')
    hyperparam_manager = ConfigManager(config_path= training_config_dir)
    
    training_dataset = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'dt', interval= [0, 0.8])
    test_dataset = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'dt', interval= [0.8, 1])
    X_train = training_dataset.directions
    y_train = training_dataset.labels
    X_test = test_dataset.directions
    y_test = test_dataset.labels
    

    if args.use_gpu and torch.cuda.is_available():
        logger.info('Since you requested for gpu and gpu is availabel, gpu will be used')
        if_use_gpu = 1
    else:
        if_use_gpu = 0
        logger.info('GPU is not available, thus training will be done on CPU')
    
    train_results = train_wf_model(
                                           logger= logger, 
                    training_loop= tt_training_loop  ,                                                                                                          
                    num_classes= cm.MON_SITE_NUM,
                    input_processor= tt_input_processor,
                    hyperparam_manager= hyperparam_manager,
                    save_model= False,
                    x_train = X_train,
                    y_train = y_train,
                    x_test= X_test,
                    y_test= y_test,
                    x_val= X_test,
                    y_val= y_test,
                    cuda_id= args.cuda_id,
                    if_use_gpu= if_use_gpu,
                    report_train_accuracy= True,
                    wf_model_type= 'tt')
    val_accuracy_list = train_results['val_accuracy_list']
    test_accuracy = train_results['test_accuracy']
    train_losses = train_results['training_losses'] 
    train_accuracies = train_results['training_accuracies'] 
    confusion_matrix = train_results['confusion_matrix']
    
    result_dir = os.path.join(cm.BASE_DIR, 'results', 'original_models', 'tik_tok',cm.data_set_folder )
    save_accuracy_plot(accuracies= val_accuracy_list, n = 1, save_path= result_dir, k = args.k, file_name= 'accuracy_val.png')
    plot_train_metrics(training_loss= train_losses, training_accuracy= train_accuracies, save_path= result_dir)
    plot_model_stats(train_results= train_results, model_type= f'{args.classifier}_{args.regulator_config}', save_path= result_dir)

    # python3 -m models.Tik_Tok.temp_train