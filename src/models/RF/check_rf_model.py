# a simple code for training the RF model on specified data and checking the results
import utils.config_utils as cm
import numpy as np
from utils.trace_dataset import TraceDataset
import argparse
from os.path import join
from training.config_manager import ConfigManager
from tqdm import tqdm
from models.RF.rf_train import  rf_training_loop, RF_input_processor
from training.train_utils import train_wf_model, stratified_split
from training.train_utils import save_accuracy_plot, plot_train_metrics, plot_model_stats
import os
from utils.parser_utils import str2bool
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training an RF model')

    parser.add_argument('-config', help='which config file to use for the data', default = 'Tik_Tok')
    parser.add_argument('-train_config', help='which config file to use for training the model', default = 'RF_Tik_Tok')
    parser.add_argument('-use_gpu', type=str2bool, nargs='?', const=True, default= True, help='whether gpu should be used for training (if availabe)')
    parser.add_argument('-e', '--extract_ds', help='should we extract the dataset or is it already stored', 
                        default = False)
    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False, help='whether we should save the model')
    parser.add_argument('--cuda_id', default=0, type=int, help='CUDA device ID')
    


    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio") # test ration will be 1 - train - val

    logger = cm.init_logger(name = 'Training an RF model')
    args = parser.parse_args()
    cm.initialize_common_params(args.config)
    
    
    # in case we want to save the model

    model_save_path = os.path.join(cm.BASE_DIR, 'models', 'RF_Original', f'{cm.data_set_folder}')
    if args.save:
        logger.info(f'the model will be saved at {model_save_path}')

    training_config_dir = join(cm.BASE_DIR, 'configs', 'training', args.train_config + '.yaml')
    hyperparam_manager = ConfigManager(config_path= training_config_dir)
    
    training_dataset = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'tam')

    traces = np.array(training_dataset.directions)
    labels = np.array(training_dataset.labels)
    X_train, y_train, X_val, y_val, X_test, y_test, train_indices, val_indices, test_indices = stratified_split(traces, labels,
                                                                                                                train_ratio= args.train_ratio,
                                                                                                                val_ratio= args.val_ratio,
                                                                                                                test_ratio= 1 - args.train_ratio - args.val_ratio)

    
    train_results = train_wf_model(logger= logger, 
                    training_loop= rf_training_loop  ,                                                                                                          
                    num_classes= cm.MON_SITE_NUM,
                    input_processor= RF_input_processor,
                    hyperparam_manager= hyperparam_manager,
                    save_model= args.save,
                    x_train = X_train,
                    y_train = y_train,
                    x_test= X_test,
                    y_test= y_test,
                    x_val= X_val,
                    y_val= y_val,
                    cuda_id= args.cuda_id,
                    use_wandb= False,
                    topk_s= None,
                    class_rankings_val= None,
                    class_rankings_test= None,
                    if_use_gpu= args.use_gpu,
                    report_train_accuracy= True,
                    wf_model_type= 'RF',
                    actual_websites_test= None,
                    model_save_path= model_save_path)
    
    result_dir = os.path.join(cm.BASE_DIR,  'results', 'original models', 'RF', cm.data_set_folder)
    val_accuracy_list = train_results['val_accuracy_list']
    train_losses = train_results['training_losses'] 
    train_accuracies = train_results['training_accuracies'] 
    save_accuracy_plot(accuracies= val_accuracy_list, n = 1, save_path= result_dir)
    plot_train_metrics(training_loss= train_losses, training_accuracy= train_accuracies, save_path= result_dir)
    
    plot_model_stats(train_results= train_results, model_type= 'RF Original', save_path= result_dir)
    
# python3 -m models.RF.check_rf_model