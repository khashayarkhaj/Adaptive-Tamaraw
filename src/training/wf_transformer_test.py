# testing if wf transformer works well on df18

import utils.config_utils as cm
from utils.trace_dataset import TraceDataset
from models.WF_Transformer.common_layer import StepOpt, LabelSmoothing
from models.WF_Transformer.UTransformer import UTransformer
from models.WF_Transformer.wf_transformer_train import train_wft
from models.WF_Transformer.wf_transformer_train import wft_training_loop
from models.WF_Transformer.wf_transform_utils import WFT_input_processor
from training.train_utils import train_wf_model, stratified_split
import torch
from torch import nn
import numpy as np
import random
from training.config_manager import ConfigManager
from os.path import join
import argparse
from training.train_utils import save_accuracy_plot, plot_per_class_accuracy, plot_train_metrics, plot_model_stats, plot_website_accuracies
import os
from utils.parser_utils import str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training an RF model')

    parser.add_argument('-config', help='which config file to use for the data', default = 'DF18')
    parser.add_argument('-train_config', help='which config file to use for training the model', default = 'WFT_DF')
    parser.add_argument('-use_gpu', type = bool, help='whether gpu should be used for training (if availabe)', default = True)
    parser.add_argument('-e', '--extract_ds', help='should we extract the dataset or is it already stored', 
                        default = False)
    parser.add_argument('-save', type=str2bool, nargs='?', const=True, default=False, help='whether we should save the model')
    parser.add_argument('--cuda_id', default=0, type=int, help='CUDA device ID')
    parser.add_argument('-cc', '--compute_canada', type=str2bool, nargs='?', const=True, default= True,
                         help='Whether we are using compute canada')


    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio") # test ration will be 1 - train - val

    seed = 0#buneng 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


    args = parser.parse_args()
    cm.initialize_common_params(args.config)
    
    logger = cm.init_logger(name = 'Train WF Transformer on given dataset')
    is_gpu = torch.cuda.is_available()
    gpu_nums = torch.cuda.device_count()
    logger.info(f'Do we have gpu: {is_gpu}')
    logger.info(f'Number of gpus {gpu_nums}')

    cm.compute_canada = args.compute_canada

    if args.config == 'DF18':
        ds_train = TraceDataset(extract_traces= False, trace_mode= 'cell', split= 'train')
        ds_train.trim_or_pad_traces(trace_length= cm.trace_length) #probably 5000
        ds_train.replace_negative_ones(2)
        ds_train.to_numpy()
        X_train,Y_train = ds_train.directions, ds_train.labels


        ds_val = TraceDataset(extract_traces= False, trace_mode= 'cell', split= 'val')
        ds_val.trim_or_pad_traces(trace_length= cm.trace_length) #probably 5000
        ds_val.replace_negative_ones(2)
        ds_val.to_numpy()
        X_val,Y_val = ds_val.directions, ds_val.labels

        print('X_train shape {0} x_val.shape:{1}'.format(X_train.shape,X_val.shape))
        print('y_train shape {0} y_val.shape:{1}'.format(Y_train.shape,Y_val.shape))

        ds_test = TraceDataset(extract_traces= False, trace_mode= 'cell', split= 'test')
        ds_test.trim_or_pad_traces(trace_length= cm.trace_length) #probably 5000
        ds_test.replace_negative_ones(2)
        ds_test.to_numpy()
        X_test,Y_test = ds_test.directions, ds_test.labels

    else:
        training_dataset = TraceDataset(extract_traces= args.extract_ds, trace_mode= 'cell')
        training_dataset.trim_or_pad_traces(trace_length= cm.trace_length)
        training_dataset.replace_negative_ones(2)
        traces = np.array(training_dataset.directions)
        labels = np.array(training_dataset.labels)
        X_train, Y_train, X_val, Y_val, X_test, Y_test, train_indices, val_indices, test_indices = stratified_split(traces, labels,
                                                                                                                    train_ratio= args.train_ratio,
                                                                                                                    val_ratio= args.val_ratio,
                                                                                                                    test_ratio= 1 - args.train_ratio - args.val_ratio)
    training_config_dir = join(cm.BASE_DIR, 'configs', 'training', args.train_config + '.yaml')
    hyperparam_manager = ConfigManager(config_path= training_config_dir)
    # batchsize = hyperparam_manager.get('train.batch_size')
    # trainepoch = hyperparam_manager.get('train.num_epochs')
    classes = cm.MON_SITE_NUM
    lens = cm.trace_length

    wft_model = UTransformer(num_vocab=3, 
                        embedding_size=128, 
                        hidden_size=1024, 
                        num_layers=1,
                        num_heads=1, 
                        total_key_depth=512, 
                        total_value_depth=512,
                        filter_size=512,
                        classes=classes,
                        lens=lens,
                        input_dropout=0.0,
                        layer_dropout=0.0, 
                        attention_dropout=0.1,
                        relu_dropout=0.1)
    # criterion = LabelSmoothing(smoothing=0.05)
    # optimizer = StepOpt(len(X_train),batchsize, torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9,0.999), weight_decay=5e-2, amsgrad = True))
    # train_wft(model,criterion,optimizer,X_train,Y_train,X_val,Y_val,epoch=trainepoch,batchsize=batchsize,classes=classes,gpu=is_gpu, logger= logger)

    train_results = train_wf_model(logger= logger, 
                                wf_model= wft_model,
                        training_loop= wft_training_loop,                                                                                                            
                        num_classes= cm.MON_SITE_NUM,
                        input_processor= WFT_input_processor,
                        hyperparam_manager= hyperparam_manager,
                        save_model= False,
                        x_train = X_train,
                        y_train = Y_train,
                        x_test= X_test,
                        y_test= Y_test,
                        x_val= X_val,
                        y_val= Y_val,
                        cuda_id= None,
                        use_wandb= False,
                        topk_s= None,
                        class_rankings_val= None,
                        class_rankings_test= None,
                        if_use_gpu= True,
                        wf_model_type= 'WFT',
                        convert_data_to_torch= True,
                        report_train_accuracy= True)
    result_dir = os.path.join(cm.BASE_DIR,  'results', 'original models', 'WFT', cm.data_set_folder)
    val_accuracy_list = train_results['val_accuracy_list']
    train_losses = train_results['training_losses'] 
    train_accuracies = train_results['training_accuracies'] 
    save_accuracy_plot(accuracies= val_accuracy_list, n = 1, save_path= result_dir)
    plot_train_metrics(training_loss= train_losses, training_accuracy= train_accuracies, save_path= result_dir)
    
    plot_model_stats(train_results= train_results, model_type= 'WFT Original', save_path= result_dir)
#python3 -m training.wf_transformer_test
#python3 -m training.wf_transformer_test -config Tik_Tok