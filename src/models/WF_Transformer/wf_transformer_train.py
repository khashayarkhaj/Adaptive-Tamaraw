# Function to train WF transformer and evaluate it
import torch.utils.data as Data
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import time
from models.WF_Transformer.common_layer import StepOpt, LabelSmoothing
from models.WF_Transformer.wf_transform_utils import WFT_input_processor
import utils.config_utils as cm
from training.loss_functions import MultitaskLoss

def train_wft(model,criterion,optimizer,X_train,Y_train,X_valid,Y_valid,epoch,batchsize,classes,logger, gpu=True):

    logger.info(f'if use gpu: {gpu}')
    logger.info(f'if cuda is available: {torch.cuda.is_available()}')
    lens = X_train.shape[1]

    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    X_valid = torch.tensor(X_valid)
    Y_valid = torch.tensor(Y_valid)


    traindata = Data.TensorDataset(X_train, Y_train)
    train = Data.DataLoader(traindata, batch_size=batchsize, shuffle=False)
    validdata = Data.TensorDataset(X_valid, Y_valid)
    valid = Data.DataLoader(validdata, batch_size=batchsize, shuffle=False)
    if gpu and torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    logger.info(f'The model is being trained with {next(model.parameters()).device}')
    start_time = time.time()
    for enum in range(0, epoch):
        logger.info(f"WF Transformer train epoch: {enum}/{epoch}")
        epoch_start_time = time.time()  # Start time of the epoch
        e = 0
        eacc = 0
        model.train()
        batch_losses = []
        for i, (x_data,y_data) in enumerate(tqdm(train, desc= f'Going through all {len(train)} batches for epoch {enum}')):
            if gpu:
                x_data = x_data.cuda()
                y_data = y_data.cuda()
            x = x_data.view([-1,lens,1]).long()
            y = y_data.view([-1]).long()
            output = model(x)
            loss=criterion(output,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            predict = nn.functional.softmax(output, dim=1)
            output = torch.max(predict, 1)[1]
            acc = torch.sum(torch.eq(output,y))
            eacc = eacc+acc
            e = e + len(y)
            batch_losses.append(loss.item())
        
        epoch_loss = sum(batch_losses) / len(batch_losses)
        train_accuracy = eacc/e
        
        logger.info("Train Accuracy:"+str(eacc/e))
        with torch.no_grad(): # TODO replace this with your own validation code
            val_e = 0
            val_eacc = 0
            model.eval()
            for i, (x_data,y_data) in enumerate(tqdm(valid, desc= 'validating WFT Transfomer')):
                if gpu:
                    x_data = x_data.cuda()
                    y_data = y_data.cuda()
                x = x_data.view([-1,lens,1]).long()
                y = y_data.view([-1]).long()
                output = model(x)
                predict = nn.functional.softmax(output, dim=1)
                output = torch.max(predict, 1)[1]
                acc = torch.sum(torch.eq(output,y))
                val_eacc = val_eacc+acc
                val_e = val_e + len(y)
        epoch_end_time = time.time()  # End time of the epoch
        epoch_duration = epoch_end_time - epoch_start_time  # Calculate duration
        val_accuracy = val_eacc/val_e
        logger.info(f'Epoch: {enum + 1}, epoch loss: {epoch_loss} , train accuracy: {train_accuracy}, val accuracy: {val_accuracy}, epoch duration: {epoch_duration:.2f}s')
    total_time = time.time() - start_time

    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("Training of WF Transformer completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def wft_training_loop(**kwargs):
        
        #unpacking the arguments passed by layers above
        model = kwargs['wf_model']
        gpu = kwargs['if_use_gpu']
        train_loader = kwargs['train_loader']
        current_epoch = kwargs['current_epoch']
        optimizer = kwargs['optimizer']
        input_processor = kwargs['input_processor'] if 'input_processor' in kwargs.keys() else None
        lens = cm.trace_length
        # using the tools that WF_Transformer used
        

        multi_task = kwargs['multi_task']
        if not multi_task:
            criterion = LabelSmoothing(smoothing=0.05)
        else:
            model.enable_multi_task()
            hyperparam_manager = kwargs['hyperparam_manager']
            main_task_weight = None
            if hyperparam_manager is not None:
                main_task_weight = hyperparam_manager.get('train.mt_weight')
            if main_task_weight:
                criterion = MultitaskLoss(criterion = LabelSmoothing(smoothing=0.05), alpha= main_task_weight)
            else:
                criterion = MultitaskLoss(criterion = LabelSmoothing(smoothing=0.05))
        model.train()
        batch_losses = []
        e = 0
        eacc = 0
        for i, training_resources in enumerate(tqdm(train_loader, desc= f'Going through all {len(train_loader)} batches for epoch {current_epoch}')):
            
            if multi_task:
                x_data,y_data, y2_data = training_resources
            else:
                x_data,y_data = training_resources

            if gpu:
                x_data = x_data.cuda()
                y_data = y_data.cuda()
                if multi_task:
                    y2_data = y2_data.cuda()
                    
            if input_processor is not None:
                x,y = input_processor(x_data, y_data)
                if multi_task:
                    _, y2 = input_processor(None, y2_data)
            else:
                x = x_data
                y = y_data
                if multi_task:
                    y2 = y2_data

            if multi_task:
                output, output2 = model(x)
                loss = criterion(output, output2, y, y2)
            else:        
                output = model(x)
                loss=criterion(output,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            predict = nn.functional.softmax(output, dim=1)
            output = torch.max(predict, 1)[1]
            acc = torch.sum(torch.eq(output,y))
            eacc = eacc+acc.item()
            e = e + len(y)
            batch_losses.append(loss.item())
        
        training_stats = {}
        training_stats['batch_losses'] = batch_losses
        train_accuracy = eacc/e
        #training_stats['train_accuracy'] = train_accuracy don't return this to check actual train accuracy of wft according to my code
        
        if multi_task:
            model.disable_multi_task()# the multi_task_enabled will be set again to true whenever needed
        return training_stats