# encoding: utf8
# the code for training an RF model. This file has been slightly changed to be compatible to this codebase
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import os
from models.RF.RF_model import getRF
from models.RF.rf_utils import RF_input_processor
import utils.config_utils as cm
from utils.trace_dataset import TraceDataset
import argparse
from os.path import join
from training.config_manager import ConfigManager
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import time
from ray import train, tune
# from training.train_utils import evaluate_model
from utils.parser_utils import str2bool
from training.loss_functions import MultitaskLoss


def adjust_learning_rate(optimizer, echo, learning_rate, epoch, annealing_rate = 0.2):
    lr = learning_rate * (annealing_rate ** (echo / epoch)) # this was initially 0.2
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr



def rf_training_loop(wf_model, device, train_loader, 
                     scheduler_method, optimizer, learning_rate, 
                     current_epoch, total_epochs, if_use_gpu = True, input_processor = None, annealing_rate = 0.2, **kwargs):
    wf_model.to(device)
    wf_model.train()
    # if we want to do multi tasking, we expect kwargs to have multi_task and also hyperparam_manager to get the weights of tasks
    multi_task = kwargs['multi_task']
    if not multi_task:
        loss_func = nn.CrossEntropyLoss()
    else:
        wf_model.enable_multi_task()
        hyperparam_manager = kwargs['hyperparam_manager']
        main_task_weight = None
        if hyperparam_manager is not None:
            main_task_weight = hyperparam_manager.get('train.mt_weight')
        
        if main_task_weight:
            loss_func = MultitaskLoss(criterion = nn.CrossEntropyLoss(), alpha= main_task_weight)
        else:
            loss_func = MultitaskLoss(criterion = nn.CrossEntropyLoss())
    
    batch_losses = []

    if scheduler_method == 'default': # setting the default learning rate scheduler in rf
        adjust_learning_rate(optimizer, echo= current_epoch, learning_rate= learning_rate, epoch= total_epochs, annealing_rate= annealing_rate)
    
    tqdm_disabled = False
    if 'training_loop_tqdm' in kwargs.keys():
        tqdm_disabled = not kwargs['training_loop_tqdm']
    for step, training_resources in enumerate(tqdm(train_loader, desc= f'Going through all {len(train_loader)} batches', disable= tqdm_disabled)):

        if multi_task:
            tr_x, tr_y1, tr_y2 = training_resources
            batch_x = Variable(tr_x.to(device))       
            batch_y1 = Variable(tr_y1.to(device))
            batch_y2 = Variable(tr_y2.to(device))
            output1, output2 = wf_model(batch_x)
            del batch_x    
            loss = loss_func(output1, output2, batch_y1, batch_y2)
            del batch_y1
            del batch_y2
        else:
            tr_x, tr_y = training_resources
            batch_x = Variable(tr_x.to(device))   
            batch_y = Variable(tr_y.to(device))
            output = wf_model(batch_x)
            del batch_x  
            loss = loss_func(output, batch_y)
            del batch_y
            del output


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        
        training_stats = {}
        training_stats['batch_losses'] = batch_losses
        
    if multi_task:
        wf_model.disable_multi_task()# the multi_task_enabled will be set again to true whenever needed

    return training_stats


def rf_mt_training_loop(wf_model, device, train_loader, 
                     scheduler_method, optimizer, learning_rate, 
                     current_epoch, total_epochs, if_use_gpu = True, input_processor = None, annealing_rate = 0.2):
    wf_model.multi_task_enabled = True
    wf_model.to(device)
    wf_model.train()
    loss_func = MultitaskLoss(criterion = nn.CrossEntropyLoss())
    batch_losses = []

    if scheduler_method == 'default': # setting the default learning rate scheduler in rf
        adjust_learning_rate(optimizer, echo= current_epoch, learning_rate= learning_rate, epoch= total_epochs, annealing_rate= annealing_rate)
    for step, (tr_x, tr_y1, tr_y2) in enumerate(tqdm(train_loader, desc= f'Going through all {len(train_loader)} batches')):

        batch_x = Variable(tr_x.to(device))
        
        batch_y1 = Variable(tr_y1.to(device))
        batch_y2 = Variable(tr_y2.to(device))
        output1, output2 = wf_model(batch_x)

        del batch_x
        
        loss = loss_func(output1, output2, batch_y1, batch_y2)
        del batch_y1
        del batch_y2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        del output1
        del output2
        training_stats = {}
        training_stats['batch_losses'] = batch_losses
        
    wf_model.multi_task_enabled = False

    return training_stats
    

