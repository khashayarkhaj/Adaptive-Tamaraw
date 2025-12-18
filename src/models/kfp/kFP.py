import csv
import sys
from sys import stdout
import numpy as np
#import matplotlib.pylab as plt
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import tree
import sklearn.metrics as skm
import scipy
import random
import os
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
import code
import time

# re-seed the generator
#np.random.seed(1234)


############ Feeder functions ############

def chunks(labels, features):
    """ Calculates the websites instances boundry and chunks the labels and features list based on that.
    This function assumes lables are already sorted tuples based on websites and features index maps
    to labels index"""

    # I have adjusted this because the initial (first) label may not always be 0.
    # If so, last_label = 0, will break the logic
    last_label = labels[0][0]

    instance_start = 0
    instance_end = 0
    labels_chunked = []
    features_chunked = []
    for index, label in enumerate(labels):
        if label[0] != last_label:
            instance_start = instance_end
            instance_end = index
            last_label = label[0]
            labels_chunked.append(labels[instance_start:instance_end])
            features_chunked.append(features[instance_start:instance_end])
    labels_chunked.append(labels[instance_end:])
    features_chunked.append(features[instance_end:])
    return labels_chunked, features_chunked

def checkequal(lst):
    return lst[1:] == lst[:-1]


############ Non-Feeder functions ########

def mon_train_test_references(dic, mon_train_inst, shuffle=True):
    """Prepare monitored data in to training and test sets."""

    split_target, split_data = chunks(dic['alexa_label'], dic['alexa_feature'])

    training_data = []
    training_label = []
    test_data = []
    test_label = []
    for i in range(len(split_data)):
        temp = list(zip(split_data[i], split_target[i]))
        if shuffle:
            random.shuffle(temp)
        data, label = list(zip(*temp))
        training_data.extend(data[:mon_train_inst])
        training_label.extend(label[:mon_train_inst])
        test_data.extend(data[mon_train_inst:])
        test_label.extend(label[mon_train_inst:])

    flat_train_data = []
    flat_test_data = []
    
    for tr in training_data:
        flat_train_data.append(list(sum(tr, ())))
    for te in test_data:
        flat_test_data.append(list(sum(te, ())))
    training_features =  list(zip(flat_train_data, training_label))
    test_features =  list(zip(flat_test_data, test_label))
    return training_features, test_features

def unmon_train_test_references(dic, unmon_train):
    """Prepare unmonitored data in to training and test sets."""

    training_data = []
    training_label = []
    test_data = []
    test_label = []

    unmon_data = dic['unmonitored_feature']
    unmon_label = dic['unmonitored_label']
    unmonitored = list(zip(unmon_data, unmon_label))
    random.shuffle(unmonitored)
    u_data, u_label = list(zip(*unmonitored))

    training_data.extend(u_data[:unmon_train])
    training_label.extend(u_label[:unmon_train])

    test_data.extend(u_data[unmon_train:])
    test_label.extend(u_label[unmon_train:])

    flat_train_data = []
    flat_test_data = []
    for tr in training_data:
        flat_train_data.append(list(sum(tr, ())))
    for te in test_data:
        flat_test_data.append(list(sum(te, ())))
    training_features =  list(zip(flat_train_data, training_label))
    test_features =  list(zip(flat_test_data, test_label))
    return training_features, test_features


def RF_closedworld(training, test, num_trees=100, compute_cross_val = False, verbose = True, return_predictions = False):
    '''Closed world RF classification of data -- only uses sk.learn classification - does not do additional k-nn.'''

    tr_data, tr_label1 = list(zip(*training))
    tr_label = list(zip(*tr_label1))[0]
    te_data, te_label1 = list(zip(*test))
    te_label = list(zip(*te_label1))[0]

    if verbose:
        print("Training ...")
    verbose_num = 0
    if verbose:
        verbose_num = 2
    start_time = time.time()
    model = RandomForestClassifier(n_jobs=-1, n_estimators=num_trees, oob_score = True, verbose= verbose_num)
    model.fit(tr_data, tr_label)
    accuracy_train = model.score(tr_data, tr_label)
    accuracy_test = model.score(te_data, te_label)
    # Calculate and print elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    if verbose:
        print(f"Training time: {elapsed_time:.2f} seconds")
        print("RF accuracy train = ", accuracy_train)
        print("RF accuracy test = ", accuracy_test)
    

    if compute_cross_val:
        print(f"Feature importance scores: {model.feature_importances_}")
        scores = cross_val_score(model, np.array(tr_data), np.array(tr_label))
        print("cross_val_score = ", scores.mean())
        print("OOB score = ", model.oob_score_)
    
    kfp_predictions = None
    if return_predictions:
        # Get predictions for each test instance
        kfp_predictions = model.predict(te_data)
    
    return accuracy_test, model, kfp_predictions



def RF_openworld(mon_training, mon_test, unmon_training, unmon_test, num_trees):
    '''Produces leaf vectors used for classification.'''

    training = mon_training + unmon_training
    test = mon_test + unmon_test

    tr_data, tr_label1 = list(zip(*training))
    tr_label = list(zip(*tr_label1))[0]
    te_data, te_label1 = list(zip(*test))
    te_label = list(zip(*te_label1))[0]

    print("Training ...")
    model = RandomForestClassifier(n_jobs=-1, n_estimators=num_trees, oob_score=True)
    model.fit(tr_data, tr_label)

    train_leaf = list(zip(model.apply(tr_data), tr_label))
    test_leaf = list(zip(model.apply(te_data), te_label))
    return train_leaf, test_leaf, len(mon_test)


def distances(mon_training, mon_test, unmon_training, unmon_test, keep_top=100, num_trees=1000):
    """ This uses the above function to calculate distance from test instance between each training instance (which are used as labels) and writes to file
        Default keeps the top 100 instances closest to the instance we are testing.
        -- Saves as (distance, true_label, predicted_label) --
    """

    train_leaf, test_leaf, test_mon_size = RF_openworld(mon_training, mon_test, unmon_training, unmon_test, num_trees)

    # Make into numpy arrays
    train_leaf = [(np.array(l, dtype=int), v) for l, v in train_leaf]
    test_leaf = [(np.array(l, dtype=int), v) for l, v in test_leaf]

    print('Computing Distance for Monitor Instances...')
    distance_mon = {}
    pbar = tqdm(test_leaf[:test_mon_size], unit="instance", total=test_mon_size)
    for instance in pbar:

        temp = []
        for item in train_leaf:
            # vectorize the average distance computation
            d = np.sum(item[0] != instance[0]) / float(item[0].size)
            if d == 1.0:
                continue
            temp.append((d, item[1]))
            
        if not instance[1] in distance_mon:
            distance_mon[instance[1]] = [sorted(temp)[:keep_top]]
        else:
            distance_mon[instance[1]].extend([sorted(temp)[:keep_top]])


    print('Computing Distance for Unmonitored Instances...')
    distance_unmon = {}
    pbar = tqdm(test_leaf[test_mon_size:], unit="instance", total=len(test_leaf[test_mon_size:]))
    for instance in pbar:

        temp = []
        for item in train_leaf:
            # vectorize the average hamming distance computation
            d = np.sum(item[0] != instance[0]) / float(item[0].size)
            if d == 1.0:
                continue
            temp.append((d, item[1]))

        if not instance[1] in distance_unmon:
            distance_unmon[instance[1]] = [sorted(temp)[:keep_top]]
        else:
            distance_unmon[instance[1]].extend([sorted(temp)[:keep_top]])

    return distance_mon, distance_unmon


def distance_stats(dict_mon, dict_unmon, knn=3):
    """
        For each test instance this picks out the minimum training instance distance, checks (for mon) if it is the right label and checks if it's knn are the same label
    """

    TP=0
    instance_count = 0
    for site in dict_mon.keys():
        for instance in dict_mon[site]:
            instance_count += 1
            match_count = 0
            for i in instance[:knn]:
                if site == i[1]:
                    match_count += 1
            if match_count == knn:
                TP += 1
    TPR = TP/instance_count
    print(f"Instances: {instance_count} - TP: {TP} - TPR: {TPR}")

    FP = 0
    instance_count = 0
    last_mon_label = max(dict_mon.keys())
    for site in dict_unmon.keys():
        for instance in dict_unmon[site]:
            instance_count += 1
            internal_test = []
            for i in instance[:knn]:
                internal_test.append(i[1])
            if checkequal(internal_test) == True and internal_test[0] <= last_mon_label:
                FP += 1
    FPR = FP/instance_count
    print(f"Instances: {instance_count} - FP: {FP} - FPR: {FPR}")

    return TPR, FPR


if __name__ == "__main__":
    exit()

