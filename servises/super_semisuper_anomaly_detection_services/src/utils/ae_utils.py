'''
ae_utils.py
================================================================================
The module containing miscellaneous utility functions for the anomaly detection
method based on both the i) semi-supervised autoencoder and the ii) unsupervised
autoencoder plus the supervised classifier

Copyright 2020 - The IoTwins Project Consortium, Alma Mater Studiorum Università
di Bologna. All rights reserved.
'''
import os
import sys
import math
from decimal import *
import collections
import pickle
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.python.keras.models import load_model, Sequential, Model
from tensorflow.python.keras import optimizers, initializers, regularizers
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.layers import UpSampling1D, Dropout, Lambda
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.losses import mse, binary_crossentropy
import pandas as pd
import time
import subprocess
import numpy as np
import itertools as it

def autoencoder(_n_features, _hparams):
    ''' 
    ae_utils.autoencoder::Build the autoencoder models
    
    :param _n_features: int
        number of input features of the data
    :param _hparams: dict
        network hyperparameters
    
    :return: Keras model
        the build autoencoder
    :return: Keras model
        the build encoder
    '''
    input_layer = Input(shape=(_n_features,))
    loss_func = _hparams['loss']

    if _hparams['overcomplete']:
        nl = _hparams['nl_o'] 
        nnl = _hparams['nnl_o'] * _n_features
    else:
        nl = _hparams['nl_u']
        if _hparams['nnl_u'] < _n_features:
            nnl = _n_features // _hparams['nnl_u']
        else:
            print('[ae_util:autoencoder] Error: nnl_u {} should be smaller \
                    than n_features {}'.format(_hparams['nnl_u'], _n_features))
            nnl = _n_features // 2

    # encoder
    for layer in range(1, int(nl)+1):
        if _hparams['overcomplete']:
            hidden = Dense(units=int(nnl), 
                    activation=_hparams['actv'], 
                    activity_regularizer=regularizers.l1(_hparams['l1_reg']))(
                            input_layer if layer == 1 else hidden)
        else:
            hidden = Dense(units=int(nnl), activation=_hparams['actv'])(
                            input_layer if layer == 1 else hidden)

        if _hparams['drop_enabled']:
            hidden = Dropout(rate=_hparams['drop_factor'])(hidden)

    encoder = Model(input_layer, hidden)
   
    # decoder
    for layer in reversed(range(1, int(nl)+1)):
        if _hparams['overcomplete']:
            hidden = Dense(units=int(nnl), activation=_hparams['actv'], 
                    activity_regularizer=regularizers.l1(_hparams['l1_reg']))(
                            hidden)
        else:
            hidden = Dense(units=int(nnl), activation=_hparams['actv']
                    )(hidden)

        if _hparams['drop_enabled']:
            hidden = Dropout(rate=_hparams['drop_factor'])(hidden)
        
    output_layer = Dense(_n_features, activation=_hparams['actv'])(hidden)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer=_hparams['optimizer'], loss=loss_func)
    
    return autoencoder, encoder


def create_classifier_on_encoder(_n_features, _hparams, _encoder):
    ''' 
    ae_utils.create_classifier_on_encoder::Build a binary classifier layers on
    top of an existing encoder

    :param _n_features: int
        number of input features of the data
    :param _hparams: dict
        network hyperparameters
    :param _encoder: Keras model
        the trained encoder
    
    :return: Keras model
        the built classifier
    '''
    input_layer = Input(shape=(_n_features,))
    loss_func = _hparams['loss']

    nl = _hparams['nl']
    if _hparams['nnl'] < _n_features:
        nnl = _n_features // _hparams['nnl']
    else:
        print('[ae_util:create_classifier_on_encoder] Error: nnl {} should be \
                smaller than n_features {}'.format(_hparams['nnl'], 
                    _n_features))
        nnl = _n_features // 2

    # classifier
    for layer in range(1, int(nl)+1):
        hidden = Dense(units=int(nnl), activation=_hparams['actv'])(
                        input_layer if layer == 1 else hidden)

        if _hparams['drop_enabled']:
            hidden = Dropout(rate=_hparams['drop_factor'])(hidden)

    output_layer = Dense(1, activation=_hparams['actv'])(hidden)
    classifier = Model(input_layer, output_layer)
    classifier.compile(optimizer=_hparams['optimizer'], loss=loss_func)
    
    return classifier


def unix_time_millis(dt):
    '''
    ae_utils.unix_time_millis::convert from UNIX date time to milliseconds
    
    :param dt: datetime
        date time in UNIX time

    :return: int
        date time converted in milliseconds
    '''
    return long((dt - epoch).total_seconds() * 1000.0)


def millis_unix_time(millis):
    '''
    ae_utils.millis_unix_time::convert from milliseconds to UNIX date time
    
    :param millis : int
        date time in milliseconds

    :return: datetime
        date time converted in UNIX date time
    '''
    seconds = millis / 1000
    return epoch + datetime.timedelta(seconds=seconds)


def drop_stuff(df, features_to_be_dropped):
    '''
    ae_utils.:drop_stuff:Drop unused features and rows with NaN
    
    :param df: dataframe
        the data
    :param features_to_be_dropped: list of strings
        list of features/column to be dropped

    :return: dataframe
        cleaned up data frame
    '''
    for fd in features_to_be_dropped:
        if fd in df:
            del df[fd]
    new_df = df.dropna(axis=0, how='all')
    new_df = new_df.dropna(axis=1, how='all')
    new_df = new_df.fillna(0)
    return new_df

def evaluate_predictions_semisup(predicted, actual):
    '''
    ae_utils.evaluate_predictions_semisup::Evaluate prediction of the
    semi-supervised autoencoder

    :param predicted : numpy array
        autoencoder predicted labels
    :param actual : numpy array 
        true labels

    :return : Python dictionary
        the detailed evaluation metrics computed on the autoencoder 
    '''
    MAE = []
    MSE = []
    RMSE = []
    NRMSE = []
    CVRMSE = []
    MAPE = []
    SMAPE = []

    nb_samples, nb_series = actual.shape
    abs_errors = {}
    p_abs_errors = {}
    sp_abs_errors = {}
    squared_errors = {}
    MAE = {}
    MAPE = {}
    SMAPE = {}
    MSE = {}
    RMSE = {}

    actual_t = {}
    pred_t = {}

    for j in range(nb_series):
        abs_errors[j] = []
        p_abs_errors[j] = []
        sp_abs_errors[j] = []
        squared_errors[j] = []
        actual_t[j] = []
        pred_t[j] = []

    for i in range(nb_samples):
        for j in range(nb_series):
            abs_errors[j].append(abs(predicted[i][j] - actual[i][j]))
            squared_errors[j].append((predicted[i][j] - actual[i][j])*
                (predicted[i][j] - actual[i][j]))
            if actual[i][j] != 0:
                p_abs_errors[j].append((abs(predicted[i][j]-actual[i][j]))* 
                        100 / actual[i][j])
            sp_abs_errors[j].append((abs(predicted[i][j]-actual[i][j])) * 100 / 
                (predicted[i][j] + actual[i][j]))
            actual_t[j].append(actual[i][j])
            pred_t[j].append(predicted[i][j])

    for j in range(nb_series):
        MAE[j] = Decimal(np.mean(np.asarray(abs_errors[j])))
        MAPE[j] = Decimal(np.mean(np.asarray(p_abs_errors[j])))
        SMAPE[j] = Decimal(np.nanmean(np.asarray(sp_abs_errors[j])))
        MSE[j] = Decimal(np.mean(np.asarray(squared_errors[j])))
        RMSE[j] = Decimal(math.sqrt(MSE[j]))

    stats_res = {}
    stats_res["MAE"] = MAE
    stats_res["MSE"] = MSE
    stats_res["RMSE"] = RMSE
    stats_res["MAPE"] = MAPE
    stats_res["SMAPE"] = SMAPE
    stats_res["ABS_ERRORS"] = abs_errors
    stats_res["P_ABS_ERRORS"] = p_abs_errors
    stats_res["SP_ABS_ERRORS"] = sp_abs_errors
    stats_res["SQUARED_ERRORS"] = squared_errors

    cumul_errors = [0] * len(stats_res["ABS_ERRORS"][0])
    std_feature_errors = [0] * len(stats_res["ABS_ERRORS"][0])
    avg_feature_errors = [0] * len(stats_res["ABS_ERRORS"][0])
    for j in stats_res["ABS_ERRORS"].keys():
        for i in range(len(stats_res["ABS_ERRORS"][j])):
            cumul_errors[i] += stats_res["ABS_ERRORS"][j][i]

    # 'normalize' cumulated errors
    cumul_errors_norm = []
    for ce in cumul_errors:
        cumul_errors_norm.append(ce / len(stats_res["ABS_ERRORS"]))
    stats_res["CUMUL_ABS_ERRORS"] = cumul_errors
    stats_res["CUMUL_ABS_ERRORS_NORM"] = cumul_errors_norm

    return stats_res


def pairwise(iterable):
    '''
    ae_utils.pairwise::Convert a list in a iterable collection of pairwise
    tuples: from s = [s0, s1, s2, s3, s4,  ..] to s = (s0, s1), (s1, s2), (s2,
    s3), ...

    :param iterable : list
        the list

    :return : Python iterable
        iterable collection of tuples
    '''

    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def is_in_timeseries(timestamps, ts, sampling_time):
    '''
    ae_utils.is_in_timeseries::Check if a particular datetime is inside a
    time-series of timestamps

    :param timestamps : Datetime list
        list of timestamps present in the time-series
    :param ts : datetime
        the timestamp to be searched for in the time-series
    :param sampling_time : int
        the sampling time of the timeseries timestamp (in seconds)

    :return : bool
        True if the timestamp is present, False otherwise
    '''
    for ts_1, ts_2 in pairwise(timestamps):
        if(ts_1 <= ts <= ts_2 and 
                (ts_1 - ts_2).total_seconds() <= sampling_time):
            return True
    return False

def find_gaps_timeseries(timestamps, sampling_time):
    '''
    ae_utils.find_gaps_timeseries::Find gaps in time series. Given a set of time
    stamps, find missing values 

    :param timestamps : Datetime list
        list of timestamps
    :param sampling_time : int
        the sampling time of the timeseries timestamp (in seconds)

    :return : list
        the list of the gaps in the time-series
    '''
    gap_list = []
    for ts1, ts2 in pairwise(timestamps):
        tdif = (ts2 - ts1).total_seconds()
        if tdif > 1.1*sampling_time:
            gap_list.append((ts1,ts2))
    return gap_list

def detection_exploreThresholds(actual_normal, pred_normal, 
        actual_anomal, pred_anomal, actual_normal_all, pred_normal_all):
    '''
    ae_utils.detection_exploreThresholds::Analyse reconstruction errors
    distributions and explore varying detection thresholds.

    :params actual_normal : numpy array
        true values of normal data points (only those not used to train the
        model)

    :params pred_normal : numpy array
        predicted values for normal data points (only those not used to train
        the model)

    :params actual_anomal : numpy array
        true values of anomalous data points
    :params pred_anomal : numpy array
        predicted values for anomalous data points

    :params actual_normal_all : numpy array
        true values of normal data points (both those used to train the model
        and those used to test it)

    :params pred_normal_all : numpy array
        predicted values for normal data points (both those used to train the
        model and those used to test it)

    :return : int
        best percentile threshold
    :return : Python dictionary
        recap statistics obtained with the best threshold
    '''
    msk = np.random.rand(len(actual_anomal)) < 0.7
    validation_set_actual_A = actual_anomal[msk]
    test_set_actual_A = actual_anomal[~msk]
    validation_set_pred_A = actual_anomal[msk]
    test_set_pred_A = actual_anomal[~msk]

    actual_anomal_redux = actual_anomal[~msk]
    actual_anomal = actual_anomal_redux

    nn_samples, nn_series = actual_normal.shape
    errors_normal = [0] * nn_samples
    abs_errors_normal = {}
    squared_errors_normal = {}

    for j in range(nn_series):
        abs_errors_normal[j] = []
        squared_errors_normal[j] = []
    for i in range(nn_samples):
        for j in range(nn_series):
            abs_errors_normal[j].append(
                    abs(pred_normal[i][j] - actual_normal[i][j]))
            squared_errors_normal[j].append((
                pred_normal[i][j] - actual_normal[i][j])*
                (pred_normal[i][j] - actual_normal[i][j]))

    na_samples, na_series = actual_anomal.shape
    errors_anomal = [0] * na_samples
    abs_errors_anomal = {}
    squared_errors_anomal = {}

    for j in range(na_series):
        abs_errors_anomal[j] = []
        squared_errors_anomal[j] = []
    for i in range(na_samples):
        for j in range(na_series):
            abs_errors_anomal[j].append(
                    abs(pred_anomal[i][j] - actual_anomal[i][j]))
            squared_errors_anomal[j].append((
                pred_anomal[i][j] - actual_anomal[i][j])*
                (pred_anomal[i][j] - actual_anomal[i][j]))

    nn_all_samples, nn_all_series = actual_normal_all.shape
    errors_normal_all = [0] * nn_all_samples
    abs_errors_normal_all = {}
    squared_errors_normal_all = {}

    for j in range(nn_all_series):
        abs_errors_normal_all[j] = []
        squared_errors_normal_all[j] = []
    for i in range(nn_all_samples):
        for j in range(nn_all_series):
            abs_errors_normal_all[j].append(
                    abs(pred_normal_all[i][j] - actual_normal_all[i][j]))
            squared_errors_normal_all[j].append((
                pred_normal_all[i][j] - actual_normal_all[i][j])*
                (pred_normal_all[i][j] - actual_normal_all[i][j]))

    # max abs error 
    for j in abs_errors_normal.keys():
        for i in range(len(abs_errors_normal[j])):
            if errors_normal[i] < abs_errors_normal[j][i]:
                errors_normal[i] = abs_errors_normal[j][i]
    for j in abs_errors_normal_all.keys():
        for i in range(len(abs_errors_normal_all[j])):
            if errors_normal_all[i] < abs_errors_normal_all[j][i]:
                errors_normal_all[i] = abs_errors_normal_all[j][i]
    for j in abs_errors_anomal.keys():
        for i in range(len(abs_errors_anomal[j])):
            if errors_anomal[i] < abs_errors_anomal[j][i]:
                errors_anomal[i] = abs_errors_anomal[j][i]

    n_perc_min = 85
    n_perc_max = 99
    classes_normal = [0] * nn_samples
    classes_anomal = [1] * na_samples
    errors = errors_normal + errors_anomal
    classes = classes_normal + classes_anomal

    best_threshold = n_perc_max
    fscore_A_best = 0
    fscore_N_best = 0
    fscore_W_best = 0
    fps = []
    fns = []
    tps = []
    tns = []
    n_percs = []
    precs = []
    recalls = []
    fscores = []
    for n_perc in range(n_perc_min, n_perc_max+2):
        error_threshold = np.percentile(np.asarray(errors_normal_all), n_perc)
        predictions = []
        for e in errors:
            if e > error_threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
                classes, predictions, average='binary', pos_label=0)
        precision_A, recall_A, fscore_A, xyz = precision_recall_fscore_support(
                classes, predictions, average='binary', pos_label=1)
        precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
                classes, predictions, average='weighted')

        conf_mat = confusion_matrix(classes, predictions).ravel()
        if len(conf_mat) == 4:
            tn, fp, fn, tp = conf_mat
        else:
            tn, fp, fn, tp = conf_mat, 0, 0, 0

        fscores.append(fscore_W)
        precs.append(precision_W)
        recalls.append(recall_W)
        n_percs.append(n_perc)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)
        tns.append(tn)
 
        if fscore_W > fscore_W_best:
            precision_W_best = precision_W
            precision_N_best = precision_N
            precision_A_best = precision_A
            recall_W_best = recall_W
            recall_N_best = recall_N
            recall_A_best = recall_A
            fscore_W_best = fscore_W
            fscore_N_best = fscore_N
            fscore_A_best = fscore_A
            best_threshold = n_perc
            best_err_threshold = error_threshold

    recap_stat = {'precision_N' : precision_N_best, 'recall_N': recall_N_best,
            'fscore_N': fscore_N_best, 'precision_A': precision_A_best,
            'recall_A': recall_A_best, 'fscore_N': fscore_N_best, 
            'precision_W': precision_W_best, 'recall_W': recall_W_best,
            'fscore_W': fscore_W_best, 'err_threshold': best_err_threshold,
            'n_perc': best_threshold}

    return best_threshold, recap_stat

def detection_withThreshold(percentile, actual_normal, pred_normal, 
        actual_anomal, pred_anomal, actual_normal_all, pred_normal_all):
    '''
    ae_utils.detection_withThreshold::Analyse reconstruction errors
    distributions and explore varying detection thresholds.
    
    :params percentile : int
        the percentile to be used to compute the detection threshold
    :params actual_normal : numpy array
        true values of normal data points (only those not used to train the
        model)

    :params pred_normal : numpy array
        predicted values for normal data points (only those not used to train
        the model)

    :params actual_anomal : numpy array
        true values of anomalous data points
    :params pred_anomal : numpy array
        predicted values for anomalous data points

    :params actual_normal_all : numpy array
        true values of normal data points (both those used to train the model
        and those used to test it)

    :params pred_normal_all : numpy array
        predicted values for normal data points (both those used to train the
        model and those used to test it)

    :return : Python dictionary
        recap statistics obtained with the given threshold
    '''

    msk = np.random.rand(len(actual_anomal)) < 0.7
    validation_set_actual_A = actual_anomal[msk]
    test_set_actual_A = actual_anomal[~msk]
    validation_set_pred_A = actual_anomal[msk]
    test_set_pred_A = actual_anomal[~msk]

    actual_anomal_redux = actual_anomal[~msk]
    actual_anomal = actual_anomal_redux

    nn_samples, nn_series = actual_normal.shape
    errors_normal = [0] * nn_samples
    abs_errors_normal = {}
    squared_errors_normal = {}

    for j in range(nn_series):
        abs_errors_normal[j] = []
        squared_errors_normal[j] = []
    for i in range(nn_samples):
        for j in range(nn_series):
            abs_errors_normal[j].append(
                    abs(pred_normal[i][j] - actual_normal[i][j]))
            squared_errors_normal[j].append((
                pred_normal[i][j] - actual_normal[i][j])*
                (pred_normal[i][j] - actual_normal[i][j]))

    na_samples, na_series = actual_anomal.shape
    errors_anomal = [0] * na_samples
    abs_errors_anomal = {}
    squared_errors_anomal = {}

    for j in range(na_series):
        abs_errors_anomal[j] = []
        squared_errors_anomal[j] = []
    for i in range(na_samples):
        for j in range(na_series):
            abs_errors_anomal[j].append(
                    abs(pred_anomal[i][j] - actual_anomal[i][j]))
            squared_errors_anomal[j].append((
                pred_anomal[i][j] - actual_anomal[i][j])*
                (pred_anomal[i][j] - actual_anomal[i][j]))

    nn_all_samples, nn_all_series = actual_normal_all.shape
    errors_normal_all = [0] * nn_all_samples
    abs_errors_normal_all = {}
    squared_errors_normal_all = {}

    for j in range(nn_all_series):
        abs_errors_normal_all[j] = []
        squared_errors_normal_all[j] = []
    for i in range(nn_all_samples):
        for j in range(nn_all_series):
            abs_errors_normal_all[j].append(
                    abs(pred_normal_all[i][j] - actual_normal_all[i][j]))
            squared_errors_normal_all[j].append((
                pred_normal_all[i][j] - actual_normal_all[i][j])*
                (pred_normal_all[i][j] - actual_normal_all[i][j]))

    # max abs error 
    for j in abs_errors_normal.keys():
        for i in range(len(abs_errors_normal[j])):
            if errors_normal[i] < abs_errors_normal[j][i]:
                errors_normal[i] = abs_errors_normal[j][i]
    for j in abs_errors_normal_all.keys():
        for i in range(len(abs_errors_normal_all[j])):
            if errors_normal_all[i] < abs_errors_normal_all[j][i]:
                errors_normal_all[i] = abs_errors_normal_all[j][i]
    for j in abs_errors_anomal.keys():
        for i in range(len(abs_errors_anomal[j])):
            if errors_anomal[i] < abs_errors_anomal[j][i]:
                errors_anomal[i] = abs_errors_anomal[j][i]

    classes_normal = [0] * nn_samples
    classes_anomal = [1] * na_samples
    errors = errors_normal + errors_anomal
    classes = classes_normal + classes_anomal

    n_perc = percentile
    error_threshold = np.percentile(np.asarray(errors_normal_all), n_perc)
    predictions = []
    for e in errors:
        if e > error_threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    precision_N, recall_N, fscore_N, xyz = precision_recall_fscore_support(
            classes, predictions, average='binary', pos_label=0)
    precision_A, recall_A, fscore_A, xyz = precision_recall_fscore_support(
            classes, predictions, average='binary', pos_label=1)
    precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
            classes, predictions, average='weighted')

    conf_mat = confusion_matrix(classes, predictions).ravel()
    if len(conf_mat) == 4:
        tn, fp, fn, tp = conf_mat
    else:
        tn, fp, fn, tp = conf_mat, 0, 0, 0

    recap_stat = {'precision_N' : precision_N, 'recall_N': recall_N,
            'fscore_N': fscore_N, 'precision_A': precision_A,
            'recall_A': recall_A, 'fscore_N': fscore_N, 
            'precision_W': precision_W, 'recall_W': recall_W,
            'fscore_W': fscore_W, 'err_threshold': error_threshold,
            'n_perc': n_perc}

    return recap_stat

def split_dataset_semisup(df, labels):
    '''
    ae_utils.split_dataset::Randomly split data sets in training and test set.
    Create a set of subsets for later reuse; it separates normal points that are
    going to be used for training from anomalous ones (semisupervised approach)

    :param df : dataframe
        data to be split (no labels)
    :param labels : Panda series
        the labels corresponding to the data  

    :return : python lists and dataframe
        split subsets
    '''
    # the number of labels must match the number of examples
    if len(df) != len(labels):
        print("[ae_util:split_dataset] len(df) {} != len(labels) {}".format(
            len(df), len(labels)))
        return [], [], [], [], [], []

    anomalies = []
    all_idxs = []
    for idx in range(len(labels)):
        all_idxs.append(idx)
    for idx in range(len(labels)):
        if labels[idx] != 0:
            anomalies.append(idx)

    no_anomalies = list(set(all_idxs) - set(anomalies))

    df_anomalies = df[anomalies]
    df_noAnomalies = df[no_anomalies]

    msk = np.random.rand(len(df_noAnomalies)) < 0.7
    train = df_noAnomalies[msk]
    test = df_noAnomalies[~msk]

    msk_idxs = {}
    for i in range(len(msk)):
        msk_idxs[no_anomalies[i]] = msk[i]

    test_noAnomalies = test
    test = np.concatenate((test, df_anomalies), axis=0)

    return (train, test, df_noAnomalies, 
            df_anomalies, test_noAnomalies, df_anomalies)


def assign_class_sup(_predicted_probabilities, _threshold=0.5):
    '''
    ae_utils.assign_class_sup::Assign the corresponding class (normal or
    anomaly) to the prediction made by the NN classifier

    :param _predicted_probabilities : Python list/numpy array
        predictions made by the NN classifier
    :param _threshold : float
        classifying threshold - predictions higher than this value correspond to
        anomalies, normal points otherwise

    :return : numpy array
        list of classes assigned to the input examples
    '''
    classes = []
    for i in range(len(_predicted_probabilities)):
        if(_predicted_probabilities[i] >= _threshold):
            classes.append(1)
        else:
            classes.append(0)
    return np.asarray(classes)


def evaluate_predictions_sup(predicted, actual):
    '''
    ae_utils.evaluate_predictions_sup::Evaluate prediction of the
    supervised model (autoencoder + classifier)

    :param predicted : numpy array
        autoencoder predicted labels
    :param actual : numpy array 
        true labels

    :return : Python dictionary
        the detailed evaluation metrics computed 
    '''
    count_diff=0
    tp = tn = fp = fn = 0
    for i in range(len(actual)):
        count_diff += abs(actual[i] - predicted[i])

        if((actual[i] == predicted[i]) & (actual[i] == 0)):
            tn += 1
        elif((actual[i] == predicted[i]) & (actual[i] == 1)):
            tp += 1
        elif(actual[i] == 1):
            fn += 1
        else:
            fp += 1
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score=2*(precision*recall)/(precision+recall)

    stats_res = {}
    stats_res['accuracy_alt'] = (len(actual) - int(count_diff)
            ) / len(actual) * 100
    stats_res['accuracy'] = accuracy
    stats_res['precision'] = precision
    stats_res['recall'] = recall
    stats_res['f1_score'] = f1_score
    stats_res['fp'] = fp
    stats_res['fn'] = fn
    stats_res['tp'] = tp
    stats_res['tn'] = tn

    return stats_res

