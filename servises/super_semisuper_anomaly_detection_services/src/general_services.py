'''
general_services.py
================================================================================
The module containing services of general utility

Copyright 2020 - The IoTwins Project Consortium, Alma Mater Studiorum
Università di Bologna. All rights reserved.
'''
#!/usr/bin/python3.6
import numpy as np
import pandas as pd
from sklearn import preprocessing as sklpre
from sklearn import model_selection as sklms 
from imblearn import over_sampling as imbover
from imblearn import under_sampling as imbunder
import re, pickle, sys
import os.path
import ast

volume_dir = '../data'  # '/root/mlservice_volume'

def _get_categorical_continuous_features(df):
    categorical_features = []
    continuous_features = []
    for c in df.columns:
        if df[c].dtype == 'object':
            categorical_features.append(c)
        else:
            continuous_features.append(c)
    return categorical_features, continuous_features

def _encode_categorical_feature(df_in, cat_feat):
    if cat_feat not in df_in:
        return df_in
    dummy_cols =  pd.get_dummies(df_in[cat_feat], dummy_na=False)
    df = pd.concat([df_in, dummy_cols], axis=1)
    del df[cat_feat]
    return df

def _oneHot_encode(df, cat_feats=[]):
    # if needed, idenfify continuous and categorical_features
    if(len(cat_feats) == 0):
        cat_feats, cont_feats = _get_categorical_continuous_features(df)

    # encode categorical fetures one by one
    for catf in cat_feats:
        df = _encode_categorical_feature(df, catf)

    return df

def _normalize(df, cat_feats=[], cont_feats=[], scaler_in=None,
        scalerType='minmax'):

    # if needed, idenfify continuous and categorical_features
    if(len(cont_feats) == 0 and len(cat_feats) == 0):
        cat_feats, cont_feats = _get_categorical_continuous_features(df)

    if scaler_in == None:
        # choose normalization type
        if scalerType == 'minmax':
            scaler = sklpre.MinMaxScaler(feature_range=(0, 1))
        elif scalerType == 'std':
            scaler = sklpre.StandardScaler()
        else:
            print('[gs:normalize] Unsupported scaler type: {}'.format(
                scalerType))
            return df, None
    else: 
        scaler = scaler_in

    df[cont_feats] = scaler.fit_transform(df[cont_feats])
    return df, scaler

def _preprocess(df, y='', scaler_in=None):
    # drop NaN values
    df = df.dropna()

    if y != '' and y in df:
        target = df[y]
        del df[y]
    else: 
        target = []
    
    # idenfify continuous and categorical_features
    catf, contf = _get_categorical_continuous_features(df)

    # encode categorical features 
    df = _oneHot_encode(df, catf)

    # normalize data 
    df, scaler = _normalize(df, catf, contf, scaler_in)

    return df, target, scaler

def _split_data_noLabel(df, ratio = 0.7):
    df = df.values

    # create training set split
    df_train, df_temp = sklms.train_test_split(df, train_size=ratio,
            random_state=42)

    # create test set and validation set splits
    df_test, df_val = sklms.train_test_split(df_temp, train_size=0.8,
            random_state=42)

    return df_train, df_test, df_val

def _split_data_withLabel(df, y, ratio = 0.7):
    df = df.values

    # create training set split
    df_train, df_temp, y_train, y_temp = sklms.train_test_split(df, y,
            train_size=ratio, random_state=42)

    # create test set and validation set splits
    df_test, df_val, y_test, y_val = sklms.train_test_split(df_temp, y_temp,
            train_size=0.8, random_state=42)

    return df_train, df_test, df_val, y_train, y_test, y_val

def _oversample(X, y, method='SMOTE', strat='not majority'):
    # compute minimum number of samples per class
    min_samples = len(y)
    for l in set(y):
        if y.tolist().count(l) < min_samples:
            min_samples = y.tolist().count(l)
    if min_samples <= 5:
        method='RNDM'

    if method == 'ADASYN':
        ios = imbover.ADASYN(sampling_strategy=strat, random_state=42)
    elif method == 'SMOTE':
        ios = imbover.SMOTE(sampling_strategy=strat, random_state=42)
    elif method == 'SMOTENC':
        ios = imbover.SMOTENC(sampling_strategy=strat, random_state=42)
    elif method == 'BORDERSMOTE':
        ios = imbover.BorderlineSMOTE(sampling_strategy=strat, random_state=42)
    elif method == 'SVMSMOTE':
        ios = imbover.SVMSMOTE(sampling_strategy=strat, random_state=42)
    elif method == 'KMEANSSMOTE':
        ios = imbover.KMeansSMOTE(sampling_strategy=strat, random_state=42)
    elif method == 'RNDM':
        ios = imbover.RandomOverSampler(sampling_strategy=strat, random_state=42)

    X_resampled, y_resampled = ios.fit_resample(X, y)
    return X_resampled, y_resampled


def serialize_py_obj(obj, obj_fname):
    '''
    general_services.serialize_py_obj::Serialize a Python object

    :param obj : Python object
        object to be serialized
    :param obj_fname : str
        name of the file of the serialized object
    '''
    with open(obj_fname, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_py_obj(obj_fname):
    '''
    general_services.load_py_obj::Load a serialized Python object

    :param obj_fname : str
        name of the csv file of the serialized object

    :return : Python object
        de-serialized object
    '''
    if os.path.isfile(obj_fname):
        with open(obj_fname, 'rb') as handle:
            obj = pickle.load(handle)
    else:
        print('[gs:load_py_obj] File {} not found!'.format(obj_fname))
        obj = None
    return obj

def get_categorical_continuous_features(df_fname):
    '''
    general_services.get_categorical_continuous_features::Identify categorical
    and continuous features in a data frame ** This method is not completely
    reliable as it relies on .info() to get the real data type of the values of
    a feature -- as some missing values that are represented as strings in a
    continuous feature can coerce it to read them as object dtypes --> a
    preliminary analysis of the data should be performed **

    :param df_fname : str
        name of the csv file with data to be analysed

    :return : lists of strings
        column names of categorical features and list of column names of
        categorical features
    '''
    df_fname = '{}/{}'.format(volume_dir, df_fname)
    df = pd.read_csv(df_fname)
    cat_feats, cont_feats = _get_categorical_continuous_features(df)
    return cat_feats, cont_feats  # categorical_features, continuous_features non è un errore???

def encode_categorical_feature(df_fname, cat_feat):
    '''
    general_services.encode_categorical_feature::Encode a categorical feature
    into with one-hot encoding

    :param df_fname : str
        name of the csv file with data to be encoded
    :param cat_feat : str
        name of the categorical feature to be encoded

    :return : dataframe 
        data with encoded feature
    '''
    df_fname = '{}/{}'.format(volume_dir, df_fname)
    df_in = pd.read_csv(df_fname)
    df = _encode_categorical_feature(df_in)
    return df

def oneHot_encode(df_fname, cat_feats=[]):
    '''
    general_services.oneHot_encode::Categorical feature encoding - one hot
    encoding

    :param df_fname : str
        name of the csv file with data to be encoded
    :param (optional) cat_feats : list of strings 
        column names of categorical features

    :return : dataframe 
        data with one-hot encoded categorical features
    '''
    df_fname = '{}/{}'.format(volume_dir, df_fname)
    df = pd.read_csv(df_fname)
    df = _oneHot_encode(df, cat_feats)
    return df

def normalize(df_fname, cat_feats=[], cont_feats=[], scaler_in=None,
        scalerType='minmax'):
    '''
    general_services.normalize::Data normalization

    :param df_fname : str
        name of the csv file with data to be normalized
        ** assumption no NaN value in the data frame **
    :param cat_feats : list of strings 
        (optional) column names of categorical features
    :param cont_feats : list of strings 
        (optional) column names of continuous features
    :params scaler_in : Sklearn scaler object
        (optional) the scaler used to normalize the data
    :param scalerType : string
        (optional) scaling type - supported types 1) 'minmax' and 2) 'std'

    :return : dataframe
        normalized data 
    :return : Sklearn scaler object
        scaler object used to normalized the data
        - if a scaler was provided as input, the same object is returned
    '''
    df_fname = '{}/{}'.format(volume_dir, df_fname)
    df = pd.read_csv(df_fname)
    df, scaler = _normalize(df, cat_feats=[], cont_feats=[], scaler_in=None,
            scalerType='minmax')
    return df, scaler

def preprocess(df_fname, y='', scaler_in=None):
    '''
    general_services.preprocess::Preprocess data frame: remove NaN values,
    one-hot encoding of categorical features and normalization of continuous
    features (using minmax scaling)

    :param df_fname : str
        name of the csv file with data to be preprocessed
    :param (optional) y : string
        name of the feature to be used as label/target
    :params scaler_in : Sklearn scaler object
        (optional) the scaler used to normalize the data

    :return : dataframe
        preprocessed data 
    :return : numpy array, panda series
        list for targets if the label/target option has been specified
        * otherwise the target list is a empty list
        * labels are not preprocessed (neither scaled nor encoded)
    :return : scikit-learn object
        the scaler object used to normalize the data
        - if a scaler was provided as input, the same object is returned
    '''
    df_fname = '{}/{}'.format(volume_dir, df_fname)
    df = pd.read_csv(df_fname)
    df, target, scaler = _preprocess(df, y,  scaler_in)
    # save the results 
    df_fname_norm = '{}_norm.csv'.format(volume_dir, df_fname.split('.')[0])
    df.to_csv(df_fname_norm)
    norm_target_fname = '{}/normalized_label.pickle'.format(volume_dir)
    serialize_py_obj(target, norm_target_fname)
    scaler_fname = '{}/scaler.pickle'.format(volume_dir)
    serialize_py_obj(scaler, scaler_fname)
    return df, target, scaler

def split_data_noLabel(df_fname, ratio = 0.7):
    '''
    general_services.split_data_noLabel::Split data  frame in train, test and
    validation sets. This function does not expect labels, it operates on a
    single data frame

    :param df_fname : str
        name of the csv file with data to be split
        * Allowed inputs are lists, numpy arrays, scipy-sparse matrices or
        pandas dataframes

    :param (optional) ratio: float
        the proportion of the data set to be included in the training set split
        -- default value = 0.7

    :return : python list 
        list containing train-test split of inputs 
    '''
    df_fname = '{}/{}'.format(volume_dir, df_fname)
    df = pd.read_csv(df_fname)
    df_train, df_test, df_val = _split_data_noLabel(df)
    return df_train, df_test, df_val

def split_data_withLabel(df_fname, y, ratio = 0.7):
    ''' 
    general_services.split_data_withLabel::Split data frame in train, test and
    validation sets. This functions works with labeled data; it _expects_ a list
    (numpy array, pandas Series) of labels

    :param df_fname : str
        name of the csv file with data to be split
        * Allowed inputs are lists, numpy arrays, scipy-sparse matrices or
        pandas dataframes
    :param y : str
        name of the label column 
    :param ratio : float
        the proportion of the data set to be included in the training set split
        -- default value = 0.7

    :return : python list
        list containing train-test split of inputs features and labels
    '''
    df_fname = '{}/{}'.format(volume_dir, df_fname)
    df = pd.read_csv(df_fname)
    df_train, df_test, df_val, y_train, y_test, y_val = _split_data_withLabel(
            df, y, ratio)
    return df_train, df_test, df_val, y_train, y_test, y_val

def oversample(X_fname, y_fname, method='SMOTE', strat='not majority'):
    '''
    general_services.oversample:: oversampling function -- exploits imbalance
    learn python module

    :param X_fname : str
        name of the file containing the Python list or numpy array of features
        to be oversampled 

    :param y_fname : str
        name of the file containing the Python list of labels to be oversampled
        (associated to the input features)

    :param method : string
        oversampling method -- supported methods: 'ADASYN', 'SMOTE',
        'KMEANSMOTE', 'RNDM', 'SMOTENC', 'SVMSMOTE', 'BORDERSMOTE' 

    :param strat : string
        oversampling strategy -- supported strategies: 'minority', 'not
        minority', 'not majority', 'all' 

    :return : numpy arrays
        data (input features) after oversampling

    :return : numpy array
        labels after oversampling

    '''
    # load Python objects
    X = load_py_obj(X_fname)
    y = load_py_obj(y_fname)
    if X == None:
        print('[gs:oversample] Problem with features to be oversampled')
        return None, None
    if y == None:
        print('[gs:oversample] Problem with labels to be oversampled')
        return None, None

    possible_algs = ['ADASYN', 'SMOTE', 'KMEANSMOTE', 'RNDM', 'SMOTENC',
            'SVMSMOTE', 'BORDERSMOTE']
    possible_strats = ['minority', 'not minority', 'not majority', 'all']
    if method not in possible_algs:
        print("[gs:oversample] Unsupported oversampling algorithm {}".format(
            method))
        return X, y
    if strat not in possible_strats:
        print("[gs:oversample] Unsupported oversampling strategy {}".format(
            strat))
        return X, y 
    if len(X) != len(y):
        print("[gs:oversample] Differing lengths: len(X) ({}) != len(y) \
                ({})".format(len(X), len(y)))
        return X, y 

    X_resampled, y_resampled = _oversample(X, y, method, strat)
    return X_resampled, y_resampled

def sanitize(input_str):
    '''
    general_services.sanitize::Sanitize string by removing all non
    alphanumerical characters

    :param input_str : string
        the input string to be sanitized

    :return : string
        the sanitized string
    '''
    out_str = re.sub(r'\W+', '', input_str)
    return out_str

if __name__ == '__main__':
    service_type = int(sys.argv[1])
    service_args = sys.argv[2:]
    if service_type == 1:
        get_categorical_continuous_features(service_args[0])
    elif service_type == 2:
        encode_categorical_feature(service_args[0], service_args[1])
    elif service_type == 3:
        if len(service_args) == 1:
            oneHot_encode(service_args[0])
        elif len(service_args) == 2:
            x = ast.literal_eval(service_args[1])
            x = [n.strip() for n in x]
            oneHot_encode(service_args[0], x)
        else:
            print("[gs] Wrong arguments {}".format(service_args))
    elif service_type == 4:
        if len(service_args) == 1:
            normalize(service_args[0])
        elif len(service_args) == 2:
            normalize(service_args[0], service_args[1])
        elif len(service_args) == 3:
            normalize(service_args[0], service_args[1], service_args[2])
        elif len(service_args) == 4:
            normalize(service_args[0], service_args[1], service_args[2], 
                    service_args[3])
        elif len(service_args) == 5:
            normalize(service_args[0], service_args[1], service_args[2], 
                    service_args[3], service_args[4])
        else:
            print("[gs] Wrong arguments {}".format(service_args))
    elif service_type == 5:
        if len(service_args) == 1:
            preprocess(service_args[0])
        elif len(service_args) == 2:
            preprocess(service_args[0], service_args[1])
        elif len(service_args) == 3:
            preprocess(service_args[0], service_args[1], service_args[2])
        else:
            print("[gs] Wrong arguments {}".format(service_args))
    elif service_type == 6:
        if len(service_args) == 1:
            split_data_noLabel(service_args[0])
        elif len(service_args) == 2:
            split_data_noLabel(service_args[0], service_args[1])
    elif service_type == 7:
        if len(service_args) == 2:
            split_data_withLabel(service_args[0], service_args[1])
        elif len(service_args) == 3:
            split_data_withLabel(service_args[0], service_args[1], 
                    service_args[2])
        else:
            print("[gs] Wrong arguments {}".format(service_args))
    elif service_type == 8:
        if len(service_args) == 2:
            oversample(service_args[0], service_args[1])
        elif len(service_args) == 3:
            oversample(service_args[0], service_args[1], service_args[2])
        elif len(service_args) == 4:
            oversample(service_args[0], service_args[1], service_args[2],
                    service_args[3])
        else:
            print("[gs] Wrong arguments {}".format(service_args))
    elif service_type == 9:
        sanitize(service_args[0])
    else:
        print('[gs] Unsupported option')
