'''
anomalyDetection.py
================================================================================
The module containing services specific for supervised and semi-supervised
anomaly detection with Deep Learning techniques

Copyright 2020 - The IoTwins Project Consortium, Alma Mater Studiorum
Università di Bologna. All rights reserved.
'''
#!/usr/bin/python3.6

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing as sklpre
from sklearn import model_selection as sklms
from tensorflow.python.keras.optimizers import Adam, Adadelta
from tensorflow.python.keras.callbacks import Callback, EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import servises.super_semisuper_anomaly_detection_services.src.utils.ae_utils as ae
import servises.super_semisuper_anomaly_detection_services.src.general_services as gs
import sys
import joblib


volume_dir = '..'  # '/root/mlservice_volume'
data_dir = '{}/data'.format(volume_dir)
trained_models_dir = '{}/trained_models/'.format(volume_dir)
out_dir = '{}/out/'.format(volume_dir)

'''
anomalyDetection.default_hparams_semisup::Default parameters for the
semi-supervised autoencoder
'''
default_hparams_semisup = {
    'epochs': 20,           # number of training epochs
    'batch_size': 32,       # batch size
    'shuffle': True,        # shuffle data during training
    'overcomplete': False,   # the autoencoder can be overcomplete or
                            # undercomplete
    'nl_o': 3,              # number of layers in the overcomplete case
    'nl_u': 4,              # number of layers in the undercomplete case
    'nnl_o': 10,            # the number of neurons per layer in the
                            # overcomplete autoencoder is computed as n_features
                            # * nnl_o 
    'nnl_u': 2,             # the number of neurons per layer in the
                            # overcomplete autoencoder is computed as n_features
                            # // nnl_u
    'actv': 'relu',         # activation function
    'loss': 'mae',          # loss function
    'l1_reg': 0.00001,      # l1 regularization factor (only for overcomplete
                            # case)
    'lr': 0.0015,           # learning rate
    'optimizer': 'adam',    # optimizer
    'drop_enabled': False,  # add Dropout layer
    'drop_factor': 0.1      # Dropout rate (only if drop==True)
}


'''
anomalyDetection.default_hparams_sup::Default parameters for the unsupervised
autoencoder (to be used in conjunction with the classifier)
'''
default_hparams_sup = {
    'epochs': 20,           # number of training epochs
    'batch_size': 32,       # batch size
    'shuffle': True,        # shuffle data during training
    'overcomplete': False,  # the autoencoder can be overcomplete or
                            # undercomplete
    'nl_o': 3,              # number of layers in the overcomplete case
    'nl_u': 4,              # number of layers in the undercomplete case
    'nnl_o': 10,            # the number of neurons per layer in the
                            # overcomplete autoencoder is computed as n_features
                            # * nnl_o 
    'nnl_u': 2,             # the number of neurons per layer in the
                            # overcomplete autoencoder is computed as n_features
                            # // nnl_u
    'actv': 'relu',         # activation function
    'loss': 'mae',          # loss function
    'l1_reg': 0.00001,      # l1 regularization factor (only for overcomplete
                            # case)
    'lr': 0.0001,           # learning rate
    'optimizer': 'adam',    # optimizer
    'drop_enabled': True,   # add Dropout layer
    'drop_factor': 0.1       # Dropout rate (only if drop==True)
}

'''
anomalyDetection.default_hparams_classr_sup::Default parameters for supervised
classifier
'''
default_hparams_classr_sup = {
    'epochs': 1,                   # number of training epochs
    'batch_size': 32,               # batch size
    'shuffle': True,                # shuffle data during training
    'nl': 2,                        # number of layers 
    'nnl': 2,                       # the number of neurons per layer in the is
                                    # computed as n_features // nnl
    'actv': 'relu',                 # activation function
    'loss': 'binary_crossentropy',  # loss function
    'lr': 0.0001,                   # learning rate
    'optimizer': 'adam',            # optimizer
    'drop_enabled': False,          # add Dropout layer
    'drop_factor': 0.1               # Dropout rate (only if drop==True)
}


def _save_trained_model(model_name, model, stats, scaler):
    '''
    Save a Keras DL model already trained and stored, together with the accuracy
    stats computed during its validation and the sklearn object used for scaling
    the training data

    :param model_name : string 
        the model name - only alphanumerical characters are allowed, no file
        extension needs to be specified
    :param model : Keras model
        the trained model
    :param stats : Python dictionary
        the summary of the detailed statistics computed on the autoencoder, not
        necessarily related to the classification task (which depends on the
        threshold as well), e.g. the reconstruction error for the every data
        point, and the accuracy results
    :params scaler : Sklearn scaler object
        the scaler used to normalize the data

    :return : string
        name of the file containing saved Keras model
    :return : string 
        name of the pickle file containing Python dictionary with the summary
        of the detailed statistics
    :return : string
        name of the pickle file containing the scikit-learn object with scaler
        object used to normalize the data
    '''
    # save the Keras model
    ae_model_name = '{}{}'.format(trained_models_dir, model_name)

    ae_model_name_ext = '{}.h5'.format(ae_model_name)
    model.save(ae_model_name_ext)

    # save associated stats
    ae_stats_file = '{}{}_stats.pickle'.format(out_dir, model_name)
    gs.serialize_py_obj(stats, ae_stats_file)

    # save scaler
    ae_scaler_file = '{}{}_scaler.save'.format(trained_models_dir, model_name)
    joblib.dump(scaler, ae_scaler_file)  # ERROR non dovrebbe essere anche questo un file pickle come scritto nella documentazione?

    return ae_model_name, ae_stats_file, ae_scaler_file  # ERROR non dovrebbe tornare ae_scaler_file come 3° parametro?


def _load_trained_model(model_name):
    '''
    Load a Keras DL model already trained and stored, together with the accuracy
    stats computed during its validation and the sklearn object used for scaling
    the training data

    :param model_name : string 
        the model name - only alphanumerical characters are allowed, no file
        extension needs to be specified

    :return : Keras model
        the trained model
    :return : Python dictionary
        the summary of the detailed statistics computed on the autoencoder, not
        necessarily related to the classification task (which depends on the
        threshold as well), e.g. the reconstruction error for the every data
        point, and the accuracy results
    :return scaler : Sklearn scaler object
        the scaler used to normalize the data
    '''
    # sanitize input model name
    model_name = gs.sanitize(model_name)
    # model_name = '{}{}'.format(trained_models_dir, model_name)   # ERROR se modifico il model name dopo ho un errore per caricare il file stats salvato in output
    trained_model_name = '{}{}'.format(trained_models_dir, model_name)

    # Keras model -- the trained autoencoder
    model_name_ext = '{}.h5'.format(trained_model_name)
    # check if the trained model exists
    if os.path.isfile(model_name_ext):  
        model = ae.load_model(model_name_ext)
    else:
        print("[adssae:load_trained_model] No model with name {}".format(
            trained_model_name))
        model = None

    # stats file (python dictionary)
    stats_file = '{}{}_stats.pickle'.format(out_dir, model_name)
    # check if the stats have been correctly saved
    if os.path.isfile(stats_file):  
        # gs.load_py_obj(stats_file)  # ERROR non dovrebbe essere stats = gs... O.O???
        stats = gs.load_py_obj(stats_file)
    else:
        print("[adssae:load_trained_model] No stats file with name {}".format(
            stats_file))
        stats = {}

    scaler_file = '{}{}_scaler.save'.format(trained_models_dir, model_name)
    # check if the scaler has been correctly saved
    if os.path.isfile(scaler_file):  
        scaler = joblib.load(scaler_file) 
    else:
        print("[adssae:load_trained_model] No scaler file with name {}".format(
            scaler_file))
        scaler = None

    return model, stats, scaler


def semisup_autoencoder(df_fname, sep=',', user_id='default', task_id='0.0', 
        hparams_file=None, n_percentile=-1, save=True):
    '''
    anomalyDetection.semisup_autoencoder::creates a semi-supervised autoencoder
    to detect anomalies; the idea is to train the autoencoder on the normal data
    alone, then use the reconstruction error and a threshold to classify unseen
    examples (binary classification: allowed classes are only normal and
    anomaly).  This model does not work on time-series but rather in a
    "combinatorial" fashion: after having been trained it makes it prediction
    (thus classifying the data point) based on the single test examples fed to
    it (disregarding precious data points).  For details, see Borghesi et al.,
    2019, "A semi-supervised autoencoder-based approach for anomaly detection in
    high performance computing systems", Engineering Applications of Artificial
    Intelligence. 

    :param df_fname : str
        name of the csv file with data to be used for training and testing. One
        column has to termed "label" and it has to contains the class of the
        example - 0 means normal data point, any other integer number
        corresponds to an anomalous data point

    :params user_id : str
        user identifier

    :params task_id : str
        task identifier

    :params hparams_file: str
        name of the pickle file containing the Python dictionary with
        hyperparameters to be used to build the autoencoder 

    :params n_percentile : int
        percentile to be used to compute the detection threshold; if no
        percentile is provided all values in the range [85, 99] will be explored
        and the one providing the best accuracy results will be selected

    :return : string
        name of the file containing saved Keras model (the trained autoencoder)
    :return : string
        name of the pickle file containing the scikit-learn object with scaler
        object used to normalize the data
    :return : string 
        name of the pickle file containing Python dictionary with  the summary
        of the detailed statistics computed on the autoencoder, not  necessarily
        related to the classification task (which depends on the  threshold as
        well), e.g. the reconstruction error for the every data point, and the
        accuracy results
    '''
    # sanitize input string parameters
    user_id = gs.sanitize(user_id)
    task_id = gs.sanitize(task_id)

    # read data from file
    df_fname = '{}/{}'.format(data_dir, df_fname)
    df = pd.read_csv(df_fname, sep=sep)

    if hparams_file == None:
        hparams = default_hparams_semisup
    # read hyperparameters from binary file (pickle object)
    else:
        # hparams = gs.load_py_obj(hparams_file)  TODO just for testing purpose I can pass the params as a dictionary
        hparams = hparams_file

    df.columns = df.columns.str.replace(' ', '')
    df, labels, scaler = gs._preprocess(df, 'label')

    df_tensor = df.values
    n_samples, n_features = df_tensor.shape

    # data split in test/train
    (x_train, x_test, df_noAnomalies, df_anomalies,
            test_noAnomalies, test_anomalies) = ae.split_dataset_semisup(
                    df_tensor, labels)

    # create autoencoder model
    ae_model, _ = ae.autoencoder(n_features, hparams)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, min_delta=1e-5)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', patience=5, min_lr=1e-5, factor=0.2)

    print("[adssae:semisup_autoencoder] Autoencoder model:")
    ae_model.summary()

    print("[adssae:semisup_autoencoder] Begin training")

    history = ae_model.fit(x_train, x_train, epochs=hparams['epochs'],
            batch_size=hparams['batch_size'], shuffle=hparams['shuffle'], 
            callbacks=[early_stopping, reduce_lr],
            validation_split=0.1, verbose=0)

    print("[adssae:semisup_autoencoder] Training concluded")

    # Errors distributions
    decoded_normal = ae_model.predict(df_noAnomalies)
    decoded_anomal = ae_model.predict(df_anomalies)
    decoded_normal_test = ae_model.predict(test_noAnomalies)

    # no percentile provided: explore the [85,99] range
    if n_percentile == -1:
        best_percentile, acc_stats = ae.detection_exploreThresholds(
                test_noAnomalies, decoded_normal_test, df_anomalies,
                decoded_anomal, df_noAnomalies, decoded_normal)
    else:
        acc_stats = ae.detection_withThreshold(n_percentile, test_noAnomalies,  # percentile
                decoded_normal_test, df_anomalies, decoded_anomal,
                df_noAnomalies, decoded_normal)

    # compute detailed autoencoder statistics
    decoded_features = ae_model.predict(df_tensor)
    pred_stats = ae.evaluate_predictions_semisup(
            decoded_features, df_tensor)

    print("[adssae:semisup_autoencoder] Accuracy Statistics")
    print(acc_stats)
    #print(pred_stats)

    # ae_stats = {**acc_stats, **pred_stats}  cambiata!
    ae_stats = {**acc_stats, **pred_stats, 'val_loss': history.history['val_loss']}

    # save the trained model and associated statistics
    if save:
        print("[adssae:semisup_autoencoder] Saving model and stats")
        ae_model_name = 'semisup_ae_{}_{}'.format(user_id, task_id)
        ae_model_fname, ae_stats_file, ae_scaler_file = _save_trained_model(ae_model_name, ae_model, ae_stats, scaler)
        return ae_model_fname, ae_scaler_file, ae_stats_file

    return ae_model, scaler, ae_stats


def semisup_detection_inference(df_fname, model_name, sep=',', 
        user_id='default', task_id='0.0', anomaly_threshold=None, 
        scaler=None):
    '''
    anomalyDetection.semisup_detection_inference::Exploits a semi-supervised
    autoencoder to detect anomalies; the autoencoder model needs to have been
    previously trained and the detection-threshold computed.  This function can
    be only used at "inference" time, when new data arrive and need to be
    checked for anomalies

    :param df_fname : str
        filename of the data to be classified in either anomaly or normal points
        - no labels are provided in the data 
        - the data need to have been normalized

    :params model_name : string
        name of the model (autoencoder) already trained and stored

    :params anomaly_threshold : float
        threshold to be used for distinguish between normal and anomalous
        points. It should have been computed while training the autoencoder.
        Optional: if missing, the original threshold used during the model's
        creation will be used

    :params scaler : Sklearn scaler object
        the scaler used to normalize the data. Optional: if missing, the
        original scaler used during the model's creation will be used

    :params user_id : str
        user identifier
    :params task_id : str
        task identifier

    :return : Python list
        the computed labels expressed as a list of integer values, one label for
        each data point included in df
        - 0 indicates normal points, 1 indicates anomalous ones
    '''
    # load the trained model and corollary files
    model, stats, old_scaler = _load_trained_model(model_name)
    
    # if the user din't provide an error threshold, use the one computed during
    # the model's validation
    if anomaly_threshold == None:
        anomaly_threshold = stats['err_threshold']

    # if the user din't provide a scaler, use the one computed during
    # the model's training
    if scaler == None:
        scaler = old_scaler

    # read data from file
    df_fname = '{}/{}'.format(volume_dir, df_fname)
    df = pd.read_csv(df_fname, sep=sep)

    # preprocess the data
    df.columns = df.columns.str.replace(' ', '')
    df, labels, scaler = gs._preprocess(df, 'label', scaler)

    df_tensor = df.values
    preds = model.predict(df_tensor)

    # compute errors
    nn_samples, nn_series = df_tensor.shape

    errors = [0] * nn_samples
    abs_errors = {}
    for j in range(nn_series):
        abs_errors[j] = []
    for i in range(nn_samples):
        for j in range(nn_series):
            abs_errors[j].append(abs(preds[i][j] - df_tensor[i][j]))
    # max abs error 
    for j in abs_errors.keys():
        for i in range(len(abs_errors[j])):
            if errors[i] < abs_errors[j][i]:
                errors[i] = abs_errors[j][i]

    # classify samples
    labels = []
    for e in errors:
        if e > anomaly_threshold:
            labels.append(1)
        else:
            labels.append(0)

    #print("[adssae:semisup_detection_inference] Predicted labels:")
    #print(labels)

    print("[adssae:semisup_detection_inference] Saving labels")
    # save the predicted labels
    labels_file = '{}{}_predLabels.pickle'.format(out_dir, model_name)
    gs.serialize_py_obj(labels, labels_file)
    return labels


def sup_autoencoder_classr(df_fname, sep=',', user_id='default', task_id='0.0', 
        hparams_file_ae=None, hparams_file_classr=None, n_percentile=-1, save=True):
    '''
    anomalyDetection.sup_autoencoder_classr::Creates a supervised model for
    anomaly detection composed by an unsupervised autoencoder (feature
    extraction), then connected to a classifier layer trained in a supervised
    fashion.
    
    :param df_fname : str
        name of the csv file with data to be used for training and testing. One
        column has to termed "label" and it has to contains the classes of the
        example - 0 means normal data point, any other integer number
        corresponds to an anomalous data point

    :params user_id : str
        user identifier

    :params task_id : str
        task identifier

    :params hparams_file_ae: str
        name of the pickle file containing the Python dictionary with
        hyperparameters to be used to build the autoencoder 

    :params hparams_file_classr: str
        name of the pickle file containing the Python dictionary with
        hyperparameters to be used to build the classifier

    :return : string
        name of the file containing saved Keras model (the trained autoencoder
        plus the classifier)
    :return : string
        name of the pickle file containing the scikit-learn object with scaler
        object used to normalize the data
    :return : string 
        name of the pickle file containing Python dictionary with  the summary
        of the detailed statistics computed on the classifier
    '''
    # sanitize input string parameters
    user_id = gs.sanitize(user_id)
    task_id = gs.sanitize(task_id)

    # read data from file
    df_fname = '{}/{}'.format(data_dir, df_fname)
    df = pd.read_csv(df_fname, sep=sep)

    if hparams_file_ae == None:
        hparams_ae = default_hparams_sup
    # read hyperparameters for autoencoder from binary file (pickle object)
    else:
        # hparams_ae = gs.load_py_obj(hparams_file_ae) TODO just for testing purpose I can pass the params as a dictionary
        hparams_ae = hparams_file_ae

    if hparams_file_classr == None:
        hparams_classr = default_hparams_classr_sup
    # read hyperparameters for classifier from binary file (pickle object)
    else:
        #hparams_classr = gs.load_py_obj(hparams_file_classr) TODO just for testing purpose I can pass the params as a dictionary
        hparams_classr = hparams_file_classr

    df.columns = df.columns.str.replace(' ', '')
    df, labels, scaler = gs._preprocess(df, 'label')

    df_tensor = df.values
    n_samples, n_features = df_tensor.shape

    # data split in test/train
    x_train, x_test, y_train, y_test = sklms.train_test_split(df_tensor, 
            labels, test_size=0.3, shuffle=True, random_state=42)

    # create autoencoder model
    ae_model, enc_model = ae.autoencoder(n_features, hparams_ae)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, min_delta=1e-5) 
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', patience=5, min_lr=1e-5, factor=0.2)

    print("[adssae:sup_autoencoder_classr] Autoencoder model:")
    ae_model.summary()

    print("[adssae:sup_autoencoder_classr] Begin unsupervised training")

    history = ae_model.fit(x_train, x_train, epochs=hparams_ae['epochs'], 
            batch_size=hparams_ae['batch_size'], shuffle=hparams_ae['shuffle'], 
            callbacks=[early_stopping, reduce_lr],
            validation_split=0.1, verbose=0)

    print("[adssae:sup_autoencoder_classr] Unsupervised training concluded")

    # add classifier to encoder model
    classr_model = ae.create_classifier_on_encoder(n_features, hparams_classr, 
            enc_model)

    # freeze encoder layers weights 
    for l in enc_model.layers:
        l.trainable = False

    print("[adssae:sup_autoencoder_classr] Classifier model:")
    classr_model.summary()

    print("[adssae:sup_autoencoder_classr] Classifier training")
    history = classr_model.fit(x_train, y_train, 
            epochs=hparams_classr['epochs'], 
            batch_size=hparams_classr['batch_size'], 
            shuffle=hparams_classr['shuffle'], 
            callbacks=[early_stopping, reduce_lr],
            validation_split=0.1, verbose=0)

    print("[adssae:sup_autoencoder_classr] Classifier training concluded")

    # predictions
    preds = classr_model.predict(x_test)
    pred_classes = ae.assign_class_sup(preds)

    print("[adssae:sup_autoencoder_classr] Accuracy Statistics")
    pred_stats = ae.evaluate_predictions_sup(pred_classes, y_test.tolist())
    print(pred_stats)


    # save the trained model and associated statistics
    if save:
        print("[adssae:sup_autoencoder_classr] Saving model and stats")
        model_name = 'sup_ae_clssr_{}_{}'.format(user_id, task_id)

        model_fname, stats_file, scaler_file = _save_trained_model(model_name, classr_model, pred_stats, scaler)

    return classr_model, scaler, pred_stats


def sup_detection_inference(df_fname, model_name, sep=',', 
        user_id='default', task_id='0.0', scaler=None):
    '''
    anomalyDetection.isup_detection_inference::Exploits a supervised classifier
    to detect anomalies; the classifier model needs to have been previously
    trained. This function can be only used at "inference" time, when new data
    arrive and need to be checked for anomalies

    :param df : dataframe
        data to be classified in either anomaly or normal points
        - no labels are provided in the data 
        - the data need to have been normalized

    :params model_name : string
        name of the model already trained and stored

    :params scaler : Sklearn scaler object
        the scaler used to normalize the data. Optional: if missing, the
        original scaler used during the model's creation will be used

    :params user_id : str
        user identifier
    :params task_id : str
        task identifier

    :return : Python list
        the computed labels expressed as a list of integer values, one label for
        each data point included in df
        - 0 indicates normal points, 1 indicates anomalous ones
    '''
    # load the trained model and corollary files
    model, stats, old_scaler = _load_trained_model(model_name)
    
    # if the user din't provide a scaler, use the one computed during
    # the model's training
    if scaler == None:
        scaler = old_scaler

    # read data from file
    df_fname = '{}/{}'.format(volume_dir, df_fname)
    df = pd.read_csv(df_fname, sep=sep)

    # preprocess the data
    df.columns = df.columns.str.replace(' ', '')
    df, labels, scaler = gs._preprocess(df, '', scaler)

    df_tensor = df.values
    preds = model.predict(df_tensor)
    pred_classes = ae.assign_class_sup(preds)

    print("[adssae:sup_detection_inference] Predicted labels:")
    print(pred_classes)
    print("[adssae:sup_detection_inference] Saving labels")
    # save the predicted labels
    labels_file = '{}{}_predLabels.pickle'.format(out_dir, model_name)
    gs.serialize_py_obj(pred_classes, labels_file)


if __name__ == '__main__':
    service_type = int(sys.argv[1])
    service_args = sys.argv[2:]
    # semi-supervised anomaly detection AE, training
    if service_type == 0:  
        if len(service_args) == 1:
            semisup_autoencoder(service_args[0])
        elif len(service_args) == 2:
            semisup_autoencoder(service_args[0], service_args[1])
        elif len(service_args) == 3:
            semisup_autoencoder(service_args[0], service_args[1], 
                    service_args[2])
        elif len(service_args) == 4:
            semisup_autoencoder(service_args[0], service_args[1], 
                    service_args[2], service_args[3])
        elif len(service_args) == 5:
            semisup_autoencoder(service_args[0], service_args[1], 
                    service_args[2], service_args[3], service_args[4])
        elif len(service_args) == 6:
            semisup_autoencoder(service_args[0], service_args[1], 
                    service_args[2], service_args[3], service_args[4],
                    service_args[5])
        else:
            print("[adssae] Wrong arguments {}".format(service_args))
    # semi-supervised anomaly detection AE, inference
    elif service_type == 1:  
        if len(service_args) == 2:
            semisup_detection_inference(service_args[0], service_args[1])
        elif len(service_args) == 3:
            semisup_detection_inference(service_args[0], service_args[1],
                    service_args[2])
        elif len(service_args) == 4:
            semisup_detection_inference(service_args[0], service_args[1],
                    service_args[2], service_args[3])
        elif len(service_args) == 5:
            semisup_detection_inference(service_args[0], service_args[1],
                    service_args[2], service_args[3], service_args[4])
        elif len(service_args) == 5:
            semisup_detection_inference(service_args[0], service_args[1],
                    service_args[2], service_args[3], service_args[4], 
                    service_args[5])
        elif len(service_args) == 6:
            semisup_detection_inference(service_args[0], service_args[1],
                    service_args[2], service_args[3], service_args[4], 
                    service_args[5], service_args[5])
        else:
            print("[adssae] Wrong arguments {}".format(service_args))
    # supervised anomaly detection AE + classifier, training
    if service_type == 2:  
        if len(service_args) == 1:
            sup_autoencoder_classr(service_args[0])
        elif len(service_args) == 2:
            sup_autoencoder_classr(service_args[0], service_args[1])
        elif len(service_args) == 3:
            sup_autoencoder_classr(service_args[0], service_args[1], 
                    service_args[2])
        elif len(service_args) == 4:
            sup_autoencoder_classr(service_args[0], service_args[1], 
                    service_args[2], service_args[3])
        elif len(service_args) == 5:
            sup_autoencoder_classr(service_args[0], service_args[1], 
                    service_args[2], service_args[3], service_args[4])
        elif len(service_args) == 6:
            sup_autoencoder_classr(service_args[0], service_args[1], 
                    service_args[2], service_args[3], service_args[4],
                    service_args[5])
        elif len(service_args) == 7:
            sup_autoencoder_classr(service_args[0], service_args[1], 
                    service_args[2], service_args[3], service_args[4],
                    service_args[5], service_args[6])
        else:
            print("[adssae] Wrong arguments {}".format(service_args))
    # supervised anomaly detection AE + classifier, inference
    elif service_type == 3:  
        if len(service_args) == 2:
            sup_detection_inference(service_args[0], service_args[1])
        elif len(service_args) == 3:
            sup_detection_inference(service_args[0], service_args[1],
                    service_args[2])
        elif len(service_args) == 4:
            sup_detection_inference(service_args[0], service_args[1],
                    service_args[2], service_args[3])
        elif len(service_args) == 5:
            sup_detection_inference(service_args[0], service_args[1],
                    service_args[2], service_args[3], service_args[4])
        elif len(service_args) == 5:
            sup_detection_inference(service_args[0], service_args[1],
                    service_args[2], service_args[3], service_args[4], 
                    service_args[5])
        elif len(service_args) == 6:
            sup_detection_inference(service_args[0], service_args[1],
                    service_args[2], service_args[3], service_args[4], 
                    service_args[5], service_args[5])
        else:
            print("[adssae] Wrong arguments {}".format(service_args))

    else:
        print('[adssae] Unsupported option')
