#!/usr/bin/python3.6
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os
import sys
#sys.path.insert(0, os.path.abspath('..'))
#sys.path.insert(0, os.path.abspath('../../../servises/super_semisuper_anomaly_detection_services/src/'))
#sys.path.insert(0, os.path.abspath('../src/utils'))

from anomalyDetection import semisup_autoencoder, semisup_detection_inference

volume_dir = 'data/'
#file_name = 'ae_test.csv'
#file_name_inference = 'ae_test_inference.csv'
file_name = 'train_caravan-insurance-challenge.csv'
file_name_inference = 'test_caravan-insurance-challenge.csv'

keras_model, scaler, sum = semisup_autoencoder(volume_dir+file_name, sep=',')
print(keras_model)  # ../trained_models/semisup_ae_default_00
print(scaler)       # ../trained_models/semisup_ae_default_00_scaler.save
print(sum)          # ../out/semisup_ae_default_00_stats.pickle

semisup_detection_inference(volume_dir+file_name, 'semisup_ae_default_00', sep=',')

