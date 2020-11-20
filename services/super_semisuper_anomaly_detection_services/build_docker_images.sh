# This script builds all the Docker images corresponding to the different ML
# services. This needs to be run in the same folder containing the docker images

# General Services 
docker build -f imgs/Dockerfile_ML_getCatContFeats -t \
    localhost:5000/mlservice:getCatContFeats .
docker build -f imgs/Dockerfile_ML_encodeCatFeat -t \
    localhost:5000/mlservice:encodeCatFeat .
docker build -f imgs/Dockerfile_ML_onehotEncodeFeats -t \
    localhost:5000/mlservice:onehotEncodeFeats .
docker build -f imgs/Dockerfile_ML_norm -t localhost:5000/mlservice:norm .
docker build -f imgs/Dockerfile_ML_preproc -t localhost:5000/mlservice:preproc .
docker build -f imgs/Dockerfile_ML_splitUnlabel -t \
    localhost:5000/mlservice:splitUnlabel .
docker build -f imgs/Dockerfile_ML_splitLabel -t \
    localhost:5000/mlservice:splitLabel .
docker build -f imgs/Dockerfile_ML_oversample -t \
    localhost:5000/mlservice:oversample .

# Anomaly Detection with semisupervised autoencoder
docker build -f imgs/Dockerfile_ML_anomalyDetect_semisupAE_train -t \
    localhost:5000/mlservice:ad_ssAE_train .
docker build -f imgs/Dockerfile_ML_anomalyDetect_semisupAE_infer -t \
    localhost:5000/mlservice:ad_ssAE_infer .

# Anomaly Detection with autoencoder plus supervised classifier
docker build -f imgs/Dockerfile_ML_anomalyDetect_supAE_train -t \
    localhost:5000/mlservice:ad_sAEclassr_train .
docker build -f imgs/Dockerfile_ML_anomalyDetect_supAE_infer -t \
    localhost:5000/mlservice:ad_sAEclassr_infer .



