# Unsupervised anomaly detection services

This repository contains the implementation of the ML-based services for
unsupervised anomaly detection and status change tracking. The techniques used
to develop these services are Functional Isolation Forest (FIF) and the ADTK
module (Anomaly Detection Tool Kit). 

The unsupervised anomaly detection services can be used as standalone web
application, following the instruction contained in this document. 
The functionalities implemented here will be merged in the IoTwinws
infrastructure during the project continuation.

## Requirements 

The code requires Python >= 3.8 to be executed.
Docker needs to be installed as well. 

The following modules are required as well (can be installed as using the
following command: pip install <module_name>):
- plotly>= 4.6.0
- dash==1.14
- dash-table==4.9
- dash-upload-components==0.0.2
- dash-core-components==1.10.1
- dash-html-components==1.0.3
- Flask==1.1.2
- boto3==1.14.36
- botocore==1.17.36
- numpy==1.19.1
- pandas==1.1.0
- scipy==1.5.2
- gunicorn==19.9.0
- scikit-learn==0.23.2
- urllib3==1.25.10
- dill==0.3.2
- adtk==0.6.2
- cufflinks==0.17.3


## Installation instruction

To access the component containing the anomaly detection services, download the
compressed archive file containing the code and go to the path of the destination
directory, e.g. <tar_src_dir>
- cd <tar_src_dir>

Load the Docker image:
- sudo docker load --input <tar_src_dir>

To check that the image has been properly loaded run the command:
- sudo docker images 

Launch the container with a docker run command:
- sudo docker run -p 8058:8058 <component_name>

Now the component is now running on http://127.0.0.1:8080 address and port,
which can be opened in a browser. 


