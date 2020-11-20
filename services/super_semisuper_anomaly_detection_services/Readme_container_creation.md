# How to create a docker-based service  #

1) Create docker image file
   - Create docker file with preferred editor inside the &lt;docker_dir> directory
   - Set up desired files and scripts inside &lt;docker_dir>
   - One Docker container per service
2) Build docker image file 
   - Command: `docker build -f <docker_file> -t <IP><port><container_name> <docker_dir>`
   - Example: `docker build -f Dockerfile_ML -t localhost:5000/mlservice:ml .`
3) Run the docker file
   1) Direct (blocking) execution
       - Command: `docker run <IP><port>/<container_name>`
       - Example `docker run localhost:5000/mlservice:ml`
   2) Daemon execution
      - Command: `docker run -d <IP><port>/<service_name>`
      - Example: `docker run -d localhost:5000/mlservice:ml`
4) I/O
   1) Docker volume
      1) Create volume
         - Command: `docker volume create <vol_name>`
         - Example: `docker volume create mlservice_vol`
      2) Create and build the docker container (see points 1 and 2)
      3) Run container with mounted volume
         - Command: `docker run --mount source=<vol_name>,target=<target_mount_point> <IP><port>/<service_name>`
         - Example: `docker run --mount source=mlservice_vol,target=/root/mlservice_vol localhost:5000/mlservice:ml`
      4) Inspect volume: 
         - Command: `docker inspect <vol_name>`
         - Example: `docker inspect mlservice_vol`
      5) Look at files inside volume: 
         - Command: `docker run -it --mount source=<vol_name>,target=<target_mount_point> <IP><port>/<service_name> find <target_mount_point>`
   2) Bind mount (4.1 is preferred)
      1) Run with bind mount in host file system
         - Command: `docker run --mount='type=bind,src=<src_dir_host>,dst=<dst_dir_cointainer>' <IP><port>/<container_name>`
         - Example: `docker run --mount='type=bind,src=/home/b0rgh/ioTwins/WP3/AI_services/docker/Docker/data,dst=/data' localhost:5000/mlservice:ml`
   3) Add data to existing volume:
      1) Create temporary docker image: 
         - Command: `docker run -it --mount source=<vol_name>,target=<target_mount_point> --name <temp_img> <simple_img> true`
         - Example: `docker run -it --mount source=mlservice_volume,target=/root/mlservice_volume --name temp busybox true`
      2) Copy data: 
         - Command: `docker cp <file_to_be_copied> <temp_img>:<target_mount_point>`
         - Example: `docker cp data/ae_test.csv temp:/root/mlservice_volume`
      3) Remove temporary docker image: 
         - Command: `docker rm <temp_img>`
   4) Inspect volume content
      - Command: `docker run -it --rm --name test --mount source=<vol_name>,target=<target_mount_point> busybox /bin/sh`
      - Example: `docker run -it --rm --name test --mount source=mlservice_volume,target=/root/mlservice_volume busybox /bin/sh`

# How to launch the different services #

## General Services ##
### Get categorical continuous features
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:getCatContFeats <df_filename>
```

### Encode a precise categorical feature
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:encodeCatFeat <df_filename> <cat_feature_name>
```

### Encode list of categorical features via one-hot scheme
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:onehotEncodeFeats <df_filename> [<cat_feature_names>]
```
- `<cat_feature_names>` is a list of strings (surround by \'string '\, wrapped by
  [..] and separated by comma (no space); it contains the list of categorical
  columns

Example: 

```
docker run -it --mount
  source=mlservice_volume,target=/root/mlservice_volume \
  localhost:5000/mlservice:onehotEncodeFeats \
  flights.csv [\'carrier\',\'origin'\]
```

### Normalize data
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:norm <df_filename> [<cat_feature_names>] \
  [<cont_feature_names>] [<scaler_filename>] [<scaler_type>]
```

- `<cat_feature_names>` is a list of strings (surround by \'string '\, wrapped by
  [..] and separated by comma (no space); it contains the list of categorical
  columns
- `<cont_feature_names>` is a list of strings, wrapped by [..] and separated by
  comma (no space); it contains the list of continuous-valued columns

### Preprocess data
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:preproc <df_filename> [<label_name>] [<scaler_filename>]
```
Example:
```
docker run -it --mount
  source=mlservice_volume,target=/root/mlservice_volume \
  localhost:5000/mlservice:preproc flights.csv
```

### Split an unlabeled dataset
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:splitUnlabel <df_filename> [<split_ratio>] 
```

### Split a labeled dataset
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:splitLabel <df_filename> <label_name> [<split_ratio>] 
```

### Oversample a labeled dataset
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:oversample <X_filename> <y_filename> [<method>] \
  [<strategy>]
```

- `<X_filename>` is name of the file containing the Python list or numpy array of
  features to be oversampled 
- `<y_filename>` is name of the file containing the Python list of labels to be
  oversampled (associated to the input features)

### Sanitize a string
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:sanitize <input_string> 
```

- `<input_string>` is the string to be sanitized


## Anomaly Detection ##

### Build and train autoencoder (semi-supervised approach)
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:ad_ssAE_train <df_filename> [<separator>] [<user_id>] \
  [<task_id>] [<hyperparameters_file>] [<n_percentile>]
```

- `<separator>` indicates the separator symbol used in the csv; it must be
  specified from command by encasing in ''

Example: 
```
docker run -it --mount
  source=mlservice_volume,target=/root/mlservice_volume \
  localhost:5000/mlservice:ad_ssAE_train ae_test.csv ';'
```

### Perform inference on a trained DL model (semi-supervised approach)
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:mlservice:ad_ssAE_infer <df_filename> <trained_model> \
  [<separator>] [<user_id>] [<task_id>] [<detection_threshold>] [<scaler>]
```

- the `<detection_threshold>` is the value used to distinguish between anomalous
  and normal data points: examples with reconstruction error greater than the
`<detection_threshold>` are classified as anomalies, normal otherwise
- `<scaler>` is a scikit-learn scaler that can be passed (for instance, the same
  scaler used during training)

### Build and train autoencoder (supervised approach)
```
 docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:ad_ssAE_train <df_filename> [<separator>] [<user_id>] \
  [<task_id>] [<autoencoder_hyperparameters_file>] \
  [<classifier_hyperparameters_file>]
```

- `<separator>` indicates the separator symbol used in the csv; it must be
  specified from command by encasing in ''

```
docker run -it --mount
  source=mlservice_volume,target=/root/mlservice_volume \
  localhost:5000/mlservice:ad_sAEclassr_train ae_test.csv ';'
```

### Perform inference on a trained DL model (supervised approach)
```
docker run -it --mount
  source=<vol_name>,target=<target_mount_point> \
  <IP><port>/mlservice:mlservice:ad_sAEclassr_infer <df_filename> \
  <trained_model> [<separator>] [<user_id>] [<task_id>] [<scaler>]
```

- `<scaler>` is a scikit-learn scaler that can be passed (for instance, the same
  scaler used during training)
