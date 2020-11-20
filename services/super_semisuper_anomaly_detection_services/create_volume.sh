# This script creates the volume and populates it with required folders
docker volume create mlservice_vol

docker run -it --rm --name test --mount \
    source=mlservice_volume,target=/root/mlservice_volume busybox \
    mkdir /root/mlservice_volume/trained_models

docker run -it --rm --name test --mount \
    source=mlservice_volume,target=/root/mlservice_volume busybox \
    mkdir /root/mlservice_volume/out
