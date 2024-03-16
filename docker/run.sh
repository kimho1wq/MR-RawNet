sudo docker run --runtime=nvidia --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host \
-v $PWD:/workspace/MR-RawNet/ \
-v /home/kimho1wq/DB:/workspace/MR-RawNet/DB mr_rawnet
