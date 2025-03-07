#!/bin/bash
xhost +local:

docker run -it \
    --runtime=nvidia \
    --gpus=all \
    --shm-size=64g \
    -v /home/ivanov_la/hdmap/MapNet:/home/ivanov_la/MapNet \
    -v /datasets/nuScenes2d/can_bus:/home/ivanov_la/MapNet/data/can_bus \
    -v /datasets/nuScenes2d/nuscenes:/home/ivanov_la/MapNet/data/nuscenes \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    mapnet
