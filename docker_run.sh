#!/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

set -e

# Wrapper script for setting up `docker run` to properly
# cache downloaded files, custom extension builds and
# mount the source directory into the container and make it
# run as non-root user.

rest=$@

IMAGE="${IMAGE:-crpi-pxcwlfv1s57m718e.cn-shanghai.personal.cr.aliyuncs.com/seventen/diffuser:v1}"

CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${IMAGE} 2> /dev/null)
if [[ "${CONTAINER_ID}" ]]; then
    docker run --shm-size=2g --gpus all -it --rm -v `pwd`:/scratch --user $(id -u):$(id -g) \
        --workdir=/scratch -e HOME=/scratch $IMAGE $@
else
    echo "Unknown container image: ${IMAGE}"
    exit 1
fi