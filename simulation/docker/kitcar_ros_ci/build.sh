#!/bin/bash

CI_REGISTRY=$1
# Parent tag
PARENT_TAG=$2
# Tag name
TAG=$3
# Build the image
docker build -t "$CI_REGISTRY"/kitcar/kitcar-gazebo-simulation/kitcar_ros_ci:"$TAG" --build-arg PARENT_TAG="${PARENT_TAG}" .
