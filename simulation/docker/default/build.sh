#!/bin/bash
# Arguments:
# CI registry
CI_REGISTRY=$1
# TAG name
TAG_NAME=$2
# KITCAR_REPO_PATH within the container
KITCAR_REPO_PATH_IN_IMAGE=$3

# Before building the docker image, the init files
# need to be linked into the scope of the Dockerfile!
rm -rf kitcar-gazebo-simulation/
mkdir kitcar-gazebo-simulation
# Copy outside kitcar-gazebo-simulation repo into the scope of the dockerfile
cp -r ${KITCAR_REPO_PATH}/kitcar-gazebo-simulation /tmp/kitcar-gazebo-simulation
cp -r /tmp/kitcar-gazebo-simulation .
rm -rf /tmp/kitcar-gazebo-simulation

# Build the image
docker build -t $CI_REGISTRY/kitcar/kitcar-gazebo-simulation:$TAG_NAME  --build-arg KITCAR_REPO_PATH=$KITCAR_REPO_PATH_IN_IMAGE .

# Clean up copied files
rm -rf kitcar-gazebo-simulation/
