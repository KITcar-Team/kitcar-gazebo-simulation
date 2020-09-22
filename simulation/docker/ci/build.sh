#!/bin/bash
CI_REGISTRY=$1
TAG=$2

# If tags start with docs or machine_learning, additional packages and dependencies are installed.
if [ "$TAG" = "docs*" ] ; then BUILD_ARGS="$BUILD_ARGS --build-arg INSTALL_DOC_PACKAGES=true --build-arg INSTALL_ML_PACKAGES=true"; fi
if [ "$TAG" = "machine_learning*" ] ; then BUILD_ARGS="$BUILD_ARGS --build-arg INSTALL_ML_PACKAGES=true --build-arg INSTALL_NODE_JS=true"; fi

# Before building the docker image, the init files
# need to be linked into the scope of the Dockerfile!
rm -rf init/
mkdir init
ln ../../../init/* init/
# Build the image
docker build -t $CI_REGISTRY/kitcar/kitcar-gazebo-simulation/ci:$TAG $BUILD_ARGS .
# Clean up
rm -rf init/
