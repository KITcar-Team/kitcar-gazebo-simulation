#!/bin/bash
CI_REGISTRY=$1
# Before building the docker image, the init files
# need to be linked into the scope of the Dockerfile!
rm -rf init/
mkdir init
ln ../../../init/* init/
# Build the image
docker build -t $CI_REGISTRY/kitcar/kitcar-gazebo-simulation/ci:focal .
# Clean up
rm -rf init/
