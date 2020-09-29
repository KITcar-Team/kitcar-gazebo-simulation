#!/bin/bash
SERVICE=$1
TAG=$2
PARENT_TAG=$3

# Before building the docker image, the init files
# need to be linked into the scope of the Dockerfile!
rm -rf init/
mkdir init
ln ../../../init/* init/

# Set env variable used by docker-compose to create the image's tag
export CI_IMAGE_TAG=$TAG
export CI_PARENT_TAG=$PARENT_TAG
# Build the image
docker-compose build $SERVICE



# Clean up
rm -rf init/
