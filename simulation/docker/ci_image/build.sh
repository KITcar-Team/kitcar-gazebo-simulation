#!/bin/bash
# Before building the docker image, the init files
# need to be linked into the scope of the Dockerfile!
rm -rf init/
mkdir init
ln ../../../init/* init/
# Build the image
docker build -t git.kitcar-team.de:4567/kitcar/kitcar-gazebo-simulation:focal .
