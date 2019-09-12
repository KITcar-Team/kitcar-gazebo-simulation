#!/bin/bash

#Define gazebo-simulation and gazebo-utils-path
KITCAR_GAZEBO_SIM_PATH=$KITCAR_REPO_PATH/kitcar-gazebo-simulation
KITCAR_GAZEBO_UTILS_PATH=$KITCAR_REPO_PATH/kitcar-gazebo-utils

ROAD=$KITCAR_GAZEBO_SIM_PATH/models/env_db/$1
python3 $KITCAR_GAZEBO_UTILS_PATH/road-generation/road-generator.py $ROAD/road.xml -o $ROAD/commonroad.xml  
python3 $KITCAR_GAZEBO_UTILS_PATH/road-generation/gazebo-renderer.py $ROAD/commonroad.xml -o $ROAD -n $1 -g $KITCAR_GAZEBO_UTILS_PATH/road-generation --force --tile_size=$2

