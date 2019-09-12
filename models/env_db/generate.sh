#!/bin/bash
#Define gazebo-simulation and gazebo-utils-path
UTILS=$KITCAR_REPO_PATH/kitcar-gazebo-utils
ROAD=$KITCAR_REPO_PATH/kitcar-gazebo-simulation/models/env_db/$1

python3 $KITCAR_REPO_PATH/kitcar-gazebo-utils/road-generation/road-generator.py $ROAD/road.xml -o $ROAD/commonroad.xml  
python3 $UTILS/road-generation/gazebo-renderer.py $ROAD/commonroad.xml -o $ROAD -n $1 -g $UTILS/road-generation --force

