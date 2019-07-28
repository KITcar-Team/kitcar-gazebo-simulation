#!/bin/bash
ROAD=$KITCAR_GAZEBO_SIM_PATH/data/roads/$1
python3 $KITCAR_GAZEBO_GENERATION_PATH/road-generator.py $ROAD/road.xml -o $ROAD/commonroad.xml  
python3 $KITCAR_GAZEBO_GENERATION_PATH/gazebo-renderer.py $ROAD/commonroad.xml -o $ROAD -m $KITCAR_GAZEBO_SIM_PATH/data/materials -g $KITCAR_GAZEBO_GENERATION_PATH --force

