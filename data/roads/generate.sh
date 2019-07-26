#!/bin/bash
ROAD=$1
python3 $KITCAR_GAZEBO_GENERATION_PATH/road-generator.py $KITCAR_GAZEBO_DATA_PATH/roads/$ROAD/road.xml -o $KITCAR_GAZEBO_DATA_PATH/roads/$ROAD/commonroad.xml  
python3 $KITCAR_GAZEBO_GENERATION_PATH/gazebo-renderer.py $KITCAR_GAZEBO_DATA_PATH/roads/$ROAD/commonroad.xml -o $KITCAR_GAZEBO_DATA_PATH/roads/$ROAD -m $KITCAR_GAZEBO_DATA_PATH/materials -g $KITCAR_GAZEBO_GENERATION_PATH --force

