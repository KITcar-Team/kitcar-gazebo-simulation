#!/bin/bash

ROAD_GENERATION_PATH=$KITCAR_REPO_PATH/kitcar-gazebo-simulation/utils #Path to road-generation folder
#Read input flags
FLAGS=""
while getopts "st:br:" option
do
	case "${option}"
		in
		s) FLAGS="$FLAGS --segmentation";;
		t) FLAGS="$FLAGS --tile_size=${OPTARG}";;
		b) FLAGS="$FLAGS --background";;
		l) FLAGS="$FLAGS --lines";;
		r) ROAD_NAME="${OPTARG}";; # Name of the road e.g. default-road

	esac
done

ROAD_FOLDER=$KITCAR_REPO_PATH/kitcar-gazebo-simulation/models/env_db/$ROAD_NAME # Complete path of the road folder


#Call road generation script, which translates the road.xml into the commonroad language.
#Expected arguments are the road.xml filepath and an (optional) output flag -o which specifies where the commonroad should be written to.
python3 $ROAD_GENERATION_PATH/road-generator.py $ROAD_FOLDER/road.xml -o $ROAD_FOLDER/commonroad.xml

#Call gazebo renderer which renders the commonroad.xml into a gazebo world
#Inputs are:
# commonroad file paths
#-o output folder
#-n name of road
#-g generation_dir: directory where the road generation code is located
#--force is needed to overwrite existing files
#Optional:
#--segmentation: Enables colored rendering for color segmentation
#--tile_size=...: Quadratic tiles of given size are extracted as gazebo models
#--background: (currently broken): pictures of "maschinenhalle" are inserted in the background to simulate more realistic driving
python3 $ROAD_GENERATION_PATH/gazebo-renderer.py $ROAD_FOLDER/commonroad.xml -o $ROAD_FOLDER -n $ROAD_NAME -g $ROAD_GENERATION_PATH --force $FLAGS

