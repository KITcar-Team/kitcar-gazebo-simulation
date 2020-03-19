#!/bin/bash

GENERATION=$KITCAR_REPO_PATH/kitcar-gazebo-simulation/utils/car_model/generate_dr_drift.py
INPUT_YAML=$(pwd)/dr_drift.yaml
BASE_MODEL_SDF=$(pwd)/model_base.sdf
OUT_CAM_YAML=$KITCAR_REPO_PATH/kitcar-gazebo-simulation/src/simulation_brain_link/param/car_specs/dr_drift/camera.yaml
OUT_DEPTH_CAM_YAML=$KITCAR_REPO_PATH/kitcar-gazebo-simulation/src/simulation_brain_link/param/car_specs/dr_drift/depth_camera.yaml
OUT_MODEL_SDF=$(pwd)/model.sdf

#Call generation script, which translates the road.xml into the commonroad language.
python3 $GENERATION $INPUT_YAML -b $BASE_MODEL_SDF -c $OUT_CAM_YAML -d $OUT_DEPTH_CAM_YAML -m $OUT_MODEL_SDF

