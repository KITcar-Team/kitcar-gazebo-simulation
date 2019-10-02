#!/bin/bash

GENERATION=$KITCAR_REPO_PATH/kitcar-gazebo-utils/car_model/generate_dr_drift.py
INPUT_YAML=$(pwd)/dr_drift.yaml
OUT_CAM_YAML=$KITCAR_REPO_PATH/kitcar-gazebo-simulation/src/gazebo_simulation/param/simulation_car_specs/dr_drift/camera.yaml
OUT_MODEL_SDF=$(pwd)/model.sdf

#Call generation script, which translates the road.xml into the commonroad language.
python3 $GENERATION $INPUT_YAML -c $OUT_CAM_YAML -m $OUT_MODEL_SDF

