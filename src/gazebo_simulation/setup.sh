source /usr/share/gazebo/setup.sh
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:$(rospack find gazebo_simulation)/data/models/sensors:$(rospack find gazebo_simulation)/data/models/vehicles:$(rospack find gazebo_simulation)/data/models/misc:$(rospack find gazebo_simulation)/data/models/objects
export GAZEBO_RESOURCE_PATH=${GAZEBO_RESOURCE_PATH}:$(rospack find gazebo_simulation)/data/worlds:$(rospack find gazebo_simulation)/data/media

