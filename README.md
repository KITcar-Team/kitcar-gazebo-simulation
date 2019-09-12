# KITcar gazebo simulation

## Setup

This repository can be used in combination with `kitcar-ros` and `kitcar-gazebo-utils`. Therefore the init script must be called. Change into `kitcar-gazebo-simulation` folder and run the script:
```
cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation
./init/init.sh
```
Furthermore the following packages must be installed:
```
sudo apt-get install ros-melodic-gazebo-ros-control ros-melodic-gazebo-ros-pkgs
```
and `kitcar-gazebo-simulation` build with `catkin_make`.
Other information can be found in [KITcar-Gazebo-Simulation Wiki](https://wiki.kitcar-team.de/doku.php?id=konzept_und_umsetzung:gazebo_simulation:kitcar-gazebo-simulation).

