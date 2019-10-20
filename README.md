# KITcar gazebo simulation

## Setup

This repository can be used in combination with `kitcar-ros` and `kitcar-gazebo-utils`. Therefore the init script must be called. Change into `kitcar-gazebo-simulation` folder and run the script:
```
cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation
./init/init.sh
```
Furthermore the following packages must be installed:
```
sudo apt-get install -y \
ros-melodic-gazebo-ros-control ros-melodic-gazebo-ros-pkgs \
python3 gazebo9 libgazebo9-dev python-opencv python3-pip \
libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0
```
and the python requirements installed:
```
pip3 install -r requirements.txt
```
Then build with `catkin_make`.
Other information can be found in [KITcar-Gazebo-Simulation Wiki](https://wiki.kitcar-team.de/doku.php?id=technik:repos:kitcar-gazebo-simulation:start).

