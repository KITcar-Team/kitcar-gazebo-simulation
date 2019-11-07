# KITcar gazebo simulation

## Setup

Clone this repository in the same directory as `kitcar-ros`. Make sure that the path variable *$KITCAR_REPO_PATH* is set to the repository where your `kitcar-ros` installation is.

Therefore the init script must be called. Change into `kitcar-gazebo-simulation` folder and run the script:
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

At this point make sure that you have the latest version of `libignition-math2` installed by running:
```sudo apt upgrade libignition-math2 libcairo2-dev```

The next step is to install needed python packages by calling:

```
pip3 install -r requirements.txt
```
(*Install pip3 if not already installed. Ignore any error thrown by pip when trying to install pygobject, it seems to be irrelevant.*)

Then build `kitcar-gazebo-repository` by calling `catkin_make` in the root folder of this repository.

Other information can be found in [KITcar-Gazebo-Simulation Wiki](https://wiki.kitcar-team.de/doku.php?id=technik:systemstruktur:simulation:start).

