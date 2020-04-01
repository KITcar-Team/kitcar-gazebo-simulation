# KITcar gazebo simulation

## Setup

- **KITcar internal**: Clone this repository in the same directory as `kitcar-ros`. Make sure that the path variable *$KITCAR_REPO_PATH* is set to the directory where your `kitcar-ros` and `kitcar-gazebo-simulation` installations are.
- **External:** Set up the environment variable *$KITCAR_REPO_PATH* such that
```
$KITCAR_REPO_PATH/kitcar-gazebo-simulation == /path/to/kitcar-gazebo-simulation
```

#### Python
Ensure that your Python-Version is 3.8 or larger:
```
python3 -V
```
If not, upgrade your system to Python 3.8. (*KITcar internal: Goto [KITcar-Gazebo-Simulation Python](https://wiki.kitcar-team.de/doku.php?id=teams:simulation:python)*)

#### Init-Script
Therefore the init script must be called. Change into `kitcar-gazebo-simulation` folder and run the script:
```
cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation
./init/init.sh
```
(*Ignore any error thrown by pip when trying to install pygobject, it seems to be irrelevant.*)

If there is an error with `libignition-math2` make sure you have the latest version installed by running:
```sudo apt upgrade libignition-math2 libcairo2-dev```

#### Build
Then build `kitcar-gazebo-repository` by calling `catkin_make` in the simulation folder of this repository.

(*KITcar internal: Other information can be found in [KITcar-Gazebo-Simulation Wiki](https://wiki.kitcar-team.de/doku.php?id=technik:systemstruktur:simulation:start)*).

