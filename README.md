# KITcar gazebo simulation

## Setup

Clone this repository in the same directory as `kitcar-ros`. Make sure that the path variable *$KITCAR_REPO_PATH* is set to the repository where your `kitcar-ros` installation is.

Therefore the init script must be called. Change into `kitcar-gazebo-simulation` folder and run the script:
```
cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation
./init/init.sh

```
(* Ignore any error thrown by pip when trying to install pygobject, it seems to be irrelevant.*)

If there is an error with `libignition-math2` make sure you have the latest version installed by running:
```sudo apt upgrade libignition-math2 libcairo2-dev```


Then build `kitcar-gazebo-repository` by calling `catkin_make` in the root folder of this repository.

Other information can be found in [KITcar-Gazebo-Simulation Wiki](https://wiki.kitcar-team.de/doku.php?id=technik:systemstruktur:simulation:start).

