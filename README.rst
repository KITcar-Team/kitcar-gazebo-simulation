kitcar-gazebo-simulation
========================

The simulation has been developed and primarily used on Ubuntu 18.04.
Other Linux distributions were not tested.
Additionally, `ROS melodic <http://wiki.ros.org/melodic/Installation/Ubuntu>`_ \
must be installed with all dependencies.

Clone
-----

The first step is of course to clone the repository.
These are some ways to get it:

* **KITcar internal**. Clone this repository in the same directory as `kitcar-ros`.
  Change into the correct directory. By default it's ``/home/<USERNAME>/kitcar`` and run::

   git clone git@git.kitcar-team.de:kitcar/kitcar-gazebo-simulation.git


$KITCAR_REPO_PATH
-----------------

The environment variable **$KITCAR_REPO_PATH** must contain the directory in which you've cloned **kitcar-gazebo-simulation**.

Make sure that **$KITCAR_REPO_PATH** is set to the directory where you've cloned **kitcar-gazebo-simulation** into::

  cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation

should put you into the root directory of **kitcar-gazebo-simulation**.

If it doesn't work, create the variable with::


   export KITCAR_REPO_PATH=<DIRECTORY WHERE kitcar-gazebo-simulation IS>


Adding

::

  export KITCAR_REPO_PATH=<DIRECTORY WHERE kitcar-gazebo-simulation IS>

to your ``.bashrc`` ensures that the path is always set.

Python
------

Ensure that your Python-Version is 3.8 or larger::

   python3 -V

If not, upgrade your system to Python 3.8.
(*KITcar internal: Goto* `KITcar-Gazebo-Simulation Python <https://wiki.kitcar-team.de/doku.php?id=teams:simulation:python>`_ )

Init-Script
-----------

Therefore the init script must be called.
Change into `kitcar-gazebo-simulation` folder and run the script::

   cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation
   ./init/init.sh

(*Ignore any error thrown by pip when trying to install pygobject,
it seems to be irrelevant.*)

If there is an error with `libignition-math2` make sure \
you have the latest version installed by running::

   sudo apt upgrade libignition-math2 libcairo2-dev

Build
-----

Then build `kitcar-gazebo-repository` by calling `catkin_make` in the simulation folder \
of this repository.

(*KITcar internal: Other information can be found in* \
`KITcar-Gazebo-Simulation Wiki <https://wiki.kitcar-team.de/doku.php?id=technik:systemstruktur:simulation:start>`_ ).

Git Lfs
-------

Images and other large files are tracked using `Git LFS <https://git-lfs.github.com/>`_, \
to download them locally, you need to pull them::

   git lfs pull

Test
----

Let's test if everything works. Open up a new terminal.

#. Before being able to start the simulation, we must generate (build + render) a road::

      python3 -m generate_road default_road

#. That's it. You can now start the simulation with

  ::

     roslaunch gazebo_simulation master.launch

Gazebo should now open with the car and a road.
