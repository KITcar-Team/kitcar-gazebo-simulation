Renderer
========

Creating and displaying roads in Gazebo is an important part of the simulation.
The :py:class:`simulation.src.simulation_groundtruth.src.groundtruth.renderer.Renderer`
is started as a part of the
:py:class:`simulation.src.simulation_groundtruth.src.groundtruth.node.GroundtruthNode`.

The road is drawn onto the ground as an image.
I.e. all road lines are just an image that is displayed in Gazebo.
However, because roads can be very large, it is better to split up the road into equally
sized :py:class:`simulation.utils.road.renderer.tile.Tile`.

Additionally, there are some optimizations:

#. Only tiles with visible sections of the road are created
#. A road is rendered only once. If it is opened again, without modifying the road file,
   the previously rendered tiles are reused.

Obstacles and traffic signs must be created as well.
After the renderer has created it's groundplane,
:py:class:`simulation.src.simulation_groundtruth.src.groundtruth.object_controller.ObjectController`
spawns all obstacles and traffic signs.

Putting the pieces together; these are the steps taken to create and populate the Gazebo world:

#. Import the road from ``simulation/models/env_db/<ROAD_NAME>.py``,
#. check results from previous renderings are available, otherwise
#. distribute the road onto multiple equally sized tiles on the ground,
#. draw each tile and save the results to a file,
#. spawn the tiles in Gazebo, and
#. spawn obstacles and traffic signs with the ObjectController.
