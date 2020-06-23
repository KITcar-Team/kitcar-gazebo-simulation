"""The CarStateNode publishes up to date information about the simulated vehicle.

Thereby allowing easy access to the vehicle’s position and speed,
but also the vehicle’s frame and view cone.

The CarStateVisualizationNode then processes this information
and publishes messages which can be displayed in RVIZ.
"""

from typing import List, Any

# Decorator to easily add classes/functions to the geometry module.
def export(define):
    # Add new object to globals
    globals()[define.__name__] = define
    # Append the object to __all__ which is visible when the module is imported
    __all__.append(define.__name__)
    return define


__all__: List[
    Any
] = []  # Will hold all classes/functions which can be imported with 'from simulation.utils.geometry import ...'

# import all files which are part of the geometry module
import simulation.src.gazebo_simulation.src.car_state.node  # noqa:402
import simulation.src.gazebo_simulation.src.car_state.visualization  # noqa:402
