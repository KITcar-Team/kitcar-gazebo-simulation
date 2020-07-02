"""The CarStateNode publishes up to date information about the simulated vehicle.

Thereby allowing easy access to the vehicle’s position and speed,
but also the vehicle’s frame and view cone.

The CarStateVisualizationNode then processes this information
and publishes messages which can be displayed in RVIZ.
"""

from .node import CarStateNode
from .visualization import CarStateVisualizationNode

__all__ = ["CarStateNode", "CarStateVisualizationNode"]
