from typing import List, Callable
from dataclasses import dataclass

import rospy

from simulation.utils.road.road import Road
from simulation.utils.road.sections import StaticObstacle
import simulation.utils.road.renderer.obstacle as render_obstacle
import simulation.utils.road.renderer.traffic_sign as render_sign

from gazebo_simulation.msg import CarState as CarStateMsg


@dataclass
class ObjectController:
    """ROS node to place obstacles and other objects (e.g. traffic signs) in Gazebo."""

    road: Road
    spawn_model: Callable[[str], None]
    remove_model: Callable[[str], None]
    _obstacle_idx: int = 0
    _sign_idx: int = 0

    @property
    def obstacle_name(self):
        self._obstacle_idx += 1
        return f"obstacle_{self._obstacle_idx}"

    @property
    def sign_name(self):
        self._sign_idx += 1
        return f"signs_{self._sign_idx}"

    def _load_static_obstacles_from_groundtruth(self) -> List[StaticObstacle]:
        return sum((section.obstacles for section in self.road.sections), [])

    def load(self, model_names: List[str]):
        """Reload all objects."""
        # Update obstacles
        rospy.loginfo(f"Starting to place obstacles and signs.")
        objects_to_remove = [
            model_name
            for model_name in model_names
            if ("obstacle" in model_name or "sign" in model_name)
        ]
        for model in objects_to_remove:
            self.remove_model(model)
        rospy.logdebug(f"All obstacles and signs: {objects_to_remove}")

        obstacles = self._load_static_obstacles_from_groundtruth()
        for i, obs in enumerate(obstacles):
            self.spawn_model(render_obstacle.draw(self.obstacle_name, obs))

        signs = sum((sec.traffic_signs for sec in self.road.sections), [])
        for i, sign in enumerate(signs):
            self.spawn_model(render_sign.draw(self.sign_name, sign))

    def update(self, car_state: CarStateMsg):
        """Update the object controller.

        Use this function to move dynamic obstacles around.

        Args:
            car_state: Current car state containing the position of the car.
        """
        pass
