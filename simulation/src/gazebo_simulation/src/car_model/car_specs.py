from dataclasses import dataclass
import yaml

from simulation.utils.geometry import Point, Vector
from simulation.utils.urdf import JointDynamics, JointLimit


@dataclass
class CarSpecs:
    wheelbase: float
    track: float
    vehicle_width: float
    distance_cog_front: float
    distance_cog_rear: float
    cog_height: float
    moment_of_inertia: float
    tire_stiffness_front: float
    tire_stiffness_rear: float
    mass: float
    tire_radius: float
    distance_to_front_bumper: float
    distance_to_rear_bumper: float
    max_steering_angle_front_left: float
    max_steering_angle_front_right: float
    max_steering_angle_back_left: float
    max_steering_angle_back_right: float

    tire_mass: float = 0.5

    @classmethod
    def from_file(cls, file_path: str) -> "CarSpecs":
        with open(file_path) as file:
            car_specs = CarSpecs(**yaml.load(file, Loader=yaml.SafeLoader))
        return car_specs

    def save(self, file_path: str):
        """Save to file.

        Args:
            file_path: Path to file.
        """
        with open(file_path, "w+") as file:
            yaml.dump(
                dict(self.__dict__), file, Dumper=yaml.SafeDumper, default_flow_style=False
            )

    @property
    def body_height(self):
        return self.cog_height * 2

    @property
    def tire_width(self):
        return (self.vehicle_width - self.track) / 2

    @property
    def center_rear_axle(self) -> Point:
        """Point: Center of rear axle in COG coordinate frame."""
        return Point(-self.distance_cog_rear, 0, self.tire_radius)

    @property
    def center_rear_left_tire(self) -> Point:
        """Point: Left rear tire in COG coordinate frame."""
        return self.center_rear_axle + Vector(0, self.track / 2 + self.tire_width, 0)

    @property
    def center_rear_right_tire(self) -> Point:
        """Point: Right rear tire in COG coordinate frame."""
        return self.center_rear_axle - Vector(0, self.track / 2 + self.tire_width, 0)

    @property
    def center_front_axle(self) -> Point:
        """Point: Center of front axle in COG coordinate frame."""
        return Point(self.distance_cog_front, 0, self.tire_radius)

    @property
    def center_front_left_tire(self) -> Point:
        """Point: Left front tire in COG coordinate frame."""
        return self.center_front_axle + Vector(0, self.track / 2 + self.tire_width, 0)

    @property
    def center_front_right_tire(self) -> Point:
        """Point: Right front tire in COG coordinate frame."""
        return self.center_front_axle - Vector(0, self.track / 2 + self.tire_width, 0)

    @property
    def steering_limit_rear(self) -> JointLimit:
        return JointLimit(
            lower=-1 * self.max_steering_angle_back_right,
            upper=self.max_steering_angle_back_left,
            effort=1,
            velocity=100,
        )

    @property
    def steering_limit_front(self) -> JointLimit:
        return JointLimit(
            lower=-1 * self.max_steering_angle_front_right,
            upper=self.max_steering_angle_front_left,
            effort=100,
            velocity=100,
        )

    @property
    def tire_dynamics(self):
        return JointDynamics(damping=1, friction=1)
