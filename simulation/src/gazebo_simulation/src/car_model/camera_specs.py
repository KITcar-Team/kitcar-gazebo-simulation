from typing import Dict
from dataclasses import dataclass, field
from functools import partial
import math

import numpy as np
import yaml

from simulation.utils.geometry import Vector

from .specs import Specs


@dataclass
class CameraSpecs(Specs):
    """Camera specifications usually defined in kitcar-ros.

    With this class, the real calibration is used to create the simulated camera model.

    This way the camera specifications and calibration are very similar.
    The calibrations are still not exactly the same
    and therefore a separate calibration file is loaded!

    There are, however, some nuances:

    * Instead of considering the complete calibration matrix,
      only the *pitch* angle is used to rotate the camera model
      around the y axis in simulation.
    * The real camera crops the image before publishing the image. This is reflected in
      the different :py:attr:`capture_size` and :py:attr:`output_size`.
    """

    focal_length_x: float
    focal_length_y: float
    optical_center_x: float
    optical_center_y: float

    r11: float
    r12: float
    r13: float
    r21: float
    r22: float
    r23: float
    r31: float
    r32: float
    r33: float
    t1: float
    t2: float
    t3: float

    capture_size: Dict[str, float] = field(
        default_factory=partial(dict, {"width": 1280, "height": 1024})
    )
    capture_format: str = "L8"
    clip: Dict[str, float] = field(default_factory=partial(dict, {"near": 0.1, "far": 4}))

    output_size: Dict[str, float] = field(
        default_factory=partial(dict, {"width": 1280, "height": 650})
    )

    update_rate: float = 60

    image_topic: str = "/simulation/sensors/raw/camera"
    info_topic: str = "/simulation/sensors/camera/info"

    @property
    def R(self) -> np.matrix:
        """np.matrix: Rotation matrix of the camera."""
        return np.matrix(
            [
                [self.r11, self.r12, self.r13],
                [self.r21, self.r22, self.r23],
                [self.r31, self.r32, self.r33],
            ]
        )

    @property
    def t_vehicle(self) -> np.ndarray:
        """np.ndarray: Translation vector to the camera in vehicle coordinates."""
        return np.linalg.inv(self.R) @ np.array([self.t1, self.t2, self.t3])

    @property
    def xyz(self) -> Vector:
        """Vector: Translation vector to the camera in vehicle coordinates."""
        t = (
            -1 / 1000 * np.linalg.inv(self.R) @ np.array([self.t1, self.t2, self.t3])
        ).tolist()[0]
        return Vector(*t)

    @property
    def rpy(self) -> Vector:
        """Vector: Vector of roll, pitch, yaw angles of the camera's pose."""
        # return Vector(0, math.asin(-1 * self.r21), 0)
        return Vector(0, math.asin(-self.r21), 0)

    @property
    def horizontal_fov(self) -> float:
        return 2 * math.atan(self.capture_size["width"] / 2 / self.focal_length_x)

    def simplify(self):
        """Create a simplified calibration from the real one."""

        # Camera is only rotated around the y axis
        s = math.sin(self.rpy.y)
        c = math.cos(self.rpy.y)
        self.r11, self.r12, self.r13 = 0, -1, 0
        self.r21, self.r22, self.r23 = -s, 0, -c
        self.r31, self.r32, self.r33 = c, 0, -s

        self.optical_center_x = self.output_size["width"] - self.capture_size["width"] / 2
        self.optical_center_y = self.output_size["height"] - self.capture_size["height"] / 2

        # By creating a camera with horizontal_fov, the focal_length_x is implicitly used for y as well!
        self.focal_length_y = self.focal_length_x

    def save(self, file_path: str, simplify=True, remove_clip=True, remove_sizes=True):
        """Save to file.

        Args:
            file_path: Path to file.
            simplify: Simplify the car specs before saving.
            remove_clip: Don't save clip.
            remove_sizes: Don't save capture and output_size.
        """
        if simplify:
            self.simplify()
        data = dict(self.__dict__)
        if remove_clip:
            del data["clip"]
        if remove_sizes:
            del data["capture_size"]
            del data["output_size"]

        with open(file_path, "w+") as file:
            yaml.dump(data, file, Dumper=yaml.SafeDumper, default_flow_style=False)
