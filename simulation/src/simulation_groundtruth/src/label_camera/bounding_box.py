import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from simulation_groundtruth.msg import LabeledBoundingBox as LabeledBoundingBoxMsg

from simulation.utils.geometry import Point, Polygon, Transform, Vector


@dataclass
class BoundingBox:
    """Container for a visible object on the camera image."""

    class_id: int
    """Id of the object's class."""
    class_description: str
    """Readable description of the object's class."""
    world_points: List[Point]
    """Object's coordinates in the world."""

    world_vehicle_tf = Transform([0, 0], 0)
    """Transformation from world to vehicle coordinates."""
    vehicle_pixel_matrix = np.zeros((2, 3))
    """Transformation matrix from vehicle coordinates to camera pixels.

    This is not a normal Transform, but a numpy matrix because it is not an affine
    transformation.
    """

    def set_tfs(
        world_vehicle_tf: Transform,
        vehicle_pixel_matrix: np.matrix,
    ):
        """Update transformations."""
        BoundingBox.world_vehicle_tf = world_vehicle_tf
        BoundingBox.vehicle_pixel_matrix = vehicle_pixel_matrix

    @property
    def vehicle_points(self) -> List[Vector]:
        return list(BoundingBox.world_vehicle_tf * Vector(p) for p in self.world_points)

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get the bounds of this box in the camera image.

        Return:
            x1, y1, x2, y2
        """
        vehicle_points = self.vehicle_points

        center_ground: Vector = 1 / len(vehicle_points) * sum(vehicle_points, Vector(0, 0))
        center_ground = Vector(center_ground.x, center_ground.y)

        pixels = []
        for p in vehicle_points + [center_ground]:
            hom_vec = np.ones(4)
            hom_vec[:3] = p.to_numpy()
            pixel_vec = np.dot(BoundingBox.vehicle_pixel_matrix, hom_vec).T

            pixel_vec *= 1 / float(pixel_vec[2])
            pixels.append(Vector(pixel_vec[0], pixel_vec[1]))

        pixel_ground_center = pixels[-1]
        del pixels[-1]

        pixel_poly = Polygon(pixels)
        x1, y1, x2, y2 = pixel_poly.bounds

        x1, x2 = round(min(x1, x2)), round(max(x1, x2))
        y1, y2 = round(min(y1, y2)), round(max(y1, y2))

        return x1, y1, x2, y2, round(pixel_ground_center.x), round(pixel_ground_center.y)

    def to_msg(
        self,
    ) -> LabeledBoundingBoxMsg:
        """Create a msg from this object."""
        x1, y1, x2, y2, x_ground, y_ground = self.get_bounds()

        return LabeledBoundingBoxMsg(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            x_ground=x_ground,
            y_ground=y_ground,
            class_id=self.class_id,
            class_description=self.class_description,
        )

    @property
    def distance(self):
        """float: Shortest distance between the car and this object."""
        return min(abs(p) for p in self.vehicle_points)

    @property
    def angle(self):
        """float: Minimum angle between the car's direction and the object."""
        e_1 = Vector(1, 0)
        return math.acos(min(1 / abs(p) * p * e_1 for p in self.vehicle_points))

    @property
    def orientation(self) -> float:
        """float: Orientation between car and front of the object."""
        p0, _, _, p3, *_ = self.vehicle_points
        dir_vec = Vector(p3) - Vector(p0)
        orientation = dir_vec.cross(Vector(0, 0, 1)).argument
        return orientation

    def __lt__(self, other_bb: "BoundingBox"):
        return self.distance < other_bb.distance


@dataclass
class VisualBoundingBox:
    """Visualizable bounding box with a label and color."""

    bounds: Tuple[int, int, int, int]
    label: str
    color: Tuple[int, int, int]

    def draw(
        self,
        img: np.ndarray,
    ):
        """Draw the bounding box into the given image."""
        x1, y1, x2, y2, x_ground, y_ground = self.bounds

        if x1 == -1 and x2 == -1:
            return

        cv2.rectangle(img, (x1, y1), (x2, y2), self.color, 2)

        # Label
        (test_width, text_height), baseline = cv2.getTextSize(
            self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img,
            (x1, y1),
            (x1 + test_width, y1 - text_height - baseline),
            self.color,
            thickness=cv2.FILLED,
        )
        cv2.drawMarker(
            img,
            (x_ground, y_ground),
            self.color,
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
        )
        cv2.putText(
            img,
            self.label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
