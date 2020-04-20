"""StaticObstacle on road and ParkingObstacle on ParkingSpot."""

from dataclasses import dataclass
import math

from geometry import Polygon, Point, Transform, Vector

from road import schema


@dataclass
class _StaticObstacle:

    center: Point = Point(0.4, -0.2)
    """Center point of the obstacle."""
    width: float = 0.2
    """Width of the obstacle."""
    depth: float = 0.2
    """Width of the obstacle."""
    angle: float = 0

    """Frame of the obstacle."""
    height: float = 0.2
    """Height of the obstacle."""
    transform: Transform = None
    """Transform to coordinate system in which frame is given."""
    normalize_x: bool = True
    """If true, all x-values are substracted by the lowest x-value."""

    def __post_init__(self):
        # prevents execution when building documentation
        if self.transform is None:
            self.transform = Transform([0, 0], 0)


class StaticObstacle(_StaticObstacle):
    @property
    def radians(self) -> float:
        """float: Angle converted into radians."""
        return math.radians(self.angle)

    @property
    def center(self) -> Point:
        """Point: Center of the obstacle in global coordinates."""
        tf = Transform([-self._center.x, 0], 0) if self.normalize_x else 1
        return Point(self.transform * (tf * Vector(self._center)))

    @center.setter
    def center(self, c: Point):
        self._center = c

    @property
    def frame(self) -> Polygon:
        """Polygon: Frame of the obstacle in global coordinates."""
        return Transform(self.center, self.radians + self.transform.get_angle()) * Polygon(
            [
                [-self.depth / 2, self.width / 2],
                [self.depth / 2, self.width / 2],
                [self.depth / 2, -self.width / 2],
                [-self.depth / 2, -self.width / 2],
            ]
        )

    def export(self) -> schema.obstacle:
        # width and depth interchanged because of orientation in renderer
        rect = schema.rectangle(
            length=self.width,
            width=self.depth,
            orientation=self.transform.get_angle() + self.angle,
            centerPoint=self.center.to_schema(),
        )
        obstacle = schema.obstacle(
            role="static", type="parkedVehicle", shape=schema.shape()
        )
        obstacle.shape.rectangle.append(rect)

        return obstacle


@dataclass
class _ParkingObstacle(_StaticObstacle):
    center: Point = Point(0.2, -0.2)
    """Center point of the obstacle."""
    width: float = 0.15
    """Width of the obstacle."""
    depth: float = 0.15
    """Width of the obstacle."""
    normalize_x: bool = False


class ParkingObstacle(StaticObstacle, _ParkingObstacle):
    pass
