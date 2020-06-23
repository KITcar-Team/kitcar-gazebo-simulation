"""Quadratic- and CubicBezierCurves."""

from typing import Union, Sequence
from dataclasses import dataclass
import numpy as np

from simulation.utils.geometry import Point, Line

import simulation.utils.road.sections.type as road_section_type
from simulation.utils.road.sections.road_section import RoadSection


def _read_point(p: Union[Point, Sequence[float]]) -> np.ndarray:
    """Get numpy array from point or coordinate sequence."""
    try:
        return p.to_numpy()[:2]
    except AttributeError:
        return np.array(p)


def _compute_cubic_bezier(t, p0, p1, p2, p3):
    c0 = (1 - t) * p0 + t * p1
    c1 = (1 - t) * p1 + t * p2
    c2 = (1 - t) * p2 + t * p3
    d0 = (1 - t) * c0 + t * c1
    d1 = (1 - t) * c1 + t * c2
    x = (1 - t) * d0 + t * d1
    return x


@dataclass
class BezierCurve(RoadSection):

    p0: Union["Point", Sequence[float]] = Point(0, 0)
    """Control point 0."""
    p1: Union["Point", Sequence[float]] = None
    """Control point 1."""
    p2: Union["Point", Sequence[float]] = None
    """Control point 2."""

    def __post_init__(self):
        assert self.p0 is not None, f"Missing start point for {self.__class__}."
        assert self.p1 is not None, f"Missing first point for {self.__class__}."
        assert self.p2 is not None, f"Missing second point for {self.__class__}."
        self._p0 = _read_point(self.p0)
        self._p1 = _read_point(self.p1)
        self._p2 = _read_point(self.p2)

        super().__post_init__()


@dataclass
class QuadBezier(BezierCurve):
    """Quadratic bezier curve, defined by two control points.

    Args:
        p0 (Union[Point, Sequence[float]]) = [0,0]: Control point 0.
        p1 (Union[Point, Sequence[float]]): Control point 1.
        p2 (Union[Point, Sequence[float]]): Control point 2.
    """

    TYPE = road_section_type.QUAD_BEZIER

    @property
    def middle_line(self) -> Line:
        points = []
        t = 0
        while t <= 1:
            c0 = (1 - t) * self._p0 + t * self._p1
            c1 = (1 - t) * self._p1 + t * self._p2
            x = (1 - t) * c0 + t * c1
            points.append(Point(*x))
            t += 0.01
        t = 1
        c0 = (1 - t) * self._p0 + t * self._p1
        c1 = (1 - t) * self._p1 + t * self._p2
        x = (1 - t) * c0 + t * c1
        points.append(Point(*x))
        return self.transform * Line(points)


@dataclass
class CubicBezier(BezierCurve):
    """Cubic bezier curve, defined by three control points.

    Args:
        p0 (Union[Point, Sequence[float]]) = [0,0]: Control point 0.
        p1 (Union[Point, Sequence[float]]): Control point 1.
        p2 (Union[Point, Sequence[float]]): Control point 2.
        p3 (Union[Point, Sequence[float]]): Control point 3.
    """

    TYPE = road_section_type.CUBIC_BEZIER

    p3: Union[Point, Sequence[float]] = None
    """Control point 3."""

    def __post_init__(self):
        assert self.p3 is not None, "Missing third point for CubicBezier."
        self._p3 = _read_point(self.p3)

        super().__post_init__()

    @property
    def middle_line(self) -> Line:
        points = []
        t = 0
        while t <= 1:
            points.append(
                Point(*_compute_cubic_bezier(t, self._p0, self._p1, self._p2, self._p3))
            )
            t += 0.01
        t = 1
        points.append(
            Point(*_compute_cubic_bezier(t, self._p0, self._p1, self._p2, self._p3))
        )
        return self.transform * Line(points)
