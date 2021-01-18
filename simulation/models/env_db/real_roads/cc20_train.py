import math
import os

from simulation.utils.geometry import Point
from simulation.utils.road.road import Road  # Definition of the road class
from simulation.utils.road.sections import (
    CubicBezier,
    CustomSection,
    Intersection,
    RightCircularArc,
    StraightRoad,
)

road = Road()

custom_part_one: CustomSection = CustomSection.from_yaml(
    os.path.join(os.path.dirname(__file__), "./cc20_scans/training_part1.yaml")
)
road.append(custom_part_one)
road.append(StraightRoad(length=0.1))
road.append(RightCircularArc(radius=3, angle=math.radians(17)))
road.append(StraightRoad(length=0.08))
road.append(
    CubicBezier(
        p1=Point(0.1, 0),
        p2=Point(0.9, -0.05),
        p3=Point(1.0, -0.05),
        left_line_marking=StraightRoad.MISSING_LINE_MARKING,
        right_line_marking=StraightRoad.MISSING_LINE_MARKING,
        middle_line_marking=StraightRoad.MISSING_LINE_MARKING,
    )
)
road.append(
    CustomSection.from_yaml(
        os.path.join(os.path.dirname(__file__), "./cc20_scans/training_loop.yaml")
    )
)
road.append(StraightRoad(length=0.85))
road.append(
    Intersection(size=1.1, angle=math.radians(82), exit_direction=Intersection.STRAIGHT)
)
road.append(
    CustomSection.from_yaml(
        os.path.join(os.path.dirname(__file__), "./cc20_scans/training_part2.yaml")
    )
)
road.close_loop()
