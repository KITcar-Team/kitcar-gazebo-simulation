from simulation.utils.geometry import Line, Polygon, Transform, Point

from simulation.src.simulation_groundtruth.src.groundtruth.node import GroundtruthNode
from simulation.src.simulation_groundtruth.src.groundtruth.test import (
    road_section_mocks as section_mocks,
)

import simulation.utils.road.sections.type as road_section_type
from simulation.utils.road.sections.line_tuple import LineTuple

from simulation_groundtruth.srv import IntersectionSrvResponse
from simulation_groundtruth.msg import Parking as ParkingMsg

from unittest.mock import Mock

PKG = "simulation_groundtruth"


def create_straight_section(id: int, obstacle: bool = False):
    """Mock straight section to be used for testing.

    Args:
        id: ID of the section.
        obstacle: Indicate whether there should be an obstacle on the right lane.
        lane_width: width of right and left lane.
    """
    tf = Transform([2 * id, 0], 0)
    obstacles = None
    if obstacle:
        obstacles = [tf * Polygon([[1, 0], [1, -0.3], [1.4, -0.3], [1.4, 0]])]

    middle_line = tf * Line([Point(r=i, phi=0) for i in range(3)])
    left_line = middle_line.parallel_offset(0.4, side="left")
    right_line = middle_line.parallel_offset(0.4, side="right")
    return section_mocks.mock_generic_section(
        id=id,
        type_=road_section_type.STRAIGHT_ROAD,
        left_line=left_line,
        middle_line=middle_line,
        right_line=right_line,
        obstacles=obstacles,
    )


def create_parking_section(id: int, left: bool = True, right: bool = True, lane_width=0.4):
    """Mock parking section to be used for testing.

    Args:
        id: ID of the section.
        left: Indicate whether a parking lot on the left side should be created.
        right: Indicate whether a parking lot on the right side should be created.
        lane_width: width of right and left lane.
    """
    tf = Transform([2 * id, 0], 0)
    middle_line = tf * Line([[0, 0], [2, 0]])
    left_line = middle_line.parallel_offset(0.4, side="left")
    right_line = middle_line.parallel_offset(0.4, side="right")

    left_lots = [
        (
            tf
            * Line(
                [
                    [0, -lane_width],
                    [0, -2 * lane_width],
                    [2, -2 * lane_width],
                    [2, -lane_width],
                ]
            ),
            [
                (
                    ParkingMsg.SPOT_FREE,
                    tf
                    * Polygon(
                        [
                            [0, -lane_width],
                            [1, -lane_width],
                            [1, -2 * lane_width],
                            [0, -2 * lane_width],
                        ]
                    ),
                ),
                (
                    ParkingMsg.SPOT_X,
                    tf
                    * Polygon(
                        [
                            [1, -lane_width],
                            [2, -lane_width],
                            [2, -2 * lane_width],
                            [1, -2 * lane_width],
                        ]
                    ),
                ),
            ],
        )
    ]
    right_lots = [
        (
            tf
            * Line(
                [
                    [0, lane_width],
                    [0, 2 * lane_width],
                    [2, 2 * lane_width],
                    [2, lane_width],
                ]
            ),
            [
                (
                    ParkingMsg.SPOT_FREE,
                    tf
                    * Polygon(
                        [
                            [0, lane_width],
                            [1, lane_width],
                            [1, 2 * lane_width],
                            [0, 2 * lane_width],
                        ]
                    ),
                ),
                (
                    ParkingMsg.SPOT_OCCUPIED,
                    tf
                    * Polygon(
                        [
                            [1, lane_width],
                            [2, lane_width],
                            [2, 2 * lane_width],
                            [1, 2 * lane_width],
                        ]
                    ),
                ),
            ],
        )
    ]

    return section_mocks.mock_parking_section(
        id=id,
        type_=road_section_type.PARKING_AREA,
        left_line=left_line,
        middle_line=middle_line,
        right_line=right_line,
        left_lots=left_lots if left else None,
        right_lots=right_lots if right else None,
    )


def create_intersection(
    id: int, rule: int = IntersectionSrvResponse.EQUAL, lane_width: float = 0.4
):
    """Mock intersection to be used for testing.

    Args:
        id: ID of the section.
        rule: Priority rule of intersection.
        lane_width: width of right and left lane.
    """

    tf = Transform([2 * id, 0], 0)
    middle_line = tf * Line([[0, 0], [2, 0]])
    left_line = middle_line.parallel_offset(lane_width, side="left")
    right_line = middle_line.parallel_offset(lane_width, side="right")

    south_middle = tf * Line([[0, 0], [0.75, 0]])
    north_middle = tf * Line([[1.25, 0], [2, 0]])
    return section_mocks.mock_intersection(
        id=1,
        type_=road_section_type.INTERSECTION,
        left_line=left_line,
        middle_line=middle_line,
        right_line=right_line,
        turn=IntersectionSrvResponse.STRAIGHT,
        rule=rule,
        south=LineTuple(
            south_middle.parallel_offset(lane_width, side="left"),
            south_middle,
            south_middle.parallel_offset(lane_width, side="right"),
        ),
        west=LineTuple(
            tf * Line([[1 - lane_width, lane_width], [1 - lane_width, 3 * lane_width]]),
            tf * Line([[1, lane_width], [1, 3 * lane_width]]),
            tf * Line([[1 + lane_width, lane_width], [1 + lane_width, 3 * lane_width]]),
        ),
        east=LineTuple(
            tf * Line([[1 - lane_width, -lane_width], [1 - lane_width, -3 * lane_width]]),
            tf * Line([[1, -lane_width], [1, -3 * lane_width]]),
            tf * Line([[1 + lane_width, -lane_width], [1 + lane_width, -3 * lane_width]]),
        ),
        north=LineTuple(
            north_middle.parallel_offset(lane_width, side="left"),
            north_middle,
            north_middle.parallel_offset(lane_width, side="right"),
        ),
        # surface_markings=[
        #    (
        #        tf * Polygon([[0.3, -0.1], [0.3, -0.3], [0.6, -0.3], [0.6, -0.1]]),
        #        LabeledPolygonMsg.RIGHT_TURN_MARKING,
        #    )
        # ],
    )


class GroundtruthMockNode(GroundtruthNode):
    """Subclass of GroundtruthNode which loads mock road sections for testing.

    The parameter *road* can be used to specify which road should be created.
    All options are explained in the parameter file.

    """

    def _load_road(self):
        road_sections = []
        road_sections.append(create_straight_section(0))

        if self.param.road <= 1:
            road_sections.append(
                create_straight_section(1, obstacle=(self.param.road == 1))
            )
        elif self.param.road == 2:
            road_sections.append(create_parking_section(1))
        elif self.param.road == 3:
            road_sections.append(
                create_intersection(1, rule=IntersectionSrvResponse.PRIORITY_YIELD)
            )
        elif self.param.road == 4:
            road_sections.append(create_intersection(1, rule=IntersectionSrvResponse.YIELD))
        elif self.param.road == 5:
            road_sections.append(create_intersection(1, rule=IntersectionSrvResponse.STOP))

        road_sections.append(create_straight_section(2))

        m = Mock()
        m.sections = road_sections
        return m
