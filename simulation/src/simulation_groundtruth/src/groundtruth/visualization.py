#!/usr/bin/env python3
"""Node that publishes groundtruth information for rviz."""

import rospy
from typing import List

from simulation.utils.geometry import Polygon, Point

# Received from simulation.src.simulation_groundtruth.src.groundtruth extractor
from simulation_groundtruth.srv import (
    SectionSrv,
    LaneSrv,
    ParkingSrv,
    LabeledPolygonSrv,
    IntersectionSrv,
)
from simulation_groundtruth.msg import (
    Parking as ParkingMsg,
    LabeledPolygon as LabeledPolygonMsg,
)
from geometry_msgs.msg import Point32 as GeomPoint


import simulation.utils.road.sections.type as road_section_type

from simulation.utils.ros_base.node_base import NodeBase
import simulation.utils.ros_base.visualization as visualization

from visualization_msgs.msg import Marker


class GroundtruthVisualizationNode(NodeBase):
    """ROS node to visualize groundtruth objects in RVIZ.

    Attributes:
        publishers (Dict[rospy.Publisher]): Automatically generated dictionary \
                containing all publishers. A publisher is created for every visualization \
                topic listed in the parameter file.
        get_sections, get_lane, get_parking, get_obstacles, get_intersection \
                (rospy.ServiceProxy): Can be called for groundtruth information.

        """

    def __init__(self):

        super().__init__(name="groundtruth_visualization_node")

        # Run the update function with a rate of self.param.rate
        self.run(function=self.update, rate=self.param.rate)

    def start(self):
        """Start the node by creating publishers and service proxies."""
        rospy.wait_for_service(self.param.topics.section)

        # For all entries in topics/visualization a publisher is created!
        # and added to the publishers dictionary.
        rospy.loginfo(
            f"Creating publishers for these visualization topics: \
                    {self.param.topics.visualization.as_dict()}"
        )

        self.publishers = {
            name: rospy.Publisher(topic, Marker, queue_size=100)
            for name, topic in self.param.topics.visualization.as_dict().items()
        }

        self.get_sections = rospy.ServiceProxy(self.param.topics.section, SectionSrv)
        self.get_lane = rospy.ServiceProxy(self.param.topics.lane, LaneSrv)
        self.get_parking = rospy.ServiceProxy(self.param.topics.parking, ParkingSrv)
        self.get_obstacles = rospy.ServiceProxy(
            self.param.topics.obstacle, LabeledPolygonSrv
        )
        self.get_intersection = rospy.ServiceProxy(
            self.param.topics.intersection, IntersectionSrv
        )

    def stop(self):
        """Stop the node by unregistering publishers."""
        for _, publisher in self.publishers.items():
            publisher.unregister()

    def _publish_point_marker(
        self,
        points: List[GeomPoint],
        publisher_name: str,
        id: int,
        rgba: List[float],
        ns="simulation/groundtruth",
    ):
        """Publish an RVIZ marker on the publisher's topic.

        Args:
            points: Points to be published.
            publisher_name: Key of the publisher in the publisher dictionary.
            id: RVIZ marker id.
            rgba: List of the marker color.
            ns: RVIZ namespace of the marker.
        """
        marker = visualization.get_marker_for_points(
            (Point(p) for p in points),
            frame_id="simulation",
            rgba=rgba,
            id=id,
            ns=ns,
            duration=5 / self.param.rate,  # Groundtruth is too slow otherwise!
        )
        self.publishers[publisher_name].publish(marker)

    def _show_lane_markers(self):
        """Publish markers for all lane markings of road."""
        for id in (sec.id for sec in self.get_sections().sections):

            lane_msg = self.get_lane(id).lane_msg

            self._publish_point_marker(
                lane_msg.left_line, "left_line", id, self.param.colors.left_line,
            )

            self._publish_point_marker(
                lane_msg.middle_line, "middle_line", id, self.param.colors.middle_line,
            )

            self._publish_point_marker(
                lane_msg.right_line, "right_line", id, self.param.colors.right_line,
            )

    def _show_parking_markers(self):
        """Publish markers for all parking lines and spots of road."""
        parking_sections = [
            s
            for s in self.get_sections().sections
            if s.type == road_section_type.PARKING_AREA
        ]

        parking_counter = 0
        spot_counter = 0

        for sec in parking_sections:
            parking_srv = self.get_parking(sec.id)
            left_msg, right_msg = parking_srv.left_msg, parking_srv.right_msg

            def show_borders(borders, topic_name, ns):
                nonlocal parking_counter
                for border in borders:
                    self._publish_point_marker(
                        border.points,
                        topic_name,
                        parking_counter,
                        self.param.colors.parking_border,
                        ns="/simulation/groundtruth/" + ns,
                    )
                    parking_counter += 1

            show_borders(left_msg.borders, "parking_line", "parking_left")
            show_borders(right_msg.borders, "parking_line", "parking_right")

            spots = []
            spots.extend(left_msg.spots)
            spots.extend(right_msg.spots)
            for spot in spots:
                # Extract all four points
                points = Polygon(spot.frame.points).to_geometry_msg().points

                # self.rviz_parking_spot_publisher.publish(poly)

                color = self.param.colors.parking_spot_free

                if spot.type == ParkingMsg.SPOT_X:
                    points = [
                        points[0],
                        points[1],
                        points[3],
                        points[2],
                        points[0],
                        points[3],
                        points[2],
                        points[1],
                    ]  # Create an x in the visualization
                    color = self.param.colors.parking_spot_x
                elif spot.type == ParkingMsg.SPOT_OCCUPIED:
                    color = self.param.colors.parking_spot_occupied

                self._publish_point_marker(points, "parking_spot", spot_counter, color)

                spot_counter += 1

    def _show_obstacle_markers(self):
        """Publish polygons for all obstacles on the road."""
        obstacles = sum(
            (
                [
                    Polygon(msg.frame).to_geometry_msg().points
                    for msg in self.get_obstacles(sec.id).polygons
                    if msg.type == LabeledPolygonMsg.OBSTACLE
                ]
                for sec in self.get_sections().sections
            ),
            [],
        )

        for id, obstacle in enumerate(obstacles):
            self._publish_point_marker(obstacle, "obstacle", id, self.param.colors.obstacle)

    def _show_intersection_markers(self):
        """Publish lanes and surface markings of all intersections."""
        intersec_id = 0

        def show_lines(lane_msg):
            nonlocal intersec_id
            self._publish_point_marker(
                lane_msg.left_line,
                "intersection",
                intersec_id,
                self.param.colors.left_line,
                ns="/simulation/groundtruth/intersection",
            )
            intersec_id += 1

            self._publish_point_marker(
                lane_msg.middle_line,
                "intersection",
                intersec_id,
                self.param.colors.middle_line,
                ns="/simulation/groundtruth/intersection",
            )
            intersec_id += 1

            self._publish_point_marker(
                lane_msg.right_line,
                "intersection",
                intersec_id,
                self.param.colors.right_line,
                ns="/simulation/groundtruth/intersection",
            )
            intersec_id += 1

        for intersection in (
            self.get_intersection(sec.id)
            for sec in self.get_sections().sections
            if sec.type == road_section_type.INTERSECTION
        ):
            # Show all lines
            show_lines(intersection.south)
            show_lines(intersection.west)
            show_lines(intersection.east)
            show_lines(intersection.north)

    def update(self):
        """Update all markers."""
        rospy.logdebug("Updating groundtruth visualization.")
        self._show_lane_markers()
        self._show_obstacle_markers()
        self._show_parking_markers()
        self._show_intersection_markers()
