from typing import List, Dict
import functools
from dataclasses import dataclass
import random
import math

import rospy
import std_msgs
import geometry_msgs.msg
from tf2_msgs.msg import TFMessage

from simulation.utils.ros_base.node_base import NodeBase
from simulation.utils.geometry import Vector, Transform, Line, Pose
from simulation_brain_link.msg import State as StateEstimationMsg
from simulation_groundtruth.srv import (
    SectionSrv,
    LaneSrv,
)


@dataclass
class DrivingState:
    distance_driven: float
    time: float


class AutomaticDriveNode(NodeBase):
    """ROS node to drive the car along the right side of the road.

    Instead of directly modifying the car's position and speed.
    The vehicle_simulation's output is emulated.
    I.e. the transform from the vehicle to it's world coordinate system is published
    and a state_estimation message published.

    This enables to use the vehicle_simulation_link_node to move the car
    and only replace KITcar_brain + vehicle_simulation!

    Attributes:
        pub_tf (rospy.publisher): Publishes the new vehicle/world transform.
        state_estimation_publisher (rospy.Publisher): Publishes state estimation messages.
        section_proxy (rospy.ServiceProxy): Connection to groundtruth section service.
        lane_proxy (rospy.ServiceProxy): Connection to groundtruth lane service.
        _driving_state (DrivingState): Keep track of time and distance along the road.
    """

    def __init__(self):

        super().__init__(
            name="automatic_drive_node", log_level=rospy.INFO
        )  # Name can be overwritten in launch file

        self.run(function=self.update, rate=float(self.param.rate))

    def start(self):
        """Start node."""
        self.pub_tf = rospy.Publisher(
            "/tf", TFMessage, queue_size=100
        )  # See: https://github.com/ros/geometry2/blob/melodic-devel/tf2_ros/src/tf2_ros/transform_broadcaster.py

        self.state_estimation_publisher = rospy.Publisher(
            self.param.topics.vehicle_simulation_link.state_estimation,
            StateEstimationMsg,
            queue_size=1,
        )

        groundtruth_topics = self.param.topics.groundtruth

        rospy.wait_for_service(groundtruth_topics.section, timeout=30)

        # Create groundtruth service proxies
        self.section_proxy = rospy.ServiceProxy(groundtruth_topics.section, SectionSrv)
        self.lane_proxy = rospy.ServiceProxy(groundtruth_topics.lane, LaneSrv)

        # Calculate the driving line once, so that it is cached!!
        self.driving_line

        # Read initial position from vehicle simulation link parameters
        try:
            initial = self.param.vehicle_simulation_link.initial_pose
            if len(initial) > 3:
                angle = initial[3]
                del initial[3]
            else:
                angle = 0
            pos = Vector(initial)
            self.initial_tf = Transform(pos, angle)
        except KeyError:
            self.initial_tf = None

        self._driving_state = DrivingState(0, rospy.Time.now().to_sec())

        super().start()

    def stop(self):
        self.state_estimation_publisher.unregister()
        self.pub_tf.unregister()
        super().stop()

    @functools.cached_property
    def middle_line(self) -> Line:
        """Line: Line in the middle of the road."""
        # Get all sections
        sections: List[int] = self.section_proxy().sections

        assert len(sections) > 0, (
            "There must be atleast one road section. "
            "(The groundtruth node might not be working correctly.)"
        )

        # Concatenate the middle line of all sections
        return sum(
            (Line(self.lane_proxy(sec.id).lane_msg.middle_line) for sec in sections), Line()
        )

    @functools.cached_property
    def driving_line(self) -> Line:
        """Line: Line where car drives."""

        path = Line()

        def append(offset, segment):
            nonlocal path

            if offset > 0:
                segment = segment.parallel_offset(offset, "left")
            elif offset < 0:
                segment = segment.parallel_offset(-offset, "right")

            path += segment

        if self.param.randomize_path:
            # Stitch a line together from varied offsets along the middle line
            length = self.middle_line.length
            x = 0
            offset = 0
            max_step = 4
            road_width = 0.4
            while x < length:
                offset = max(
                    min((0.5 - random.random()) * 2 * road_width, road_width), -road_width,
                )

                p = self.middle_line.interpolate_pose(x)
                orth = Vector(1, 0, 0).rotated(p.get_angle() + math.pi / 2)
                path += Line([p.position + offset * orth, p.position + offset * orth])

                x += max_step * random.random()
        else:
            param_path: List[Dict[str, float]] = self.param.path

            current_start = param_path[0]["start"]
            current_offset = param_path[0]["offset"]

            param_path.remove(param_path[0])

            # Read the path from the parameters
            for obj in param_path:
                end_arc_length = obj["start"]
                before_end_line = Line.cut(self.middle_line, end_arc_length)[0]
                current_segment = Line.cut(before_end_line, current_start)[1]

                append(current_offset, current_segment)

                current_offset = obj["offset"]
                current_start = obj["start"]

            current_segment = Line.cut(self.middle_line, current_start)[1]

            append(current_offset, current_segment)

        return path

    def update(self):
        """Calculate and publish new car state information."""
        # Update the driving state
        current_time = rospy.Time.now().to_sec()
        self._driving_state.distance_driven += (
            current_time - self._driving_state.time
        ) * self.param.speed
        self._driving_state.distance_driven %= self.driving_line.length
        self._driving_state.time = current_time

        rospy.logdebug(f"Current driving state: {self._driving_state}")

        # Calculate position, speed, and yaw
        position = self.driving_line.interpolate(self._driving_state.distance_driven)

        # Depending on the align_with_middle_line parameter, the car is always parallel to the
        # middle line or to the driving line.
        alignment_line = (
            self.middle_line if self.param.align_with_middle_line else self.driving_line
        )

        # Always let the car face into the direction of the middle line.
        pose = Pose(
            position,
            alignment_line.interpolate_direction(alignment_line.project(position)),
        )

        speed = Vector(self.param.speed, 0)  # Ignore y component of speed
        # Yaw rate = curvature * speed
        yaw_rate = (
            alignment_line.interpolate_curvature(
                min(self._driving_state.distance_driven, alignment_line.length)
            )
            * self.param.speed
        )

        # Publish up to date messages!
        self.update_world_vehicle_tf(
            self.initial_tf.inverse * Transform(pose, pose.get_angle())
        )
        self.update_state_estimation(speed, yaw_rate)

    def update_state_estimation(self, speed: Vector, yaw_rate: float):
        """Publish new state estimation message.

        Args:
            speed: Current speed in vehicle coordinates.
            yaw_rate: Yaw rate of the car.
        """
        msg = StateEstimationMsg()
        msg.speed_x = speed.x
        msg.speed_y = speed.y
        msg.yaw_rate = yaw_rate

        self.state_estimation_publisher.publish(msg)

    def update_world_vehicle_tf(self, vehicle_world_tf: Transform):
        """Publish up to date world to vehicle transformation to /tf.

        Args:
            vehicle_world_tf(Transform): Transformation between vehicle and world frames.
        """
        tf_stamped = geometry_msgs.msg.TransformStamped()

        tf_stamped.header = std_msgs.msg.Header()
        tf_stamped.header.stamp = rospy.Time.now()
        # Transform from world to vehicle
        tf_stamped.header.frame_id = self.param.vehicle_simulation_link.frame.world
        tf_stamped.child_frame_id = self.param.vehicle_simulation_link.frame.vehicle

        # Transformation from world to vehicle
        tf_stamped.transform = (vehicle_world_tf).to_geometry_msg()

        self.pub_tf.publish(TFMessage([tf_stamped]))
