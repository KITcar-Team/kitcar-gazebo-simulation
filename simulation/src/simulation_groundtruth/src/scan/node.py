import math

import rospy
import tf2_ros

from simulation.utils.geometry import Pose, Transform, Vector
from simulation.utils.road.sections import CustomSection
from simulation.utils.ros_base.node_base import NodeBase


class ScanNode(NodeBase):
    """ROS node providing services to access the road's groundtruth."""

    def __init__(self, name="scan_node", log_level=rospy.INFO):
        super().__init__(name=name, log_level=log_level)

        self.section = CustomSection()
        self.run(function=self.update_middle_line, rate=self.param.rate)

    def start(self):
        self.listener = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.listener)

    def stop(self):
        self.save_section()

    def save_section(self):
        """Save the current custom section to a file."""
        if len(self.section.middle_line_points) > 1:
            # Adjust initial position to zero.
            initial_pose = Pose(self.section.middle_line.interpolate_pose(0))
            tf = Transform(initial_pose.position, initial_pose.orientation).inverse
            self.section.middle_line_points = [
                tf * p for p in self.section.middle_line_points
            ]
        self.section.save_as_yaml(self.param.file_path)

    def get_current_pose(self):
        """Try to get the current pose from /tf."""
        try:
            tf_transform = self.listener.lookup_transform(
                "odom", "vehicle", rospy.Time.now(), timeout=rospy.Duration(0.01)
            )
            return Pose(tf_transform.transform)
        except Exception as e:
            rospy.logerr(f"Could not lookup transform {e}")
            return

    def update_middle_line(self):
        """Update the middle line using the current position."""

        current_pose = self.get_current_pose()
        if current_pose is None:
            return

        middle_line_position = current_pose.position + Vector(0, 0.2).rotated(
            current_pose.orientation
        ).rotated(math.pi / 2)

        if (
            len(self.section.middle_line_points) == 0
            or middle_line_position.distance(self.section.middle_line_points[-1])
            > self.param.min_point_distance
        ):
            self.section.middle_line_points.append(middle_line_position)
            rospy.logdebug(f"Add new point: {middle_line_position}")
