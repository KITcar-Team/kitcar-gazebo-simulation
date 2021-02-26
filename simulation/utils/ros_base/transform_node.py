"""ROS transform base node class with tf2 transform interface."""

from typing import Optional

import rospy
import tf2_ros

from simulation.utils.geometry import Transform
from simulation.utils.ros_base.node_base import NodeBase


class TransformNode(NodeBase):
    """Extends NodeBase with a tf2 transform handler.

    Args:
        name (str): Name of the node
        parameter_cache_time (int) = 1: Duration for which parameters will be cached
            for performance
        log_level (int) = rospy.INFO: Loglevel with which the node works.
    """

    def __init__(
        self, *, name: str, parameter_cache_time: float = 1, log_level: int = rospy.INFO
    ):
        super().__init__(
            name=name, parameter_cache_time=parameter_cache_time, log_level=log_level
        )

        self.__tf2 = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.__tf2)

    def get_transformation(
        self,
        target_frame: str,
        source_frame: str,
        timestamp: rospy.rostime.Time,
        timeout: rospy.rostime.Duration = rospy.Duration(0.1),
    ) -> Optional[Transform]:
        """tf2 transform handler.

        Arguments:
            target_frame (str): Name of the target frame
            source_frame (str): Name of the source frame
            timestamp (rospy.rostime.Time): The time in the buffer
            timeout (rospy.rostime.Duration): Lookup timeout

        Returns:
            Returns the transformation.
        """
        try:
            tf_transform = self.__tf2.lookup_transform(
                target_frame, source_frame, timestamp, timeout
            )
            return Transform(tf_transform.transform)
        except Exception as e:
            rospy.logerr(f"Could not lookup transform for message {e}")
            return
