"""Gazebo does not provide distance sensors out of the box.

As a workaround, the simulated `Dr. Drift` is equipped with depth cameras.
The depth camera sensor data is then converted into a distance
by extracting the closest point inside the depth cameras point cloud.

This is done separately for each time of flight sensor through an instance
of the SensorTofNode.
"""

import rospy

# Messages
import sensor_msgs.point_cloud2
from sensor_msgs.msg import Range, PointCloud2

from simulation.utils.geometry.vector import Vector

from simulation.utils.ros_base.node_base import NodeBase

__copyright__ = "KITcar"


class SensorTofNode(NodeBase):
    """ROS node which receives a pointcloud and publishes the point with minimum norm of that pointcloud.
    This is done to emulate a time of flight sensor.

    Attributes:
        subscriber (rospy.subscriber): Subscribes to depth camera sensor from gazebo
        publisher (rospy.publisher): Publishes distance to object on tof topic
    """

    def __init__(self):
        """ initialize the node"""

        super().__init__(name="tof_sensor_node")  # Name can be overwritten in launch file

        self.run()

    def start(self):
        """Start node."""
        self.publisher = rospy.Publisher(self.param.topics.tof_sensor, Range, queue_size=1)
        self.subscriber = rospy.Subscriber(
            self.param.topics.depth_camera,
            PointCloud2,
            callback=self.pointcloud_cb,
            queue_size=1,
        )
        super().start()

    def stop(self):
        """Turn off node."""
        self.subscriber.unregister()
        self.publisher.unregister()
        super().stop()

    def pointcloud_cb(self, pc: PointCloud2):
        """Process new sensor information of depth camera."""

        out_msg = Range()

        out_msg.field_of_view = self.param.tof.field_of_view
        out_msg.min_range = self.param.tof.min_range
        out_msg.max_range = self.param.tof.max_range
        out_msg.header.frame_id = self.param.frame_id

        vecs = (
            Vector(p)
            for p in sensor_msgs.point_cloud2.read_points(pc, field_names=("x", "y", "z"))
        )
        out_msg.range = min(v.norm for v in vecs)

        rospy.logdebug(f"Pointcloud received in {rospy.get_name()}:{vecs}")
        rospy.logdebug(f"Publishing range: {out_msg.range}")

        self.publisher.publish(out_msg)
