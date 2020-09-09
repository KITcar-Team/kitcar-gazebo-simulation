"""Prepare Gazebo’s camera image for kitcar-ros."""

import rospy
from cv_bridge.core import CvBridge
from sensor_msgs.msg import Image as ImageMsg

from simulation.utils.ros_base.node_base import NodeBase


class SensorCameraNode(NodeBase):
    """ROS node which receives the camera image, crops it and can perform preprocessing.

    Attributes:
        subscriber (rospy.subscriber): Subscribes to camera image from gazebo
        publisher (rospy.publisher): Publishes camera image.
    """

    def __init__(self):

        super().__init__(
            name="sensor_camera_node"
        )  # Name can be overwritten in launch file

        if self.param.apply_gan:
            # Import the ros connector here because it requires the "torch" package.
            # The torch package is quite heavy and should only be necessary if the
            # neural network is used.
            from simulation.utils.machine_learning.cycle_gan.ros_connector import (
                RosConnector,
            )

            self.gan_connector = RosConnector(self.param.use_wasserstein_gan)

        self.run()

    def start(self):
        """Start node."""
        self.publisher = rospy.Publisher(
            self.param.topics.pub_topic, ImageMsg, queue_size=1
        )
        self.subscriber = rospy.Subscriber(
            self.param.topics.sub_topic,
            ImageMsg,
            callback=self.receive_image,
            queue_size=1,
        )
        super().start()

    def stop(self):
        """Turn off node."""
        self.subscriber.unregister()
        self.publisher.unregister()
        super().stop()

    def receive_image(self, img_msg: ImageMsg):
        """Process and republish new camera image.

        The image is cropped to simulate the internal preprocessing of our real camera.

        Args:
            img_msg: Published image message."""

        br = CvBridge()
        cv_img = br.imgmsg_to_cv2(img_msg)

        new_img = cv_img[
            self.param.output_start_y : self.param.output_end_y,
            self.param.output_start_x : self.param.output_end_x,
        ]

        if self.param.apply_gan:
            new_img = self.gan_connector(new_img)

        out_msg = br.cv2_to_imgmsg(new_img, encoding="mono8")
        out_msg.header = img_msg.header

        self.publisher.publish(out_msg)
