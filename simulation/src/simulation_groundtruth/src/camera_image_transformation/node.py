"""Calculates the coordinates inside the world coordinate-system."""

import numpy as np
import rosparam
import rospy
from geometry_msgs.msg import Point as PointMsg
from numpy.linalg import inv

from simulation.src.gazebo_simulation.src.car_model.camera_specs import CameraSpecs
from simulation.utils.geometry import Vector
from simulation.utils.ros_base.transform_node import TransformNode


class CameraImageTransformation(TransformNode):
    """ROS node which calculates the coordinates inside the world coordinate-system.

    when clicking on the camera image inside rqt_image_view - /camera/image_raw_mouse_left
    need to be activated inside rqt_image_view

    Attributes:
        world_publisher (rospy.Publisher): Publishes the world coordinates
        camera_mouse_subscriber (rospy.Subscriber): Subscribes to image_raw_mouse_left
    """

    def __init__(self):
        """Initialize the node."""
        super().__init__(name="camera_image_transformation")
        self.run()

    def start(self):
        """Start node."""

        param = rosparam.get_param("/camera/")
        self.__camera_specs = CameraSpecs.from_dict(**param)

        self.camera_mouse_subscriber = rospy.Subscriber(
            self.param.topics.camera_mouse_left, PointMsg, self.on_mouse_click
        )

        self.world_publisher = rospy.Publisher(
            self.param.topics.coordinates, PointMsg, queue_size=10
        )

        super().start()

    def stop(self):
        """Turn off node."""

        self.camera_mouse_subscriber.unregister()

        self.world_publisher.unregister()

        super().stop()

    def on_mouse_click(self, msg: PointMsg):
        """Calculate and publish the coordinates inside the world coordinate-system when
        parsed the coordinates of a pixel on the camera image.

        Arguments.
            msg: Message with information about the pixel position on the camera image.
        """
        # Coordinates mouse click
        point_image = np.array([msg.x, msg.y, 1]).T
        # Coordinates vehicle (back axis)
        h_inv = inv(self.__camera_specs.H)
        point_car_ = np.array(h_inv @ point_image).reshape(3)
        # Normalize array
        point_car = point_car_ / point_car_[2]

        # Convert to sim vector
        sim_vec = Vector(point_car[:2])
        # Coordinates sim world
        world = self.get_transformation("odom", "vehicle", rospy.Time.now()) * sim_vec

        # Publish world coordinates
        self.world_publisher.publish(world.to_geometry_msg())
