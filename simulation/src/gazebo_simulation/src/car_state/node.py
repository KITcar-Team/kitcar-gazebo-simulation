import rospy
import numpy as np

from simulation.utils.ros_base.node_base import NodeBase
from simulation.utils.geometry import Vector, Point, Polygon, Transform

import geometry_msgs.msg
from gazebo_simulation.msg import CarState as CarStateMsg

from . import export

from simulation.src.gazebo_simulation.src.car_model.car_specs import CarSpecs
from simulation.src.gazebo_simulation.src.car_model.camera_specs import CameraSpecs

__copyright__ = "KITcar"


@export
class CarStateNode(NodeBase):
    """ROS node which publishes information about the model in a CarState.

    Attributes:
        car_frame (shapely.geom.Polygon): Frame of car in vehicle coordinate system
        model_pose_subscriber (rospy.Subscriber): Receive the cars pose
        model_twist_subscriber (rospy.Subscriber): Receive the cars twist
        get_vehicle_twist (Callable[[],None]: Returns current vehicle twist by calling the service proxy
        publisher (rospy.publisher): CarStateMsg) publishes real time information about the car
    """

    def __init__(self):
        """ initialize the node"""

        super(CarStateNode, self).__init__(name="car_state_node")

        self.read_car_config()

        # Start running node.
        self.run(function=self.state_update, rate=self.param.max_rate)

    def receive_pose(self, pose):
        self.latest_vehicle_pose = pose

    def receive_twist(self, twist):
        self.latest_vehicle_twist = twist

    def start(self):
        """Start node."""

        self.model_pose_subscriber = rospy.Subscriber(
            self.param.topics.model_plugin.namespace
            + "/"
            + self.param.car_name
            + "/"
            + self.param.topics.model_plugin.get.pose,
            geometry_msgs.msg.Pose,
            self.receive_pose,
            queue_size=1,
        )

        self.model_twist_subscriber = rospy.Subscriber(
            self.param.topics.model_plugin.namespace
            + "/"
            + self.param.car_name
            + "/"
            + self.param.topics.model_plugin.get.twist,
            geometry_msgs.msg.Twist,
            self.receive_twist,
            queue_size=1,
        )

        rospy.wait_for_message(self.model_pose_subscriber.name, geometry_msgs.msg.Pose)
        rospy.wait_for_message(self.model_twist_subscriber.name, geometry_msgs.msg.Twist)

        self.publisher = rospy.Publisher(
            self.param.topics.car_state, CarStateMsg, queue_size=1
        )

        super().start()

    def stop(self):
        """Turn off node."""
        super().stop()
        self.model_pose_subscriber.unregister()
        self.model_twist_subscriber.unregister()
        self.publisher.unregister()

    def read_car_config(self):
        """Process car parameters.

        """
        car_specs = CarSpecs.from_file(self.param.car_specs_path)
        camera_specs = CameraSpecs.from_file(self.param.camera_specs_path)

        """ Car frame config """
        chassis_size = Vector(
            car_specs.distance_to_rear_bumper
            # + car_specs.wheelbase
            + car_specs.distance_cog_front,
            car_specs.vehicle_width,
        )
        chassis_position = Point(car_specs.center_rear_axle.x, car_specs.center_rear_axle.y)

        # get dimensions
        x_span = Vector(
            0.5 * chassis_size.x, 0
        )  # Vector in x direction of length = width/2
        y_span = Vector(0, 0.5 * chassis_size.y)
        self.car_frame = Polygon(
            [
                chassis_position + x_span + y_span,  # Front right
                chassis_position - x_span + y_span,  # Front left
                chassis_position - x_span - y_span,  # Back left
                chassis_position + x_span - y_span,  # Back right
            ]
        )

        """ Camera config """
        # This parameter tells how far the camera can see
        view_distance: float = camera_specs.clip["far"]

        # field of view (opening angle of camera)
        fov: float = camera_specs.horizontal_fov

        if self.param.cone_points == 0:
            self.view_cone = None
            return

        # Calculate a few points to approximate view frame
        # Add points on horizon of our camera (at view_distance away from vehicle) /approximates a view cone

        # Create geom.Polygon from points
        self.view_cone = Polygon(
            [Point(0, 0)]
            + [
                Point(r=view_distance, phi=alpha)
                for alpha in np.linspace(-fov / 2, fov / 2, self.param.cone_points)
            ]
        )

    def state_update(self):
        """Publish new CarState with updated information."""

        # Request current pose and twist from model_interface
        pose: geometry_msgs.msg.Pose = self.latest_vehicle_pose
        twist: geometry_msgs.msg.Twist = self.latest_vehicle_twist

        # Transform which is used to calculate frame and view cone
        tf = Transform(pose)

        rospy.logdebug(f"State update transform: {tf.rotation}")

        # Create message
        msg = CarStateMsg()
        msg.pose = pose
        msg.twist = twist
        msg.frame = (tf * self.car_frame).to_geometry_msg()

        if self.view_cone:
            msg.view_cone = (tf * self.view_cone).to_geometry_msg()

        if not rospy.is_shutdown():
            self.publisher.publish(msg)
