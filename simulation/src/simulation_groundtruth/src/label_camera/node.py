"""ROS node that connects multiple speakers to ROS topics."""

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge.core import CvBridge
from gazebo_simulation.msg import CarState as CarStateMsg
from sensor_msgs.msg import Image as ImageMsg
from simulation_groundtruth.msg import GroundtruthStatus
from simulation_groundtruth.msg import ImageLabels as ImageLabelsMsg
from simulation_groundtruth.srv import LabeledPolygonSrv, LaneSrv, SectionSrv

# Messages
from std_srvs.srv import Empty as EmptySrv

from simulation.src.gazebo_simulation.src.car_model.camera_specs import CameraSpecs
from simulation.utils.geometry import Transform
from simulation.utils.ros_base.node_base import NodeBase

from .bounding_box import BoundingBox, VisualBoundingBox
from .label_speaker import LabelSpeaker


class LabelCameraNode(NodeBase):
    def __init__(self):
        """Initialize the node."""
        super().__init__(name="label_camera_node", log_level=rospy.DEBUG)

        # Load world to camera transformation
        self.camera_specs = CameraSpecs(**self.param.camera.as_dict())

        self.run()

    def start(self):
        groundtruth_topics = self.param.topics.groundtruth

        # Create groundtruth service proxies
        section_proxy = rospy.ServiceProxy(
            groundtruth_topics.section, SectionSrv, persistent=True
        )
        lane_proxy = rospy.ServiceProxy(groundtruth_topics.lane, LaneSrv, persistent=True)
        obstacle_proxy = rospy.ServiceProxy(
            groundtruth_topics.obstacle, LabeledPolygonSrv, persistent=True
        )
        surface_marking_proxy = rospy.ServiceProxy(
            groundtruth_topics.surface_marking, LabeledPolygonSrv, persistent=True
        )
        traffic_sign_proxy = rospy.ServiceProxy(
            groundtruth_topics.traffic_sign, LabeledPolygonSrv, persistent=True
        )
        self.pause_physics_proxy = rospy.ServiceProxy(
            self.param.topics.pause_gazebo, EmptySrv
        )
        self.unpause_physics_proxy = rospy.ServiceProxy(
            self.param.topics.unpause_gazebo, EmptySrv
        )
        self._proxies = [
            section_proxy,
            lane_proxy,
            obstacle_proxy,
            surface_marking_proxy,
            traffic_sign_proxy,
            self.unpause_physics_proxy,
            self.pause_physics_proxy,
        ]

        rospy.wait_for_service(groundtruth_topics.section)

        self.groundtruth_status_subscriber = rospy.Subscriber(
            self.param.topics.groundtruth.status,
            GroundtruthStatus,
            callback=self.receive_groundtruth_update,
        )

        # Add all speakers
        self.label_speaker = LabelSpeaker(
            section_proxy=section_proxy,
            lane_proxy=lane_proxy,
            obstacle_proxy=obstacle_proxy,
            surface_marking_proxy=surface_marking_proxy,
            sign_proxy=traffic_sign_proxy,
        )

        # Subscribe to carstate
        self.car_state_subscriber = rospy.Subscriber(
            self.param.topics.car_state.car_state,
            CarStateMsg,
            self.receive_car_state,
            queue_size=1,
        )
        self.image_subscriber = rospy.Subscriber(
            "/camera/image_raw", ImageMsg, self.receive_image, queue_size=1
        )
        self.image_publisher = rospy.Publisher(
            self.param.topics.debug_image, ImageMsg, queue_size=1
        )
        self.label_publisher = rospy.Publisher(
            self.param.topics.image_labels, ImageLabelsMsg, queue_size=100
        )

        self.listener = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.listener)

        rospy.wait_for_message(self.param.topics.car_state.car_state, CarStateMsg)

    def stop(self):
        self.subscriber.unregister()

        for p in self._proxies:
            p.close()

        self.groundtruth_status_subscriber.unregister()
        self.car_state_subscriber.unregister()

        self.image_subscriber.unregister()
        self.image_publisher.unregister()
        self.label_publisher.unregister()

    def receive_groundtruth_update(self, msg: GroundtruthStatus):
        """Receive GroundtruthStatus message.

        Args:
            msg: New GroundtruthStatus message
        """
        self.param.active = msg.status == GroundtruthStatus.READY

    def receive_car_state(self, msg: CarStateMsg):
        """Receive CarState message and update transformations and label speaker."""
        self._latest_car_state_msg = msg

    def receive_image(self, msg: ImageMsg):
        """Receive new camera image and publish corresponding labels."""
        try:
            tf_transform = self.listener.lookup_transform(
                "vehicle", "sim_world", msg.header.stamp, timeout=rospy.Duration(0.1)
            )
            current_tf = Transform(tf_transform.transform)
        except Exception as e:
            rospy.logerr(f"Could not lookup transform {e}")
            return

        # Pass latest car state to the speaker to ensure tf and
        # car state are approximately synchronized
        self.label_speaker.listen(self._latest_car_state_msg)

        try:
            # Pause physics to prevent the car from moving any further
            # while calculating the bounding boxes
            self.pause_physics_proxy()

            image_size = (
                self.camera_specs.output_size["width"],
                self.camera_specs.output_size["height"],
            )

            # Update transformations
            BoundingBox.set_tfs(current_tf, self.camera_specs.P)

            visible_bbs = self.label_speaker.speak(
                image_size, self.camera_specs.horizontal_fov
            )

            self.label_publisher.publish(
                ImageLabelsMsg(
                    img_header=msg.header,
                    bounding_boxes=[bb.to_msg() for bb in visible_bbs],
                )
            )

            # Optionally publish an image with bounding boxes drawn into it
            if self.param.publish_debug_image:
                colors = self.param.colors.as_dict()

                visualization_bbs = (
                    VisualBoundingBox(
                        bb.get_bounds(),
                        label=bb.class_description,
                        color=colors[str(bb.class_id // 100)]
                        if str(bb.class_id // 100) in colors
                        # ID Namespaces are in steps of 100.
                        else colors[str(-1)],
                    )
                    for bb in visible_bbs
                )

                br = CvBridge()
                cv_img = br.imgmsg_to_cv2(msg)
                cv_img: np.ndarray = cv_img.copy()
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

                for bb in visualization_bbs:
                    bb.draw(cv_img)
                out_msg = br.cv2_to_imgmsg(cv_img, encoding="rgb8")
                out_msg.header = msg.header

                self.image_publisher.publish(out_msg)
        finally:
            self.unpause_physics_proxy()
