import functools
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, List

import rospy
import yaml
from simulation_evaluation.msg import SignEvaluation as SignEvaluationMsg
from simulation_evaluation.msg import TrafficSigns
from simulation_groundtruth.srv import LabeledPolygonSrv, SectionSrv
from visualization_msgs.msg import Marker

import simulation.utils.ros_base.visualization as visualization
from simulation.src.simulation_evaluation.src.sign_evaluation import plots
from simulation.utils.geometry import Point, Polygon
from simulation.utils.ros_base.transform_node import TransformNode


@dataclass
class Sign:
    position: Point
    desc: str
    visible: bool
    detections: List[Any] = field(default_factory=list)

    def evaluate(self):
        """Evaluate this sign.

        Returns:
            the true_positive false_positive and average_distance.
        """
        false_positive = sum(1 for _, desc in self.detections if desc != self.desc)
        true_positive = len(self.detections) - false_positive
        average_distance = (
            (
                sum(
                    self.position.distance(point)
                    for point, kind in self.detections
                    if kind == self.desc
                )
                / true_positive
            )
            if true_positive > 0
            else -1
        )
        return true_positive, false_positive, average_distance

    def to_msg(self):
        """Create the sign evaluation message."""
        return SignEvaluationMsg(
            self.position.to_geometry_msg(), self.desc, *self.evaluate()
        )

    @staticmethod
    def closest(signs: List["Sign"], point: Point) -> "Sign":
        def get_min(a, b):
            return a if a.position.distance(point) < b.position.distance(point) else b

        return functools.reduce(get_min, signs)

    @staticmethod
    def sum_evaluations(evaluations):
        """Sum over the evaluations.

        Args:
            evaluations: list of tp, fp and distance per sign

        Returns:
            The summed evaluations
        """

        def add_evaluations(e1, e2):
            """Add two evaluations.

            If Signs do not have detections the distance is -1, therefore the distance needs
            to be handled separately.
            """
            true_positive = e1[0] + e2[0]
            false_positive = e1[1] + e2[1]
            distance = e1[2] + e2[2]
            if e1[2] < 0 and e2[2] < 0:
                distance = -1
            elif e1[2] < 0:
                distance = e2[2]
            elif e2[2] < 0:
                distance = e1[2]
            return true_positive, false_positive, distance

        length = len(evaluations)
        tp, fp, d = functools.reduce(add_evaluations, evaluations)
        return tp, fp, float(d) / float(length)


class SignEvaluationNode(TransformNode):
    """Evaluate the sign detection of the simulated car."""

    def __init__(self):
        super().__init__(name="sign_evaluation_node", log_level=rospy.DEBUG)

        with open(os.path.join(os.path.dirname(__file__), "label_conversion.yaml")) as file:
            self.name_conversion = yaml.load(file, Loader=yaml.SafeLoader)

        self.name_conversion = {int(k[:2]): v for k, v in self.name_conversion.items()}

        self.run()

    def start(self):
        super().start()
        self.detections_subscriber = rospy.Subscriber(
            name=self.param.topics.detections,
            data_class=TrafficSigns,
            callback=self.save_msg,
        )
        self.publish_counter = 0
        self.evaluation_publisher = rospy.Publisher(
            self.param.topics.evaluation, data_class=SignEvaluationMsg, queue_size=10
        )
        self.marker_publisher = rospy.Publisher(
            self.param.topics.sign_marking, data_class=Marker, queue_size=10
        )

        rospy.wait_for_service(self.param.topics.traffic_sign)
        sign_proxy = rospy.ServiceProxy(self.param.topics.traffic_sign, LabeledPolygonSrv)

        rospy.wait_for_service(self.param.topics.section)
        sections = rospy.ServiceProxy(self.param.topics.section, SectionSrv)().sections

        self.signs = [
            Sign(Point(Polygon(msg.frame).centroid), msg.desc, msg.visible)
            for sec in sections
            for msg in sign_proxy(sec.id).polygons
        ]

    def stop(self):
        super().stop()
        self.detections_subscriber.unregister()
        self.marker_publisher.unregister()
        self.create_plots()

    def get_evaluations(self, ids):
        """Calculate the evaluation for the given ids.

        Returns:
            evaluations: The summed true_positive, false_positive, distances per sign.
            descriptions: The list of sign descriptions.
        """
        evaluations = [(self.signs[i].evaluate(), self.signs[i].desc) for i in ids]
        descriptions = list({self.signs[i].desc for i in ids})
        evaluations_per_sign = [
            ([e for e, desc in evaluations if desc == description], description)
            for description in descriptions
        ]
        summed_evaluations = [
            Sign.sum_evaluations(per_sign) for per_sign, _ in evaluations_per_sign
        ]
        return summed_evaluations, descriptions

    def create_plots(self):
        """Create the plots from the detected signs."""
        shutil.rmtree(self.param.path, ignore_errors=True)
        os.makedirs(self.param.path)

        ids = list(range(len(self.signs)))

        """True positives"""
        values, kinds = self.get_evaluations(ids)
        plots.create_plot(
            kinds,
            [e[0] for e in values],  # True positives
            save_dir=self.param.path,
            y_label="number_tp",
            file_name="number_tp",
            title="Amount of true positives",
        )

        # Only signs with at least one detection!
        ids = [i for i, _ in enumerate(self.signs) if self.signs[i].evaluate()[2] > 0]
        values, kinds = self.get_evaluations(ids)

        """Distance"""
        plots.create_plot(
            kinds,
            values=[e[2] for e in values],  # Distances
            save_dir=self.param.path,
            y_label="distance",
            file_name="distance",
            title="Distance",
        )
        """Precision"""
        plots.create_plot(
            kinds,
            # Precision signs with at least one detection are used, e[0]+e[1] > 0)
            values=[e[0] / (e[0] + e[1]) for e in values],
            save_dir=self.param.path,
            y_label="precision",
            file_name="precision",
            title="Precision",
        )

    def _publish_point_marker(
        self,
        point: Point,
        id: int,
        ns="simulation/sign_evaluation",
    ):
        """Publish an RVIZ marker on the publisher's topic.

        Args:
            point: Point to be published.
            id: RVIZ marker id.
            ns: RVIZ namespace of the marker.
        """
        rospy.logdebug(f"display point {point}")

        marker = visualization.get_marker(
            frame_id="sim_world",
            rgba=[255, 0, 255, 255],
            id=id,
            type=2,
            ns=ns,
            scale=0.05,
            duration=1 / self.param.rate,
        )
        marker.pose.position = point.to_geometry_msg()
        try:
            self.marker_publisher.publish(marker)
        except Exception as err:
            rospy.logerr(err)

    def add_detection(self, point: Point, desc: str):
        """Add the detection to the closest sign.

        Args:
            point: The center point of the detection.
            desc: The description of the found sign.
        """

        closest_sign = Sign.closest(self.signs, point)
        if closest_sign.position.distance(point) < self.param.distance_threshold:
            closest_sign.detections.append((point, desc))
        self.evaluation_publisher.publish(closest_sign.to_msg())

    def save_msg(self, msg: TrafficSigns):
        """Add a message to the signs.

        Args:
            msg: The traffic sign massage from the detection.
        """
        if msg.sub_messages:
            tf = self.get_transformation(
                "sim_world",
                msg.header.frame_id,
                msg.sub_messages[0].header.stamp,
                timeout=rospy.Duration(0.1),
            )
            if tf is not None:
                rospy.logdebug(f"point before transformation {msg.sub_messages}")
                points = [
                    (tf * Point(m.pose.position), self.name_conversion[m.type])
                    for m in msg.sub_messages
                ]
                for point, class_desc in points:
                    self.add_detection(point, class_desc)
                    self.publish_counter += 1
                    self._publish_point_marker(point, self.publish_counter)
            else:
                rospy.logerr(
                    f"Error: could not get a transformation, message: {msg.sub_messages}"
                )
