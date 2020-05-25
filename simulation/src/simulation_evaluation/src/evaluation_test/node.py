from typing import List, Tuple
from geometry_msgs.msg import Twist
import rospy

from simulation.utils.ros_base.node_base import NodeBase
from simulation.utils.geometry import Line, Polygon, Transform, Vector
from gazebo_simulation.msg import CarState as CarStateMsg


class EvaluationTestNode(NodeBase):
    """Node primarily useful to test and debug nodes in the evaluation pipeline.

    The node has a number of predefined paths and is able to fake CarState messages
    as if the car was driving on one of those paths.
    """

    def __init__(self):
        NodeBase.__init__(self, name="test_evaluation_pipeline_node", log_level=rospy.DEBUG)

        # Loop the drive function!
        self.run(function=self.drive, rate=10)

    def start(self):
        self.car_state_publisher = rospy.Publisher(
            self.param.topics.car_state.car_state, CarStateMsg
        )

    def fake_car_state(self, path: Line, arc_length: float, speed: float):
        """Publish a CarState message depending on situation.

        Args:
            path: The car drives on this path.
            arc_length: Current arc length of the car on the path.
            speed: Speed the car drives with.
        """
        pose = path.interpolate_pose(arc_length=arc_length)
        msg = CarStateMsg()
        msg.pose = pose.to_geometry_msg()
        msg.frame = (
            Transform(pose.to_geometry_msg())
            * Polygon([[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1], [0.1, -0.1]])
        ).to_geometry_msg()
        msg.twist = Twist()
        msg.twist.linear = Vector(speed, 0, 0).to_geometry_msg()
        self.car_state_publisher.publish(msg)

    def _get_path(self) -> Line:
        """Create a predefined path.

        Depending on the *path* parameter,
        one of the predefined paths is created.
        """
        right_center = -1 / 2 * self.param.lane_width  # -0.2 by default
        left_center = 1 / 2 * self.param.lane_width  # 0.2 by default

        right_parking = -3 / 2 * self.param.lane_width
        left_parking = 3 / 2 * self.param.lane_width

        length = 6

        if self.param.path == 0:  # Straight
            return Line([[0.2, right_center], [length - 0.2, right_center]])
        if self.param.path == 1:  # Off road right
            return Line([[0.2, right_center], [length, -3]])
        if self.param.path == 2:  # Overtaking
            return Line(
                [
                    [0.2, right_center],
                    [2, right_center],
                    [2, left_center],
                    [4, left_center],
                    [4, right_center],
                    [length - 0.2, right_center],
                ]
            )
        if self.param.path == 3:  # Parking front right
            return Line(
                [
                    [0.2, right_center],
                    [2.5, right_center],
                    [2.5, right_parking],
                    [2.5, right_center],
                    [length - 0.2, right_center],
                ]
            )
        if self.param.path == 4:  # Parking front right
            return Line(
                [
                    [0.2, right_center],
                    [3.5, right_center],
                    [3.5, right_parking],
                    [3.5, right_center],
                    [length - 0.2, right_center],
                ]
            )
        if self.param.path == 5:  # Parking front left
            return Line(
                [
                    [0.2, right_center],
                    [2.5, right_center],
                    [2.5, left_parking],
                    [2.5, right_center],
                    [length - 0.2, right_center],
                ]
            )
        if self.param.path == 6:  # Parking back left
            return Line(
                [
                    [0.2, right_center],
                    [3.5, right_center],
                    [3.5, left_parking],
                    [3.5, right_center],
                    [length - 0.2, right_center],
                ]
            )

    def _get_stops(self) -> List[Tuple[float, float]]:
        """Load stops from parameters.

        A stop can be configured inside a parameter file by appending to the stops list.
        The stop list should be a list of lists with two values. E.g.

        stops: [[1, 2], [3, 1]]

        is interpretated as a stop after 1m for 2 seconds and another stop after 3 meters \
                for 1 second.
        Return:
            Sorted list of all stops as tuples with arc_length and duration.
        """
        return [(stop[0], stop[1]) for stop in sorted(self.param.stops, key=lambda x: x[0])]

    def drive(self):
        """Drive along a path."""
        path = self._get_path()
        stops = self._get_stops()

        start_time = rospy.Time.now().to_sec()
        time = start_time

        def length_speed(duration) -> Tuple[float, float]:
            """Calculate current arc_length and speed."""
            nonlocal start_time

            speed = self.param.speed
            length = duration * speed

            while len(stops) > 0 and stops[0][0] < length:
                length -= stops[0][1] * self.param.speed
                length = max(stops[0][0], length)
                speed = 0
                if length > stops[0][0]:
                    start_time += stops[0][1]
                    del stops[0]
            return length, speed

        while length_speed(time - start_time)[0] < path.length:

            self.fake_car_state(path, *length_speed(time - start_time))

            rospy.sleep(1 / self.param.update_rate)
            time = rospy.Time.now().to_sec()
