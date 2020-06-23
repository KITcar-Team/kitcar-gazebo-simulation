import unittest
import random
import functools

from simulation.utils.geometry import Line, Pose, Polygon, Point

from simulation.src.simulation_evaluation.src.speaker.speakers import AreaSpeaker

from . import fake_msgs
from . import utils


from simulation_evaluation.msg import Speaker as SpeakerMsg
from gazebo_simulation.msg import CarState as CarStateMsg


class ModuleTest(unittest.TestCase):
    def setUp(self):

        n_section = 1
        n_points = 100
        lane_width = 0.4

        # Create points
        lines = fake_msgs.create_points(
            section_count=n_section,
            point_count=n_points,
            offset_right=lane_width,
            direction=0,
            deviation=0,
        )

        self.right_border = Line(
            [[0, -lane_width], [1, -2 * lane_width], [2, -2 * lane_width], [3, -lane_width]]
        )
        self.left_border = Line(
            [[0, lane_width], [1, 2 * lane_width], [2, 2 * lane_width], [3, lane_width]]
        )

        # Fake section and lane msg proxies / usually in the groundtruth package
        section_msg_proxy = functools.partial(
            fake_msgs.section_msgs, section_count=n_section
        )
        lane_msg_proxy = functools.partial(fake_msgs.lane_msgs, lines)
        parking_msg_proxy = functools.partial(
            fake_msgs.parking_msgs, [], [], self.right_border, self.left_border
        )

        self.speaker = AreaSpeaker(
            section_proxy=section_msg_proxy,
            lane_proxy=lane_msg_proxy,
            parking_proxy=parking_msg_proxy,
            min_wheel_count=3,
            area_buffer=0.001,
        )

    def test_speaker_properties(self):
        """Test properties and speak function of event speaker."""

        self.assertTrue(
            utils.polygon_list_almost_equal(
                [self.speaker.right_corridor],
                [Polygon(self.speaker.middle_line, self.speaker.right_line)],
            )
        )
        self.assertTrue(
            utils.polygon_list_almost_equal(
                [self.speaker.left_corridor],
                [Polygon(self.speaker.left_line, self.speaker.middle_line)],
            )
        )
        self.assertTrue(
            utils.polygon_list_almost_equal(
                self.speaker.parking_lots,
                [
                    Polygon(self.right_border.get_points()),
                    Polygon(self.left_border.get_points()),
                ],
            )
        )

    def test_speak_function(self):
        """Test speaker msg """
        # Car msg
        pose = Pose(Point(1, 3), 0)

        # Frames
        frames = []

        frames.append(
            (
                Polygon(
                    [Point(0.1, 0.1), Point(0.1, 0.3), Point(0.5, 0.3), Point(0.5, 0.1)]
                ),
                SpeakerMsg.LEFT_LANE,
            )
        )  # On left side
        frames.append(
            (
                Polygon(
                    [Point(0, -0.3), Point(0, -0.01), Point(0.5, -0.01), Point(0.5, -0.3)]
                ),
                SpeakerMsg.RIGHT_LANE,
            )
        )  # On right side
        frames.append(
            (
                Polygon([Point(0, -0.3), Point(0, 0.3), Point(0.5, 0.3), Point(0.5, -0.3)]),
                SpeakerMsg.LEFT_LANE,
            )
        )  # Between left and right should return left!
        frames.append(
            (
                Polygon(
                    [Point(-0.5, -0.3), Point(-0.5, 0.3), Point(0.5, 0.3), Point(0.5, -0.3)]
                ),
                SpeakerMsg.OFF_ROAD,
            )
        )  # Partly in partly out
        frames.append(
            (
                Polygon([Point(10, 2), Point(10, 3), Point(15, 3), Point(15, 2)]),
                SpeakerMsg.OFF_ROAD,
            )
        )  # Off road

        # Parking
        frames.append(
            (
                Polygon([Point(1, 0.4), Point(1, 0.7), Point(2, 0.7), Point(2, 0.4)]),
                SpeakerMsg.PARKING_LOT,
            )
        )  # On left parking
        frames.append(
            (
                Polygon([Point(1, -0.4), Point(1, -0.7), Point(1, -0.7), Point(2, -0.4)]),
                SpeakerMsg.PARKING_LOT,
            )
        )  # On right parking
        frames.append(
            (
                Polygon([Point(1, 0), Point(1, -0.7), Point(2, -0.7), Point(2, 0)]),
                SpeakerMsg.PARKING_LOT,
            )
        )  # On right side and parking lot

        car_msg = CarStateMsg()
        car_msg.pose = pose.to_geometry_msg()

        for frame, expected in frames[-1:]:
            print(f"Expecting msg {expected} for frame {frame}.")
            self.speaker.listen(car_msg)
            self.assertTrue(utils.assert_msgs_for_pos(self.speaker, frame, expected))


if __name__ == "__main__":
    random.seed("KITCAR")
    unittest.main()
