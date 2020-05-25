import unittest
from simulation.src.simulation_evaluation.src.speaker.speakers import SpeedSpeaker
from simulation.utils.geometry import Point

from gazebo_simulation.msg import CarState as CarStateMsg
from simulation_evaluation.msg import Speaker as SpeakerMsg


class ModuleTest(unittest.TestCase):
    def test_speed_speaker(self):

        speed_speaker = SpeedSpeaker(time=0)

        car_msg = CarStateMsg()

        speeds = []

        # speed, (time_before, time_after), expected result
        speeds.append((0, (0, 2), SpeakerMsg.SPEED_HALTED))
        speeds.append((0, (0, 4), SpeakerMsg.SPEED_STOPPED))
        speeds.append((0, (0, 0), SpeakerMsg.SPEED_0))
        speeds.append((3, (0, 0), SpeakerMsg.SPEED_1_10))
        speeds.append((11, (0, 0), SpeakerMsg.SPEED_11_20))
        speeds.append((25, (0, 0), SpeakerMsg.SPEED_21_30))
        speeds.append((32, (0, 0), SpeakerMsg.SPEED_31_40))
        speeds.append((44, (0, 0), SpeakerMsg.SPEED_41_50))
        speeds.append((55, (0, 0), SpeakerMsg.SPEED_51_60))
        speeds.append((65, (0, 0), SpeakerMsg.SPEED_61_70))
        speeds.append((73, (0, 0), SpeakerMsg.SPEED_71_80))
        speeds.append((83, (0, 0), SpeakerMsg.SPEED_81_90))
        speeds.append((99, (0, 0), SpeakerMsg.SPEED_91_))

        for speed, time, expected in speeds:

            speed_speaker.listen(car_msg)
            speed_speaker.speak(current_time=time[0])

            car_msg.twist.linear = Point(speed / 3.6 / 10, 0, 0).to_geometry_msg()  # 1 m/s
            speed_speaker.listen(car_msg)

            response = speed_speaker.speak(current_time=time[1])
            self.assertEqual(response[0].type, expected)


if __name__ == "__main__":
    unittest.main()
