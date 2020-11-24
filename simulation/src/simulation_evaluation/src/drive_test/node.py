import rospy
from simulation_brain_link.msg import MissionMode
from simulation_evaluation.msg import Referee as RefereeMsg
from simulation_groundtruth.msg import GroundtruthStatus
from std_msgs.msg import String as StringMsg

from simulation.utils.ros_base.node_base import NodeBase


class DriveTestNode(NodeBase):
    """Complete an automatic test using this node.

    Also the gazebo simulation, KITcar_brain and the CarStateNode needs to be running.
    """

    def __init__(self, run: bool = True):
        """Initialize the node and optionally start it as well.

        Args:
            run: Indicate wether to call :py:func:`run`.
                 In a test it can be useful to do this manually.
        """
        super().__init__(name="drive_test_node", log_level=rospy.DEBUG)

        self._groundtruth_status = GroundtruthStatus.READY
        self._previous_state_machine_msg = None
        self._state_machine_msg = None
        self.state = -1
        self.parking_successes = -1

        if run:
            self.run()

    def start(self):

        self.mission_mode_publisher = rospy.Publisher(
            self.param.topics.mission_mode, MissionMode, queue_size=1
        )

        self.groundtruth_status_subscriber = rospy.Subscriber(
            self.param.topics.groundtruth.status,
            GroundtruthStatus,
            callback=self.receive_groundtruth_update,
        )

        # 1. Ensure that evaluation pipeline is started up
        # Otherwise it can happen that the groundtruth is reloaded
        # while the speaker is starting up.
        # Then the speakers will operate on a false groundtruth!
        rospy.wait_for_message(self.param.topics.referee.info, RefereeMsg)

        # For some reason this is necessary for the mission mode publisher to work correctly
        rospy.sleep(0.1)

        self.referee_subscriber = rospy.Subscriber(
            self.param.topics.referee.info,
            RefereeMsg,
            callback=self.referee_cb,
            queue_size=1,
        )  # Subscribe to referee

        self.sm_info_subscriber = rospy.Subscriber(
            self.param.topics.state_machine.info,
            StringMsg,
            callback=self.receive_state_machine_info,
            queue_size=1,
        )  # Subscribe to sm updates

        rospy.wait_for_message(self.param.topics.referee.info, RefereeMsg)

        # Goooo
        self.update_mission_mode(self.param.mission_mode)

    def stop(self):
        self.referee_subscriber.unregister()
        self.sm_info_subscriber.unregister()
        self.groundtruth_status_subscriber.unregister()
        self.mission_mode_publisher.unregister()
        self.renderer_reload_publisher.unregister()

    def receive_groundtruth_update(self, msg: GroundtruthStatus):
        """Receive GroundtruthStatus message.

        Args:
            msg: New GroundtruthStatus message
        """
        self._groundtruth_status = msg.status

    def receive_state_machine_info(self, msg: StringMsg):
        """Receive info message when state machines change.

        Args:
            msg: New string message
        """
        if msg.data == self._state_machine_msg:
            return
        if self._state_machine_msg is not None:
            self._previous_state_machine_msg = self._state_machine_msg
        self._state_machine_msg = msg.data

    def update_mission_mode(self, mission_mode: int):
        """Change the car's mission mode.

        Args:
            mission_mode: Desired mission mode.
        """
        rospy.loginfo(f"Updating the mission mode to {mission_mode}.")
        msg = MissionMode()
        msg.header.stamp = rospy.Time.now()
        msg.mission_mode = mission_mode
        self.mission_mode_publisher.publish(msg)

    def referee_cb(self, msg: RefereeMsg):
        """Listen for changes in the referee status."""
        rospy.logdebug(f"Referee state: {msg.state}")

        if msg.state == RefereeMsg.FAILED or msg.state == RefereeMsg.COMPLETED:
            # Drive is over: save result and shutdown!

            self.state = msg.state
            self.parking_successes = msg.parking_successes
            self.last_state_machine_transition = (
                self._previous_state_machine_msg,
                self._state_machine_msg,
            )

            # Shutdown node...
            rospy.signal_shutdown("Drive is finished...")
