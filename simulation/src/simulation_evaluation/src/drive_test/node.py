import rospy

from simulation_brain_link.msg import MissionMode
from simulation_evaluation.msg import Referee as RefereeMsg
from simulation_groundtruth.msg import GroundtruthStatus

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

        # For some reason this is necessary for the mission mode publisher to work correctly!
        rospy.sleep(0.1)

        self.referee_subscriber = rospy.Subscriber(
            self.param.topics.referee.info,
            RefereeMsg,
            callback=self.referee_cb,
            queue_size=1,
        )  # Subscribe to referee

        rospy.wait_for_message(self.param.topics.referee.info, RefereeMsg)

        # Goooo
        self.update_mission_mode(self.param.mission_mode)

    def stop(self):
        self.referee_subscriber.unregister()
        self.groundtruth_status_subscriber.unregister()
        self.mission_mode_publisher.unregister()
        self.renderer_reload_publisher.unregister()

    def receive_groundtruth_update(self, msg: GroundtruthStatus):
        """Receive GroundtruthStatus message.

        Args:
            msg: New GroundtruthStatus message
        """
        self._groundtruth_status = msg.status

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

            # Shutdown node...
            rospy.signal_shutdown("Drive is finished...")
