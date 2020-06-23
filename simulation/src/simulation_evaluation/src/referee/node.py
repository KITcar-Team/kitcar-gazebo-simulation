from typing import Tuple

import rospy
from std_msgs.msg import Bool as BoolMsg

from simulation.utils.ros_base.node_base import NodeBase

from simulation_evaluation.msg import (
    State as StateMsg,
    Broadcast as BroadcastMsg,
    Referee as RefereeMsg,
)

from simulation.src.simulation_evaluation.src.referee.referee import (
    Referee,
    StateMachineConnector,
)


class RefereeNode(NodeBase):
    """Evaluate the drive.

    Keep track of the state machines and the broadcast speaker to calculate a score.
    """

    def __init__(self):
        super().__init__(name="referee_node")
        self.run()

    def start(self):

        self.info_publisher = rospy.Publisher(
            self.param.topics.info, RefereeMsg, queue_size=1
        )

        # State machine subscribers
        self.progress_handler = self._connect_state_machine(
            self.param.topics.state_machine.progress
        )
        self.overtaking_handler = self._connect_state_machine(
            self.param.topics.state_machine.overtaking
        )
        self.parking_handler = self._connect_state_machine(
            self.param.topics.state_machine.parking
        )
        self.priority_handler = self._connect_state_machine(
            self.param.topics.state_machine.priority
        )

        self.referee = Referee(
            progress=self.progress_handler[2],
            overtaking=self.overtaking_handler[2],
            parking=self.parking_handler[2],
            priority=self.priority_handler[2],
        )

        self.broadcast_subscriber = rospy.Subscriber(
            self.param.topics.speaker.broadcast, BroadcastMsg, self.update
        )

        self.reset_subscriber = rospy.Subscriber(
            self.param.topics.reset, BoolMsg, self.reset_cb
        )

        super().start()

    def stop(self):
        self.broadcast_subscriber.unregister()
        self.reset_subscriber.unregister()

        self.info_publisher.unregister()

        def stop_handler(h):
            h[0].unregister()
            h[1].unregister()

        stop_handler(self.progress_handler)
        stop_handler(self.overtaking_handler)
        stop_handler(self.parking_handler)
        stop_handler(self.priority_handler)

        super().stop()

    def _connect_state_machine(
        self, sm_topics
    ) -> Tuple[rospy.Subscriber, rospy.Publisher, StateMachineConnector]:

        set_topic = sm_topics.set
        state_topic = sm_topics.get

        publisher = rospy.Publisher(set_topic, StateMsg, queue_size=1)

        def set(state: int):
            """Change state of state machine."""
            msg = StateMsg()
            msg.state = state
            publisher.publish(msg)
            rospy.loginfo(f"Manually setting {state} in {sm_topics}")

        connector = StateMachineConnector(state=0, set_state=set)

        def update_cb(msg: StateMsg):
            """Listen for changes in state machine."""
            connector.state = msg.state
            rospy.loginfo(f"State changed to {connector.state} in {sm_topics}")

        subscriber = rospy.Subscriber(state_topic, StateMsg, update_cb)

        return subscriber, publisher, connector

    def update(self, broadcast: BroadcastMsg):
        """Update referee output with new broadcast msg."""

        # Update referee with current states
        output = self.referee.update(rospy.Time.now().to_sec(), broadcast.distance,)

        if output:
            rospy.logdebug(output)

        # Create msg
        msg = RefereeMsg()
        msg.state = self.referee.state

        observation = self.referee.observation
        msg.distance = self.referee.observation.distance
        msg.duration = observation.duration
        msg.mistakes = observation.mistakes
        msg.multiplicator = observation.multiplicator
        msg.score = observation.score
        msg.parking_successes = observation.parking_successes

        self.info_publisher.publish(msg)

        rospy.logdebug(
            f"Publishing referee info with state {msg.state} and time {msg.duration}"
        )

    def reset_cb(self, msg: BoolMsg):
        """Reset the referee.

        Args:
            msg: Reset observation's or only the state.
        """
        self.referee.reset(
            initial_observation=self.referee.observation if msg.data else None
        )
