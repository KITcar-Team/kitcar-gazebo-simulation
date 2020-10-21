"""Adjusts Gazebos update rate to guarantee sensor updates with desired frequency."""

import contextlib

import rospy
import rostopic
from gazebo_msgs.srv import (
    GetPhysicsProperties,
    GetPhysicsPropertiesRequest,
    SetPhysicsProperties,
    SetPhysicsPropertiesRequest,
)

from simulation.utils.ros_base.node_base import NodeBase


class GazeboRateControlNode(NodeBase):
    """Control gazebos update rate to meet desired sensor update rates.

    Attributes:
        relevant_targets (List[Dict[str, Any]]): Topics that should be monitored
            and update rate that the topic should have.
        _last_target_update (float): Last time the targets were updated.
        _last_target_frequencies (List[float]): Last known update rate of each target.
    """

    def __init__(self):
        """Initialize the node."""

        super().__init__(name="gazebo_rate_control_node")

        self._last_target_frequencies = {}
        self._last_target_update = 0
        self.relevant_targets = []

        super().run(function=self.update, rate=self.param.update_rate.control.rate)

    def start(self):

        rospy.wait_for_service(self.param.topics.get_physics, timeout=5)
        self.set_physics = rospy.ServiceProxy(
            self.param.topics.set_physics, SetPhysicsProperties, persistent=True
        )
        self.get_physics = rospy.ServiceProxy(
            self.param.topics.get_physics, GetPhysicsProperties, persistent=True
        )

        self.rater = rostopic.ROSTopicHz(5)

        # Start in very slow mode to ensure that everything is started before speeding up
        self._update_properties(update_rate=self.param.update_rate.min)

        self.subscribers = {}
        for target in self.param.targets:
            topic = target["topic"]
            self.subscribers[topic] = rospy.Subscriber(
                topic,
                rospy.AnyMsg,
                self.rater.callback_hz,
                callback_args=topic,
            )
            # Wait for atleast one message on every target topic
            # Continue of none is received. The topic will then be disregarded (for now).
            with contextlib.suppress(rospy.ROSException):
                rospy.wait_for_message(topic, rospy.AnyMsg, timeout=0.1)

        super().start()

    def stop(self):
        for sub in self.subscribers.values():
            sub.unregister()
        self.subscribers.clear()

        self.set_physics.close()
        self.get_physics.close()
        super().stop()

    def _calculate_update_rate(
        self,
        update_rate: float,
        frequency: float,
        desired_frequency: float,
    ) -> float:
        """Calculate new update rate.

        Args:
            update_rate: Gazebo's current maximum update_rate
            frequency: Current frequency
            desired_frequency: Optimal frequency
        Return:
            New maximum update rate.
        """
        if frequency == 0:
            return

        # Calculate new update rate
        if frequency < desired_frequency:
            return max(
                update_rate
                - (update_rate - self.param.update_rate.min)
                * self.param.update_rate.control.down
                * desired_frequency
                / frequency,
                self.param.update_rate.min,
            )
        elif frequency > desired_frequency:
            return min(
                update_rate
                + (self.param.update_rate.max - update_rate)
                * self.param.update_rate.control.up
                * desired_frequency
                / frequency,
                self.param.update_rate.max,
            )

    def _update_properties(self, update_rate):
        current_properties = self.get_physics(GetPhysicsPropertiesRequest())

        new_properties = SetPhysicsPropertiesRequest()
        new_properties.gravity = current_properties.gravity
        new_properties.ode_config = current_properties.ode_config
        new_properties.time_step = current_properties.time_step
        new_properties.max_update_rate = update_rate

        self.set_physics(new_properties)

    def update(self):
        """Adjust Gazebos update rate to meet desired output frequency of the target
        topic."""
        old_update_rate = self.get_physics(GetPhysicsPropertiesRequest()).max_update_rate

        # Update target topics every second
        if rospy.Time.now().to_sec() - self._last_target_update > 1:
            # Find available topics
            all_topics = set(t[0] for t in rospy.get_published_topics())
            rospy.logdebug(all_topics)
            self.relevant_targets = [
                target for target in self.param.targets if target["topic"] in all_topics
            ]
            self._last_target_update = rospy.Time.now().to_sec()

        # Calculate new update rate considering all targets
        update_rates = []
        for target in self.relevant_targets:
            topic = target["topic"]
            desired_frequency = target["desired"]

            frequency = self.rater.get_hz(topic)
            frequency = (
                frequency[0] if frequency else self._last_target_frequencies.get(topic, 0)
            )
            self._last_target_frequencies[topic] = frequency

            rate = self._calculate_update_rate(
                old_update_rate,
                frequency=frequency,
                desired_frequency=desired_frequency,
            )
            if rate is not None:
                rospy.logdebug(
                    f"Frequency of topic {topic}: {frequency}"
                    f" (supposed: {desired_frequency})"
                )
                update_rates.append(rate)

        if len(update_rates) > 0:
            new_update_rate = min(update_rates)
        else:
            return

        rospy.logdebug(
            ("Lowering" if new_update_rate < old_update_rate else "Increasing")
            + f" Gazebo's update rate: {old_update_rate}"
            f" -> {new_update_rate}"
        )

        self._update_properties(new_update_rate)
