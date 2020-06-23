#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track the state of a simulated drive."""

from typing import Callable

import rospy
from simulation.utils.ros_base.node_base import NodeBase

# Messages
from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_evaluation.msg import State as StateMsg
from simulation.src.simulation_evaluation.src.state_machine.state_machines.overtaking import (
    OvertakingStateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.state_machines.parking import (
    ParkingStateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.state_machines.priority import (
    PriorityStateMachine,
)

# StateMachines
from simulation.src.simulation_evaluation.src.state_machine.state_machines.progress import (
    ProgressStateMachine,
)
from simulation.src.simulation_evaluation.src.state_machine.state_machines.state_machine import (
    StateMachine,
)

__copyright__ = "KITcar"


def log(logger: Callable[[str], None] = rospy.loginfo):
    """Log the state of each state machine.

    Arguments:
        logger: Function to which the log message should be sent
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            logmsg = "\n### CURRENT STATES ###\n"
            for sm in self.state_machines:
                logmsg += f"{sm.__class__.__name__}: {sm.info()}\n"

            logger(logmsg)

            return result

        return wrapper

    return decorator


class StateMachineNode(NodeBase):
    """ROS node which tracks the state of a simulated drive.

    Attributes:
        zone_subscriber (rospy.Subscriber): Subscribes to zone of speaker
        location_subscriber (rospy.Subscriber): Subscribes to location of speaker
        events_subscriber (rospy.Subscriber): Subscribes to events of speaker
        speed_subscriber (rospy.Subscriber): Subscribes to speed of speaker
        set_subscribers (List[rospy.Subscriber]): Subscribes to .../set
        sm_publishers (List[rospy.Publisher]): Publishes to .../state
        state_machines (List[StateMachine]): Array of all StateMachine
    """

    def __init__(self):
        """Initialize the node."""
        super(StateMachineNode, self).__init__(name="state_machine")
        self.run()

    def start(self):
        """Start node."""
        self.initalize_state_machines()

        # Subscribe to speaker
        speaker_topics = self.param.topics.speaker
        self.zone_subscriber = rospy.Subscriber(
            speaker_topics.zone, SpeakerMsg, self.on_msg
        )
        self.location_subscriber = rospy.Subscriber(
            speaker_topics.area, SpeakerMsg, self.on_msg
        )
        self.events_subscriber = rospy.Subscriber(
            speaker_topics.event, SpeakerMsg, self.on_msg
        )
        self.speed_subscriber = rospy.Subscriber(
            speaker_topics.speed, SpeakerMsg, self.on_msg
        )

        super().start()

    def initalize_state_machines(self):
        """Init each state machine.

        Creates a list with each StateMachine in .state_machines, creates a publisher for each get topic in \
            .sm_publishers and creates a subscriper for each set topic in .set_publishers.
        """
        t = self.param.topics
        definitions = []
        # Add new StateMachine here
        # Usage: (StateMachineObject, Topic path for publisher, Topic path for manually setting the state machine)
        definitions.append((ProgressStateMachine, t.progress.get, t.progress.set))
        definitions.append((OvertakingStateMachine, t.overtaking.get, t.overtaking.set))
        definitions.append((ParkingStateMachine, t.parking.get, t.parking.set))
        definitions.append((PriorityStateMachine, t.priority.get, t.priority.set))

        state_machine_classes, topics, topics_set = zip(*definitions)

        self.state_machines = [
            StateMachineClass(callback=self.on_state_machine_update)
            for StateMachineClass in state_machine_classes
        ]
        self.sm_publishers = [
            rospy.Publisher(topic, StateMsg, queue_size=10) for topic in topics
        ]
        self.set_subscribers = [
            rospy.Subscriber(
                topic, StateMsg, self.set_state_machine, (self.state_machines[i])
            )
            for i, topic in enumerate(topics_set)
        ]

    def stop(self):
        """Turn off node."""
        for publisher in self.sm_publishers:
            publisher.unregister()

        for subscriber in self.set_subscribers:
            subscriber.unregister()

        self.zone_subscriber.unregister()
        self.location_subscriber.unregister()
        self.events_subscriber.unregister()
        self.speed_subscriber.unregister()

        super().stop()

    @log(rospy.loginfo)
    def on_state_machine_update(self):
        """Update each state machine."""
        for i, publisher in enumerate(self.sm_publishers):
            publisher.publish(self.state_machines[i].msg())

    @log(rospy.logdebug)
    def on_msg(self, msg: SpeakerMsg):
        """Call every state machine.

        Arguments.
            msg: Message parsed to every state machine
        """
        for sm in self.state_machines:
            sm.run(msg.type)

    def set_state_machine(self, new_msg: StateMsg, state_machine: StateMachine):
        """Update state machine manually.

        Arguments:
            new_msg: Message of the state to which the state machine should be set to
            state_machine (StateMachine): State machine to be changed
        """
        if not state_machine.set(new_msg):
            rospy.logwarn(f"Can't find {new_msg} in {state_machine.__class__.__name__}.")
