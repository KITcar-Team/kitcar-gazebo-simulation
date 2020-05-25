"""ROS node that connects multiple speakers to ROS topics."""

import rospy

from simulation.utils.ros_base.node_base import NodeBase
from simulation.src.simulation_evaluation.src.speaker.speakers import (
    AreaSpeaker,
    ZoneSpeaker,
    EventSpeaker,
    SpeedSpeaker,
    BroadcastSpeaker,
)

# Messages
from gazebo_simulation.msg import CarState as CarStateMsg
from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_evaluation.msg import Broadcast as BroadcastMsg
from simulation_groundtruth.srv import (
    SectionSrv,
    LaneSrv,
    ParkingSrv,
    LabeledPolygonSrv,
    IntersectionSrv,
)

from simulation_groundtruth.msg import GroundtruthStatus


class SpeakerNode(NodeBase):
    """ROS node that integrates all speakers into the ROS framework.

    Attributes:
        speakers (List[Tuple[Speaker, rospy.Message, rospy.Publisher]]): All speakers \
                with their message type and a publisher that publishes messages created \
                by the speaker.
        subscriber (rospy.Subscriber): Receive CarState messages with position and speed \
                of the car.
        section_proxy, lane_proxy, parking_proxy, obstacle_proxy, intersection_proxy \
                (rospy.ServiceProxy): Proxies that the speakers need to request \
                groundtruth information.
    """

    def __init__(self):
        """Initialize the node."""
        super().__init__(name="speaker_node")

        # Msg names
        self.msg_names = {
            val: name
            for name, val in SpeakerMsg.__dict__.items()
            if isinstance(val, int) and not name[0] == "_"
        }

        self.run(function=self.publish_speakers, rate=30)

    def start(self):
        rospy.logdebug("STARTING SPEAKER")
        groundtruth_topics = self.param.topics.groundtruth

        # Create groundtruth service proxies
        self.section_proxy = rospy.ServiceProxy(groundtruth_topics.section, SectionSrv)
        self.lane_proxy = rospy.ServiceProxy(groundtruth_topics.lane, LaneSrv)
        self.parking_proxy = rospy.ServiceProxy(groundtruth_topics.parking, ParkingSrv)
        self.obstacle_proxy = rospy.ServiceProxy(
            groundtruth_topics.obstacle, LabeledPolygonSrv
        )
        self.intersection_proxy = rospy.ServiceProxy(
            groundtruth_topics.intersection, IntersectionSrv
        )

        rospy.wait_for_service(groundtruth_topics.section)

        self.groundtruth_status_subscriber = rospy.Subscriber(
            self.param.topics.groundtruth.status,
            GroundtruthStatus,
            callback=self.receive_groundtruth_update,
        )

        # Add all speakers
        self.speakers = []

        def append_speaker(speaker, topic, msg_type=SpeakerMsg):
            self.speakers.append(
                (
                    speaker,
                    rospy.Publisher(topic, msg_type, queue_size=self.param.queue_size),
                )
            )

        append_speaker(
            AreaSpeaker(
                section_proxy=self.section_proxy,
                lane_proxy=self.lane_proxy,
                parking_proxy=self.parking_proxy,
                min_wheel_count=self.param.area.min_wheels,
                area_buffer=self.param.area.buffer,
            ),
            self.param.topics.area,
        )

        append_speaker(
            ZoneSpeaker(
                section_proxy=self.section_proxy,
                lane_proxy=self.lane_proxy,
                parking_proxy=self.parking_proxy,
                obstacle_proxy=self.obstacle_proxy,
                intersection_proxy=self.intersection_proxy,
                overtaking_buffer=self.param.zone.overtaking_buffer,
                start_zone_buffer=self.param.zone.start_buffer,
                end_zone_buffer=self.param.zone.end_buffer,
                yield_distance=(
                    self.param.zone.yield_distance[0],
                    self.param.zone.yield_distance[1],
                ),
            ),
            self.param.topics.zone,
        )
        append_speaker(
            EventSpeaker(
                section_proxy=self.section_proxy,
                lane_proxy=self.lane_proxy,
                obstacle_proxy=self.obstacle_proxy,
                parking_proxy=self.parking_proxy,
                parking_spot_buffer=self.param.event.parking_spot_buffer,
                min_parking_wheels=self.param.event.min_parking_wheels,
            ),
            self.param.topics.event,
        )

        append_speaker(
            SpeedSpeaker(
                stop_threshold=self.param.speed.stop_threshold,
                halt_time=self.param.speed.halt_time,
                stop_time=self.param.speed.stop_time,
            ),
            self.param.topics.speed,
        )

        append_speaker(
            BroadcastSpeaker(section_proxy=self.section_proxy, lane_proxy=self.lane_proxy,),
            self.param.topics.broadcast,
            msg_type=BroadcastMsg,
        )

        # Subscribe to carstate
        self.subscriber = rospy.Subscriber(
            self.param.topics.car_state.car_state, CarStateMsg, self.carstate_cb
        )
        rospy.logdebug("STARTED SPEAKER")
        rospy.wait_for_message(self.param.topics.car_state.car_state, CarStateMsg)

    def stop(self):
        self.subscriber.unregister()

        # Stop all speaker publisher
        for _, publisher in self.speakers:
            publisher.unregister()

    def carstate_cb(self, msg: CarStateMsg):
        """Receive CarState message and pass it to all speakers.

        Args:
            msg: New CarState message

        """
        # Pipe car state msg to every speaker handler
        for speaker, _ in self.speakers:
            speaker.listen(msg)
        rospy.logdebug("RECEIVED CARSTATE")

    def receive_groundtruth_update(self, msg: GroundtruthStatus):
        """Receive GroundtruthStatus message.

        Args:
            msg: New GroundtruthStatus message
        """
        self.param.active = msg.status == GroundtruthStatus.READY

    def publish_speakers(self):
        """Publish the output of all speakers."""
        # Publish in all speaker handlers
        for speaker, publisher in self.speakers:
            msgs = speaker.speak()

            for msg in msgs:
                rospy.logdebug(f"PUBLISHING {msg} on speaker {speaker.__class__.__name__}")
                if type(msg) == SpeakerMsg:
                    msg.name = self.msg_names[msg.type]
                publisher.publish(msg)
