import rospy
from typing import List, NewType

from simulation_groundtruth.srv import (
    SectionSrv,
    SectionSrvRequest,
    SectionSrvResponse,
    LaneSrv,
    LaneSrvRequest,
    LaneSrvResponse,
    ParkingSrv,
    ParkingSrvRequest,
    ParkingSrvResponse,
    LabeledPolygonSrv,
    LabeledPolygonSrvResponse,
    IntersectionSrv,
    IntersectionSrvRequest,
    IntersectionSrvResponse,
)


from ros_base.node_base import NodeBase
from groundtruth.groundtruth import Groundtruth
import generate_road

RoadSection = NewType("RoadSection", int)  # FIXME: RoadSection type


class GroundtruthNode(NodeBase):
    """ROS node providing services to access the road's groundtruth.

    Attributes:
        groundtruth (Groundtruth): Stores road sections \
                and allows to access groundtruth information.
        section_srv (rospy.Service): ROS service to access section ids and types.
        lane_srv (rospy.Service): ROS service to access the lanes of a road section.
        obstacle_srv (rospy.Service): ROS service to access obstacles in a road section.
        parking_srv (rospy.Service): ROS service to access parking spots \
                and other parking information.
        intersection_srv (rospy.Service): ROS service to access information of an intersection.
    """

    def __init__(self):
        super().__init__(name="groundtruth_node")
        self.run()

    def _load_road_sections(self) -> List[RoadSection]:
        """Load road sections from current road."""
        seed = self.param.seed if self.param.seed != "__no_value__" else None

        return generate_road.load_road(self.param.road, seed=seed).sections

    def start(self):
        self.groundtruth = Groundtruth(self._load_road_sections())

        # Advertise section
        self.section_srv = rospy.Service(
            self.param.topics.section, SectionSrv, self.get_sections
        )
        self.lane_srv = rospy.Service(self.param.topics.lane, LaneSrv, self.get_lanes)
        self.parking_srv = rospy.Service(
            self.param.topics.parking, ParkingSrv, self.get_parking
        )
        self.obstacle_srv = rospy.Service(
            self.param.topics.obstacle, LabeledPolygonSrv, self.get_obstacles
        )
        self.intersection_srv = rospy.Service(
            self.param.topics.intersection, IntersectionSrv, self.get_intersection
        )

    def stop(self):
        """Turn off."""
        self.section_srv.shutdown()
        self.lane_srv.shutdown()
        self.parking_srv.shutdown()
        self.obstacle_srv.shutdown()
        self.intersection_srv.shutdown()

    def get_sections(self, request: SectionSrvRequest) -> SectionSrvResponse:
        """Answer section service request."""
        response = SectionSrvResponse()
        response.sections = self.groundtruth.get_section_msgs()
        rospy.logdebug(f"Answering section request {response.sections}")
        return response

    def get_lanes(self, request: LaneSrvRequest) -> LaneSrvResponse:
        """Answer lane service request."""
        response = LaneSrvResponse()
        response.lane_msg = self.groundtruth.get_lane_msg(request.id)
        rospy.logdebug(f"Answering lane request {response.lane_msg}")
        return response

    def get_parking(self, request: ParkingSrvRequest) -> ParkingSrvResponse:
        """Answer parking service request."""
        response = ParkingSrvResponse()
        response.left_msg, response.right_msg = self.groundtruth.get_parking_msg(request.id)
        rospy.logdebug(f"Answering parking request {response.left_msg, response.right_msg}")
        return response

    def get_obstacles(self, request: LabeledPolygonSrv) -> LabeledPolygonSrvResponse:
        """Answer obstacle service request."""
        response = LabeledPolygonSrvResponse()
        response.polygons = self.groundtruth.get_obstacle_msgs(request.id)
        rospy.logdebug(f"Answering obstacle request {response.polygons}")
        return response

    def get_intersection(self, request: IntersectionSrvRequest) -> IntersectionSrvResponse:
        """Answer intersection service request."""
        response = IntersectionSrvResponse()
        (
            response.turn,
            response.rule,
            response.south,
            response.west,
            response.east,
            response.north,
        ) = self.groundtruth.get_intersection_msg(request.id)
        rospy.logdebug(f"Answering intersection request {response}")

        return response
