import rospy
from std_srvs.srv import Empty as EmptySrv
from std_msgs.msg import Empty as EmptyMsg, String as StringMsg
from gazebo_msgs.msg import ModelStates

from gazebo_simulation.msg import CarState as CarStateMsg
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

from simulation.utils.geometry import Vector
from simulation.utils.ros_base.node_base import NodeBase

import simulation.utils.road.road as road_module

from simulation.src.simulation_groundtruth.src.groundtruth.groundtruth import Groundtruth
from simulation.src.simulation_groundtruth.src.groundtruth.renderer import Renderer
from simulation.src.simulation_groundtruth.src.groundtruth.object_controller import (
    ObjectController,
)

from simulation_groundtruth.msg import GroundtruthStatus


class GroundtruthNode(NodeBase):
    """ROS node providing services to access the road's groundtruth.

    Attributes:
        groundtruth (Groundtruth): Stores road sections \
                and allows to access groundtruth information.
        renderer (Renderer): Render and update the road in Gazebo.
        object_controller (ObjectController): Create and track objects (e.g. obstacles, signs) \
                in Gazebo.
        section_srv (rospy.Service): ROS service to access section ids and types.
        lane_srv (rospy.Service): ROS service to access the lanes of a road section.
        obstacle_srv (rospy.Service): ROS service to access obstacles in a road section.
        parking_srv (rospy.Service): ROS service to access parking spots \
                and other parking information.
        intersection_srv (rospy.Service): ROS service to access information of an intersection.
    """

    def __init__(self, name="groundtruth_node", log_level=rospy.INFO):
        super().__init__(name=name, log_level=log_level)

        # Initialize without a road
        # The road is loaded in self.update()
        road = road_module.Road()
        self.groundtruth = Groundtruth(road=road)
        self.renderer = Renderer(
            road,
            remove_model=self._remove_model,
            spawn_model=self._spawn_model,
            pause_gazebo=self._pause_gazebo,
            unpause_gazebo=self._unpause_gazebo,
            info_callback=self.update_groundtruth_status,
            tile_size=Vector(self.param.tile_size),
            tile_resolution=Vector(self.param.tile_resolution),
        )
        self.object_controller = ObjectController(
            road, remove_model=self._remove_model, spawn_model=self._spawn_model,
        )

        self.model_names = []
        self.car_state = CarStateMsg()
        self._groundtruth_status = GroundtruthStatus()

        self.run()

    def _load_road(self) -> road_module.Road:
        """Load road sections from current road."""
        seed = (
            self.param.seed
            if self.param.seed != "__no_value__"
            else self.param.default_seed
        )

        return road_module.load(self.param.road, seed=seed)

    def _ensure_gazebo_startup(self):
        """Make sure that gazebo is correctly launched.

        Waiting for Gazebo is not as easy as it should be.
        **This function, however, is a workaround.**
        It attempts to spawn a model until it is there. Once the model has been spawned,
        it is directly removed again. At this point, Gazebo has been completely launched.

        Beware: The model has no collision box and is not visible.
        It is therefore not problematic if it is not correctly removed!
        """
        model_base_name = "__start_up_box__"
        rospy.wait_for_message(self.param.topics.gazebo_models, ModelStates)
        i = 0
        while True:
            # Get all models currently listed in Gazebo's models
            # that contain the model_base_name in their name.
            boxes = list(b for b in self.model_names if model_base_name in b)

            if len(boxes) != 0:
                for b in boxes:
                    self._remove_model(b)
                return
            i += 1
            self._spawn_model(f"""<model name="{model_base_name}{i}"></model>""")
            rospy.sleep(0.1)
            rospy.wait_for_message(self.param.topics.gazebo_models, ModelStates)

    def start(self):

        # Gazebo
        self.spawn_publisher = rospy.Publisher(
            self.param.topics.spawn_model, StringMsg, queue_size=1000
        )

        self.remove_publisher = rospy.Publisher(
            self.param.topics.remove_model, StringMsg, queue_size=1000
        )

        self.groundtruth_status_publisher = rospy.Publisher(
            self.param.topics.status, GroundtruthStatus, queue_size=100
        )

        self.pause_physics_proxy = rospy.ServiceProxy(
            self.param.topics.pause_gazebo, EmptySrv
        )
        self.unpause_physics_proxy = rospy.ServiceProxy(
            self.param.topics.unpause_gazebo, EmptySrv
        )

        # Updates
        self.gazebo_subscriber = rospy.Subscriber(
            self.param.topics.gazebo_models,
            ModelStates,
            callback=self.receive_model_states,
        )

        self.car_state_subscriber = rospy.Subscriber(
            self.param.topics.car_state.car_state,
            CarStateMsg,
            callback=self.receive_car_state,
        )

        # Requests
        self.interrupt_subscriber = rospy.Subscriber(
            self.param.topics.renderer.interrupt,
            EmptyMsg,
            callback=self.receive_interrupt_request,
        )
        self.reload_subscriber = rospy.Subscriber(
            self.param.topics.renderer.reload,
            EmptyMsg,
            callback=self.receive_reload_request,
        )

        self.update()

        # Groundtruth services
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

        self.pause_physics_proxy.close()
        self.unpause_physics_proxy.close()

        self.interrupt_subscriber.unregister()
        self.reload_subscriber.unregister()
        self.gazebo_subscriber.unregister()
        self.car_state_subscriber.unregister()

        self.spawn_publisher.unregister()
        self.remove_publisher.unregister()
        self.renderer_info_publisher.unregister()
        self.groundtruth_status_publisher.unregister()

    def update(self):
        """Update the groundtruth including renderer and object controller."""

        self.update_groundtruth_status(status=GroundtruthStatus.LOAD_NEW_ROAD)

        road = self._load_road()

        if self.param.start_groundtruth:
            self.groundtruth.road = road
        if self.param.start_renderer or self.param.start_object_controller:
            self._ensure_gazebo_startup()
        if self.param.start_renderer:
            self.renderer.road = road
            self.renderer.load(self.model_names)
        if self.param.start_object_controller:
            self.object_controller.road = road
            self.object_controller.load(self.model_names)

        self.update_groundtruth_status(status=GroundtruthStatus.READY)

    """Groundtruth Services."""

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

    # Receive updates
    def receive_model_states(self, model_states: ModelStates):
        self.model_names = model_states.name

    def receive_car_state(self, car_state: CarStateMsg):
        self.car_state = car_state

    # Gazebo interactions
    def _spawn_model(self, model_xml: str):
        """Spawn a model in Gazebo.

        Args:
            model_xml: XML String that defines the model. (Without sdf declaration!)
        """
        self.spawn_publisher.publish(f'<sdf version="1.6">{model_xml}</sdf>')

    def _remove_model(self, name):
        """Remove a model from Gazebo.

        Args:
            name: Name of the model in Gazebo.
        """
        self.remove_publisher.publish(name)

    def _pause_gazebo(self):
        """Pause Gazebo."""
        self.pause_physics_proxy()

    def _unpause_gazebo(self):
        """Request to start Gazebo."""
        self.unpause_physics_proxy()

    """Handle requests."""

    def receive_reload_request(self, _: EmptyMsg):
        """Receive requests to reload the world."""
        self.update()

    def receive_interrupt_request(self, _: EmptyMsg):
        """Receive requests to stop rendering processes."""
        self.renderer.interrupt()

    """Publish groundtruth update."""

    def update_groundtruth_status(
        self, status=None, processed_tiles=None, number_of_tiles=None
    ):
        self._groundtruth_status.road = self.param.road
        self._groundtruth_status.seed = self.param.seed

        if status is not None:
            self._groundtruth_status.status = status
        if processed_tiles is not None:
            self._groundtruth_status.processed_tiles = processed_tiles
        if number_of_tiles is not None:
            self._groundtruth_status.number_of_tiles = number_of_tiles

        self.groundtruth_status_publisher.publish(self._groundtruth_status)
