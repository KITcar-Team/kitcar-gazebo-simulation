"""Class definitions of tags used to define cameras."""
from dataclasses import dataclass
from simulation.utils.geometry import Vector

from .core import (
    Attribute,
    XmlObject,
    Origin,
)

from .link import (
    Box,
    Material,
    Visual,
    Collision,
    Link,
    Geometry,
)

from .gazebo import (
    Plugin,
    Gazebo,
    Sensor,
)

from .joint import (
    Parent,
    Child,
    Joint,
)


@dataclass
class CameraProperties:
    horizontal_fov: float
    update_rate: float
    image_width: float
    image_height: float
    image_format: float
    clip_near: float
    clip_far: float
    image_topic: str
    info_topic: str
    frame_name: str
    optical_center_x: float = None
    optical_center_y: float = None


@dataclass
class DepthCameraProperties(CameraProperties):
    point_cloud_cutoff: float = None
    point_cloud_topic_name: str = None


@dataclass
class CameraPlugin(Plugin):

    alwaysOn: float = 1
    updateRate: float = 0
    frameName: str = None
    cameraName: str = None

    CxPrime: float = None
    Cx: float = None
    Cy: float = None


@dataclass
class MonoCameraPlugin(CameraPlugin):

    name: Attribute = "camera_plugin"
    filename: Attribute = "libgazebo_ros_camera.so"

    imageTopicName: str = None
    cameraInfoTopicName: str = None


@dataclass
class DepthCameraPlugin(CameraPlugin):

    name: Attribute = "camera_plugin"
    filename: Attribute = "libgazebo_ros_openni_kinect.so"

    depthImageTopicName: str = None
    depthImageCameraInfoTopicName: str = None
    pointCloudTopicName: str = None
    pointCloudCutoff: float = None


@dataclass
class CameraImage(XmlObject):
    TAG = "image"

    width: int
    height: int
    format: str


@dataclass
class CameraClip(XmlObject):
    TAG = "clip"

    near: float
    far: float


@dataclass
class CameraDefinition(XmlObject):
    TAG = "camera"

    horizontal_fov: float
    image: CameraImage
    clip: CameraClip = None


@dataclass
class CameraSensor(Sensor):
    name: Attribute
    type: Attribute
    plugin: CameraPlugin
    update_rate: float
    camera: CameraDefinition


@dataclass
class _Camera(XmlObject):
    link: Link
    joint: Joint
    plugin: Gazebo
    sensor: CameraSensor

    def __init__(
        self, name: str, origin: Origin, size: Vector, mass: float, chassis_link: Link,
    ):

        box = Geometry(Box(size))
        material = Material("mat", color=Vector(1, 1, 1))
        self.link = Link(
            name + "_link",
            collision=Collision(geometry=box),
            visual=Visual(geometry=box, material=material),
        )
        self.link.use_inertial_from_collision(mass)

        self.joint = Joint(
            name=name + "_joint",
            parent=Parent(chassis_link.name),
            child=Child(self.link.name),
            origin=origin,
        )


@dataclass
class MonoCamera(_Camera):
    def __init__(
        self,
        name: str,
        origin: Origin,
        size: Vector,
        mass: float,
        properties: CameraProperties,
        chassis_link: Link,
    ):
        super().__init__(name, origin, size, mass, chassis_link)

        self.sensor = Gazebo(
            CameraSensor(
                name=name,
                type="camera",
                update_rate=properties.update_rate,
                plugin=MonoCameraPlugin(
                    imageTopicName=properties.image_topic,
                    cameraInfoTopicName=properties.info_topic,
                    frameName=properties.frame_name,
                    cameraName=name,
                    # CxPrime=properties.optical_center_x,
                    # Cy=properties.optical_center_y,
                ),
                camera=CameraDefinition(
                    horizontal_fov=properties.horizontal_fov,
                    image=CameraImage(
                        properties.image_width,
                        properties.image_height,
                        properties.image_format,
                    ),
                    clip=CameraClip(properties.clip_near, properties.clip_far),
                ),
            ),
            reference=self.link.name,
        )


@dataclass
class DepthCamera(_Camera):
    def __init__(
        self,
        name: str,
        origin: Origin,
        size: Vector,
        mass: float,
        properties: DepthCameraProperties,
        chassis_link: Link,
    ):
        super().__init__(name, origin, size, mass, chassis_link)

        self.sensor = Gazebo(
            CameraSensor(
                name=name,
                type="depth",
                update_rate=properties.update_rate,
                plugin=DepthCameraPlugin(
                    depthImageTopicName=properties.image_topic,
                    depthImageCameraInfoTopicName=properties.info_topic,
                    pointCloudTopicName=properties.point_cloud_topic_name,
                    pointCloudCutoff=properties.point_cloud_cutoff,
                    frameName=properties.frame_name,
                    cameraName=name,
                ),
                camera=CameraDefinition(
                    horizontal_fov=properties.horizontal_fov,
                    image=CameraImage(
                        properties.image_width,
                        properties.image_height,
                        properties.image_format,
                    ),
                    clip=CameraClip(properties.clip_near, properties.clip_far),
                ),
            ),
            reference=self.link.name,
        )
