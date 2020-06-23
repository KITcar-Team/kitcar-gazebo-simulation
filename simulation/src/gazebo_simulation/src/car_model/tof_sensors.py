from dataclasses import dataclass

from simulation.utils.urdf import DepthCameraProperties, Origin
from simulation.utils.geometry import Vector


def _get_depth_properties(name: str) -> DepthCameraProperties:
    NS = "/simulation/sensors/"
    return DepthCameraProperties(
        horizontal_fov=0.0349,
        update_rate=20,
        image_width=2,
        image_height=2,
        image_format="L8",
        clip_near=0.01,
        clip_far=2,
        point_cloud_cutoff=0.005,
        image_topic=NS + f"raw/distance_{name}",
        info_topic=NS + f"info/distance_{name}",
        frame_name=f"ir_{name}",
        point_cloud_topic_name=NS + f"raw/distance_{name}_points",
    )


@dataclass
class TofSensor:
    name: str
    origin: Origin
    properties: DepthCameraProperties
    size: Vector = Vector(0.02, 0.02, 0.02)
    mass: float = 1e-5

    @classmethod
    def with_default_properties(cls, name: str, origin: Origin) -> "TofSensor":
        return cls(name, origin, properties=_get_depth_properties(name))

    @property
    def full_name(self) -> str:
        return f"ir_{self.name}"
