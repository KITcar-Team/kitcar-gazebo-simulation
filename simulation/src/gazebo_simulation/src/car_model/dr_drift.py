from dataclasses import dataclass
from typing import List

import yaml

from simulation.utils.geometry import Vector
from simulation.utils.urdf import (
    Attribute,
    Box,
    CameraProperties,
    Collision,
    DepthCamera,
    Gazebo,
    Geometry,
    Link,
    Material,
    Mesh,
    MonoCamera,
    Origin,
    Plugin,
    Visual,
    XmlObject,
)

from .camera_specs import CameraSpecs
from .car_specs import CarSpecs
from .tof_sensors import TofSensor


@dataclass
class DrDrift(XmlObject):
    """Representation of the Dr. Drift vehicle.

    This class can be used to create a gazebo model of our Dr. Drift car.
    Within the initializer all pieces that make up the simulated model of Dr. Drift
    are created.

    Information about the sensors' specifications and positions are taken from
    the **car_specs** ROS package located in **KITcar_brain**.
    """

    TAG = "robot"

    name: Attribute
    version: Attribute

    def __init__(
        self,
        specs: CarSpecs,
        camera_specs: CameraSpecs,
        tof_sensors: List[TofSensor],
        name: str = "dr_drift",
        version="1.0.0",
    ):
        self.name = name
        self.version = version

        """Chassis."""

        # Box between box axis of the vehicle.
        chassis_box = Box(
            size=Vector(specs.wheelbase, specs.vehicle_width, specs.body_height)
        )

        # By default the origin of a geometry is in its center.
        # This sets the transformation to the origin of the vehicle which means
        # that the resulting origin is on the rear axis!
        chassis_origin = Origin([specs.wheelbase / 2, 0, specs.cog_height])

        # The visual origin is slightly adjusted by hand because the model
        # is a little bit off.
        visual_origin = Origin(
            xyz=[chassis_origin.xyz[0] - 0.025, 0, chassis_origin.xyz[2] + 0.01]
        )
        chassis_visual = Visual(
            origin=visual_origin,
            geometry=Geometry(Mesh(filename="file://meshes/dr_drift.stl")),
            material=Material("mat", color=Vector(1, 1, 1)),
        )
        chassis_collision = Collision(origin=chassis_origin, geometry=Geometry(chassis_box))

        self.chassis_link = Link(
            "chassis", collision=chassis_collision, visual=chassis_visual
        )
        self.chassis_link.use_inertial_from_collision(specs.mass)

        # Create cameras
        camera_origin = Origin(
            xyz=camera_specs.xyz.to_numpy(), rpy=camera_specs.rpy.to_numpy()
        )
        self.front_camera = MonoCamera(
            name="front_camera",
            origin=camera_origin,
            mass=0.001,
            size=Vector(0.05, 0.04, 0.04),
            properties=CameraProperties(
                camera_specs.horizontal_fov,
                camera_specs.update_rate,
                camera_specs.capture_size["width"],
                camera_specs.capture_size["height"],
                camera_specs.capture_format,
                optical_center_x=camera_specs.optical_center_x,
                optical_center_y=camera_specs.optical_center_y,
                clip_near=camera_specs.clip["near"],
                clip_far=camera_specs.clip["far"],
                image_topic=camera_specs.image_topic,
                info_topic=camera_specs.info_topic,
                frame_name="front_camera",
            ),
            chassis_link=self.chassis_link,
        )

        # Time of flight sensors are implemented using depth cameras
        for tof in tof_sensors:
            depth_cam = DepthCamera(
                tof.full_name + "_sensor",
                origin=tof.origin,
                size=tof.size,
                properties=tof.properties,
                chassis_link=self.chassis_link,
                mass=tof.mass,
            )
            setattr(self, tof.full_name, depth_cam)

        self.model_plugin = Gazebo(Plugin("model_plugin_link", "libmodel_plugin_link.so"))


def load_dr_drift(
    car_specs_path: str,
    camera_specs_path: str,
    coordinate_systems_path: str,
    save_car_specs: str = None,
    save_camera_calibration: str = None,
    save_static_coordinates: str = None,
) -> DrDrift:
    """Create a DrDrift instance with the information located in the provided files.

    Args:
        car_specs_path: Path to a yaml file that can be converted into a CarSpecs instance.
        camera_specs_path: Path to a yaml file that can be converted
                           into a CameraSpecs instance.
        coordinate_systems_path: Path to a file defining the positions
                                 time of flight sensors.
        save_car_specs: Path to where the resulting car_specs
                                 should be saved.
        save_camera_calibration: Path to where the resulting camera calibration
                                 should be saved.
        save_static_coordinates: Path to where the used static coordinate systems
                                 should be saved.
    """

    car_specs = CarSpecs.from_file(car_specs_path)
    camera_specs = CameraSpecs.from_file(camera_specs_path)

    if save_car_specs is not None:
        car_specs.save(save_car_specs)

    if save_camera_calibration is not None:
        camera_specs.save(save_camera_calibration)

    # Load time of flight sensors positions from coordinate systems definition
    with open(
        coordinate_systems_path,
        "r",
    ) as file:
        co_system_dict = yaml.load(file, Loader=yaml.SafeLoader)

    if save_static_coordinates is not None:
        with open(
            save_static_coordinates,
            "w",
        ) as file:
            file.write(yaml.dump(co_system_dict, Dumper=yaml.SafeDumper))

    tof_origins = {
        key: Origin(
            xyz=[val["x"], val["y"], val["z"]],
            rpy=[val["roll"], val["pitch"], val["yaw"]],
        )
        for key, val in co_system_dict.items()
    }

    def tof_from_name(name) -> TofSensor:
        return TofSensor.with_default_properties(
            name=name, origin=tof_origins[f"vehicle_to_ir_{name}"]
        )

    tof_sensors = [tof_from_name(name) for name in ("ahead", "front", "middle", "back")]

    return DrDrift(specs=car_specs, camera_specs=camera_specs, tof_sensors=tof_sensors)


if __name__ == "__main__":
    dr_drift = load_dr_drift(
        save_camera_calibration="../../param/car_specs/dr_drift/camera.yaml",
        save_car_specs="../../param/car_specs/dr_drift/car_specs.yaml",
    )
    dr_drift.save("../../param/car_specs/dr_drift/model.urdf")
