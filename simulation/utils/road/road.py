"""Road class used to define roads as a python class.

A simulated road can be defined through an object of type Road. It contains all sections of
that road as a list.
"""

import importlib
import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from simulation.utils.geometry import Point, Pose, Transform, Vector
from simulation.utils.road.sections.bezier_curve import CubicBezier
from simulation.utils.road.sections.road_section import RoadSection


@dataclass
class Road:
    """Container object for roads.

    A road consists of multiple road sections that are concatenated. The sections attribute
    contains these sections in the correct order.
    """

    _name: str = field(default=None, init=False)
    """Name of the road.

    The name attribute is determined by the name of the file.
    It is filled in when the road is generated.
    """
    _seed: str = field(default=None, init=False)
    """Seed used when generating the road.

    Determined when generating the road.
    """

    use_seed: bool = True
    """Use a default seed if none is provided.

    By default the `use_seed` attribute is true and a seed is set before creating \
    the road. If `use_seed` is set to False, the seed is invalidated and there will \
    be different random values every time the road is created.
    """

    sections: List[RoadSection] = field(default_factory=list)
    """All sections of the road."""

    length: float = 0
    """Length of road."""

    def __post_init__(self):
        """Set random seed if specified."""
        if not self.use_seed:
            # Overwrite seed
            random.seed()

    def append(self, section: RoadSection):
        """Append a road section. Determine id of the section.

        Args:
            section: New road section.
        """
        section.id = len(self.sections)
        if section.id == 0:
            section._is_start = True
        if section.id > 0:
            # Pass ending of last section as the transformation to next section
            ending: Tuple[Pose, float] = self.sections[-1].get_ending()
            section.transform = Transform(ending[0], ending[0].orientation)
            section.prev_length = self.length
        self.length = self.length + section.middle_line.length
        self.sections.append(section)

    def close_loop(self, p_curvature: float = 2):
        """Append a road section that connects the last section to the beginning.

        The road's beginning and it's end are connected using a cubic bezier curve.

        Args:
            p_curvature: Scale the curvature of the resulting loop.
        """

        # Global position of the end of the road
        end_pose_global, _ = self.sections[-1].get_ending()

        # Inverse of the end == start pose in local coordinates
        start_pose = Pose(Transform(end_pose_global).inverse)

        approximate_radius = abs(start_pose.position) / p_curvature

        section = CubicBezier(
            p1=Point(approximate_radius, 0),
            p2=start_pose.position
            - Vector(approximate_radius, 0).rotated(start_pose.orientation),
            p3=start_pose.position,
        )
        self.append(section)


DEFAULT_ROAD_DIR = os.path.join(
    os.environ.get("KITCAR_REPO_PATH"),
    "kitcar-gazebo-simulation",
    "simulation",
    "models",
    "env_db",
)


def _get_module(name: str):

    if not os.path.isabs(name):
        name = os.path.join(DEFAULT_ROAD_DIR, name)

    if not name.endswith(".py"):
        name += ".py"

    # Remove .py
    module_name = os.path.basename(name)[:-3]

    try:
        spec = importlib.util.spec_from_file_location(module_name, name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, name, module_name
    except AttributeError or ModuleNotFoundError:
        raise ValueError(f"Road {module_name} not found at {name}.")


def load(road_name: str, seed: str = "KITCAR") -> Road:
    """Load road object from file.

    Args:
        road_name: Name of the file containing the road definition.
        seed: Predetermine random values.
    """

    # Set random seed
    # set it at this point because the module is reloaded afterwards
    # the above import is just to ensure that the road is in the module cache
    random.seed(seed)

    # Ensure that the road is up to date
    road_module, file_path, road_name = _get_module(road_name)

    road = road_module.road
    road._file_path = file_path
    road._name = road_name
    road._seed = seed

    return road_module.road
