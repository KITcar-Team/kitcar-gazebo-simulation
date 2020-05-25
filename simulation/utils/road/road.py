"""Road class used to define roads as a python class.

A simulated road can be defined through an object of type Road.
It contains all sections of that road as a list.
"""

import sys
import random
import importlib
import os
from dataclasses import dataclass, field
from typing import List, Tuple

from simulation.utils.road.sections.road_section import RoadSection

from simulation.utils.geometry import Transform, Pose


@dataclass
class Road:
    """Container object for roads.

    A road consists of multiple road sections that are concatenated.
    The sections attribute contains these sections in the correct order.
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
            section.transform = Transform(ending[0], ending[0].get_angle())
            section.prev_length = self.length
        self.length = self.length + section.middle_line.length
        self.sections.append(section)


DEFAULT_ROAD_DIR = os.path.join(
    os.environ.get("KITCAR_REPO_PATH"),
    "kitcar-gazebo-simulation",
    "simulation",
    "models",
    "env_db",
)


def load(road_name: str, seed: str = "KITCAR") -> Road:
    """Load road object from file.

    Args:
        road_name: Name of the file containing the road definition.
        seed: Predetermine random values.
    """

    sys.path.append(DEFAULT_ROAD_DIR)

    try:
        road_module = importlib.import_module(road_name, DEFAULT_ROAD_DIR)
    except ModuleNotFoundError:
        raise ValueError(f"Road of name {road_name} not found in {DEFAULT_ROAD_DIR}.")

    # Set random seed
    # set it at this point because the module is reloaded afterwards
    # the above import is just to ensure that the road is in the module cache
    random.seed(seed)

    # Ensure that the road is up to date
    importlib.reload(road_module)

    road = road_module.road
    road._name = road_name
    road._seed = seed

    return road_module.road
