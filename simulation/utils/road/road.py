"""Road class used to define roads as a python class.

A simulated road can be defined through an object of type Road.
It contains all sections of that road as a list.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple

from road.sections.road_section import RoadSection

from geometry import Transform, Pose


@dataclass
class Road:
    """Container object for roads.

    A road consists of multiple road sections that are concatenated.
    The sections attribute contains these sections in the correct order.
    """

    use_seed: bool = True
    """Use a default seed if none is provided.

    By default the `use_seed` attribute is true and a seed is set before creating \
    the road. If `use_seed` is set to False, the seed is invalidated and there will \
    be different random values every time the road is created.
    """

    sections: List[RoadSection] = field(default_factory=list)
    """All sections of the road."""

    def __post_init__(self):
        """Set random seed if specified."""
        if not self.use_seed:
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

        self.sections.append(section)
