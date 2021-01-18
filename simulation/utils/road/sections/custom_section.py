from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import yaml

import simulation.utils.road.sections.type as road_section_type
from simulation.utils.basics.init_options import InitOptions
from simulation.utils.basics.save_options import SaveOptions
from simulation.utils.geometry import Line, Transform
from simulation.utils.road.sections.road_section import RoadSection


@dataclass
class CustomSection(RoadSection, InitOptions, SaveOptions):

    TYPE = road_section_type.CUSTOM

    middle_line_points: List[np.array] = field(default_factory=list)
    """Points that make up the middle line."""

    @property
    def middle_line(self) -> Line:
        return self.transform * Line(self.middle_line_points)

    def save_as_yaml(self, file_path: str):
        """Save the section as a yaml file."""

        custom_dict = dict(self.__dict__)
        if "transform" in custom_dict:
            del custom_dict["transform"]
        super().save_as_yaml(file_path, custom_dict, dumper=yaml.Dumper)

    @classmethod
    def from_yaml(cls, file_path: str):
        """Load from a yaml file."""
        return super(CustomSection, cls).from_yaml(file_path, loader=yaml.Loader)

    def split_by(
        self, sections: List[Tuple[int, RoadSection]], x_buffer=1
    ) -> List[RoadSection]:
        """Add a number of other sections along this section.

        Args:
            sections: Road sections and arc lengths at which they should be added.
            x_buffer: Buffer after a road section to improve the adjustment of the end
                pose.

        Return:
            New road sections that replace this section.
        """
        if len(sections) == 0:
            return [self]
        assert 0 < sections[0][0] < self.middle_line.length
        x_beginning, section = sections[0]

        x_end = x_beginning + section.middle_line.length
        x_buffered_end = x_end + x_buffer
        start_pose = self.middle_line.interpolate_pose(x_beginning)
        end_pose = self.middle_line.interpolate_pose(x_buffered_end)

        beginning = CustomSection(
            middle_line_points=Line.cut(self.middle_line, x_beginning)[0].get_points()
        )

        intermediate_section = RoadSection.fit_ending(
            Transform(start_pose.position, start_pose.orientation)
            * (section.get_ending()[0]),
            end_pose,
        )

        tf = Transform(end_pose.position, end_pose.orientation).inverse
        remaining_custom_section = CustomSection(
            middle_line_points=[
                tf * p for p in Line.cut(self.middle_line, x_buffered_end)[1].get_points()
            ]
        )

        def distribute_objects(attribute_name: str):
            """Distribute objects (traffic signs, obstacles, surface_markings, ...) to
            resulting sections."""
            for obj in self.__getattribute__(attribute_name):
                x = self.middle_line.project(obj.center)
                if x < x_beginning:
                    x_shift = 0
                    beginning.__getattribute__(attribute_name).append(obj)
                elif x_beginning < x < x_end:
                    x_shift = -x_beginning
                    section.__getattribute__(attribute_name).append(obj)
                elif x_end < x < x_buffered_end:
                    x_shift = -x_end
                    intermediate_section.__getattribute__(attribute_name).append(obj)
                else:
                    x_shift = -x_buffered_end
                    remaining_custom_section.__getattribute__(attribute_name).append(obj)
                obj._frame = Transform([x_shift, 0], 0) * obj._frame

        distribute_objects("obstacles")
        distribute_objects("surface_markings")
        distribute_objects("traffic_signs")

        return [
            beginning,
            section,
            intermediate_section,
        ] + remaining_custom_section.split_by(
            sections=[(x - x_buffered_end, section) for x, section in sections[1:]]
        )
