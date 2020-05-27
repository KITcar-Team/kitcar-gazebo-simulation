import errno
from dataclasses import dataclass, field
from typing import Tuple, Dict
import os
import hashlib

import cairo

from simulation.utils.geometry import Vector, Transform, Polygon
import simulation.utils.road.renderer.utils as utils
from simulation.utils.road.sections.road_section import RoadSection
import simulation.utils.road.renderer.surface_markings as render_surface_markings


@dataclass
class Tile:
    """Piece of the groundplane with lines used to display road lines on the ground.

    The groundplane in simulation is made out of many rectangular tiles.
    Each tile displays an image on the ground.
    """

    COUNTER = 0
    """Static variable used to generate unique model names in Gazebo.

    To ensure that Gazebo doesn't get confused with multiple tiles that have
    the same name, e.g. when reloading, the counter is increased every time a model
    name is generated.
    """

    index: Tuple[int, int]
    """Position within the lattice of tiles on the groundplane."""
    size: Vector
    """Size of the tile."""
    resolution: Vector
    """Resolution of the tile's image."""
    road_folder_name: str = field(default=None, repr=False)
    """Name of the folder in which all tiles of the current road are.

    (Not the complete path, just the name of the folder!)
    """
    sections: Dict[int, RoadSection] = field(default_factory=set)
    """All sections that are (atleast partly) on this tile."""
    id: str = None
    """ID of the tile.

    Automatically generated when rendering."""
    already_rendered: bool = False
    """Indicate whether the tile has been rendered before."""

    @property
    def name(self) -> str:
        """str: Name of the tile's model when spawned in Gazebo."""
        return f"tile_{self.index[0]}x{self.index[1]}"

    @property
    def transform(self) -> Transform:
        """Transform: Transform to the center of the tile."""
        return Transform([(self.index[0]) * self.size.x, (self.index[1]) * self.size.y], 0)

    @property
    def frame(self) -> Polygon:
        """Polygon: Frame of the tile."""
        return (
            self.transform
            * Transform([-self.size.x / 2, -self.size.y / 2], 0)
            * Polygon([[0, 0], [self.size.x, 0], self.size, [0, self.size.y]])
        )

    def get_material_string(self) -> str:
        """Content of the tile's material file.

        See: http://gazebosim.org/tutorials?tut=color_model
        """
        return """material {name}
        {{
            technique
            {{
                pass
                {{
                    texture_unit
                    {{
                        texture {texture_name} PF_RGB8
                    }}
                }}
            }}
        }}""".format(
            name=self.id, texture_name=self.id + ".png"
        )

    def get_model_string(self) -> str:
        """Get a model string that can be spawned in Gazebo."""
        Tile.COUNTER += 1
        return f"""
        <model name='{self.name + "x"+ str(Tile.COUNTER)}'>
          <static>1</static>
          <link name='link'>
            <collision name='collision'>
              <geometry>
                <plane>
                  <normal>0 0 1</normal>
                  <size>{self.size.x} {self.size.y}</size>
                </plane>
              </geometry>
              <surface>
                <friction>
                  <ode>
                    <mu>100</mu>
                    <mu2>50</mu2>
                  </ode>
                  <torsional>
                    <ode/>
                  </torsional>
                </friction>
                <contact>
                  <ode/>
                </contact>
                <bounce/>
              </surface>
              <max_contacts>10</max_contacts>
            </collision>
            <visual name='visual'>
              <cast_shadows>0</cast_shadows>
              <geometry>
                <plane>
                  <normal>0 0 1</normal>
                  <size>{self.size.x} {self.size.y}</size>
                </plane>
              </geometry>
              <material>
                <script>
                  <uri>model://{self.road_folder_name}/{self.id}</uri>
                  <name>{self.id}</name>
                </script>
              </material>
            </visual>
            <self_collide>0</self_collide>
            <enable_wind>0</enable_wind>
            <kinematic>0</kinematic>
          </link>
          <pose frame=''>{self.transform.x} {self.transform.y} 0 0 -0 0</pose>
        </model>
        """

    def render_to_file(self, roads_path: str):
        """Render an image of the tile and save it to a file.

        Args:
            roads_path: Directory in which all roads are located.
        """
        surface = cairo.ImageSurface(
            cairo.FORMAT_RGB24, int(self.resolution.x), int(self.resolution.y)
        )
        ctx = cairo.Context(surface)

        # Adjust scale
        ctx.scale(self.resolution.x / self.size.x, self.resolution.y / self.size.y)
        # Inverse y-axis
        ctx.translate(0, self.size.y / 2)
        ctx.scale(1, -1)
        ctx.translate(0, -self.size.y / 2)

        # Move to center of the tile
        ctx.translate(self.size.x / 2, self.size.y / 2)

        # Create black background
        ctx.set_source_rgb(0, 0, 0)
        ctx.rectangle(0, 0, self.size.x, self.size.y)
        ctx.fill()

        # Invert the render transform
        ctx.translate(-self.transform.x, -self.transform.y)

        # Draw lines for all sections
        for sec in self.sections.values():
            for line in sec.lines:
                utils.draw_line(ctx, line)
            for marking in sec.surface_markings:
                render_surface_markings.draw(ctx, marking)

        sha_256 = hashlib.sha256()
        sha_256.update(surface.get_data())
        hash = sha_256.hexdigest()

        self.id = "tile-{0}".format(hash)

        dir = os.path.join(roads_path, self.road_folder_name, self.id)
        if not os.path.exists(dir):
            try:
                os.makedirs(dir)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        surface.write_to_png(os.path.join(dir, self.id + ".png"))

        with open(os.path.join(dir, self.id + ".material"), "w+") as file:
            file.write(self.get_material_string())
