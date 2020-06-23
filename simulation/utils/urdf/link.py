"""Class definitions making up the <link>.

Example:

    A simple link with the shape of a rectangular box
    can be defined as follows:

    >>> from simulation.utils.geometry import Vector
    >>> from simulation.utils.urdf import Geometry, Box, Material, Link, Collision, Visual
    >>> box = Geometry(Box(Vector(2, 2, 2)))
    >>> material = Material("mat", color=Vector(1, 1, 1))
    >>> link = Link(
    ...     "link",
    ...     collision=Collision(geometry=box),
    ...     visual=Visual(geometry=box, material=material),
    ... )
    >>> link.use_inertial_from_collision(mass=1.0)
    ...  # Calculate inertial tensor from simulation.utils.geometry and mass
"""

from dataclasses import dataclass, field
import xml.etree.cElementTree as cET
from typing import Tuple

from simulation.utils.geometry import Vector

from .core import XmlObject, Attribute, Origin


def _origin_default_list():
    return [0, 0, 0]


@dataclass
class Color(XmlObject):
    TAG = "color"

    xyz: Attribute = field(default_factory=_origin_default_list)


@dataclass
class Inertia(XmlObject):
    TAG = "inertia"

    ixx: Attribute = 0
    ixy: Attribute = 0
    ixz: Attribute = 0
    iyx: Attribute = 0
    iyy: Attribute = 0
    iyz: Attribute = 0
    izx: Attribute = 0
    izy: Attribute = 0
    izz: Attribute = 0


@dataclass
class Inertial(XmlObject):
    TAG = "inertial"

    mass: float
    inertia: Inertia
    origin: Origin = None

    def create_xml(self, root_el=None):
        m = self.mass
        del self.mass
        el = super().create_xml(root_el)
        cET.SubElement(el, "mass", attrib={"value": str(m)})
        return el


@dataclass
class Geometry(XmlObject):
    TAG = "geometry"

    value: XmlObject


@dataclass
class Box(XmlObject):
    TAG = "box"

    size: Attribute

    def calculate_inertia(self, mass) -> Inertia:
        def rect_inertia(l1, l2):
            return 1 / 12 * mass * (l1 * l1 + l2 * l2)

        return Inertia(
            ixx=rect_inertia(self.size.y, self.size.z),
            iyy=rect_inertia(self.size.x, self.size.z),
            izz=rect_inertia(self.size.x, self.size.y),
        )


@dataclass
class Sphere(XmlObject):
    TAG = "sphere"

    radius: Attribute

    def calculate_inertia(self, mass) -> Inertia:
        ixx = 1 / 12 * mass * (3 * pow(self.radius, 2) + pow(self.radius, 2))
        return Inertia(ixx=ixx, iyy=ixx, izz=1 / 2 * mass * pow(self.radius, 2))


@dataclass
class Cylinder(XmlObject):
    TAG = "cylinder"

    radius: Attribute
    length: Attribute

    def calculate_inertia(self, mass) -> Inertia:
        ixx = 1 / 12 * mass * (3 * pow(self.radius, 2) + pow(self.length, 2))
        return Inertia(ixx=ixx, iyy=ixx, izz=1 / 2 * mass * pow(self.radius, 2))


@dataclass
class Mesh(XmlObject):
    TAG = "mesh"

    filename: Attribute
    scale: Vector = None


@dataclass
class Collision(XmlObject):
    TAG = "collision"

    origin: Origin = None
    geometry: Geometry = None


@dataclass
class Material(XmlObject):
    TAG = "material"

    name: Attribute = None
    color: Vector = None
    texture: str = None


@dataclass
class Visual(XmlObject):
    TAG = "visual"

    origin: Tuple = None
    geometry: Geometry = None
    material: Material = None


@dataclass
class Link(XmlObject):
    TAG = "link"

    name: Attribute

    collision: Collision = None
    visual: Collision = None
    inertial: Inertial = None

    def use_inertial_from_collision(self, mass: float):
        self.inertial = Inertial(
            mass=mass,
            inertia=self.collision.geometry.value.calculate_inertia(mass),
            origin=self.collision.origin,
        )
