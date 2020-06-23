"""Class definitions making up the <joint>.

Example:

    A simple joint connecting :py:attr:`link2` to
    :py:attr:`link1` with a revolute joint that can rotate along the z-axis:

    >>> from simulation.utils.urdf import Origin, Link, Joint, Parent, Child, Axis, JointLimit, JointDynamics
    >>> link1 = Link("link_1")
    >>> link2 = Link("link_2")
    >>> simple_joint = Joint(
    ...     name="simple_joint",
    ...     parent=Parent(link1.name),
    ...     child=Child(link2.name),
    ...     type=Joint.REVOLUTE,
    ...     origin=Origin([1,3,2]),
    ...     axis=Axis([0, 0, 1]),
    ...     limit=JointLimit(effort=1000, lower=-10, upper=10),
    ...     dynamics=JointDynamics(damping=1, friction=1),
    ... )
"""

from dataclasses import dataclass

from simulation.utils.geometry import Vector

from .core import XmlObject, Attribute, Origin


@dataclass
class JointDynamics(XmlObject):
    TAG = "dynamics"

    damping: Attribute = None
    friction: Attribute = None


@dataclass
class JointLimit(XmlObject):
    TAG = "limit"

    effort: Attribute = None
    lower: Attribute = None
    upper: Attribute = None
    velocity: Attribute = None


@dataclass
class Parent(XmlObject):
    TAG = "parent"

    link: Attribute


@dataclass
class Child(XmlObject):
    TAG = "child"

    link: Attribute


@dataclass
class Joint(XmlObject):
    TAG = "joint"

    # Possible joint types
    FIXED = "fixed"
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"

    name: Attribute

    parent: Parent
    child: Child

    type: Attribute = FIXED
    axis: Vector = None

    limit: JointLimit = None
    dynamics: JointDynamics = None
    origin: Origin = None
