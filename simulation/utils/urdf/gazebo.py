"""Class definitions of tags used only for gazebo."""
from dataclasses import dataclass

from .core import Attribute, XmlObject


@dataclass
class Plugin(XmlObject):
    TAG = "plugin"

    name: Attribute
    filename: Attribute


@dataclass
class Gazebo(XmlObject):
    TAG = "gazebo"

    inner: XmlObject
    reference: Attribute = None


@dataclass
class Sensor(XmlObject):
    TAG = "sensor"

    name: Attribute
    type: Attribute
