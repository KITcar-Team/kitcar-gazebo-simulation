"""Class definitions of core building blocks of the package."""
import xml.etree.cElementTree as cET
import xml.etree.ElementTree
import xml.dom.minidom
from dataclasses import dataclass, field

from simulation.utils.geometry import Vector

from typing import Any, List
from collections import Iterable


class Attribute(str):
    """Basic class that will be converted into a xml attribute.

    **Important:** Within a subclass of :py:class:`XmlObject` an attribute can be defined
    by simply annotating a class variable with py:class:`Attribute`!

    Example:
        >>> from dataclasses import dataclass
        >>> from urdf.core import XmlObject, Attribute
        >>> @dataclass
        ... class Example(XmlObject):
        ...     TAG = "example"
        ...     name: Attribute
        ...     value1: str
        ...     value2: float = None
        >>> ex = Example("example_name",value1="I'm an example.")

        The code block defines a simple example class that results in the xml string:

        .. code-block:: xml

           <example name="example_name">
             <value1>I'm an example.</value1>
           </example>

    Args:
        val: Attribute value.
        name: *Optionally* the name of the can be provided. By default the name of the \\
                Attribute instance defines the name of the resulting xml attribute.
    """

    def __new__(cls, val: str, name: str = None):
        attr = super().__new__(cls, val)
        attr.name = name
        return attr


@dataclass
class XmlObject:
    """Base class for all urdf xml objects."""

    TAG = None

    def _get_val(self, val: Any) -> str:
        """Convert a value to a string that is included into the xml."""
        if issubclass(type(val), Vector):
            xyz = (val.x, val.y, val.z)
            return " ".join(str(x) for x in xyz)
        elif not isinstance(val, str) and isinstance(val, Iterable):
            return " ".join(str(x) for x in val)
        else:
            return str(val)

    def create_xml(self, root_el: cET.Element = None) -> cET.Element:
        """Convert this instance into a xml element.

        * If the class of this instance does not define a tag,
          new xml elements from this instance are appended to the `root_el`. Otherwise,
          a sub element is created and returned.
        * Additionally to all instance attributes, a dictionary :py:attr:`parameters` can be
          used to define sub elements.
        * A :py:class:`Attribute` type annotation for any of the object`s attributes,
          will result in a xml attribute.

        Args:
            root_el: *Optional* element that is used a parent xml element when provided.
        """
        assert (
            self.__class__.TAG is not None or root_el is not None
        ), "XmlObject class TAG must be set or a root_el provided!"
        tag = self.__class__.TAG

        if tag is None:
            el = root_el
        elif root_el is None:
            el = cET.Element(tag)
        else:
            el = cET.SubElement(root_el, tag)

        d = self.__dict__

        if "parameters" in self.__dict__:
            d.update(self.__dict__["parameters"])
            del d["parameters"]

        for key, val in self.__dict__.items():
            if val is None:
                continue
            if self.__annotations__.get(key, None) is Attribute:
                if isinstance(val, Attribute):
                    el.set(val.name if val.name is not None else key, self._get_val(val))
                else:
                    el.set(key, self._get_val(val))
            elif isinstance(val, XmlObject):
                # sub_el = cET.SubElement(el, key)
                if val.__class__.TAG is None:
                    val.create_xml(el)
                else:
                    el.append(val.create_xml())
            else:
                cET.SubElement(el, key).text = self._get_val(val)

        return el if root_el is None else root_el

    def save(self, file_path: str):
        """Save this instance as a xml string to a file.

        Args:
            file_path: Location to store the xml.
        """
        with open(file_path, "w+") as file:
            elem = self.create_xml()
            rough_string = xml.etree.ElementTree.tostring(elem, "utf-8")
            reparsed = xml.dom.minidom.parseString(rough_string)
            # Get rid of empty lines (introduced by parsing a pretty xml)
            xml_str = "\n".join(
                [
                    line
                    for line in reparsed.toprettyxml(indent=" " * 2).split("\n")
                    if line.strip()
                ]
            )
            file.write(xml_str)


def _origin_default_list():
    return [0, 0, 0]


@dataclass
class Origin(XmlObject):
    TAG = "origin"

    xyz: Attribute = field(default_factory=_origin_default_list)
    rpy: Attribute = field(default_factory=_origin_default_list)

    @classmethod
    def from_vector(cls, vector: Vector):
        return cls([vector.x, vector.y, vector.z])

    @classmethod
    def from_list(cls, l: List[float]):
        return cls(l[:3], l[3:])


@dataclass
class Axis(XmlObject):
    TAG = "axis"

    xyz: Attribute = field(default_factory=_origin_default_list)
