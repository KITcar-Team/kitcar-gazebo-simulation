"""Module package to create .urdf model definitions.

The package contains a number of classes that enable to define urdf robots through python.

**For what?** The `urdf <http://wiki.ros.org/urdf>`_ is very repetetive to write
and requires prior knowledge on available tags and how they can be combined to a fully
specified robot.
Additionally, the commonly used `xacro <http://wiki.ros.org/xacro>`_ is less powerful than
python and, again, another tool.

The basic idea used throughout the package is the inheritance from :py:class:`XmlObject`.
:py:func:`XmlObject.create_xml` creates a :py:class:`xml.etree.cElementTree.Element`
from the class instance.
It does so by converting all instance attributes into xml attributes and sub elements.
If a class attribute is an instance of a subclass of :py:class:`XmlObject`,
the objects :py:func:`XmlObject.create_xml` is called recursively.

The :py:class:`XmlObject` in combination with
`dataclasses <https://docs.python.org/3/library/dataclasses.html>`_
allows to create lightweight classes that define urdf components when converted to xml.
"""

from .camera import (  # noqa: 402, 401
    CameraClip,
    CameraDefinition,
    CameraImage,
    CameraPlugin,
    CameraProperties,
    CameraSensor,
    DepthCamera,
    DepthCameraPlugin,
    DepthCameraProperties,
    MonoCamera,
)
from .core import Attribute, Axis, Origin, XmlObject  # noqa: 402, 401
from .gazebo import Gazebo, Plugin, Sensor  # noqa: 402, 401
from .joint import Child, Joint, JointDynamics, JointLimit, Parent  # noqa: 402, 401
from .link import (  # noqa: 402, 401
    Box,
    Collision,
    Color,
    Cylinder,
    Geometry,
    Inertia,
    Inertial,
    Link,
    Material,
    Mesh,
    Sphere,
    Visual,
)

__all__ = [
    "Attribute",
    "XmlObject",
    "Origin",
    "Axis",
    "Color",
    "Inertia",
    "Inertial",
    "Box",
    "Sphere",
    "Cylinder",
    "Geometry",
    "Mesh",
    "Material",
    "Visual",
    "Collision",
    "Link",
    "JointDynamics",
    "JointLimit",
    "Parent",
    "Child",
    "Joint",
    "Plugin",
    "Gazebo",
    "Sensor",
    "MonoCamera",
    "DepthCamera",
    "CameraClip",
    "CameraDefinition",
    "CameraImage",
    "CameraPlugin",
    "DepthCameraPlugin",
    "CameraSensor",
    "CameraProperties",
    "DepthCameraProperties",
]
