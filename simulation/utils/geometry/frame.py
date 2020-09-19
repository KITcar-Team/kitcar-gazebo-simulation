"""Coordinate frame class that can be connected to other frames.

The idea of this class is to provide a simple API of dealing with geometric objects in
multiple coordinate frames easily.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, NewType

TransformType = NewType("Transform", Any)


@dataclass
class Frame:

    name: str
    """Name of the frame."""

    _connected_frames: Dict[str, TransformType] = field(init=False, default_factory=dict)
    """Dictionary with other connected frames."""

    def connect_to(self, frame: "Frame", *, transformation_to_frame: TransformType):
        """Connect this frame to another frame through a transformation.

        This also connects the other frame to this one.
        """
        if frame.name == self.name:
            raise ValueError("Cannot connect to a frame with the same name.")

        self._connected_frames[frame.name] = transformation_to_frame.inverse
        frame._connected_frames[self.name] = transformation_to_frame

    def __call__(self, geom_obj):
        """Calling an instance transforms the argument's coordinates into this frame."""
        if not hasattr(geom_obj, "_frame"):
            raise ValueError("Cannot transform object that has no frame.")
        elif geom_obj._frame is self:
            return geom_obj
        if geom_obj._frame.name not in self._connected_frames:
            raise ValueError("Transformation to frame {geom_obj._frame} is unknown.")

        transformed = self._connected_frames[geom_obj._frame.name] * geom_obj
        transformed._frame = self
        return transformed


def validate_and_maintain_frames(func):
    """Ensure that both objects are in the same coordinate frame."""

    def decorator(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            obj2 = None
        elif len(args) == 0:
            obj2 = next(iter(kwargs.values()))
        else:
            obj2 = args[0]

        if (
            not hasattr(obj2, "_frame")
            or (self._frame is None or obj2._frame is None)
            or (self._frame == obj2._frame)
        ):
            result = func(self, *args, **kwargs)
            if hasattr(result, "_frame"):
                result._frame = self._frame
                if result._frame is None and hasattr(obj2, "_frame"):
                    result._frame = obj2._frame

            return result
        else:
            raise ValueError(
                f"The objects {self} and {obj2} are defined in different coordinate frames."
            )

    return decorator
