from dataclasses import dataclass

from simulation.utils.geometry import Transform


@dataclass
class Transformable:
    """Object which defines a transform property.

    The transform can only be modified through the :py:func:`set_transform`.
    """

    __transform: Transform = None
    """Transform to origin of the object.

    The name of the transform starts with __ to prevent subclasses
    from changing it's values without running :py:func:`set_transform`.
    """

    def __post_init__(self):
        if self.__transform is not None:
            self.set_transform(self.__transform)
        else:
            self.__transform = Transform([0, 0], 0)

    @property
    def transform(self):
        return self.__transform

    def set_transform(self, new_tf: Transform):
        assert type(new_tf) is Transform
        self.__transform = new_tf
