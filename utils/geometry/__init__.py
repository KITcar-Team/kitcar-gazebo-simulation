"""Geometry module init.py

Collect geometry classes which are decorated with @export.
"""

# Decorator to easily add classes/functions to the geometry module.
def export(define):
    # Add new object to globals
    globals()[define.__name__] = define
    # Append the object to __all__ which is visible when the module is imported
    __all__.append(define.__name__)
    return define


__all__ = []

# Collect the classes defined in geometry
from .vector import *
from .point import *
from .transform import *
from .pose import *
from .line import *
from .polygon import *
