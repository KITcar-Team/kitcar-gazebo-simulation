"""Definition of the geometry module

Collect classes and functions which should be included in the geometry module.
"""
from typing import List, Any

# Decorator to easily add classes/functions to the geometry module.
def export(define):
    # Add new object to globals
    globals()[define.__name__] = define
    # Append the object to __all__ which is visible when the module is imported
    __all__.append(define.__name__)
    return define


__all__: List[Any] = []  # Will hold all classes/functions which can be imported with 'from geometry import ...'

# import all files which are part of the geometry module
import geometry.vector  # noqa:402
import geometry.point  # noqa:402
import geometry.transform  # noqa:402
import geometry.pose  # noqa:402
import geometry.line  # noqa:402
import geometry.polygon  # noqa:402
