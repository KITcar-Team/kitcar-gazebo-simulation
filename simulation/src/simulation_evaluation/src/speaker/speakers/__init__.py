"""The speakers package contains all speaker classes.
The common base class is :py:class:`Speaker`.
The speakers are used to combine the current CarState with known groundtruth information.
"""
from typing import List, Any

# Decorator to easily add classes/functions to the geometry module.
def export(define):
    # Add new object to globals
    globals()[define.__name__] = define
    # Append the object to __all__ which is visible when the module is imported
    __all__.append(define.__name__)
    return define


__all__: List[Any] = []  # Will hold all classes/functions which can be imported with

# import all files which are part of the geometry module
import simulation.src.simulation_evaluation.src.speaker.speakers.speaker  # noqa:
import simulation.src.simulation_evaluation.src.speaker.speakers.event  # noqa:402
import simulation.src.simulation_evaluation.src.speaker.speakers.area  # noqa:402
import simulation.src.simulation_evaluation.src.speaker.speakers.speed  # noqa:402
import simulation.src.simulation_evaluation.src.speaker.speakers.zone  # noqa:402
import simulation.src.simulation_evaluation.src.speaker.speakers.broadcast  # noqa:402
