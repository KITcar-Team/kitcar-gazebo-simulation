from simulation.utils.road.road import Road
from simulation.utils.road.sections import StraightRoad

# Quite a boring road.
# It is used to evaluate if the AutomaticDriveNode works.
# Using two separate StraightRoad is an attempt to ensure
# that it also works with multiple road sections.
road = Road()
road.append(StraightRoad(length=3))
road.append(StraightRoad(length=2))
