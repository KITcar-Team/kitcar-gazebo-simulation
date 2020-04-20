from road.sections import StraightRoad, StaticObstacle
from road.road import Road

from geometry import Point

road = Road()
road.append(StraightRoad(length=2))
road.append(StraightRoad(length=2, right_line_marking=StraightRoad.MISSING_LINE_MARKING))
road.append(StraightRoad(length=2, obstacles=[StaticObstacle(Point(1, -0.2))],))

import textwrap  # noqa: 402

print(textwrap.fill(str(road), width=92))
