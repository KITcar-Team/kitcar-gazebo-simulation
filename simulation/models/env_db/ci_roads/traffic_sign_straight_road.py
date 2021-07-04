import os
import random

from simulation.utils.road.road import Road
from simulation.utils.road.sections import StraightRoad, TrafficSign
from simulation.utils.road.sections.traffic_sign import get_all_signs

SIZE = int(os.environ.get("SIZE", 1))
signs = SIZE * get_all_signs()
random.shuffle(signs)
y = 0.5
road = Road()
straight = StraightRoad(length=len(signs))
for i, sign in enumerate(signs):
    y *= -1
    straight.traffic_signs.append(
        TrafficSign(kind=sign, arc_length=i, angle=0, normalize_x=False, y=y, visible=True)
    )

road.append(straight)
