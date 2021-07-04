import math
import os
import random

import numpy as np

from simulation.utils.road.road import Road
from simulation.utils.road.sections import RightCircularArc, StraightRoad, TrafficSign

SIZE = int(os.environ.get("SIZE", 1))
EDGES = int(os.environ.get("EDGES", 9))

signs_dict = TrafficSign.__dict__
signs = SIZE * [signs_dict[s] for s in signs_dict if not s.startswith("__")]
random.shuffle(signs)
signs_list = np.array_split(signs, EDGES)

straight_length = len(signs_list[0])
y = 0.5

road = Road()
for s, signs in enumerate(signs_list):
    straight = StraightRoad(length=straight_length)
    for i, sign in enumerate(signs):
        y *= -1
        straight.traffic_signs.append(
            TrafficSign(
                kind=sign,
                arc_length=i,
                angle=0,
                normalize_x=False,
                y=y,
            )
        )
    road.append(straight)
    if s != len(signs_list) - 1:
        road.append(RightCircularArc(radius=2.81, angle=math.radians(360 / EDGES)))

road.close_loop()
