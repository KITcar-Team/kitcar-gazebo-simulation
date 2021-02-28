import math
import os
import random

import numpy as np

from simulation.utils.road.road import Road
from simulation.utils.road.sections import RightCircularArc, StraightRoad, TrafficSign

SIZE = int(os.environ.get("SIZE", 1))
EDGES = int(os.environ.get("EDGES", 9))
road = Road()
signs_dict = TrafficSign.__dict__
signs_ = SIZE * [signs_dict[s] for s in signs_dict if not s.startswith("__")]
random.shuffle(signs_)
signs_list = np.array_split(signs_, EDGES)
right = False
y = 0.5
largest_road = max(len(signs) for signs in signs_list)
for s, signs in enumerate(signs_list):
    straight = StraightRoad(length=3 * largest_road)
    for i, sign in enumerate(signs):
        y *= -1
        straight.traffic_signs.append(
            TrafficSign(
                kind=sign,
                arc_length=i * 3,
                angle=0,
                normalize_x=False,
                y=y,
            )
        )
    road.append(straight)
    if s != len(signs_list) - 1:
        road.append(RightCircularArc(radius=2.81, angle=math.radians(360 / EDGES)))

road.close_loop()
