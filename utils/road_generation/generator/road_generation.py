from road_generation.generator import primitive, preset_parser
import math
#import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def norm_angle(angle):
    while angle > 2 * math.pi:
        angle -= 2 * math.pi
    while angle < 0:
        angle += 2 * math.pi
    return angle

def generate_road(primitives, padding):
    new_primitives = [primitives[0]]

    for i in range(1, len(primitives)):
        last_primitive = new_primitives[i-1]
        current_primitive = primitives[i]

        (point, angle, curv) = last_primitive.get_ending()
        target_point = point + np.array([
            math.cos(angle) * padding,
            math.sin(angle) * padding
        ])
        target_angle = norm_angle(angle + math.pi)
        (begin_point, begin_angle, begin_curv) = current_primitive.get_beginning()
        new_primitives.append(primitive.TransrotPrimitive(current_primitive,
            target_point - begin_point, target_angle - begin_angle))

    return new_primitives

def check_intersections(road, road_width):
    for i in range(len(road)-1):
        for j in range(i+2, len(road)):
            p1 = road[i].get_bounding_box(road_width)
            p2 = road[j].get_bounding_box(road_width)
            if p1.intersects(p2):
                return True
    return False

def generate(root, seed = None):
    while True:
        preset = preset_parser.eval(root)
        
        seed = None if preset.seed is None else preset.seed.text

        random.seed(seed)
        primitives = preset.primitives
        road = generate_road(primitives, 0)
        if not check_intersections(road, preset.road_width):
            break

    return road
