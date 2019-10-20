from road_generation.generator import primitive
import random

class Preset:
    def __init__(self):
        # set sensitve defaults
        self.road_width = 1
        self.primitives = []

def eval(root):
    preset = Preset()
    preset.road_width = 0.4 # TODO
    preset.seed = root.find("seed")
    random.seed(None if preset.seed is None else preset.seed.text)
    preset.primitives = eval_element(root.find("sequence"))
    return preset

def eval_element(el):
    if el.tag == "line":
        return [
            primitive.StraightLine(el.attrib)
        ]
    elif el.tag == "leftArc":
        return [
            primitive.LeftCircularArc(el.attrib)
        ]
    elif el.tag == "rightArc":
        return [
            primitive.RightCircularArc(el.attrib)
        ]
    elif el.tag == "quadBezier":
        return [
            primitive.QuadBezier(el.attrib)
        ]
    elif el.tag == "cubicBezier":
        return [
            primitive.CubicBezier(el.attrib)
        ]
    elif el.tag == "blockedArea":
        return [
            primitive.BlockedAreaObstacle(el.attrib)
        ]
    elif el.tag == "trafficIsland":
        return [
            primitive.TrafficIsland(el.attrib)
        ]
    elif el.tag == "intersection":
        return [
            primitive.Intersection(el.attrib)
        ]
    elif el.tag == "staticObstacle":
        return [
            primitive.StraightLineObstacle(el.attrib)
        ]
    elif el.tag == "trafficSign":
        return [
            primitive.TrafficSign(el.attrib)
        ]
    elif el.tag == "ramp":
        return [
            primitive.Ramp(el.attrib)
        ]
    elif el.tag == "zebraCrossing":
        return [
            primitive.ZebraCrossing(el.attrib)
        ]
    elif el.tag == "startLane":
        return [
            primitive.StartLane(el.attrib)
        ]
    elif el.tag == "parkingArea":
        return [
            primitive.ParkingArea(el.attrib)
        ]
    elif el.tag == "parkingLot":
        return [
            primitive.ParkingLot(el.attrib)
        ]
    elif el.tag == "parkingLotStart":
        return [
            primitive.ParkingLotStart(el.attrib)
        ]
    elif el.tag == "parkingObstacle":
        return [
            primitive.ParkingObstacle(el.attrib)
        ]
    elif el.tag == "sequence":
        return [x for child in el for x in eval_element(child)]
    elif el.tag == "optional":
        if random.random() < float(el.attrib["p"]):
            return [x for child in el for x in eval_element(child)]
        else:
            return []
    elif el.tag == "repeat":
        if "min" in el.attrib and "max" in el.attrib:
            n = random.randint(int(el.attrib["min"]), int(el.attrib["max"]))
        else:
            n = int(el.attrib["n"])
        return [x for _ in range(n) for child in el for x in eval_element(child)]
    elif el.tag == "select":
        total = sum([float(case.attrib["w"]) for case in el])
        target = random.random() * total
        current_total = 0
        for case in el:
            current_total += float(case.attrib["w"])
            if target < current_total:
                return [x for child in case for x in eval_element(child)]
                break
    elif el.tag == "shuffle":
        children = list(el)
        random.shuffle(children)
        return [x for child in children for x in eval_element(child)]
    else:
        return []
