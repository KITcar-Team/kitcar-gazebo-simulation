from road import schema
from functools import reduce

class BoundingBox:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def union(self, box):
        return BoundingBox(
            min(self.x_min, box.x_min),
            min(self.y_min, box.y_min),
            max(self.x_max, box.x_max),
            max(self.y_max, box.y_max))

    def __repr__(self):
        return "BoundingBox({0}, {1}, {2}, {3})".format(self.x_min, self.y_min,
            self.x_max, self.y_max)

def get_bounding_box(object):
    if isinstance(object, schema.boundary):
        all_x = list(map(lambda p: p.x, object.point))
        all_y = list(map(lambda p: p.y, object.point))
        return BoundingBox(min(all_x), min(all_y), max(all_x), max(all_y))
    elif isinstance(object, schema.lanelet):
        left = get_bounding_box(object.leftBoundary)
        right = get_bounding_box(object.rightBoundary)
        return left.union(right)
    elif isinstance(object, schema.rectangle):
        radius = math.sqrt((object.width/2)**2 + (object.length/2)**2)
        return BoundingBox(
            object.centerPoint.x - radius,
            object.centerPoint.y - radius,
            object.centerPoint.x + radius,
            object.centerPoint.y + radius)
    elif isinstance(object, schema.circle):
        return BoundingBox(
            object.centerPoint.x - radius,
            object.centerPoint.y - radius,
            object.centerPoint.x + radius,
            object.centerPoint.y + radius)
    elif isinstance(object, schema.polygon):
        all_x = list(map(lambda p: p.x, object.point))
        all_y = list(map(lambda p: p.y, object.point))
        return BoundingBox(min(all_x), min(all_y), max(all_x), max(all_y))
    elif isinstance(object, schema.shape):
        # TODO
        raise NotImplementedError()
    elif isinstance(object, schema.obstacle):
        # TODO
        raise NotImplementedError()
    elif isinstance(object, schema.CommonRoad):
        return reduce(
            lambda x,y: x.union(y),
            map(lambda l: get_bounding_box(l), object.lanelet))
    else:
        raise NotImplementedError()
