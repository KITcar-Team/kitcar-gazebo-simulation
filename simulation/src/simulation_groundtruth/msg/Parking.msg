#Define spot types
int8 SPOT_FREE = 0
int8 SPOT_OCCUPIED = 1
int8 SPOT_X = 2
int8 SPOT_UNKNOWN =3


# Definition of a msg describing a parking lot

Line[] borders  # Outside limitation of parking lot
LabeledPolygon[] spots
