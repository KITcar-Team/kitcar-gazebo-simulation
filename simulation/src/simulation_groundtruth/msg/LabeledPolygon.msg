#Generic msg to describe polygons with a defined type (e.g. parking spot with free/occupied/x)
int8 OBSTACLE = 0

int8 RIGHT_TURN_MARKING = 1
int8 LEFT_TURN_MARKING = 2

geometry_msgs/Polygon frame
uint8 type
