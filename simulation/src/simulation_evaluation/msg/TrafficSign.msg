#Stolen from kitcar-ros/KITcar_brain/src/messages/perception_msgs/msg/TrafficSign.msg
uint8 ZONE_30 = 1
uint8 ZONE_30_END = 2
uint8 HIGHWAY = 3
uint8 HIGHWAY_END = 4
uint8 NO_PASSING = 5
uint8 NO_PASSING_END = 6
uint8 DRIVING_DIRECTION_LEFT = 7
uint8 DRIVING_DIRECTION_RIGHT = 8
uint8 INCLINATION = 9
uint8 SLOPE = 10
uint8 TURN_LEFT = 11
uint8 TURN_RIGHT = 12
uint8 TURN_TABLE_LEFT = 13
uint8 TURN_TABLE_RIGHT = 14
uint8 CROSSWALK = 15
uint8 PARKING = 16
uint8 STOP = 17
uint8 GIVE_WAY = 18
uint8 RIGHT_OF_WAY = 19
uint8 ROAD_CLOSURE = 20
uint8 PASS_RIGHT = 21
uint8 ZONE_20 = 22
uint8 ZONE_20_END = 23

Header header
geometry_msgs/Pose pose
uint8 type
