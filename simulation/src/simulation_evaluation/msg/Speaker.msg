# Definition of a msg published by speaker and received from the state machine(s)

# LOCATION TYPE

int8 OFF_ROAD = 0
int8 RIGHT_LANE = 1
int8 LEFT_LANE = 2
int8 PARKING_LOT = 3


# ZONES
# =====
# Zones are used to determine what the car is currently allowed to do.
# E.g. in a parking area the car is allowed to cross the left side of the road to park.
int8 PARKING_ZONE = 10
int8 NO_PARKING_ZONE = 11

int8 OVERTAKING_ZONE = 12
int8 NO_OVERTAKING_ZONE = 13

int8 START_ZONE = 14  # Car is at beginning of the road
int8 DRIVING_ZONE = 15  # Car is somewhere driving
int8 END_ZONE = 16  # Car reached end of road

# Zones that are used to determine if the car stops/halts when it should.
# (e.g. before an intersection)
int8 STOP_ZONE = 17  # Car needs to stop here
int8 HALT_ZONE = 18  # Just halt
int8 NO_STOP_ZONE = 19  # The car currently does not need to stop or halt!


# SPEED ZONES

int8 SPEED_UNLIMITED_ZONE = 30
int8 SPEED_10_ZONE = 31
int8 SPEED_20_ZONE = 32
int8 SPEED_30_ZONE = 33
int8 SPEED_40_ZONE = 34
int8 SPEED_50_ZONE = 35
int8 SPEED_60_ZONE = 36
int8 SPEED_70_ZONE = 37
int8 SPEED_80_ZONE = 38
int8 SPEED_90_ZONE = 39

# EVENTS

int8 COLLISION = 40  # Car has collided with an obstacle
int8 PARKING_SPOT = 41  # Car is inside a parking spot
int8 PRIORITY_AREA = 42  # Car intersects with a priority area.
# A priority area is the area before an obstacle in a situation
# where the obstacle has the right of way (e.g. crosswalk)

# SPEED

int8 SPEED_STOPPED = 60
int8 SPEED_HALTED = 61
int8 SPEED_0 = 62
int8 SPEED_1_10 = 63
int8 SPEED_11_20 = 64
int8 SPEED_21_30 = 65
int8 SPEED_31_40 = 66
int8 SPEED_41_50 = 67
int8 SPEED_51_60 = 68
int8 SPEED_61_70 = 69
int8 SPEED_71_80 = 70
int8 SPEED_81_90 = 71
int8 SPEED_91_ = 72


# ---- Definition ---- #

int8 type
string name  # Is only used for debugging
