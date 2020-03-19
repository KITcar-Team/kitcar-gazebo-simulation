# The fields variable is used to specify which fields of the new twist should actually be updated.
# This enables to not change the speed in z-direction
# The usage is as followed:
# Each digit represents one direction:
# linear.x, linear.y, linear.z, angular.x, angular.y, angular.z
uint8 POSITION_X = 0
uint8 POSITION_Y = 1
uint8 POSITION_Z = 2

uint8 ORIENTATION_W = 3
uint8 ORIENTATION_X = 4
uint8 ORIENTATION_Y = 5
uint8 ORIENTATION_Z = 6

# New twist
int8[] keys
float32[] values
