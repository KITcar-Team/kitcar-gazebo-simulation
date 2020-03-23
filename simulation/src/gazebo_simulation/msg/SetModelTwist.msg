# The fields variable is used to specify which fields of the new twist should actually be updated.
# This enables to not change the speed in z-direction
# The usage is as followed:
# Each digit represents one direction:
# linear.x, linear.y, linear.z, angular.x, angular.y, angular.z
uint8 LINEAR_X = 0
uint8 LINEAR_Y = 1
uint8 LINEAR_Z = 2

uint8 ANGULAR_X = 3
uint8 ANGULAR_Y = 4
uint8 ANGULAR_Z = 5

# New twist
int8[] keys
float32[] values
