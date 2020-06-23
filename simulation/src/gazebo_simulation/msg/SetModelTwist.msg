# The keys list is a list of integers specifying which variables should updated.
# See SetModelPose.msg for an explanation on how to use the keys/values lists.
uint8 LINEAR_X = 0
uint8 LINEAR_Y = 1
uint8 LINEAR_Z = 2

uint8 ANGULAR_X = 3
uint8 ANGULAR_Y = 4
uint8 ANGULAR_Z = 5

# New twist
int8[] keys
float32[] values
