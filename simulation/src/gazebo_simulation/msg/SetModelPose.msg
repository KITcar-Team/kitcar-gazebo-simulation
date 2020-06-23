# The keys list is a list of integers specifying which variables should updated.
#
# Example
# -------
# If the message
#
# msg = SetModelPose()
# msg.keys = [SetModelPose.POSITION_X, SetModelPose.POSITION_Y]
# msg.values = [1,2]
#
# is received by a model plugin link, the corresponding model's
# position's x- and y-coordinate will be updated to x=1, y=2 while the z-coordinate remains
# unchanged.
#
uint8 POSITION_X = 0
uint8 POSITION_Y = 1
uint8 POSITION_Z = 2

uint8 ORIENTATION_W = 3
uint8 ORIENTATION_X = 4
uint8 ORIENTATION_Y = 5
uint8 ORIENTATION_Z = 6

int8[] keys
float32[] values
