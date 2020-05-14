# !!COPIED FROM KITCAR ROS!!
# This message is copied from the state_estimation_msgs in kitcar ros.
# Why? By copying the message definition,
# there's no dependency from kitcar-gazebo-simulation on kitcar-ros.
# However: Whenever the message definition in kitcar ros is changed,
# the changes must also be done here!!

# Message for estimated state data send by state_estimation_node

Header header

float32 speed_x
float32 speed_y
float32 yaw_rate
float32 acceleration
float32 steering_angle_back
float32 steering_angle_front
