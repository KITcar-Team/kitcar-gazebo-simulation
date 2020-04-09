# Used to pass information about the cars current state (e.g. velocity, position, frame, fov, ...)

geometry_msgs/Pose pose
geometry_msgs/Twist twist

#Current 2d frame of the car
geometry_msgs/Polygon frame

#Current field of view of the car in simulation coordinates
geometry_msgs/Polygon view_cone
