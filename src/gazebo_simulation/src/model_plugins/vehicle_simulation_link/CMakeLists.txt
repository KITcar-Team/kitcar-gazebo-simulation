
## Declare a C++ library
 add_library(vehicle_simulation_link
#   src/${PROJECT_NAME}/gazebo_simulation.cpp
   vehicle_simulation_link.cpp
 )
 
## Specify libraries to link a library or executable target against
 target_link_libraries(vehicle_simulation_link
   ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES}
 )
