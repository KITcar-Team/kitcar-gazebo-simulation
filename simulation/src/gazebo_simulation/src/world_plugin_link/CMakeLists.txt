## Declare a C++ library
 add_library(world_plugin_link
   world_plugin_link.cpp
 )

add_dependencies(world_plugin_link gazebo_simulation_generate_messages_cpp)

## Specify libraries to link a library or executable target against
 target_link_libraries(world_plugin_link
   ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES}
 )
