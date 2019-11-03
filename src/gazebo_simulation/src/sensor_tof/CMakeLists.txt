
##Declare Nodelet
add_executable(sensor_tof
  src/sensor_tof/sensor_tof.cpp
  src/sensor_tof/sensor_tof_node.cpp
)

##Declare Nodelet
add_library(sensor_tof_nodelet
  src/sensor_tof/sensor_tof.cpp
  src/sensor_tof/sensor_tof_node.cpp
)
## set compiler definitions for nodelet
set_target_properties(sensor_tof_nodelet PROPERTIES COMPILE_DEFINITIONS "BUILD_NODELET")

if(${PROJECT_NAME}_gencfg)
  add_dependencies(sensor_tof ${PROJECT_NAME}_gencfg)
  add_dependencies(sensor_tof_nodelet ${PROJECT_NAME}_gencfg)
endif(${PROJECT_NAME}_gencfg)

add_dependencies(sensor_tof ${catkin_EXPORTED_TARGETS})
add_dependencies(sensor_tof_nodelet ${catkin_EXPORTED_TARGETS})

target_link_libraries(sensor_tof
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)
target_link_libraries(sensor_tof_nodelet
  ${catkin_LIBRARIES}
  ${common_LIBRARIES}
)

target_link_libraries(sensor_tof ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${common_LIBRARIES})
target_link_libraries(sensor_tof_nodelet ${catkin_LIBRARIES} ${common_LIBRARIES} ${PCL_LIBRARIES})

